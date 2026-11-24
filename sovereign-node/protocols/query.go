package protocols

import (
	"context"
	"crypto/ed25519"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"

	"sovereignnode/audit"
	"sovereignnode/bridge"
)

// maxRequestBytes bounds how much we read off an inbound stream before giving up, so a
// hostile peer can't exhaust memory with an unbounded JSON body.
const maxRequestBytes = 4 << 20 // 4 MiB

// retrieveTimeout bounds a single local rag retrieval triggered by a federated query and
// doubles as the stream read/write deadline.
const retrieveTimeout = 25 * time.Second

// Retriever is the slice of the bridge client the query handler needs. Defined as an
// interface so tests can inject a fake without a live rag-api. *bridge.Client satisfies
// it.
type Retriever interface {
	Retrieve(ctx context.Context, kbID, query string, topK int) (*bridge.RetrieveResponse, error)
}

// DeptDirectory is the slice of node.Registry the handlers need: resolve an issuer's
// grant-signing public key, and map an inbound peer id back to a department id. Declared
// here (not imported from node) to avoid an import cycle node -> protocols -> node.
// *node.Registry satisfies it.
type DeptDirectory interface {
	PublicKeyFor(dept string) (ed25519.PublicKey, bool)
	DeptForPeer(p peer.ID) (string, bool)
	// IsGrantRevoked reports whether a grant id is on the responder-enforced revocation list.
	IsGrantRevoked(grantID string) bool
}

// QueryService owns the state the /rag/query/1.0 handler needs.
type QueryService struct {
	// LocalDept is THIS node's department id — the peer_dept_id stamped on responses and
	// the department whose local KB we retrieve from.
	LocalDept string
	// Registry resolves issuer public keys and maps peers to departments.
	Registry DeptDirectory
	// Bridge retrieves spans from the LOCAL rag-api when a grant checks out.
	Bridge Retriever
	// Replay, when set, rejects stale or replayed queries (freshness + seen-nonce). nil
	// disables the check (used by unit tests that don't exercise replay).
	Replay *ReplayGuard
}

// RegisterQueryHandler installs the inbound stream handler for QueryProtocol on h.
func RegisterQueryHandler(h host.Host, svc *QueryService) {
	h.SetStreamHandler(QueryProtocol, svc.handleQuery)
}

// handleQuery is the fail-closed core. It NEVER writes a transport error to the wire: on
// any authorisation failure it replies with GrantValid:false + empty spans and returns.
func (s *QueryService) handleQuery(stream network.Stream) {
	defer stream.Close()

	remote := stream.Conn().RemotePeer()
	_ = stream.SetReadDeadline(time.Now().Add(retrieveTimeout))

	var req FederatedQueryRequest
	dec := json.NewDecoder(io.LimitReader(stream, maxRequestBytes))
	if err := dec.Decode(&req); err != nil {
		// Malformed request — fail closed, do not leak details.
		s.writeResponse(stream, s.denied())
		return
	}

	s.writeResponse(stream, s.authorizeAndRetrieve(remote, &req))
}

func (s *QueryService) denied() FederatedQueryResponse {
	return FederatedQueryResponse{Spans: nil, PeerDeptID: s.LocalDept, GrantValid: false}
}

// authorizeAndRetrieve runs every fail-closed check in order and, only if ALL pass,
// performs the local retrieval. It never returns an error — every failure is a denied
// response.
func (s *QueryService) authorizeAndRetrieve(remote peer.ID, req *FederatedQueryRequest) FederatedQueryResponse {
	g := req.Grant
	// The request carries the detached signature in GrantToken; it must agree with the
	// grant's SignedToken field. Copy it in so both are consistent before verifying.
	if req.GrantToken != "" {
		g.SignedToken = req.GrantToken
	}

	// 1. Issuer public key must be known to us.
	pub, ok := s.Registry.PublicKeyFor(g.IssuerDeptID)
	if !ok || len(pub) == 0 {
		fmt.Fprintf(os.Stderr, "query: check1 FAILED pubkey lookup: local_dept=%s issuer_dept=%s found=%v keylen=%d grant_id=%s\n", s.LocalDept, g.IssuerDeptID, ok, len(pub), g.GrantID)
		audit.Query(audit.Decision{ResponderDept: s.LocalDept, RemotePeer: remote.String(), IssuerDept: g.IssuerDeptID, SubjectDept: g.SubjectDeptID, GrantID: g.GrantID, Topics: req.Topics, Allowed: false, Reason: "deny_pubkey"})
		return s.denied()
	}

	// 2. Grant signature must verify against the issuer's public key.
	if !g.Verify(pub) {
		fmt.Fprintf(os.Stderr, "query: check2 FAILED signature verify: local_dept=%s issuer_dept=%s subject_dept=%s grant_id=%s signed_token_len=%d issuer_pub=%x\n", s.LocalDept, g.IssuerDeptID, g.SubjectDeptID, g.GrantID, len(g.SignedToken), pub)
		audit.Query(audit.Decision{ResponderDept: s.LocalDept, RemotePeer: remote.String(), IssuerDept: g.IssuerDeptID, SubjectDept: g.SubjectDeptID, GrantID: g.GrantID, Topics: req.Topics, Allowed: false, Reason: "deny_signature"})
		return s.denied()
	}

	// 2b. Revocation is not in the signature, so a caller can present a still-valid-looking
	//     copy of a revoked grant. The RESPONDER enforces revocation from its own list.
	if s.Registry.IsGrantRevoked(g.GrantID) {
		fmt.Fprintf(os.Stderr, "query: check2b FAILED grant revoked: local_dept=%s grant_id=%s\n", s.LocalDept, g.GrantID)
		audit.Query(audit.Decision{ResponderDept: s.LocalDept, RemotePeer: remote.String(), IssuerDept: g.IssuerDeptID, SubjectDept: g.SubjectDeptID, GrantID: g.GrantID, Topics: req.Topics, Allowed: false, Reason: "deny_revoked"})
		return s.denied()
	}

	// 3. Grant must be active (not revoked, within [not_before, expires_at)).
	if !g.IsActive() {
		fmt.Fprintf(os.Stderr, "query: check3 FAILED grant not active: local_dept=%s grant_id=%s not_before=%s expires_at=%s revoked_at=%v now=%s\n", s.LocalDept, g.GrantID, g.NotBefore.UTC(), g.ExpiresAt.UTC(), g.RevokedAt, time.Now().UTC())
		audit.Query(audit.Decision{ResponderDept: s.LocalDept, RemotePeer: remote.String(), IssuerDept: g.IssuerDeptID, SubjectDept: g.SubjectDeptID, GrantID: g.GrantID, Topics: req.Topics, Allowed: false, Reason: "deny_inactive"})
		return s.denied()
	}

	// 4. The grant's subject MUST be the department of the peer actually calling us.
	//    Prevents a peer replaying a grant issued to a different department.
	callerDept, ok := s.Registry.DeptForPeer(remote)
	if !ok || callerDept == "" || callerDept != g.SubjectDeptID {
		fmt.Fprintf(os.Stderr, "query: check4 FAILED caller dept mismatch: local_dept=%s remote_peer=%s dept_for_peer_found=%v caller_dept=%q grant_subject_dept=%q grant_id=%s\n", s.LocalDept, remote, ok, callerDept, g.SubjectDeptID, g.GrantID)
		audit.Query(audit.Decision{ResponderDept: s.LocalDept, RemotePeer: remote.String(), CallerDept: callerDept, IssuerDept: g.IssuerDeptID, SubjectDept: g.SubjectDeptID, GrantID: g.GrantID, Topics: req.Topics, Allowed: false, Reason: "deny_subject_mismatch"})
		return s.denied()
	}

	// 4b. Replay protection: the request must be fresh and its nonce unseen (defence in
	//     depth over the grant + the peer-authenticated stream). Fail closed.
	if s.Replay != nil {
		if err := s.Replay.Check(g.SubjectDeptID, req.Nonce, req.IssuedAt); err != nil {
			fmt.Fprintf(os.Stderr, "query: replay rejected from %s: %v\n", callerDept, err)
			audit.Query(audit.Decision{ResponderDept: s.LocalDept, RemotePeer: remote.String(), CallerDept: callerDept, IssuerDept: g.IssuerDeptID, SubjectDept: g.SubjectDeptID, GrantID: g.GrantID, Topics: req.Topics, Allowed: false, Reason: "deny_replay"})
			return s.denied()
		}
	}

	// 5. Requested topics must be a subset of the grant's topics.
	if len(req.Topics) > 0 && !g.AllowsTopics(req.Topics) {
		fmt.Fprintf(os.Stderr, "query: check5 FAILED topics not allowed: local_dept=%s requested_topics=%v grant_topics=%v grant_id=%s\n", s.LocalDept, req.Topics, g.Topics, g.GrantID)
		audit.Query(audit.Decision{ResponderDept: s.LocalDept, RemotePeer: remote.String(), CallerDept: callerDept, IssuerDept: g.IssuerDeptID, SubjectDept: g.SubjectDeptID, GrantID: g.GrantID, Topics: req.Topics, Allowed: false, Reason: "deny_topics"})
		return s.denied()
	}

	// 6. Resolve the KB to retrieve from and enforce kb ∈ grant.KBIDs.
	kbID := req.KBID
	if kbID == "" {
		if len(g.KBIDs) == 0 {
			fmt.Fprintf(os.Stderr, "query: check6 FAILED no kb_id in request and grant has no kb_ids: local_dept=%s grant_id=%s\n", s.LocalDept, g.GrantID)
			audit.Query(audit.Decision{ResponderDept: s.LocalDept, RemotePeer: remote.String(), CallerDept: callerDept, IssuerDept: g.IssuerDeptID, SubjectDept: g.SubjectDeptID, GrantID: g.GrantID, Topics: req.Topics, Allowed: false, Reason: "deny_kb"})
			return s.denied()
		}
		kbID = g.KBIDs[0]
	}
	if !g.AllowsKB(kbID) {
		fmt.Fprintf(os.Stderr, "query: check6 FAILED kb not allowed: local_dept=%s requested_kb=%q grant_kb_ids=%v grant_id=%s\n", s.LocalDept, kbID, g.KBIDs, g.GrantID)
		audit.Query(audit.Decision{ResponderDept: s.LocalDept, RemotePeer: remote.String(), CallerDept: callerDept, IssuerDept: g.IssuerDeptID, SubjectDept: g.SubjectDeptID, GrantID: g.GrantID, KBID: kbID, Topics: req.Topics, Allowed: false, Reason: "deny_kb"})
		return s.denied()
	}

	// All checks passed. Retrieve, capped at the grant's max_results_per_query.
	topK := req.TopK
	if g.MaxResultsPerQuery > 0 && (topK <= 0 || topK > g.MaxResultsPerQuery) {
		topK = g.MaxResultsPerQuery
	}
	if topK <= 0 {
		topK = 1
	}

	ctx, cancel := context.WithTimeout(context.Background(), retrieveTimeout)
	defer cancel()
	ctx = bridge.WithTraceparent(ctx, req.Traceparent)

	out, err := s.Bridge.Retrieve(ctx, kbID, req.QueryText, topK)
	if err != nil {
		// Grant was valid but local retrieval failed — report honestly (grant_valid:true)
		// with an opaque error and no spans, so the caller can tell "not allowed" apart
		// from "couldn't fetch".
		fmt.Fprintf(os.Stderr, "query: retrieve kb=%s: %v\n", kbID, err)
		audit.Query(audit.Decision{ResponderDept: s.LocalDept, RemotePeer: remote.String(), CallerDept: callerDept, IssuerDept: g.IssuerDeptID, SubjectDept: g.SubjectDeptID, GrantID: g.GrantID, KBID: kbID, Topics: req.Topics, Allowed: false, Reason: "retrieval_failed"})
		return FederatedQueryResponse{Spans: nil, PeerDeptID: s.LocalDept, GrantValid: true, Error: "retrieval_failed"}
	}

	spans := make([]FederatedSpan, 0, len(out.Spans))
	for _, sp := range out.Spans {
		spans = append(spans, FederatedSpan{
			ChunkID:      sp.ChunkID,
			Text:         sp.Text,
			Score:        sp.Score,
			DocID:        sp.DocID,
			KBID:         kbID,
			SourceDeptID: s.LocalDept,
			Category:     sp.Category,
		})
	}
	audit.Query(audit.Decision{ResponderDept: s.LocalDept, RemotePeer: remote.String(), CallerDept: callerDept, IssuerDept: g.IssuerDeptID, SubjectDept: g.SubjectDeptID, GrantID: g.GrantID, KBID: kbID, Topics: req.Topics, Allowed: true, Reason: "allow", SpanCount: len(spans)})
	return FederatedQueryResponse{Spans: spans, PeerDeptID: s.LocalDept, GrantValid: true}
}

func (s *QueryService) writeResponse(stream network.Stream, resp FederatedQueryResponse) {
	_ = stream.SetWriteDeadline(time.Now().Add(retrieveTimeout))
	if err := json.NewEncoder(stream).Encode(resp); err != nil {
		fmt.Fprintf(os.Stderr, "query: write response: %v\n", err)
	}
}
