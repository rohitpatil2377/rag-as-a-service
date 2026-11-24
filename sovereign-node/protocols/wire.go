package protocols

import "sovereignnode/grant"

// Protocol IDs for the libp2p streams. Versioned so peers can negotiate.
const (
	// QueryProtocol carries a FederatedQueryRequest and returns a FederatedQueryResponse.
	QueryProtocol = "/rag/query/1.0"
	// GrantProtocol exchanges/announces signed grant tokens between peers.
	GrantProtocol = "/rag/grant/1.0"
)

// FederatedQueryRequest is sent by an initiating node to a peer over the query stream.
//
// It carries the FULL grant object plus the detached base64 signature in GrantToken so
// the receiving peer can verify the signature against the issuer's known public key
// WITHOUT any shared state — the grant is self-describing and self-authenticating.
type FederatedQueryRequest struct {
	// GrantToken is the base64 Ed25519 signature over the grant's canonical claims.
	// It is copied into Grant.SignedToken before verification (they must agree).
	GrantToken string `json:"grant_token"`
	// Grant is the full grant object being presented as authorisation.
	Grant grant.Grant `json:"grant"`
	// QueryText is the natural-language query to retrieve against.
	QueryText string `json:"query_text"`
	// QueryVector is an optional dense query vector (reserved; the rag bridge currently
	// retrieves by text, but the field is carried for future vector-only federation).
	QueryVector []float32 `json:"query_vector,omitempty"`
	// TopK is the requested number of spans (capped to the grant's max_results_per_query).
	TopK int `json:"top_k"`
	// Topics the caller wants to query; must be a subset of the grant's topics.
	Topics []string `json:"topics,omitempty"`
	// KBID is the specific knowledge base being queried; must be in the grant's kb_ids.
	// (Not in the Python FederatedQueryRequest, but required to pick a KB on this side;
	// if empty the node uses the grant's first kb_id.)
	KBID string `json:"kb_id,omitempty"`
	// Nonce is a single-use random token; IssuedAt is the RFC3339 UTC send time. Together they
	// give the responder replay protection (freshness window + seen-nonce cache) on top of the
	// grant. Stamped by the initiator's node on the outbound hop (QueryPeer).
	Nonce    string `json:"nonce,omitempty"`
	IssuedAt string `json:"issued_at,omitempty"`
	// Traceparent is the initiator's W3C trace context, propagated so the peer's rag-api
	// retrieval joins the SAME distributed trace (observability across the federation hop).
	Traceparent string `json:"traceparent,omitempty"`
}

// FederatedSpan mirrors rag_schemas.federation.FederatedSpan.
type FederatedSpan struct {
	ChunkID      string  `json:"chunk_id"`
	Text         string  `json:"text"`
	Score        float64 `json:"score"`
	DocID        string  `json:"doc_id"`
	KBID         string  `json:"kb_id"`
	SourceDeptID string  `json:"source_dept_id"`
	// Category is the span's primary FR-B7 taxonomy tag. Carried so the initiator can apply
	// grant-topic enforcement (ADR-0017 L1) to mesh spans, matching the local path.
	Category string `json:"category,omitempty"`
}

// FederatedQueryResponse is returned by a peer after local retrieval. On ANY grant
// failure it MUST be {GrantValid:false, Spans:[]} — never a transport error/500. It
// fails closed silently.
type FederatedQueryResponse struct {
	Spans      []FederatedSpan `json:"spans"`
	PeerDeptID string          `json:"peer_dept_id"`
	GrantValid bool            `json:"grant_valid"`
	Error      string          `json:"error,omitempty"`
}

// GrantAnnounce is the payload on the grant-exchange stream: a peer announcing a signed
// grant it holds (either one it issued, or one it received) so the mesh can gossip
// authorisations. The receiver verifies the signature before trusting it.
type GrantAnnounce struct {
	// FromDeptID is the announcing peer's department id.
	FromDeptID string `json:"from_dept_id"`
	// Grant is the full, signed grant being announced.
	Grant grant.Grant `json:"grant"`
}

// GrantAck is the response to a GrantAnnounce.
type GrantAck struct {
	Accepted bool   `json:"accepted"`
	Reason   string `json:"reason,omitempty"`
}
