package node

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"

	"sovereignnode/grant"
	"sovereignnode/protocols"
)

// federateRequest is the LOCAL control-API body. The department's own rag-api / Python
// MeshTransport POSTs it to ask THIS node to run an OUTBOUND federated query against a peer.
type federateRequest struct {
	PeerDeptID  string      `json:"peer_dept_id"`
	QueryText   string      `json:"query_text"`
	TopK        int         `json:"top_k"`
	Grant       grant.Grant `json:"grant"`
	GrantToken  string      `json:"grant_token,omitempty"`
	KBID        string      `json:"kb_id,omitempty"`
	Topics      []string    `json:"topics,omitempty"`
	Traceparent string      `json:"traceparent,omitempty"`
}

// PeerQuerier dials a peer and runs a federated query. It is protocols.QueryPeer in
// production; tests inject a fake so no real dialling happens.
type PeerQuerier func(
	ctx context.Context, h host.Host, p peer.ID, req *protocols.FederatedQueryRequest,
) (*protocols.FederatedQueryResponse, error)

// ControlServer is the node's LOCAL (loopback) HTTP control plane — the OUTBOUND counterpart
// to the inbound libp2p handlers. The department's own rag-api / MeshTransport POSTs
// /federate here and the node dials the named peer over the mesh, shipping the QUERY (never a
// vector — ADR-0019). It MUST bind loopback/intranet only; it is not a mesh endpoint and
// carries no auth of its own (it acts on behalf of the co-located rag-api, guarded by the
// pod's NetworkPolicy).
type ControlServer struct {
	node    *Node
	querier PeerQuerier
}

// NewControlServer builds a control server for n. Pass protocols.QueryPeer as querier in
// production.
func NewControlServer(n *Node, querier PeerQuerier) *ControlServer {
	return &ControlServer{node: n, querier: querier}
}

// Handler exposes the control routes.
func (cs *ControlServer) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/federate", cs.handleFederate)
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]string{"status": "ok", "dept": cs.node.DeptID})
	})
	return mux
}

// Serve runs the control server on addr until ctx is cancelled.
func (cs *ControlServer) Serve(ctx context.Context, addr string) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("control: listen %s: %w", addr, err)
	}
	srv := &http.Server{Handler: cs.Handler(), ReadHeaderTimeout: 5 * time.Second}
	go func() {
		<-ctx.Done()
		shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutCtx)
	}()
	if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
		return err
	}
	return nil
}

func (cs *ControlServer) handleFederate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	var body federateRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20)).Decode(&body); err != nil {
		writeJSON(w, protocols.FederatedQueryResponse{GrantValid: false, Error: "bad_request"})
		return
	}

	rec, ok := cs.node.Registry.Lookup(body.PeerDeptID)
	if !ok || rec.PeerID == "" {
		// Unknown or unrouted peer: fail closed with no spans (never a 500).
		writeJSON(w, protocols.FederatedQueryResponse{
			GrantValid: false, PeerDeptID: body.PeerDeptID, Error: "peer_unknown",
		})
		return
	}

	token := body.GrantToken
	if token == "" {
		token = body.Grant.SignedToken
	}
	req := &protocols.FederatedQueryRequest{
		Grant:       body.Grant,
		GrantToken:  token,
		QueryText:   body.QueryText,
		TopK:        body.TopK,
		Topics:      body.Topics,
		KBID:        body.KBID,
		Traceparent: body.Traceparent,
	}

	resp, err := cs.querier(r.Context(), cs.node.Host, rec.PeerID, req)
	if err != nil {
		// Dial/transport failure is availability, not authorisation: no spans, no 500.
		writeJSON(w, protocols.FederatedQueryResponse{
			GrantValid: false, PeerDeptID: body.PeerDeptID, Error: "dial_failed",
		})
		return
	}
	writeJSON(w, resp)
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
