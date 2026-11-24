package node_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"

	"sovereignnode/node"
	"sovereignnode/protocols"
)

func newTestNode(t *testing.T) *node.Node {
	t.Helper()
	n, err := node.New("finance", "/ip4/127.0.0.1/tcp/0", "")
	if err != nil {
		t.Fatalf("new node: %v", err)
	}
	t.Cleanup(func() { _ = n.Close() })
	return n
}

func postFederate(t *testing.T, h http.Handler, body map[string]any) protocols.FederatedQueryResponse {
	t.Helper()
	raw, _ := json.Marshal(body)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/federate", bytes.NewReader(raw)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d", rec.Code)
	}
	var resp protocols.FederatedQueryResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return resp
}

// A known, routed peer: the control server dials it (stubbed) and returns its spans.
func TestControlFederate_KnownPeerReturnsSpans(t *testing.T) {
	n := newTestNode(t)
	n.Registry.Register("hr", nil, n.PeerID(), "") // any non-empty PeerID makes it "routed"

	stub := func(_ context.Context, _ host.Host, _ peer.ID, _ *protocols.FederatedQueryRequest) (*protocols.FederatedQueryResponse, error) {
		return &protocols.FederatedQueryResponse{
			GrantValid: true, PeerDeptID: "hr",
			Spans: []protocols.FederatedSpan{{ChunkID: "c1", Text: "x", Score: 0.9, DocID: "d1"}},
		}, nil
	}
	h := node.NewControlServer(n, stub).Handler()

	resp := postFederate(t, h, map[string]any{"peer_dept_id": "hr", "query_text": "q", "top_k": 5})
	if !resp.GrantValid || len(resp.Spans) != 1 || resp.Spans[0].ChunkID != "c1" {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

// An unknown peer must fail closed WITHOUT ever dialling.
func TestControlFederate_UnknownPeerFailsClosed(t *testing.T) {
	n := newTestNode(t)
	called := false
	stub := func(_ context.Context, _ host.Host, _ peer.ID, _ *protocols.FederatedQueryRequest) (*protocols.FederatedQueryResponse, error) {
		called = true
		return &protocols.FederatedQueryResponse{GrantValid: true}, nil
	}
	h := node.NewControlServer(n, stub).Handler()

	resp := postFederate(t, h, map[string]any{"peer_dept_id": "nobody", "query_text": "q", "top_k": 5})
	if resp.GrantValid || resp.Error != "peer_unknown" {
		t.Fatalf("expected fail-closed peer_unknown, got %+v", resp)
	}
	if called {
		t.Fatalf("querier must not be called for an unknown peer")
	}
}

// A dial failure is availability, not authorisation: no spans, no 500.
func TestControlFederate_DialFailureNoSpans(t *testing.T) {
	n := newTestNode(t)
	n.Registry.Register("hr", nil, n.PeerID(), "")
	stub := func(_ context.Context, _ host.Host, _ peer.ID, _ *protocols.FederatedQueryRequest) (*protocols.FederatedQueryResponse, error) {
		return nil, context.DeadlineExceeded
	}
	h := node.NewControlServer(n, stub).Handler()

	resp := postFederate(t, h, map[string]any{"peer_dept_id": "hr", "query_text": "q", "top_k": 5})
	if resp.GrantValid || resp.Error != "dial_failed" || len(resp.Spans) != 0 {
		t.Fatalf("expected fail-open availability (no spans), got %+v", resp)
	}
}
