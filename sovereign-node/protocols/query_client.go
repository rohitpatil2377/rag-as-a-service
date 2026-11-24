package protocols

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
)

// maxResponseBytes bounds an inbound federated RESPONSE. A response carries spans (text), so
// it is allowed to be larger than a request, but still bounded so a hostile or buggy peer
// can't exhaust memory.
const maxResponseBytes = 8 << 20 // 8 MiB

// QueryPeer is the OUTBOUND counterpart to handleQuery: it dials peer p on QueryProtocol,
// sends req, and returns the peer's FederatedQueryResponse.
//
// Layering: this is exactly "federate the query, not the vector" (ADR-0019) — we ship
// req.QueryText and the signed grant; the peer re-embeds and searches in ITS OWN space and
// returns spans. A raw vector is never sent across a heterogeneous-embedder boundary.
//
// Error semantics mirror the mesh's fail-closed posture: a dial/encode/decode failure is a
// transport error and is returned to the caller (availability). An AUTHORISATION failure is
// NOT an error here — the peer replies with a well-formed {GrantValid:false, Spans:[]} which
// the caller inspects. The whole exchange is bounded by retrieveTimeout.
// newNonce returns a random 128-bit hex token for single-use replay protection.
func newNonce() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		// crypto/rand failure is catastrophic; a time-based fallback still varies per call.
		return fmt.Sprintf("ts-%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}

func QueryPeer(ctx context.Context, h host.Host, p peer.ID, req *FederatedQueryRequest) (*FederatedQueryResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, retrieveTimeout)
	defer cancel()

	// Stamp single-use replay-protection fields for the responder (freshness + nonce).
	if req.Nonce == "" {
		req.Nonce = newNonce()
	}
	if req.IssuedAt == "" {
		req.IssuedAt = time.Now().UTC().Format(time.RFC3339Nano)
	}

	stream, err := h.NewStream(ctx, p, QueryProtocol)
	if err != nil {
		return nil, fmt.Errorf("query: open stream to %s: %w", p, err)
	}
	defer stream.Close()

	_ = stream.SetWriteDeadline(time.Now().Add(retrieveTimeout))
	if err := json.NewEncoder(stream).Encode(req); err != nil {
		return nil, fmt.Errorf("query: encode request to %s: %w", p, err)
	}
	// Half-close the write side so the responder sees EOF after exactly one request.
	_ = stream.CloseWrite()

	_ = stream.SetReadDeadline(time.Now().Add(retrieveTimeout))
	var resp FederatedQueryResponse
	if err := json.NewDecoder(io.LimitReader(stream, maxResponseBytes)).Decode(&resp); err != nil {
		return nil, fmt.Errorf("query: decode response from %s: %w", p, err)
	}
	return &resp, nil
}
