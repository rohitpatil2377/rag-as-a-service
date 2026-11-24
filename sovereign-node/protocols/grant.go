package protocols

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"

	"sovereignnode/grant"
)

// GrantStore holds grants this node has received (and, if it is the subject, may later
// present). It is concurrency-safe and keyed by grant id.
type GrantStore struct {
	mu     sync.RWMutex
	grants map[string]grant.Grant
}

// NewGrantStore returns an empty store.
func NewGrantStore() *GrantStore {
	return &GrantStore{grants: make(map[string]grant.Grant)}
}

// Put stores (or replaces) a grant by its grant id.
func (s *GrantStore) Put(g grant.Grant) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.grants[g.GrantID] = g
}

// Get returns the stored grant for id.
func (s *GrantStore) Get(id string) (grant.Grant, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	g, ok := s.grants[id]
	return g, ok
}

// List returns a snapshot of all stored grants.
func (s *GrantStore) List() []grant.Grant {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]grant.Grant, 0, len(s.grants))
	for _, g := range s.grants {
		out = append(out, g)
	}
	return out
}

// GrantService owns the state the /rag/grant/1.0 announce handler needs.
type GrantService struct {
	// LocalDept is THIS node's department id. A grant is stored only if we are its
	// subject (or issuer) — we don't hoard grants meant for other departments.
	LocalDept string
	// Registry resolves the announcing issuer's public key to verify the grant signature.
	Registry DeptDirectory
	// Store is where accepted grants land.
	Store *GrantStore
}

// RegisterGrantHandler installs the inbound stream handler for GrantProtocol on h.
func RegisterGrantHandler(h host.Host, svc *GrantService) {
	h.SetStreamHandler(GrantProtocol, svc.handleAnnounce)
}

// handleAnnounce reads a GrantAnnounce, verifies the grant's signature against the
// issuer's public key, and — if this node is the grant's subject or issuer — stores it.
// It replies with a GrantAck. Verification failure is a rejection, never a transport
// error.
func (s *GrantService) handleAnnounce(stream network.Stream) {
	defer stream.Close()

	_ = stream.SetReadDeadline(time.Now().Add(10 * time.Second))

	var ann GrantAnnounce
	dec := json.NewDecoder(io.LimitReader(stream, maxRequestBytes))
	if err := dec.Decode(&ann); err != nil {
		s.writeAck(stream, GrantAck{Accepted: false, Reason: "malformed"})
		return
	}

	g := ann.Grant

	// Verify the signature against the issuer's registered public key.
	pub, ok := s.Registry.PublicKeyFor(g.IssuerDeptID)
	if !ok || len(pub) == 0 {
		s.writeAck(stream, GrantAck{Accepted: false, Reason: "unknown_issuer"})
		return
	}
	if !g.Verify(pub) {
		s.writeAck(stream, GrantAck{Accepted: false, Reason: "bad_signature"})
		return
	}

	// Only store grants that concern us: we are the subject (can present it later) or the
	// issuer (our own grant echoed back). Ignore grants meant for other departments.
	if g.SubjectDeptID != s.LocalDept && g.IssuerDeptID != s.LocalDept {
		s.writeAck(stream, GrantAck{Accepted: false, Reason: "not_subject"})
		return
	}

	s.Store.Put(g)
	s.writeAck(stream, GrantAck{Accepted: true})
}

func (s *GrantService) writeAck(stream network.Stream, ack GrantAck) {
	_ = stream.SetWriteDeadline(time.Now().Add(10 * time.Second))
	if err := json.NewEncoder(stream).Encode(ack); err != nil {
		fmt.Fprintf(os.Stderr, "grant: write ack: %v\n", err)
	}
}
