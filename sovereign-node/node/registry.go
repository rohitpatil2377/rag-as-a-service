// Package node builds and wires the libp2p transport for the SovereignNode: the host
// itself (identity, listen addrs, bootstrap dialling) and the department registry that
// maps a department id to the Ed25519 public key its grants are signed with plus its
// libp2p reachability.
package node

import (
	"crypto/ed25519"
	"sync"

	"github.com/libp2p/go-libp2p/core/peer"
)

// DeptRecord is what the registry knows about a department: the Ed25519 public key its
// grants are signed with (so a receiving node can verify grant.Verify against it), and
// its libp2p identity/addresses (so we can dial it and map an inbound peer id back to a
// department).
type DeptRecord struct {
	// DeptID is the department identifier (matches grant.IssuerDeptID / SubjectDeptID).
	DeptID string
	// PublicKey is the department's grant-signing Ed25519 public key.
	PublicKey ed25519.PublicKey
	// PeerID is the department node's libp2p peer id (may be empty if only the pubkey
	// is known — e.g. we can verify its grants but haven't learned its address yet).
	PeerID peer.ID
	// Addr is a dialable multiaddr string for the department node (optional).
	Addr string
}

// Registry is an in-memory, concurrency-safe directory of departments. A receiving node
// consults it to (a) find the issuer public key to verify an inbound grant against, and
// (b) map an inbound stream's remote peer id back to a department id to enforce that the
// grant's subject matches who is actually calling.
type Registry struct {
	mu     sync.RWMutex
	byDept map[string]*DeptRecord
	byPeer map[peer.ID]*DeptRecord
	// revokedDepts: a compromised department whose key must no longer verify grants or call.
	// revokedGrants: individual grant ids the RESPONDER refuses even if presented with a still
	// signature-valid copy — revocation is NOT in the signature, so it is enforced here, not
	// trusted from the caller's grant bytes (a caller can hold the pre-revocation signed grant).
	revokedDepts  map[string]bool
	revokedGrants map[string]bool
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{
		byDept:        make(map[string]*DeptRecord),
		byPeer:        make(map[peer.ID]*DeptRecord),
		revokedDepts:  make(map[string]bool),
		revokedGrants: make(map[string]bool),
	}
}

// Register records (or updates) a department's grant-signing public key and, optionally,
// its libp2p peer id and dialable multiaddr. Passing an empty peerID/addr just records
// the public key. A copy of pubkey is stored so later mutation of the caller's slice
// cannot corrupt the registry.
func (r *Registry) Register(dept string, pubkey ed25519.PublicKey, peerID peer.ID, addr string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	rec := r.byDept[dept]
	if rec == nil {
		rec = &DeptRecord{DeptID: dept}
		r.byDept[dept] = rec
	}
	if len(pubkey) > 0 {
		rec.PublicKey = append(ed25519.PublicKey(nil), pubkey...)
	}
	if peerID != "" {
		// If this dept was previously indexed under a different peer id, drop the stale
		// reverse mapping so byPeer never points at the wrong department.
		if rec.PeerID != "" && rec.PeerID != peerID {
			delete(r.byPeer, rec.PeerID)
		}
		rec.PeerID = peerID
		r.byPeer[peerID] = rec
	}
	if addr != "" {
		rec.Addr = addr
	}
}

// PublicKeyFor returns the grant-signing public key registered for dept. The bool is
// false when the department is unknown or has no public key yet — callers MUST fail
// closed (reject the grant) in that case.
func (r *Registry) PublicKeyFor(dept string) (ed25519.PublicKey, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	rec := r.byDept[dept]
	if rec == nil || len(rec.PublicKey) == 0 || r.revokedDepts[dept] {
		return nil, false
	}
	return rec.PublicKey, true
}

// Lookup returns a copy of the full record for dept.
func (r *Registry) Lookup(dept string) (DeptRecord, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	rec := r.byDept[dept]
	if rec == nil {
		return DeptRecord{}, false
	}
	return *rec, true
}

// DeptForPeer maps an inbound stream's remote peer id back to the department id it was
// registered under. This is how the query handler proves the caller's department matches
// the grant's subject_dept_id. The bool is false for an unknown peer.
func (r *Registry) DeptForPeer(p peer.ID) (string, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	rec := r.byPeer[p]
	if rec == nil || r.revokedDepts[rec.DeptID] {
		return "", false
	}
	return rec.DeptID, true
}

// RevokeDept marks a department compromised: its key stops verifying grants (PublicKeyFor)
// and it can no longer be recognised as a caller (DeptForPeer). Both fail closed.
func (r *Registry) RevokeDept(dept string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.revokedDepts[dept] = true
}

// IsDeptRevoked reports whether dept has been revoked.
func (r *Registry) IsDeptRevoked(dept string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.revokedDepts[dept]
}

// RevokeGrant adds a grant id to the responder-enforced revocation list. A grant on this
// list is refused even if the caller presents a signature-valid copy (revocation is not in
// the signature) — this is how a revoked grant is honoured across the mesh.
func (r *Registry) RevokeGrant(grantID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.revokedGrants[grantID] = true
}

// IsGrantRevoked reports whether grantID is on the revocation list.
func (r *Registry) IsGrantRevoked(grantID string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.revokedGrants[grantID]
}
