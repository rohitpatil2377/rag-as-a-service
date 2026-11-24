package node_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"sovereignnode/grant"
	"sovereignnode/node"
)

// A revoked department's key no longer verifies (PublicKeyFor) and it can no longer be
// recognised as a caller (DeptForPeer) — both fail closed (T2 zero-trust revocation).
func TestRegistry_RevokeDept(t *testing.T) {
	n := newTestNode(t)
	pub := grant.DevPublicKey("s")
	n.Registry.Register("hr", pub, n.PeerID(), "")

	if _, ok := n.Registry.PublicKeyFor("hr"); !ok {
		t.Fatalf("hr key should verify before revocation")
	}
	n.Registry.RevokeDept("hr")
	if !n.Registry.IsDeptRevoked("hr") {
		t.Fatalf("hr should read as revoked")
	}
	if _, ok := n.Registry.PublicKeyFor("hr"); ok {
		t.Fatalf("revoked dept key must not verify")
	}
	if _, ok := n.Registry.DeptForPeer(n.PeerID()); ok {
		t.Fatalf("revoked dept must not be recognised as a caller")
	}
}

// A grant on the revocation list is refused by the responder even though its signature and
// its own (unset) revoked_at would otherwise pass — revocation is enforced here, not trusted
// from the caller's grant bytes.
func TestRegistry_RevokeGrant(t *testing.T) {
	n := newTestNode(t)
	if n.Registry.IsGrantRevoked("g-1") {
		t.Fatalf("grant should not be revoked initially")
	}
	n.Registry.RevokeGrant("g-1")
	if !n.Registry.IsGrantRevoked("g-1") {
		t.Fatalf("g-1 should read as revoked")
	}
}

// The operator registry file carries revocations: a `revoked` department and a revoked_grants
// list are applied on load.
func TestApplyRegistryFile_Revocations(t *testing.T) {
	fin := newTestNode(t)
	path := filepath.Join(t.TempDir(), "registry.json")
	rf := node.RegistryFile{
		Departments: []node.RegistryEntry{
			{Dept: "hr", DevSecret: "s"},
			{Dept: "legal", DevSecret: "s2", Revoked: true},
		},
		RevokedGrants: []string{"g-bad", ""},
	}
	raw, _ := json.Marshal(rf)
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := fin.ApplyRegistryFile(t.Context(), path); err != nil {
		t.Fatalf("apply: %v", err)
	}
	// hr is live; legal is revoked; g-bad is on the grant revocation list.
	if _, ok := fin.Registry.PublicKeyFor("hr"); !ok {
		t.Fatalf("hr should be live")
	}
	if _, ok := fin.Registry.PublicKeyFor("legal"); ok {
		t.Fatalf("revoked legal must not verify")
	}
	if !fin.Registry.IsGrantRevoked("g-bad") {
		t.Fatalf("g-bad should be revoked from the file")
	}
	if fin.Registry.IsGrantRevoked("") {
		t.Fatalf("empty grant id must not be added")
	}
}
