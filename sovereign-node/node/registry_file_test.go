package node_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"sovereignnode/grant"
	"sovereignnode/node"
)

func writeRegistry(t *testing.T, rf node.RegistryFile) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "registry.json")
	raw, _ := json.Marshal(rf)
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatalf("write registry: %v", err)
	}
	return path
}

// A registry file with a dev-secret key AND a live peer address registers the key, the peer
// id, and actually connects the two nodes.
func TestApplyRegistryFile_DevSecretAndDial(t *testing.T) {
	ctx := context.Background()
	hr := newTestNode(t) // dept "finance" per newTestNode; use as the PEER here
	fin, err := node.New("finance-caller", "/ip4/127.0.0.1/tcp/0", "")
	if err != nil {
		t.Fatalf("caller node: %v", err)
	}
	t.Cleanup(func() { _ = fin.Close() })

	peerAddr := fmt.Sprintf("%s/p2p/%s", hr.Host.Addrs()[0], hr.PeerID())
	path := writeRegistry(t, node.RegistryFile{Departments: []node.RegistryEntry{
		{Dept: "hr", DevSecret: grant.DefaultDevJWTSecret, PeerAddr: peerAddr}, // dev key + dial
		{Dept: "finance-caller", DevSecret: "x"},                               // own dept -> skipped
	}})

	loaded, err := fin.ApplyRegistryFile(ctx, path)
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if loaded != 1 {
		t.Fatalf("loaded: got %d want 1 (own dept skipped)", loaded)
	}
	// Key learned (dev default secret) and it verifies as expected.
	got, ok := fin.Registry.PublicKeyFor("hr")
	if !ok || !got.Equal(grant.DevPublicKey("")) {
		t.Fatalf("hr key not registered from dev secret")
	}
	// Peer id + reverse mapping learned, and the two hosts are actually connected.
	rec, ok := fin.Registry.Lookup("hr")
	if !ok || rec.PeerID != hr.PeerID() {
		t.Fatalf("hr peer id not registered: %+v", rec)
	}
	if dept, ok := fin.Registry.DeptForPeer(hr.PeerID()); !ok || dept != "hr" {
		t.Fatalf("reverse peer->dept mapping missing")
	}
	if fin.Host.Network().Connectedness(hr.PeerID()).String() != "Connected" {
		t.Fatalf("nodes did not connect via the registry file")
	}
}

// An explicit base64 public key is accepted and takes precedence.
func TestApplyRegistryFile_PubKeyB64(t *testing.T) {
	fin := newTestNode(t)
	pub := grant.DevPublicKey("some-secret")
	path := writeRegistry(t, node.RegistryFile{Departments: []node.RegistryEntry{
		{Dept: "legal", PubKeyB64: base64.StdEncoding.EncodeToString(pub)},
	}})
	if _, err := fin.ApplyRegistryFile(context.Background(), path); err != nil {
		t.Fatalf("apply: %v", err)
	}
	got, ok := fin.Registry.PublicKeyFor("legal")
	if !ok || !got.Equal(pub) {
		t.Fatalf("legal pubkey_b64 not registered")
	}
}

// A malformed key is skipped without stopping the other entries; the file itself parses.
func TestApplyRegistryFile_SkipsBadEntry(t *testing.T) {
	fin := newTestNode(t)
	path := writeRegistry(t, node.RegistryFile{Departments: []node.RegistryEntry{
		{Dept: "bad", PubKeyB64: "!!!not-base64!!!"},
		{Dept: "good", DevSecret: "s"},
	}})
	loaded, err := fin.ApplyRegistryFile(context.Background(), path)
	if err != nil {
		t.Fatalf("apply should not error on a bad entry: %v", err)
	}
	if loaded != 1 {
		t.Fatalf("loaded: got %d want 1 (bad skipped)", loaded)
	}
	if _, ok := fin.Registry.PublicKeyFor("good"); !ok {
		t.Fatalf("good entry should have loaded")
	}
	if _, ok := fin.Registry.PublicKeyFor("bad"); ok {
		t.Fatalf("bad entry must not register a key")
	}
}

func TestLoadRegistryFile_BadPath(t *testing.T) {
	if _, err := node.LoadRegistryFile(filepath.Join(t.TempDir(), "nope.json")); err == nil {
		t.Fatalf("expected error for missing file")
	}
}
