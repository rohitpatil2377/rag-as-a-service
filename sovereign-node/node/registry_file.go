package node

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"

	"github.com/libp2p/go-libp2p/core/peer"

	"sovereignnode/grant"
)

// RegistryEntry is one department's line in an operator registry file. It gives a node the
// two things it cannot learn on its own about a peer: the grant-signing public key (to verify
// that peer's grants) and the libp2p address (to dial it). Both are optional per entry —
// supply whichever this node needs for that peer:
//
//   - PubKeyB64 — the base64 (std) raw 32-byte Ed25519 public key (production). Needed to
//     VERIFY grants a peer issued (i.e. when this node receives that peer's grants).
//   - DevSecret — a dev shared secret from which the dev key is derived (dev/POC interop with
//     the Python signer); used only when PubKeyB64 is empty. Leave both empty for an
//     address-only entry (reachability without a verify key — e.g. mapping a caller peer id
//     to its department for the inbound subject check).
//   - PeerAddr — a full “/p2p/<peerid>“ multiaddr; supplies BOTH the peer id (subject
//     check / reverse lookup) and a dialable address (outbound queries).
type RegistryEntry struct {
	Dept      string `json:"dept"`
	PubKeyB64 string `json:"pubkey_b64,omitempty"`
	DevSecret string `json:"dev_secret,omitempty"`
	PeerAddr  string `json:"peer_addr,omitempty"`
	// Revoked marks a compromised department: its key is loaded but immediately revoked, so
	// its grants fail closed and it cannot call. Ship it in the same file operators already push.
	Revoked bool `json:"revoked,omitempty"`
}

// RegistryFile is the on-disk operator directory: which departments exist, how to verify
// their grants, and how to reach them. It is the piece that lets two independently-deployed
// nodes federate without any in-process seeding — each operator ships the same file (or the
// subset of peers it is allowed to talk to).
type RegistryFile struct {
	Departments []RegistryEntry `json:"departments"`
	// RevokedGrants: grant ids the responder must refuse mesh-wide, even if a caller presents a
	// signature-valid copy (revocation is not in the signature).
	RevokedGrants []string `json:"revoked_grants,omitempty"`
}

// LoadRegistryFile reads and parses an operator registry file.
func LoadRegistryFile(path string) (*RegistryFile, error) {
	raw, err := os.ReadFile(path) //nolint:gosec // operator-provided path, by design
	if err != nil {
		return nil, fmt.Errorf("registry: read %s: %w", path, err)
	}
	var rf RegistryFile
	if err := json.Unmarshal(raw, &rf); err != nil {
		return nil, fmt.Errorf("registry: parse %s: %w", path, err)
	}
	return &rf, nil
}

// pubKey resolves an entry's Ed25519 public key: an explicit base64 key wins; otherwise the
// dev key derived from DevSecret. Returns nil if neither is usable, so the caller can decide
// whether an addr-only entry (no key) is acceptable.
func (e RegistryEntry) pubKey() (ed25519.PublicKey, error) {
	if e.PubKeyB64 != "" {
		b, err := base64.StdEncoding.DecodeString(e.PubKeyB64)
		if err != nil {
			return nil, fmt.Errorf("dept %q: bad pubkey_b64: %w", e.Dept, err)
		}
		if len(b) != ed25519.PublicKeySize {
			return nil, fmt.Errorf("dept %q: pubkey must be %d bytes, got %d", e.Dept, ed25519.PublicKeySize, len(b))
		}
		return ed25519.PublicKey(b), nil
	}
	if e.DevSecret != "" {
		return grant.DevPublicKey(e.DevSecret), nil
	}
	return nil, nil
}

// ApplyRegistryFile loads path, registers every department (key + peer id + addr), and dials
// the peers that carry an address. A single malformed/unreachable entry is logged and
// skipped — a bad line for one peer must not stop the node learning the others. The node's
// OWN department is skipped (it already registered itself). Returns an error only if the file
// itself cannot be read/parsed.
func (n *Node) ApplyRegistryFile(ctx context.Context, path string) (int, error) {
	rf, err := LoadRegistryFile(path)
	if err != nil {
		return 0, err
	}
	loaded := 0
	for _, e := range rf.Departments {
		if e.Dept == "" || e.Dept == n.DeptID {
			continue
		}
		key, err := e.pubKey()
		if err != nil {
			fmt.Fprintf(os.Stderr, "registry: skip %q: %v\n", e.Dept, err)
			continue
		}

		// Resolve the peer's libp2p identity+address once (a /p2p/ multiaddr yields both).
		var peerID peer.ID
		var addr string
		var info *peer.AddrInfo
		if e.PeerAddr != "" {
			pi, perr := peerInfoFromString(e.PeerAddr)
			if perr != nil || len(pi.Addrs) == 0 {
				fmt.Fprintf(os.Stderr, "registry: skip %q: bad peer_addr %q\n", e.Dept, e.PeerAddr)
				continue
			}
			info, peerID, addr = pi, pi.ID, pi.Addrs[0].String()
		}

		n.Registry.Register(e.Dept, key, peerID, addr)
		if e.Revoked {
			n.Registry.RevokeDept(e.Dept)
		}

		// Dial now so outbound /federate can reach the peer immediately; best-effort.
		if info != nil && info.ID != n.Host.ID() {
			n.Host.Peerstore().AddAddrs(info.ID, info.Addrs, peerstoreTTL)
			if cerr := n.Host.Connect(ctx, *info); cerr != nil {
				fmt.Fprintf(os.Stderr, "registry: dept %q not reachable yet: %v\n", e.Dept, cerr)
			}
		}
		loaded++
	}
	for _, gid := range rf.RevokedGrants {
		if gid != "" {
			n.Registry.RevokeGrant(gid)
		}
	}
	return loaded, nil
}
