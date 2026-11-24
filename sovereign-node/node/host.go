package node

import (
	"context"
	"crypto/rand"
	"fmt"
	"os"
	"path/filepath"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/control"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/peerstore"
	libp2ptls "github.com/libp2p/go-libp2p/p2p/security/tls"
	"github.com/multiformats/go-multiaddr"
)

// peerstoreTTL is how long bootstrap addresses are remembered in the peerstore.
const peerstoreTTL = peerstore.PermanentAddrTTL

// Node bundles the libp2p host with its department id and the department registry so the
// protocol handlers have a single object to hang off.
type Node struct {
	Host     host.Host
	DeptID   string
	Registry *Registry
}

// New builds a libp2p host that listens on listenAddr with a stable Ed25519 identity.
//
// The identity is loaded from keyPath if it exists; otherwise a fresh key is generated
// and persisted there (so a node keeps the same PeerID across restarts). An empty keyPath
// means "ephemeral identity, do not persist" — handy for tests and throwaway nodes.
func New(deptID, listenAddr, keyPath string) (*Node, error) {
	priv, err := loadOrCreateIdentity(keyPath)
	if err != nil {
		return nil, err
	}

	maddr, err := multiaddr.NewMultiaddr(listenAddr)
	if err != nil {
		return nil, fmt.Errorf("node: invalid listen multiaddr %q: %w", listenAddr, err)
	}

	// The registry is created BEFORE the host so the zero-trust connection gater can hold a live
	// reference to it: peers registered later (from the operator registry file) are then visible
	// to the gater at connection time.
	reg := NewRegistry()

	h, err := libp2p.New(
		libp2p.Identity(priv),
		libp2p.ListenAddrs(maddr),
		// Transport security = libp2p-TLS 1.3 (mutual TLS): both peers present a certificate bound
		// to their Ed25519 identity key, so every connection is mutually authenticated at the
		// transport layer. Pinned explicitly (not the negotiated default) so the guarantee is
		// unambiguous — this is the "mTLS" of the Federated RAG mesh.
		libp2p.Security(libp2ptls.ID, libp2ptls.New),
		// Zero-trust network layer: only a peer that maps to a known, non-revoked department in the
		// registry may open a session with us (see registryGater).
		libp2p.ConnectionGater(&registryGater{reg: reg}),
	)
	if err != nil {
		return nil, fmt.Errorf("node: create libp2p host: %w", err)
	}

	return &Node{
		Host:     h,
		DeptID:   deptID,
		Registry: reg,
	}, nil
}

// registryGater is the mesh's zero-trust network layer. libp2p-TLS has already proven WHO a
// connecting peer is (mutual TLS 1.3, the certificate bound to the peer's Ed25519 identity); this
// gater then decides WHETHER that authenticated peer may talk to us at all. An INBOUND connection
// is accepted only from a peer that resolves to a known, non-revoked department in the registry —
// so a node outside the federation, even one holding a valid libp2p identity, cannot establish a
// session. Outbound connections we initiate are always allowed: we only ever dial peers we resolved
// from the registry ourselves, and gating them would break bootstrap.
type registryGater struct{ reg *Registry }

func (g *registryGater) InterceptPeerDial(peer.ID) bool                      { return true }
func (g *registryGater) InterceptAddrDial(peer.ID, multiaddr.Multiaddr) bool { return true }
func (g *registryGater) InterceptAccept(network.ConnMultiaddrs) bool         { return true }

func (g *registryGater) InterceptSecured(dir network.Direction, p peer.ID, _ network.ConnMultiaddrs) bool {
	if dir == network.DirOutbound {
		return true
	}
	dept, ok := g.reg.DeptForPeer(p)
	return ok && !g.reg.IsDeptRevoked(dept)
}

func (g *registryGater) InterceptUpgraded(network.Conn) (bool, control.DisconnectReason) {
	return true, 0
}

// loadOrCreateIdentity returns the persisted Ed25519 identity at keyPath, generating and
// writing one if the file is absent. keyPath == "" yields an ephemeral in-memory key.
func loadOrCreateIdentity(keyPath string) (crypto.PrivKey, error) {
	if keyPath == "" {
		priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
		if err != nil {
			return nil, fmt.Errorf("node: generate ephemeral identity: %w", err)
		}
		return priv, nil
	}

	if raw, err := os.ReadFile(keyPath); err == nil {
		priv, err := crypto.UnmarshalPrivateKey(raw)
		if err != nil {
			return nil, fmt.Errorf("node: unmarshal identity key %q: %w", keyPath, err)
		}
		return priv, nil
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("node: read identity key %q: %w", keyPath, err)
	}

	// Not present — generate and persist (0600, the file is a private key).
	priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("node: generate identity: %w", err)
	}
	raw, err := crypto.MarshalPrivateKey(priv)
	if err != nil {
		return nil, fmt.Errorf("node: marshal identity: %w", err)
	}
	if dir := filepath.Dir(keyPath); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return nil, fmt.Errorf("node: create key dir %q: %w", dir, err)
		}
	}
	if err := os.WriteFile(keyPath, raw, 0o600); err != nil {
		return nil, fmt.Errorf("node: write identity key %q: %w", keyPath, err)
	}
	return priv, nil
}

// PeerID returns the host's libp2p peer id.
func (n *Node) PeerID() peer.ID {
	return n.Host.ID()
}

// ListenAddrs returns the fully-qualified /p2p/ multiaddrs another node can dial to reach
// this host (listen addrs with the peer id appended).
func (n *Node) ListenAddrs() []string {
	pidAddr, err := multiaddr.NewMultiaddr("/p2p/" + n.Host.ID().String())
	if err != nil {
		return nil
	}
	var out []string
	for _, a := range n.Host.Addrs() {
		out = append(out, a.Encapsulate(pidAddr).String())
	}
	return out
}

// ConnectBootstrap dials each bootstrap multiaddr (each of the form
// /ip4/…/tcp/…/p2p/<peerid>). Failures are collected but do not abort the others — a
// mesh should come up even if some seeds are down. It returns a non-nil error only if it
// managed to connect to NONE of a non-empty bootstrap list.
func (n *Node) ConnectBootstrap(ctx context.Context, addrs []string) error {
	if len(addrs) == 0 {
		return nil
	}
	var connected int
	var lastErr error
	for _, raw := range addrs {
		raw = trimSpace(raw)
		if raw == "" {
			continue
		}
		info, err := peerInfoFromString(raw)
		if err != nil {
			lastErr = err
			fmt.Fprintf(os.Stderr, "bootstrap: skip %q: %v\n", raw, err)
			continue
		}
		if info.ID == n.Host.ID() {
			continue // don't dial ourselves
		}
		n.Host.Peerstore().AddAddrs(info.ID, info.Addrs, peerstoreTTL)
		if err := n.Host.Connect(ctx, *info); err != nil {
			lastErr = err
			fmt.Fprintf(os.Stderr, "bootstrap: connect %s failed: %v\n", info.ID, err)
			continue
		}
		connected++
	}
	if connected == 0 && lastErr != nil {
		return fmt.Errorf("node: no bootstrap peers reachable: %w", lastErr)
	}
	return nil
}

// peerInfoFromString parses a full /p2p/ multiaddr into an AddrInfo (addr + peer id).
func peerInfoFromString(s string) (*peer.AddrInfo, error) {
	maddr, err := multiaddr.NewMultiaddr(s)
	if err != nil {
		return nil, fmt.Errorf("parse multiaddr: %w", err)
	}
	info, err := peer.AddrInfoFromP2pAddr(maddr)
	if err != nil {
		return nil, fmt.Errorf("multiaddr missing /p2p/<peerid>: %w", err)
	}
	return info, nil
}

// Close shuts down the host.
func (n *Node) Close() error {
	if n.Host == nil {
		return nil
	}
	return n.Host.Close()
}

func trimSpace(s string) string {
	start := 0
	for start < len(s) && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r') {
		start++
	}
	end := len(s)
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n' || s[end-1] == '\r') {
		end--
	}
	return s[start:end]
}
