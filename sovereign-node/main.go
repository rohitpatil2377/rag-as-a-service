// Command sovereign-node runs a Federated RAG federation node: a libp2p host that speaks
// the /rag/query/1.0 and /rag/grant/1.0 protocols. It fail-closed validates every
// inbound federated query against a cryptographic grant (byte-compatible with the Python
// control-plane signer) before retrieving from its own local rag-api.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"sovereignnode/bridge"
	"sovereignnode/grant"
	"sovereignnode/node"
	"sovereignnode/protocols"
)

func main() {
	var (
		dept          = flag.String("dept", "", "this node's department id (required)")
		listen        = flag.String("listen", "/ip4/0.0.0.0/tcp/4001", "libp2p listen multiaddr")
		ragURL        = flag.String("rag-url", "http://localhost:8000", "base URL of the local rag-api")
		ragToken      = flag.String("rag-token", "", "bearer token for the local rag-api")
		bootstrap     = flag.String("bootstrap", "", "comma-separated bootstrap peer multiaddrs (…/p2p/<peerid>)")
		keyPath       = flag.String("key", "", "path to the libp2p identity key file (generated + persisted if absent; empty = ephemeral)")
		devSecret     = flag.String("dev-secret", grant.DefaultDevJWTSecret, "dev JWT secret used to derive this dept's dev grant-signing key (interop with the Python signer)")
		controlListen = flag.String("control-listen", "", "loopback addr for the local control API that drives OUTBOUND federated queries (e.g. 127.0.0.1:7070); empty = disabled")
		registryFile  = flag.String("registry", "", "path to an operator registry JSON file (dept -> pubkey + peer address) so this node can verify and reach peer departments")
		replaySkew    = flag.Int("replay-skew-seconds", 120, "max clock skew (seconds) for accepting a federated query; also bounds the replay-nonce cache")
	)
	flag.Parse()

	if *dept == "" {
		fmt.Fprintln(os.Stderr, "error: --dept is required")
		flag.Usage()
		os.Exit(2)
	}

	if err := run(*dept, *listen, *ragURL, *ragToken, *bootstrap, *keyPath, *devSecret, *controlListen, *registryFile, *replaySkew); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}

func run(dept, listen, ragURL, ragToken, bootstrap, keyPath, devSecret, controlListen, registryFile string, replaySkewSeconds int) error {
	// Root context cancelled on SIGINT/SIGTERM so the node shuts down cleanly.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	n, err := node.New(dept, listen, keyPath)
	if err != nil {
		return err
	}
	defer n.Close()

	// Seed the registry with THIS department's own dev grant-signing public key, so a
	// grant issued by us verifies locally out of the box. Other departments' keys are
	// learned via --bootstrap wiring / grant announcements / an operator registry file.
	n.Registry.Register(dept, grant.DevPublicKey(devSecret), n.PeerID(), "")

	// Wire the retrieval bridge and both protocol handlers.
	bridgeClient := bridge.New(ragURL, ragToken)

	protocols.RegisterQueryHandler(n.Host, &protocols.QueryService{
		LocalDept: dept,
		Registry:  n.Registry,
		Bridge:    bridgeClient,
		Replay:    protocols.NewReplayGuard(time.Duration(replaySkewSeconds) * time.Second),
	})
	protocols.RegisterGrantHandler(n.Host, &protocols.GrantService{
		LocalDept: dept,
		Registry:  n.Registry,
		Store:     protocols.NewGrantStore(),
	})

	// Local control API (loopback) drives OUTBOUND federated queries: the department's own
	// rag-api / MeshTransport POSTs /federate and this node dials the named peer over the
	// mesh, shipping the QUERY not a vector (ADR-0019).
	if controlListen != "" {
		cs := node.NewControlServer(n, protocols.QueryPeer)
		go func() {
			if err := cs.Serve(ctx, controlListen); err != nil {
				fmt.Fprintf(os.Stderr, "control server: %v\n", err)
			}
		}()
		fmt.Printf("  control:    http://%s (loopback; POST /federate, GET /health)\n", controlListen)
	}

	// Learn peer departments (verify keys + dial addresses) from an operator registry file FIRST,
	// so the zero-trust connection gater knows every legitimate peer before any connection is made
	// (inbound or outbound) — otherwise a peer dialling us during startup would be refused as
	// unknown. Two independently-deployed nodes federate without in-process seeding.
	if registryFile != "" {
		loaded, err := n.ApplyRegistryFile(ctx, registryFile)
		if err != nil {
			return err
		}
		fmt.Printf("  registry:   %d peer department(s) from %s\n", loaded, registryFile)
	}

	// Dial bootstrap peers (best-effort; the mesh comes up even if some are down).
	boots := splitAddrs(bootstrap)
	if err := n.ConnectBootstrap(ctx, boots); err != nil {
		fmt.Fprintf(os.Stderr, "warning: %v\n", err)
	}

	// Announce ourselves.
	fmt.Printf("SovereignNode up\n")
	fmt.Printf("  department: %s\n", dept)
	fmt.Printf("  peer id:    %s\n", n.PeerID())
	fmt.Printf("  rag-api:   %s\n", ragURL)
	fmt.Printf("  protocols:  %s, %s\n", protocols.QueryProtocol, protocols.GrantProtocol)
	fmt.Printf("  listen addrs:\n")
	for _, a := range n.ListenAddrs() {
		fmt.Printf("    %s\n", a)
	}
	if len(boots) > 0 {
		fmt.Printf("  bootstrap:  %d peer(s)\n", len(boots))
	}
	fmt.Printf("ready — Ctrl-C to stop\n")

	// Block until a signal cancels the context.
	<-ctx.Done()
	fmt.Println("\nshutting down…")
	return nil
}

// splitAddrs splits a comma-separated multiaddr list, trimming blanks.
func splitAddrs(csv string) []string {
	if strings.TrimSpace(csv) == "" {
		return nil
	}
	parts := strings.Split(csv, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
