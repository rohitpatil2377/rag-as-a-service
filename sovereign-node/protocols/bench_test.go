package protocols

import (
	"context"
	"crypto/ed25519"
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	libp2p "github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"

	"sovereignnode/bridge"
	"sovereignnode/grant"
)

// TestFederationLatency measures the REAL mesh scatter-gather round-trip: an initiator
// dials a responder over two in-process libp2p hosts (full libp2p-TLS 1.3 mutual-auth
// handshake, noise-encrypted stream) and runs a federated query that the responder
// verifies fail-closed against a signed Ed25519 grant, then retrieves via a stub bridge.
//
// This isolates the MESH cost the proposal claims to "optimize" (dial reuse + stream RTT +
// grant verification + scatter-gather), holding the rag retrieval constant (stub) so the
// number is the federation overhead, not the LLM/vector-store time layered on top.
//
// Skipped unless BENCH=1 so it never runs in the normal unit suite. Tunables:
//
//	BENCH_ITERS  (default 500)  measured queries per peer-count
//	BENCH_PEERS  (default "1,2,4")  scatter fan-out sizes to report
//	BENCH_SPANS  (default 5)  spans the stub responder returns (payload size)
func TestFederationLatency(t *testing.T) {
	if os.Getenv("BENCH") != "1" {
		t.Skip("set BENCH=1 to run the federation latency benchmark")
	}
	iters := envInt("BENCH_ITERS", 500)
	spans := envInt("BENCH_SPANS", 5)
	peerCounts := envInts("BENCH_PEERS", []int{1, 2, 4})

	ctx := context.Background()

	priv := grant.DevSigningKey("")
	pub := priv.Public().(ed25519.PublicKey)

	// Initiator host (dept "finance" — the grant subject).
	ha, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("host A: %v", err)
	}
	defer ha.Close()

	maxPeers := 0
	for _, n := range peerCounts {
		if n > maxPeers {
			maxPeers = n
		}
	}

	// Stand up maxPeers responder hosts, each running the REAL query handler.
	stub := &fakeBridge{spans: makeSpans(spans)}
	responders := make([]host.Host, 0, maxPeers)
	for i := 0; i < maxPeers; i++ {
		hb, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
		if err != nil {
			t.Fatalf("responder %d: %v", i, err)
		}
		defer hb.Close()
		dir := &fakeDir{
			keys:  map[string]ed25519.PublicKey{"hr": pub},
			peers: map[peer.ID]string{ha.ID(): "finance"}, // caller IS the grant subject
		}
		RegisterQueryHandler(hb, &QueryService{LocalDept: "hr", Registry: dir, Bridge: stub})
		if err := ha.Connect(ctx, peer.AddrInfo{ID: hb.ID(), Addrs: hb.Addrs()}); err != nil {
			t.Fatalf("connect responder %d: %v", i, err)
		}
		responders = append(responders, hb)
	}

	g := signGrant(t, priv)

	newReq := func() *FederatedQueryRequest {
		return &FederatedQueryRequest{
			Grant: g, GrantToken: g.SignedToken, QueryText: "what is the q3 payroll variance",
			Topics: []string{"payroll"}, KBID: "kb-hr", TopK: spans,
		}
	}

	fmt.Printf("\n=== Federated RAG federation scatter-gather latency ===\n")
	fmt.Printf("transport=libp2p-TLS1.3(mTLS)  iters=%d  spans/resp=%d  (stub retrieval, mesh cost only)\n\n", iters, spans)

	for _, npeers := range peerCounts {
		if npeers > len(responders) {
			continue
		}
		targets := responders[:npeers]

		// Warmup (JIT/connection/stream-muxer priming) — excluded from the sample.
		for i := 0; i < 20; i++ {
			scatterGather(ctx, ha, targets, newReq)
		}

		lat := make([]time.Duration, 0, iters)
		for i := 0; i < iters; i++ {
			t0 := time.Now()
			if err := scatterGather(ctx, ha, targets, newReq); err != nil {
				t.Fatalf("scatterGather(peers=%d): %v", npeers, err)
			}
			lat = append(lat, time.Since(t0))
		}
		report(npeers, lat)
	}
}

// scatterGather sends the query to every target concurrently and waits for all replies
// (the gather barrier), mirroring the initiator's real fan-out. Wall-clock is the slowest
// peer, which is what the proposal's "scatter-gather latency" actually is.
func scatterGather(ctx context.Context, h host.Host, targets []host.Host, mk func() *FederatedQueryRequest) error {
	errc := make(chan error, len(targets))
	for _, tb := range targets {
		go func(id peer.ID) {
			resp, err := QueryPeer(ctx, h, id, mk())
			if err != nil {
				errc <- err
				return
			}
			if !resp.GrantValid {
				errc <- fmt.Errorf("grant invalid over wire")
				return
			}
			errc <- nil
		}(tb.ID())
	}
	for range targets {
		if err := <-errc; err != nil {
			return err
		}
	}
	return nil
}

func report(npeers int, lat []time.Duration) {
	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	pct := func(p float64) time.Duration {
		if len(lat) == 0 {
			return 0
		}
		idx := int(p / 100 * float64(len(lat)-1))
		return lat[idx]
	}
	var sum time.Duration
	for _, d := range lat {
		sum += d
	}
	mean := sum / time.Duration(len(lat))
	ms := func(d time.Duration) string { return fmt.Sprintf("%.2f", float64(d.Microseconds())/1000.0) }
	fmt.Printf("peers=%d  n=%d  p50=%sms  p95=%sms  p99=%sms  mean=%sms  max=%sms\n",
		npeers, len(lat), ms(pct(50)), ms(pct(95)), ms(pct(99)), ms(mean), ms(lat[len(lat)-1]))
}

func makeSpans(n int) []bridge.Span {
	body := strings.Repeat("federated evidence span carried across the department boundary. ", 8) // ~500B
	out := make([]bridge.Span, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, bridge.Span{
			ChunkID: fmt.Sprintf("c%d", i), Text: body, Score: 0.9 - float64(i)*0.01, DocID: fmt.Sprintf("d%d", i),
		})
	}
	return out
}

func signGrant(t *testing.T, priv ed25519.PrivateKey) grant.Grant {
	t.Helper()
	g := grant.Grant{
		GrantID:            "bench-g1",
		IssuerDeptID:       "hr",
		SubjectDeptID:      "finance",
		Topics:             []string{"payroll"},
		KBIDs:              []string{"kb-hr"},
		MaxResultsPerQuery: 50,
		NotBefore:          time.Now().Add(-time.Hour).UTC().Truncate(time.Second),
		ExpiresAt:          time.Now().Add(time.Hour).UTC().Truncate(time.Second),
	}
	tok, err := g.Sign(priv)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	g.SignedToken = tok
	return g
}

func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		var n int
		if _, err := fmt.Sscanf(v, "%d", &n); err == nil && n > 0 {
			return n
		}
	}
	return def
}

func envInts(k string, def []int) []int {
	v := os.Getenv(k)
	if v == "" {
		return def
	}
	var out []int
	for _, p := range strings.Split(v, ",") {
		var n int
		if _, err := fmt.Sscanf(strings.TrimSpace(p), "%d", &n); err == nil && n > 0 {
			out = append(out, n)
		}
	}
	if len(out) == 0 {
		return def
	}
	return out
}
