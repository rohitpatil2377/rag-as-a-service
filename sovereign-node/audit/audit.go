// Package audit emits one structured JSON line per cross-department federation decision.
//
// It is the mesh-layer audit trail. With the node running in Kubernetes, `kubectl logs` on
// the sovereign-node pod is a parseable, SIEM-forwardable record of every authorized and
// every denied federated query — the most security-critical events in the mesh. It
// COMPLEMENTS, and does not replace, the hash-chained answer/receipt audit kept by the
// Python control-plane: that trail proves what an answer was grounded on; this trail proves
// who was allowed (or refused) to read across a department boundary, and why.
//
// Depends only on the standard library (log/slog), matching the node's zero-dependency,
// air-gap-friendly posture.
package audit

import (
	"log/slog"
	"os"
	"sync"
)

var (
	initOnce sync.Once
	logger   *slog.Logger
)

func log() *slog.Logger {
	initOnce.Do(func() {
		logger = slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	})
	return logger
}

// Decision is one federation authorization outcome. Fields that are not yet resolved at the
// point a request is denied (e.g. CallerDept before the peer is mapped, KBID before the KB is
// picked) are passed empty and rendered as "".
type Decision struct {
	ResponderDept string   // THIS node's dept — the KB owner making the decision.
	RemotePeer    string   // libp2p peer id of the calling node.
	CallerDept    string   // department resolved from the calling peer ("" if unknown).
	IssuerDept    string   // grant issuer department.
	SubjectDept   string   // grant subject department (who the grant authorizes).
	GrantID       string   // grant id presented.
	KBID          string   // knowledge base targeted ("" if not yet resolved).
	Topics        []string // topics requested.
	Allowed       bool     // true = retrieval performed; false = failed closed.
	Reason        string   // machine slug: allow | deny_pubkey | deny_signature | deny_revoked | deny_inactive | deny_subject_mismatch | deny_replay | deny_topics | deny_kb | retrieval_failed.
	SpanCount     int      // spans returned on an allow.
}

// Query records one inbound federated-query decision as a single structured JSON line.
func Query(d Decision) {
	decision := "deny"
	if d.Allowed {
		decision = "allow"
	}
	log().Info("federation.query",
		slog.String("event", "federation.query"),
		slog.String("decision", decision),
		slog.String("reason", d.Reason),
		slog.String("responder_dept", d.ResponderDept),
		slog.String("remote_peer", d.RemotePeer),
		slog.String("caller_dept", d.CallerDept),
		slog.String("issuer_dept", d.IssuerDept),
		slog.String("subject_dept", d.SubjectDept),
		slog.String("grant_id", d.GrantID),
		slog.String("kb_id", d.KBID),
		slog.Any("topics", d.Topics),
		slog.Int("span_count", d.SpanCount),
	)
}
