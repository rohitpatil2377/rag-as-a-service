package bridge

import "context"

// Distributed tracing across the federation hop: the initiator's W3C ``traceparent`` rides the
// libp2p query to the peer, and the peer's node forwards it as a header on its LOCAL rag-api
// retrieve — so ONE trace spans both departments' rag-apis. Carried on the context so the
// Retriever interface (and its test fakes) stay unchanged.

type traceparentKey struct{}

// WithTraceparent attaches a W3C traceparent to ctx (no-op if empty).
func WithTraceparent(ctx context.Context, traceparent string) context.Context {
	if traceparent == "" {
		return ctx
	}
	return context.WithValue(ctx, traceparentKey{}, traceparent)
}

// traceparentFrom reads the traceparent attached to ctx, or "" if none.
func traceparentFrom(ctx context.Context) string {
	tp, _ := ctx.Value(traceparentKey{}).(string)
	return tp
}
