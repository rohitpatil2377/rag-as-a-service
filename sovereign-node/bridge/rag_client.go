// Package bridge is the SovereignNode's HTTP client to the LOCAL rag-api. When a peer's
// federated query passes the grant checks, the node calls this bridge to actually
// retrieve spans from its own department's knowledge base — the node never reaches into
// another department's data directly, only through its own local rag-api.
package bridge

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client talks to the local rag-api's retrieval endpoint.
type Client struct {
	BaseURL string // e.g. http://localhost:8000 (no trailing slash needed)
	Token   string // bearer JWT for the local rag-api (--rag-token)
	HTTP    *http.Client
}

// New builds a bridge client. A nil/zero timeout gets a sane default.
func New(baseURL, token string) *Client {
	return &Client{
		BaseURL: trimSlash(baseURL),
		Token:   token,
		HTTP:    &http.Client{Timeout: 30 * time.Second},
	}
}

func trimSlash(s string) string {
	for len(s) > 0 && s[len(s)-1] == '/' {
		s = s[:len(s)-1]
	}
	return s
}

// RetrieveRequest is the body POSTed to /v1/kb/{kb_id}/retrieve.
//
// The real rag QueryRequest carries more optional fields (mode, filters, recency, …)
// but the endpoint only requires `query`; `top_k` we set to the grant's cap. Keeping the
// body minimal avoids coupling to optional server behaviour.
type RetrieveRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k"`
}

// Span is one retrieved span from the rag-api.
//
// ASSUMPTION (documented, because the Python API can't be run here): the retrieve
// endpoint returns rag_schemas.RetrievalResult.model_dump(), i.e.
//
//	{"query": {...}, "spans": [{"chunk_id","doc_id","text","score","scorer",...}], "max_score": ..}
//
// We decode only the fields the federation response needs. Unknown fields are ignored
// by encoding/json, so extra fields (scorer, fusion_score, page, section, acl, …) are
// harmless. This matches RetrievedSpan in libs/schemas/src/rag_schemas/retrieval.py.
type Span struct {
	ChunkID  string  `json:"chunk_id"`
	DocID    string  `json:"doc_id"`
	Text     string  `json:"text"`
	Score    float64 `json:"score"`
	Category string  `json:"category,omitempty"` // FR-B7 taxonomy tag; carried for L1 topic-enforcement
}

// RetrieveResponse is the subset of RetrievalResult we consume.
type RetrieveResponse struct {
	Spans    []Span  `json:"spans"`
	MaxScore float64 `json:"max_score"`
}

// Retrieve calls POST {BaseURL}/v1/kb/{kbID}/retrieve with a bearer token and returns
// the decoded spans. topK is expected to already be capped at the grant's
// max_results_per_query by the caller.
func (c *Client) Retrieve(ctx context.Context, kbID, query string, topK int) (*RetrieveResponse, error) {
	if c == nil || c.BaseURL == "" {
		return nil, fmt.Errorf("bridge: no rag-url configured")
	}
	if topK < 1 {
		topK = 1
	}
	body, err := json.Marshal(RetrieveRequest{Query: query, TopK: topK})
	if err != nil {
		return nil, err
	}
	url := fmt.Sprintf("%s/v1/kb/%s/retrieve", c.BaseURL, kbID)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	if tp := traceparentFrom(ctx); tp != "" {
		req.Header.Set("traceparent", tp) // continue the caller's trace into this rag-api
	}

	httpc := c.HTTP
	if httpc == nil {
		httpc = http.DefaultClient
	}
	resp, err := httpc.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("rag retrieve %s: status %d: %s", kbID, resp.StatusCode, string(raw))
	}

	var out RetrieveResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("rag retrieve %s: decode response: %w", kbID, err)
	}
	return &out, nil
}
