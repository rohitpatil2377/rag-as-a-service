# SovereignNode — libp2p federation node (one per department).
# Multi-stage, static binary, VENDORED deps so the image builds fully OFFLINE
# (no Go module proxy at build time) — matches the air-gapped/sovereign posture.
FROM golang:1.26-alpine AS build
WORKDIR /src
# Vendored: copy the whole tree (incl. vendor/) and build with -mod=vendor so no network.
COPY . .
RUN CGO_ENABLED=0 GOFLAGS=-mod=vendor go build -trimpath -ldflags="-s -w" -o /out/sovereign-node .

FROM alpine:3.20
# ca-certificates for any TLS bridge call to the local rag-api; curl for the demo/healthcheck.
RUN apk add --no-cache ca-certificates curl && adduser -D -u 10001 node
COPY --from=build /out/sovereign-node /usr/local/bin/sovereign-node
# The identity key persists to a named volume at /data. A fresh named volume inherits the
# ownership of its mountpoint in the image, so create /data owned by `node` — otherwise the
# volume mounts as root and the non-root node can't write its libp2p identity (permission denied).
RUN mkdir -p /data && chown node:node /data
VOLUME /data
USER node
# libp2p listen (4001) + loopback control API (7070) are published by compose.
EXPOSE 4001 7070
ENTRYPOINT ["sovereign-node"]
