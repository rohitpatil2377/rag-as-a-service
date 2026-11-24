# RAG as a Service — admin console (FR-H3) image.
# Multi-stage: build the Vite SPA, then serve the static bundle via nginx with a
# same-origin reverse-proxy to the API (so the browser makes no cross-origin call).
# Sovereign: base images come from Acme internal mirrors in prod; nothing calls out
# at runtime. Build context is the repo ROOT (matches apps/api/Dockerfile).
#
# The console consumes the shared design tokens at repo-root `design/` via relative
# imports (../../../design/... in CSS, ../../design/... in tailwind.config.js), so the
# build stage mirrors the repo layout: console at /app/apps/console, tokens at /app/design.
FROM node:22-alpine AS build
WORKDIR /app/apps/console
COPY apps/console/package.json apps/console/package-lock.json ./
RUN npm ci
COPY design /app/design
COPY apps/console/ ./
RUN npm run build

FROM nginx:alpine
COPY apps/console/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/apps/console/dist /usr/share/nginx/html
EXPOSE 80
