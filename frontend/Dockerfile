# Atlas web, build the Vite SPA, serve it static behind nginx (which proxies /api → backend).
# Build context is the repo ROOT (pnpm workspace lives there).
#   docker build -f frontend/Dockerfile -t atlas-web .
FROM node:22-slim AS build
WORKDIR /app
# Pin pnpm to match the committed lockfile's resolver (avoids pnpm 11's release age policy rejecting
# a just published transitive dep, the lockfile was resolved with 10.x).
RUN corepack enable && corepack prepare pnpm@10.17.1 --activate
# Lockfile layer first for caching (no root package.json, pnpm workspace root is implicit).
COPY pnpm-lock.yaml pnpm-workspace.yaml ./
COPY frontend/package.json ./frontend/package.json
RUN pnpm install --frozen-lockfile
COPY frontend ./frontend
RUN pnpm -C frontend build

FROM nginx:alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/frontend/dist /usr/share/nginx/html
EXPOSE 80
