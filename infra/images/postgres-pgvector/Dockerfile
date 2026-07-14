# CNPG's operator ships its own instance manager (injected at runtime via an init container from
# the operator's own image), but no CNPG maintained Postgres image bundles pgvector: compose's
# `pgvector/pgvector` image is not CNPG shaped (CNPG has its own entrypoint/probe contract, SP5
# digest section 2). This layers pgvector onto CNPG's own base image instead of hand rolling a
# from scratch Postgres image.
#
# pg17 to match compose's pinned major (docker-compose.yml: `pgvector/pgvector@sha256:...` resolved
# from `pgvector/pgvector:pg17`). Base pinned by digest (D37): resolved from
# `ghcr.io/cloudnative-pg/postgresql:17-bookworm` at build time, same discipline as every other
# image reference in this repo (never a floating tag).
FROM ghcr.io/cloudnative-pg/postgresql@sha256:f785551be35036d65a0f1ebe71995022d261a829391fc2f14222ca19795d7c20

# The base image already configures the PGDG apt repository (it is itself built from PGDG
# packages), so `postgresql-17-pgvector` installs straight from the same source as `postgres`
# itself, no extra keyring/source list plumbing and no from source compile. Version pinned
# explicitly (never a floating candidate) so a rebuild is reproducible; bump deliberately, not by
# accident, mirroring this repo's D26 "never a floating alias" discipline for every other pin.
# `CREATE EXTENSION IF NOT EXISTS vector;` already runs idempotently at ingest time
# (testing/harness/rag_tools/ingest.py); this image only needs the shared library and control/sql
# files on disk for that statement to succeed instead of failing closed.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-17-pgvector=0.8.5-1.pgdg12+1 \
    && rm -rf /var/lib/apt/lists/*

# Back to the base image's own numeric non root user (uid 26, "postgres"); never serve as root.
USER 26
