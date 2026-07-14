# Atlas infra

One helmfile, two environments, same charts (D3): `local` is the default k3d dev/eval tier (free,
always available); `burst` is the rare, credential gated Hetzner tier (public demos, the load lane,
headline benchmarks; local numbers are never quoted). The two environments differ only in their
values layer, never in chart content.

This directory is scaffolded incrementally across the SP5 infra plan
(the SP5 plan):

- **Task 1**: the helmfile/environments/SOPS scaffold and the hermetic render test.
- **Task 2**: CNPG with pgvector on k3d, the migrate/rag-init one shot Jobs.
- **Task 3**: TEI embed/rerank, pinned digests, the indexes/ PersistentVolume/PersistentVolumeClaim
  pair (replacing Task 2's raw hostPath). TEI itself runs in cluster on burst's real amd64 nodes and
  against a real external amd64 endpoint on this arm64 k3d dev machine (see "TEI: in cluster on
  burst, external endpoint on local" below for why).
- **Task 4**: backend and web Deployments (compose parity env), a Traefik IngressRoute with a D35
  rate limit Middleware, `task k3d:up` bringing the full stack up end to end, and `task k3d:smoke`
  (the rag smoke, against the ingress).
- **Task 5** (this state): the Hetzner burst tier (OpenTofu + kube-hetzner), fully written and
  credential gated: the tofu cluster module, the wildcard cert (cert-manager DNS-01), static DNS,
  `task burst:up`/`task burst:destroy`, the weekly janitor, the R2 bucket layout, and the D41 runbook
  skeletons.

## Prerequisites

Install these before `task k3d:up` or `task infra:render`; versions are what this repo was
developed and verified against, not hard floors:

| Tool | Verified version | Install |
|---|---|---|
| Docker | 29.x (Docker Desktop) | https://docs.docker.com/get-docker/ |
| k3d | 5.8.3 | https://k3d.io/stable/#installation |
| kubectl | 1.35 | https://kubernetes.io/docs/tasks/tools/#kubectl |
| helm | 4.x | https://helm.sh/docs/intro/install/ |
| helmfile | 1.7.1 | https://helmfile.readthedocs.io/en/latest/#installation |
| sops | 3.13.2 | https://github.com/getsops/sops#download |
| age | 1.3.1 | https://github.com/FiloSottile/age#installation |

macOS: all seven are on Homebrew (`brew install docker k3d kubectl helm helmfile sops age`, or use
Docker Desktop's own installer for Docker itself). `task infra:render` only needs helmfile/sops/age
(plus the age keys, see below, plus `.env.fastlane`, see the TEI section below); `task k3d:up` needs
the full table plus a running Docker daemon.

## Layout

```
infra/
  helmfile.yaml              # environments: local, burst; releases list (grows task by task)
  .sops.yaml                 # SOPS creation rules: two age recipients on every encrypted file
  environments/
    base/values.yaml         # shared across both tiers (TEI pins, tei.mode default, CNPG shape, namespace)
    local/values.yaml        # k3d only values; postgres creds ref, generated image digest block, tei.mode: external
    local/secrets.enc.yaml   # SOPS encrypted, committed; the fake local postgres password
    burst/values.yaml        # burst sizing, the wildcard cert domain (env templated), the R2/cert refs (Task 5)
    burst/secrets.enc.yaml   # SOPS encrypted, committed; a placeholder Cloudflare DNS-01 API token (Task 5)
  images/
    postgres-pgvector/       # Dockerfile: CNPG's own pg17 base + the pgvector apt package (Task 2)
  charts/
    scaffold/                # the atlas namespace + the postgres credentials Secret (username+password)
    cnpg-cluster/             # ImageCatalog + Cluster CR (D1: instances=1, no pooler) (Task 2)
    atlas-jobs/               # checkpointer-migrate + rag-init one shot Jobs (Helm hooks) (Task 2)
    atlas-indexes/            # indexes/ PersistentVolume + PersistentVolumeClaim (Task 3)
    tei/                      # tei-embed/tei-rerank: Deployment (inCluster) or connectivity-check Job (external) (Task 3)
    atlas-backend/            # the backend Deployment + Service (compose parity env) (Task 4)
    atlas-web/                # the web Deployment + Service (nginx served SPA, unmodified image) (Task 4)
    atlas-ingress/            # Traefik IngressRoute + the D35 rate limit Middleware (Task 4)
    atlas-cert/                # ClusterIssuer + wildcard Certificate, cert-manager DNS-01, burst only (Task 5)
    phoenix/                  # Postgres backed trace storage, the ONLY deployed trace backend (SP6 task 5)
    otel-collector/           # the redacting OTel collector, fanning out to phoenix + the raw archive (SP6 task 5)
    atlas-monitoring/         # Prometheus + Alertmanager + Pushgateway + the D29 sentinel probe CronJob (SP6 task 5)
    vendor/cloudnative-pg/    # the upstream CNPG operator chart, vendored (no live network at render time)
  tofu/
    cluster/                  # kube-hetzner module invocation: 1x CX23 control plane, 2x CX33 workers (Task 5)
  scripts/
    k3d-up.sh                 # task k3d:up: cluster+registry+ingress port, build/push images, operator, cluster, jobs, TEI check, backend/web/ingress
    k3d-verify.sh              # task k3d:verify: live psql checks, TEI /info revision checks, backend/web Available, ingress HTTP 200
    k3d-smoke.sh               # task k3d:smoke: the rag smoke against the ingress (Task 4)
    k3d_smoke.py                # the retrieval/chat endpoint/chat stream/generation halves k3d-smoke.sh runs
    record_image_digests.py    # rewrites the generated image digest block in local/values.yaml
    verify_tei_revisions.py    # /info revision check for the local tier's external TEI endpoint (Task 3)
    burst-up.sh                 # task burst:up: credential gate, tofu apply, TLS secret restore, helmfile sync (Task 5)
    burst-destroy.sh            # task burst:destroy: credential gate, TLS secret backup, tofu destroy, orphan check (Task 5)
    hcloud-orphans.sh           # shared label based orphan check (burst-destroy.sh AND the janitor workflow) (Task 5)
.env.fastlane                 # repo root, gitignored: ATLAS_TEI_EMBED_URL/ATLAS_TEI_RERANK_URL (Task 3, see below)
.env.burst                    # repo root, gitignored: ATLAS_BURST_DOMAIN/ATLAS_BURST_ACME_EMAIL (Task 5, see below)
```

## Rendering (no cluster, no credentials)

```bash
task infra:render          # helmfile template for both local and burst
```

or directly:

```bash
cd infra
helmfile -e local template
helmfile -e burst template
```

`testing/tests/test_infra_manifests.py` runs the same command in the hermetic PR lane (`task test`)
and asserts it succeeds, skipping with a clear message only if `helmfile` itself is not on PATH
(never silently). It also asserts the committed secret actually decrypts through the render, not
just that `sops -d` works off to the side.

## SOPS + age: two recipients, mechanism proven now

D41: every SOPS file has two age recipients, so a single age key is never the root secret of the
whole system. Both recipients are real starting today, generated as part of this task, not deferred
until a second human operator exists:

- **primary**: the day to day operator key.
- **backup**: a second recipient so losing one key does not lose the secret. In a multi operator
  setup this key's private half belongs to a second person or a password manager, not this machine;
  today, with one operator, both live locally (see below), which is the honest state, not a gap.

Private keys live **outside the repo**, under `~/.config/atlas-age/`, and are **never committed**
(not even accidentally: nothing under a user's home directory is part of this git checkout).

```
~/.config/atlas-age/
  atlas-primary.txt   # AGE-SECRET-KEY-1..., primary recipient's private half
  atlas-primary.pub   # its public key (informational; the real copy lives in infra/.sops.yaml)
  atlas-backup.txt    # AGE-SECRET-KEY-1..., backup recipient's private half
  atlas-backup.pub    # its public key
  keys.txt            # atlas-primary.txt + atlas-backup.txt concatenated: a multi identity file
```

`keys.txt` is what `SOPS_AGE_KEY_FILE` should point at for normal use (both `sops` and `helmfile`'s
`vals` integration accept a multi identity file and try each identity in turn). `task infra:render`
and `testing/tests/test_infra_manifests.py` both default `SOPS_AGE_KEY_FILE` to this path when the
environment does not already set it, so a machine with these keys in the documented place needs no
extra setup. Set `SOPS_AGE_KEY_FILE` yourself (shell profile, direnv, `.env`, whatever you use) to
override.

**Regenerating a lost key pair, or rotating one**: generate a fresh keypair with `age-keygen`, add
its public key to `infra/.sops.yaml`'s `creation_rules`, encrypt every `*.enc.yaml` file again with
`sops updatekeys <file>` (adds/removes recipients without needing the old key present, as long as
one still valid recipient's private key is available to wrap the data key again), then retire the old
public key from `.sops.yaml` once every encrypted file has been updated. A full key rotation runbook
lands as part of the D41 runbook inventory (a later task in this plan).

## How the mechanism is proven (not just asserted)

`infra/environments/local/secrets.enc.yaml` holds one fake value, `postgresPassword:
atlas-dev-password` (the same throwaway dev password `docker-compose.yml` already uses, not a real
secret), encrypted
to both recipients above. Proof, in order of how it is actually exercised:

1. `sops -d infra/environments/local/secrets.enc.yaml` decrypts with **either** recipient's private
   key alone (verified independently for both `atlas-primary.txt` and `atlas-backup.txt`), and fails
   closed with neither present.
2. `infra/environments/local/values.yaml` references the secret via helmfile's built in `vals`
   integration: `postgres.password: ref+sops://environments/local/secrets.enc.yaml#/postgresPassword`.
   This needs no extra Helm plugin (see the note below on why); it is native to helmfile.
3. `infra/charts/scaffold/templates/secret.yaml` renders a real Kubernetes `Secret` manifest
   containing the decrypted value. `helmfile -e local template` produces this manifest with the
   plaintext password inside it; `helmfile -e burst template` does not (burst has no secrets file
   yet: wiring burst postgres credentials is a named open gap in "What Task 5 deliberately does not
   do" below, a hard prerequisite for any real burst bringup), which the render test also asserts,
   so a fake local secret can never silently leak into a burst render by accident.

So "helmfile can decrypt via the SOPS integration" is not a side claim: it is the literal mechanism
the hermetic render test exercises on every `task test` run.

### Why `vals` (`ref+sops://`), not the `helm-secrets` plugin

The more commonly documented helmfile + SOPS integration is the `helm-secrets` Helm plugin
(`environments.<env>.secrets:` / `releases[].secrets:`, which shells out to `helm secrets decrypt`).
On this machine's Helm 4 install, that plugin's legacy `plugin.yaml` only registers as a getter
plugin, not a CLI subcommand plugin (`helm secrets ...` errors "unknown command"), a real Helm 4 /
helm-secrets compatibility gap, not a workaround for a missing tool. helmfile ships its own `vals`
based secret resolution natively (`ref+sops://<path>#/<key>` inside any values file), which needs no
Helm plugin at all and was verified working end to end. If a future helm-secrets release restores
Helm 4 compatibility, both mechanisms can coexist; there is no reason to migrate off `vals` since it
is simpler and has one fewer moving part.

The `helm-secrets` plugin itself was installed globally during Task 1's investigation (before this
dead end was found) and left in place, a stray unused machine artifact noted in that task's report.
Task 2 uninstalled it (`helm plugin uninstall secrets`): nothing in this repo referenced it, and
`helm plugin list` now reports no plugins installed.

## Registry addressing (Task 2)

`task k3d:up` creates the k3d cluster with `--registry-create atlas-registry`, which produces a
plain Docker container literally named `atlas-registry` (not `k3d-atlas-registry`; the exact name
you pass to `--registry-create` is the container name) on the `k3d-atlas` docker network, and
configures every k3s node's containerd (`/etc/rancher/k3s/registries.yaml`) with a mirror for
`atlas-registry:5000`, verified by reading that file on a live node. Two different, both correct,
addresses exist for the same registry:

- **cluster internal** (`atlas-registry:5000`): what every image reference in a values file or
  manifest uses (`{{ .Values.images.backend.repository }}@{{ .Values.images.backend.digest }}`).
  Resolved by containerd via the docker network, not Kubernetes DNS (verified: CoreDNS does not, and
  should not, know about this hostname; only kubelet/containerd's own image pull path does).
- **host visible** (`localhost:<port>`, a randomly assigned port `task k3d:up` resolves via
  `docker port atlas-registry 5000/tcp`): what `docker push` uses from the host CLI. Docker
  automatically treats `localhost` as an insecure (plain HTTP) registry, so no extra TLS/insecure
  registry configuration is needed on this side either.

`task k3d:up` builds both images with `--provenance=false --sbom=false`: without those flags,
BuildKit attaches a provenance attestation and `docker push` reports a manifest LIST digest instead
of the plain image manifest digest, an unnecessary complication for a local, single platform
pipeline (verified directly: the flags produce a single, unambiguous `digest: sha256:...` line in
the push output, which `infra/scripts/k3d-up.sh` parses).

## TEI: in cluster on burst, external endpoint on local (Task 3)

`environments/base/values.yaml`'s `tei.mode` decides WHERE tei-embed/tei-rerank actually run:
`"inCluster"` (the default there) renders `infra/charts/tei`'s Deployment/Service/HF cache PVC, the
straightforward translation of `docker-compose.yml`'s own tei-embed/tei-rerank services; `"external"`
(`environments/local/values.yaml`'s own override) renders a small connectivity-check Job per role
instead and points the tier at a real, already running amd64 TEI endpoint reached over the network.
Burst never overrides `tei.mode`, so its real amd64 Hetzner nodes always run in cluster.

**Why local diverges**: this is an arm64 (Apple Silicon) k3d dev machine, and
`ghcr.io/huggingface/text-embeddings-inference` ships amd64 only. Two separate, real problems were
found live standing the in cluster path up on this machine first, not guessed in advance:

1. **A platform index pull refusal, fixable.** ghcr.io publishes the image as an OCI INDEX wrapping a
   single `linux/amd64` manifest (`docker buildx imagetools inspect` against the pinned digest shows
   this directly). k3s's own containerd CRI image pull path strictly refuses a pull mediated through
   the index when the index carries no entry matching the node's own platform, and this held true even
   after sideloading the exact bytes locally: both `k3d image import` and a direct
   `ctr images pull --local --platform linux/amd64` genuinely unpacked the real ~228MB amd64 content
   on the node (verified with `ctr images check`, which reported it fully available), yet kubelet's
   own CRI pull still failed identically afterwards ("no match for platform in manifest: not found"),
   because CRI rechecks platform compatibility against the INDEX independently of what is already
   unpacked underneath it. The actual fix (proven, not merely attempted): mirror the resolved INNER
   manifest, never the index, into a registry with `crane copy` (`crane` preserves a manifest byte for
   byte; `docker push`/`docker buildx imagetools create` both wrap even a single source back into a
   fresh index, reintroducing the same ambiguity). A bare manifest has nothing left to negotiate, so
   containerd pulled it cleanly.
2. **A real memory ceiling on tei-embed's warmup, not fixable within a bounded k8s limit on this
   machine.** Even past the platform fix, with a memory limit already raised well above
   `docker-compose.yml`'s own documented Rosetta emulation peaks (18Gi, `infra/charts/tei`'s own
   default), tei-embed kept getting OOMKilled seconds into its ONNX Runtime warmup pass (the model
   was already cache hit from the PVC, confirmed with `kubectl logs --previous`; this was never a
   download problem). `docker-compose.yml` tolerates the identical uncapped warmup only because
   compose sets no memory limit at all; a bounded k8s Deployment cannot. tei-rerank, notably, DID
   stabilize successfully under the same emulation once its own `--max-batch-tokens` cap (compose's
   own existing flag) took effect, so Rosetta execution itself was never the blocker: a large model's
   transient warmup peak, on a dev machine also running the compose stack's own already warm TEI pair
   plus other, unrelated k3d clusters at the same time, is.

Rather than run an inconsistent split (rerank in cluster, embed external), local points BOTH services
at a real external amd64 endpoint. This is genuine portfolio material, not a gap to paper over: an
arm64 dev machine cannot always run every workload it targets, and the honest response is to name the
limitation and route around it, not to force it and call the result "working."

**Provisioning the external endpoint**: create `.env.fastlane` at the repo root (gitignored, `.gitignore`
already covers it, never committed):

```bash
# .env.fastlane
ATLAS_TEI_EMBED_URL=http://<host>:<port>
ATLAS_TEI_RERANK_URL=http://<host>:<port>
```

`task k3d:up`, `task k3d:verify`, and `task infra:render` all source this file and fail with a worded
message naming it if the two variables are still unset afterward -- checked BEFORE helmfile ever
runs, because helmfile's own value resolution here (`ref+envsubst://$VAR`, the same `vals` mechanism
`postgres.password`'s own `ref+sops://` uses, since environment values files are not Go templated the
way `helmfile.yaml` itself is) is deliberately permissive: an unset variable resolves to an empty
string, not an error (verified live; there is no separate strict "fail if unset" env provider in this
`vals` build). The Taskfile/scripts are the actual fail closed gate, not helmfile.

The node's IP address is **never** hand written into any committed file: it lives only in the
gitignored `.env.fastlane` an operator provisions locally, matching the same "encryption/secrecy is
real, not faked" discipline the SOPS section above already holds itself to.

**Live verification** (`infra/scripts/verify_tei_revisions.py`, called from `task k3d:verify`): a
direct HTTP call from this operator's own host against `<url>/info` on both endpoints, asserting
`model_id`/`model_sha` match `models.lock` exactly. The in cluster half of the same promise is
`infra/charts/tei/templates/connectivity-check-job.yaml` (rendered only in `external` mode): a Helm
pre-install hook Job, using `curlimages/curl` (D37 digest pinned, and unlike the TEI image itself a
genuine manifest LIST for multiple architectures, verified with `docker buildx imagetools inspect` to include
`linux/arm64`, so it pulls natively on this node), that curls the same endpoint from inside the
cluster and fails the release's own `helmfile sync` if the model identity does not match. Both halves
passed live: the external endpoint answered with `models.lock`'s exact revisions from both this host
and a pod inside the cluster, in seconds (an HTTP call, not a model warmup).

## `task k3d:up` (Task 2 phase 1, extended by Task 3)

```bash
task k3d:up      # cluster + registry, build/push images, operator, cluster, indexes PV/PVC, jobs, TEI check
task k3d:verify  # re run the live psql + TEI checks against an already up cluster
task k3d:down    # delete the "atlas" k3d cluster (and its registry); never touches the compose stack
```

`task k3d:up` is idempotent: reruns skip cluster creation, always rebuild and repush both images (a
local dev loop should pick up new code), and both one shot Jobs are Helm hooks that
delete and recreate on every `helmfile sync` rather than hitting Kubernetes' "Job spec is immutable"
error on a plain `helm upgrade`. `infra/scripts/k3d-up.sh` is the actual orchestration (real `docker
build`, real CNPG operator pull, several `kubectl wait`s Kubernetes has no single primitive for, see
below); the Taskfile target is a one line wrapper.

Custom image (`infra/images/postgres-pgvector/Dockerfile`): CNPG ships no pgvector bundled Postgres
image of its own (compose's `pgvector/pgvector` image is not CNPG shaped, it has its own
entrypoint/probe contract), so this layers the `postgresql-17-pgvector` apt package (PGDG, the same
source CNPG's own base image already uses) onto `ghcr.io/cloudnative-pg/postgresql:17-bookworm`
(pg17 to match compose's pinned major), both pinned by digest. Built and pushed to the k3d registry
by `task k3d:up`, referenced from a CNPG `ImageCatalog` (`infra/charts/cnpg-cluster`) rather than a
bare `imageName`, so the `Cluster` CR always runs the pgvector layered image, never a bare upstream
one. (Worth noting: the CNPG base image as of this task's implementation date already bundles
pgvector 0.8.0 on its own; this custom layer upgrades it to 0.8.5 and, more importantly, rehearses
the full custom image, registry, digest, ImageCatalog pipeline D37 asks for, which stays valuable
even on a base image that happens to already carry the extension today.)

The Dockerfile pins an exact apt package version (`postgresql-17-pgvector=0.8.5-1.pgdg12+1`, D26:
never a floating alias). Nothing bumps this automatically: there is no dependency bot watching the
PGDG apt repository the way `uv.lock`/`models.lock` are watched elsewhere in this repo. Bumping it
is a manual, deliberate edit to `infra/images/postgres-pgvector/Dockerfile` followed by a
`task k3d:up` rebuild, not something to expect a PR to do on your behalf.

The CNPG operator chart itself is vendored (`infra/charts/vendor/cloudnative-pg`, pinned
`cloudnative-pg` 0.28.1) rather than added as a live Helm repository dependency, so
`helmfile template` needs zero network at render time (this repo's hermetic `task test` contract).
`infra/charts/vendor/VENDORING.md` has the exact reproducible re vendor recipe (`helm repo add` +
`helm pull` with the pinned version) and the tarball's `sha256`, so the ~19k line vendored CRD file
is verifiable against upstream rather than being archaeology, and documents the (small, grep
verified safe) post pull removals: the optional Grafana dashboard subchart dependency, which needed
a live network fetch to resolve and would otherwise have reintroduced the same hermeticity problem
this vendoring exists to avoid.

CNPG's `bootstrap.initdb`-created "atlas" role is deliberately **not** a superuser (unlike compose's
`pgvector/pgvector` image, where `POSTGRES_USER=atlas` IS the initdb superuser), so the ingest job's
own `CREATE EXTENSION IF NOT EXISTS vector;` fails closed against a bare CNPG cluster ("must be
superuser to create this extension", found live running this task, not guessed). Fixed with CNPG's
own documented mechanism for exactly this, `bootstrap.initdb.postInitApplicationSQL` (runs as
superuser in the application database right after bootstrap), not by granting the app role
superuser.

The one shot Jobs (`infra/charts/atlas-jobs`) are translated 1:1 from `docker-compose.yml`'s
`checkpointer-migrate` and `rag-init` services: same backend image (built read only from
`backend/Dockerfile`, unmodified), same `alembic upgrade head` / `rag_tools.ingest --load-existing`
commands. The one real translation choice is the DSN: compose hardcodes it as a literal env var,
here the password comes from a Kubernetes Secret (the `atlas-postgres-credentials` Secret
`charts/scaffold` already creates, extended this task with a `username` key so it doubles as CNPG's
own `bootstrap.initdb.secret`), assembled in a small shell wrapper instead.

`indexes/` reached `rag-init` via a raw `hostPath` volume into the k3d node in Task 2 (`k3d cluster
create --volume $(pwd)/indexes:/indexes@all`), that task's own simplest correct translation of
compose's `./indexes:/app/indexes:ro` bind mount. Task 3 replaces that hostPath with the proper
`PersistentVolume`/`PersistentVolumeClaim` pair the SP5 digest's own design calls for (section 2):
`infra/charts/atlas-indexes` renders a statically provisioned PV (`storageClassName: manual`, bound
by explicit `volumeName` rather than the cluster's own dynamic default StorageClass, since this needs
to stay pinned to the SAME specific k3d node path, not a freshly and emptily provisioned volume) and
a PVC that binds to it. The underlying node path is UNCHANGED (still the same
`--volume $(pwd)/indexes:/indexes@all` mount from cluster creation); only the indirection
`atlas-jobs`' rag-init Job reaches it through changed, from an inline hostPath volume to
`persistentVolumeClaim.claimName: atlas-indexes-pvc` -- the mount PATH and the ingest command/args are
byte for byte unchanged.

**Live verification** (`infra/scripts/k3d-verify.sh`, `task k3d:verify`): connects to the CNPG pod's
local socket as the `postgres` superuser (no password needed, avoids fetching the app role's
password out of the Secret for a read only check) and asserts, via `kubectl exec` + `psql`: `chunks`
has exactly 45 rows (the committed `indexes/corpus-0.1.1-bge-m3-03f983e0` build's own
`chunk_count`), all four `langgraph-checkpoint-postgres` tables exist
(`checkpoints`/`checkpoint_blobs`/`checkpoint_writes`/`checkpoint_migrations`), and the `vector`
extension is installed. The same script then runs the TEI external endpoint checks documented above.

## Backend, web, ingress (Task 4)

`infra/charts/atlas-backend` and `infra/charts/atlas-web` translate `docker-compose.yml`'s own
`backend` and `web` services 1:1 (the same compose parity acceptance bar tasks 2 and 3 already held
themselves to): the same images (`backend/Dockerfile`/`frontend/Dockerfile`, both built read only,
unmodified), the same env var names and defaults, the same `/healthz` contract. Two real translation
choices, both already established by `atlas-jobs` (Task 2) and the `tei` chart (Task 3):

- **`ATLAS_PG_DSN`** is assembled from the `atlas-postgres-credentials` Secret in a shell wrapper
  (`command: ["sh", "-c"]` re issuing `backend/Dockerfile`'s own `CMD` after exporting the DSN),
  since Kubernetes has no native string interpolation across env vars.
- **`ATLAS_TEI_EMBED_URL`/`ATLAS_TEI_RERANK_URL`** are gated by `.Values.tei.mode` (the inherited
  requirement this task carried over from the Task 3 adjudication): `"external"` (local) passes
  through `.Values.tei.external.<role>Url` unchanged, the SAME resolved URL the connectivity-check
  Job already verifies; `"inCluster"` (burst) points at the in cluster Service DNS names the `tei`
  chart's own `service.yaml` already names for exactly this purpose (`http://tei-embed:80` /
  `http://tei-rerank:80`, matching `docker-compose.yml`'s own service name shape). Both halves are
  asserted by the render test (`test_local_environment_backend_env_carries_the_tei_passthrough_urls`,
  `test_burst_environment_backend_env_points_at_in_cluster_tei_services`).

`ATLAS_INDEX_DIR` points at `/app/indexes/corpus-0.1.1-bge-m3-03f983e0`, mounted read only from the
SAME `atlas-indexes-pvc` (Task 3) `atlas-jobs`' rag-init Job already claims: `PgvectorRetriever`
reads `fingerprint.json`/`build_manifest.json` off that path at construction (D9 fail closed
discipline), so the served backend needs the identical committed index build on disk, not one baked
into its own image (that would couple an image rebuild to every index bump). Since this is a single
node k3d cluster, the same `ReadWriteOnce` PVC being mounted by both the rag-init Job and the
backend Deployment at once is safe: RWO restricts a volume to a single NODE, not a single pod, and
every pod here runs on the one node this cluster has.

`atlas-backend`'s own Service is named literally `backend`, not `atlas-backend`: `frontend/nginx.conf`
(read only reference, never edited by this task) hardcodes `proxy_pass http://backend:8000/;`, the
exact DNS shape `docker-compose.yml`'s own network already gives the `backend` compose service.
Matching that Service name is what lets the SAME built web image proxy correctly with zero edits to
`frontend/`.

**The D35 seam**: `infra/charts/atlas-ingress` renders a k3s Traefik `IngressRoute` (`entryPoints:
web`, `PathPrefix('/')` to `atlas-web`) and a rate limit `Middleware` (HLD section 4.9). k3s ships
Traefik as its default ingress controller in both tiers (SP5 digest open decision 6), so this is the
only release: no separate ingress controller chart. The Middleware is the coarse, IP scoped EDGE
control Traefik itself can enforce; the finer grained PER SESSION token/cost budget D35 also names is
backend application code (a typed ceiling enforced in FastAPI against the signed session identity,
tripping the degradation ladder to honest refusal at the per burst spend ceiling), SP4/SP6's job, not
this chart's. The seam handed to whichever sub project implements it, per the SP5 digest's own
recommendation: `ATLAS_BURST_SPEND_CEILING_USD`, `atlas-backend`'s own Deployment env, populated from
`environments/burst/values.yaml` (`backend.burstSpendCeilingUsd: 25`, a placeholder Task 5 tunes
against real burst economics) and absent from `local` (no real spend to cap in dev). Named as a
comment in `infra/charts/atlas-ingress/templates/middleware.yaml`, not implemented there.

**Ingress port mapping**: `task k3d:up` creates the cluster with
`--port "${INGRESS_HTTP_PORT}:80@loadbalancer"` (default `8090`), reaching k3s Traefik's `web`
entryPoint from the host at `http://localhost:8090`. This only takes effect at cluster CREATE time: a
cluster already running from before this task needs one `task k3d:down` then `task k3d:up` cycle to
pick it up. `8090` is not `8080` (already taken by `docker-compose.yml`'s own `web` service) or `80`
(taken by an unrelated k3d cluster on the machine this task was developed on); override with
`INGRESS_HTTP_PORT` if it collides on yours.

**`task k3d:smoke`**: the rag smoke (`testing/harness/rag_tools/smoke.py`'s own four part structure:
retrieval half always, chat endpoint/stream halves always attempted, generation half only if a
provider key is present), ported to target the k3d tier's ingress instead of a directly published
backend port. `infra/scripts/k3d-smoke.sh` opens a temporary `kubectl port-forward` to CNPG's
`atlas-pg-rw` Service (ClusterIP only, never published to the host, unlike compose's own
`postgres:5433`), torn down on exit, so `infra/scripts/k3d_smoke.py`'s retrieval half can call the
SAME `PgvectorRetriever` class the served backend uses; the chat endpoint/stream halves go through
the real ingress (`http://localhost:8090/api/...`), proving Traefik routes to `atlas-web` (nginx),
which proxies to the `backend` Service, which reaches the served backend, end to end. The generation
half is identical in spirit to `rag_tools.smoke.py`'s own (this reference system's agent graph never
binds tools to a live model, so a full live agentic turn through `/chat` is out of scope; instead it
calls the live provider directly, grounded in the SAME retrieved passages the retrieval half already
fetched), gated on a provider key present in `.env` (sourced by the Taskfile target, never read by
the script itself).

## What Task 4 deliberately does not do

- No websecure entryPoint or TLS: local has no wildcard cert (cert-manager is Task 5's job, burst
  only); the IngressRoute only wires `entryPoints: web` (plain HTTP) today.
- No Host() based routing: both tiers route `PathPrefix('/')` unconditionally. burst's own real DNS
  name (a `Host()` matcher) is Task 5's job, once real DNS exists.
- No PSS/NetworkPolicy hardening (D40): out of this task's own scope; Task 5 owns burst's workload
  posture.
- burst's own `atlas-backend`/`atlas-web`/`atlas-ingress` renders are proven by the hermetic render
  test only, not by a live Hetzner node, since the burst tier itself does not exist until Task 5.
- burst's own `tei.mode: inCluster` render (pinned digest, probes, resources, HF cache PVCs) is
  proven by the hermetic render test and by Task 3's own earlier live testing of the SAME chart on
  this machine before the platform/memory findings that task documents; it has not been exercised
  against a real Hetzner node, since the burst tier itself does not exist until Task 5.
- The burst tier's own secrets, tofu module, cert-manager wiring, and janitor are still Task 5's job;
  `cnpg-cluster`, `atlas-jobs`, `tei-embed`/`tei-rerank`, `atlas-backend`, and `atlas-web` render for
  `burst` today with placeholder or public digest values only (proving the chart, not a real burst
  deploy).

## Observability: phoenix, otel-collector, atlas-monitoring (SP6 task 5)

Not this document's own "Task 5" (the burst tier, below): these three releases landed later, in the
SP6 sub project, claiming the two helmfile slots this plan's own Task 3/Task 5 parked (SP5 digest
open decisions), and the two are numbered separately on purpose. `infra/helmfile.yaml`'s own
`needs:` graph: `phoenix` needs `atlas-scaffold` + `cnpg-cluster` (its state is a NEW `phoenix`
database inside the SAME CNPG Postgres cluster every other component uses, created by a one shot
db-init hook Job); `otel-collector` needs `phoenix` (its own config fans out to Phoenix's OTLP
receiver plus a raw archive file); `atlas-monitoring` needs `atlas-backend` (the sentinel probe
CronJob and the Prometheus scrape target both name that Service by DNS).

**C1 fix (SP6 final review)**: `task k3d:up` now syncs all three, right after `atlas-backend`
(`infra/scripts/k3d-up.sh`'s own "7.5. observability" step), waiting on each Deployment's rollout
status the same way every other release here does. Before this fix, a fresh `task k3d:down && task
k3d:up` completed green while deploying NONE of them -- `infra/scripts/k3d-up.sh` syncs one release
at a time by name and was never extended when SP6 task 5 added these three, so the entire D29
alerting surface and the Phoenix trace backend existed on this tier only if an operator hand synced
them afterward. `infra/scripts/burst-up.sh` never had this gap (see that script's own "tier parity
strategy" comment): it runs one full, unselected `helmfile -e burst sync`, so a new release added to
`infra/helmfile.yaml` reaches burst automatically, with no script edit required.

**C2 fix (SP6 final review)**: the `arizephoenix/phoenix` image ships no shell at all, so
`phoenix`'s own Deployment assembles `PHOENIX_SQL_DATABASE_URL` via Kubernetes' native, shell free
dependent environment variable expansion (`$(PGUSER)`/`$(PGPASSWORD)` inside a later `env:` entry's
own `value:`) rather than a `command: ["sh", "-c"]` wrapper, which crash looped
(`infra/charts/phoenix/templates/deployment.yaml`'s own header comment has the full mechanism).

**C3 fix (SP6 final review)**: `atlas-monitoring`'s own `PrometheusRule` custom resource
(`templates/prometheusrule.yaml`, D13/D29) needs the `monitoring.coreos.com/v1` `PrometheusRule` CRD
type registered before it can install; a fresh cluster never had it (this chart, by design, never
installs the prometheus-operator itself, `atlas-monitoring/Chart.yaml`'s own scope boundary), so
helm rejected the WHOLE release on that one unknown kind -- Prometheus/Alertmanager/Pushgateway/the
sentinel CronJob never installed either, even though they have nothing to do with that CRD.
`infra/charts/atlas-monitoring/crds/prometheusrule-crd.yaml` vendors just that one CRD definition
from upstream prometheus-operator (that file's own header names the exact source URL, version, and
sha256) into Helm's own special `crds/` chart directory, applied once, verbatim, before any of this
chart's templates, on a first install. No controller reconciles the resulting custom resource
(unchanged, and still an explicit, honest scope boundary): the D29 rule set is genuinely evaluated
by the actually running Prometheus this release deploys, reading the identical rule content as a
native rule file (`templates/prometheus-configmap.yaml`), never through this CRD.

Live verification of the D29 rule set once `atlas-monitoring` is up: `kubectl -n atlas port-forward
svc/prometheus 9090:9090`, then `curl -s http://localhost:9090/api/v1/rules` -- expect all four
`atlas.probe`/`atlas.staleness`/`atlas.resilience`/`atlas.errors` groups present, the same D29
deterministic paging set `atlas-monitoring/values.yaml`'s own `rules.groups` declares.

## Cloud portability boundary

`infra/tofu/cluster` has exactly one contract: produce a k8s cluster, a kubeconfig, and node labels.
Everything above that line, helmfile, every chart, every values file, SOPS secrets, the images
tests build, is cloud agnostic and is proven daily against k3d, a cluster tofu never touches.

Targeting AWS, GCP, or another cloud means writing a sibling module under `infra/tofu/` that
satisfies the same contract (cluster, kubeconfig, node labels); nothing above that line changes,
since helmfile already treats "burst" as one more environment values layer, not a Hetzner specific
concept.

Hetzner was chosen for cost and API simplicity, per the HLD, not for lock in. The kube-hetzner
community module (pinned by version, see `infra/tofu/cluster/main.tf`) is the one Hetzner specific
piece; nothing else in this repo names Hetzner directly.

Object storage is S3 compatible by construction: R2 today, chosen for zero egress fees (HLD section
7.2). Swapping to AWS S3 or GCS is an endpoint and credential change, not a rewrite: `ATLAS_R2_ENDPOINT`
(`infra/scripts/burst-up.sh`) and the tofu backend's own `endpoints.s3` (`infra/tofu/cluster/backend.tf`)
are the two places that URL is configured; the bucket layout below stays identical either way.

## Burst tier (Task 5)

The Hetzner burst tier: rare, credential gated, public demos and the load lane and headline
benchmarks only (local numbers are never quoted, D3). Every file is complete, reviewable code; the
gate lives ONLY at the invocation boundary (`infra/scripts/burst-up.sh`/`burst-destroy.sh`'s own
credential check), never a TODO stub.

### Prerequisites (in addition to the table above)

| Tool | Verified version | Install |
|---|---|---|
| tofu (OpenTofu) | 1.11.5 | https://opentofu.org/docs/intro/install/ |
| hcloud (Hetzner CLI) | 1.66.0 | `brew install hcloud` |
| aws (AWS CLI, used against R2's S3 compatible API) | any recent v2 | https://aws.amazon.com/cli/ |

### Provisioning an operator SSH keypair

`infra/tofu/cluster` needs an SSH public key for node access (no default; a required tofu variable,
deliberately never read via tofu's own `file()`, see `infra/tofu/cluster/variables.tf`'s own
comment). Generate one, once, at the documented path:

```bash
ssh-keygen -t ed25519 -f ~/.config/atlas-hetzner/id_ed25519 -C atlas-burst
ssh-add ~/.config/atlas-hetzner/id_ed25519
```

The private key is never read by any script in this repo, never routed through an env var, and never
written to a committed file: `infra/scripts/burst-up.sh` reads only the `.pub` file's content (into
`TF_VAR_ssh_public_key`) and leaves `ssh_private_key` at tofu's own default (`null`), kube-hetzner's
documented ssh agent based authentication path. `ssh-add` before every `task burst:up`/`burst:destroy`.

### Credentials (never committed, never in this repo's own `.env`)

`task burst:up`/`task burst:destroy` check plain, exported shell environment variables, deliberately
NOT sourced through Taskfile's `dotenv:` mechanism the way `.env`/`.env.fastlane` are: a Hetzner
token and R2 keys are billing capable cloud credentials, a different risk tier than this repo's
existing replay lane API keys, and keeping them out of any file this repo's own tooling ever opens
is a deliberate, stricter choice.

| Variable | What it is |
|---|---|
| `HCLOUD_TOKEN` | Hetzner Cloud API token, project scoped |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | R2 S3 compatible credentials (tofu state backend, TLS secret backup/restore) |
| `ATLAS_R2_ENDPOINT` | The R2 account's S3 endpoint, `https://<accountid>.r2.cloudflarestorage.com` |

Export these in your own shell profile, a password manager integration, or `direnv`, whatever you
already use for cloud credentials; not this repo's own `.env`.

### `.env.burst`: non secret, operator specific config

Same discipline `.env.fastlane` already established for the local tier's TEI endpoint (Task 3): a
real DNS domain and an ACME contact email are operator specific, not secrets, and stay out of every
committed file (`.gitignore`):

```bash
# .env.burst
ATLAS_BURST_DOMAIN=atlas.example.com
ATLAS_BURST_ACME_EMAIL=you@example.com
```

`environments/burst/values.yaml`'s own `certManager.baseDomain`/`certManager.acmeEmail` resolve these
via helmfile's `ref+envsubst://` (the SAME permissive, not fail closed at the helmfile layer,
mechanism `tei.mode: external`'s own URLs already use); the real fail closed gate, with a worded
message naming this exact file, is `infra/scripts/burst-up.sh`, checked before any tofu/hcloud/
helmfile command runs, the same pattern `.env.fastlane`'s own gate in `k3d-up.sh` established.

### The Cloudflare DNS-01 token

Not a shell environment variable: it lives in `infra/environments/burst/secrets.enc.yaml`, SOPS
encrypted to the same two age recipients `infra/.sops.yaml` already names, the exact mechanism
`environments/local/secrets.enc.yaml`'s fake postgres password already proves. Edit it once, for
real, before a real `task burst:up`:

```bash
SOPS_AGE_KEY_FILE=~/.config/atlas-age/keys.txt sops infra/environments/burst/secrets.enc.yaml
```

replacing the committed placeholder (`REPLACE_WITH_A_REAL_CLOUDFLARE_DNS_EDIT_SCOPED_API_TOKEN_NEVER_COMMIT_A_REAL_ONE`)
with a real, DNS edit scoped (not account wide) Cloudflare API token, scoped to the zone
`ATLAS_BURST_DOMAIN` lives in.

### The tofu cluster (`infra/tofu/cluster`)

kube-hetzner (`kube-hetzner/kube-hetzner/hcloud`, pinned version, D26), one CX23 control plane, two
CX33 workers, `eu-central` (the only region CX types are sold in). D40's firewall posture: the API
server and SSH are restricted to `myipv4` (kube-hetzner's own placeholder, resolved at apply time to
the machine running `tofu apply`'s own public IPv4, never a hardcoded operator IP this repo would
need to keep updated). `enable_cert_manager = true` installs the cert-manager operator/CRDs the
`atlas-cert` chart's ClusterIssuer/Certificate need; `ingress_controller = "traefik"` configures the
SAME built in Traefik `atlas-ingress` already targets on both tiers, never a second ingress
controller.

State lives in R2 (D2: "tofu state" is durable state plane inventory, not local disk), via a
deliberately PARTIAL `backend "s3" {}` block (`infra/tofu/cluster/backend.tf`): no bucket, key,
region, or endpoint is a literal in any committed file, since an R2 endpoint URL encodes the
operator's own Cloudflare account id. `infra/scripts/burst-up.sh`/`burst-destroy.sh` generate the
missing attributes into a temp file at `tofu init -backend-config=<tmpfile>` time, then delete it.

**Validating without credentials** (this task's own acceptance bar, proven with credentials
explicitly unset, never touching a real backend or a real Hetzner API call):

```bash
cd infra/tofu/cluster
env -u HCLOUD_TOKEN -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY tofu init -backend=false
env -u HCLOUD_TOKEN -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY tofu validate
```

`tofu validate` treats a required variable with no default (`hcloud_token`, `ssh_public_key`) as an
unknown value and validates structurally without one; this is the actual mechanism the acceptance bar
rests on, not a special case carved out for this task.

### `task burst:up` / `task burst:destroy`

```bash
task burst:up       # credential gate -> tofu apply -> restore the TLS secret from R2 -> helmfile -e burst sync
task burst:destroy  # credential gate -> back up the TLS secret to R2 -> tofu destroy -> the shared orphan check
```

Both refuse to run past their own credential gate, printing exactly which of the variables above are
missing, before any `tofu`/`hcloud`/`helmfile`/`aws` command executes; see `infra/scripts/burst-up.sh`
and `burst-destroy.sh`'s own header comments for the full sequence. `burst:destroy` follows "always()
discipline": every step (TLS backup, `tofu destroy`, the orphan check) is attempted regardless of an
earlier step's own failure, so a partial teardown never silently skips the destroy call itself.

### Static DNS (D3: external-dns dropped)

No automation writes a DNS record. `task burst:up` prints the Hetzner Load Balancer's public IPv4
once the cluster is up; point `ATLAS_BURST_DOMAIN` and `*.ATLAS_BURST_DOMAIN` at that address by
hand, at your own DNS provider, once. A torn down and re created burst tier gets a NEW load balancer
IP (Hetzner does not preserve it across `tofu destroy`), so this is a manual step on every real
spin up, not a one time setup; that trade off is the HLD's own explicit choice (research note 08
recommended `external-dns`, the HLD overrides it: "static DNS record to the LB; external-dns
dropped").

### The wildcard cert: DNS-01, persisted across teardowns

`infra/charts/atlas-cert` (helmfile, burst only, gated on `certManager.enabled`): a `ClusterIssuer`
using Let's Encrypt's ACME DNS-01 challenge against Cloudflare (the only challenge type that can
prove ownership of a WILDCARD name; a single HTTP-01 challenge can only prove one concrete hostname),
and a `Certificate` for `ATLAS_BURST_DOMAIN`/`*.ATLAS_BURST_DOMAIN`, targeting the Secret
`atlas-wildcard-tls` in the `atlas` namespace (where Traefik's own `IngressRoute.tls.secretName`
resolves; wiring that reference into `atlas-ingress` is the next task's job, named as a seam in that
chart's own `certificate.yaml` comment, not implemented here).

Persisted across teardowns (D3): `burst-destroy.sh` backs `atlas-wildcard-tls` up to R2
(`tls/wildcard-cert.yaml`) BEFORE `tofu destroy`; `burst-up.sh` restores it BEFORE `helmfile sync`
runs, so cert-manager finds an already valid cert on a fresh cluster and reconciles as a no op
instead of solving a fresh DNS-01 challenge (and spending a Let's Encrypt rate limit slot) on every
single spin up. A first ever spin up has nothing to restore, the honest bootstrap case, not an error.

### The weekly janitor (`.github/workflows/janitor.yml`)

HLD section 7.3's one earned cron: lists every hcloud resource labeled `cluster=atlas-burst` (the
label kube-hetzner's own module stamps on everything it creates) and fails loudly if anything is
still there, since this reference system's own burst tier is meant to be torn down between uses.
Label based, never a scan of the whole account: this is precisely what keeps the user's standing
`atlas-fastlane` box (id `152778751`, hand created, never carrying this label) out of scope by
construction, not by an incidental name difference; the same `infra/scripts/hcloud-orphans.sh` also
backs `task burst:destroy`'s own post teardown check, one definition of "orphan," never duplicated.

**Push activated by design**: this workflow file does not run anywhere until pushed to GitHub (this
implementation session never runs `git push`, per its own hard safety rules). Once pushed, it still
reports a NEUTRAL skip, not a failure, until the `HCLOUD_TOKEN` repository secret is added (an absent
credential is not the same condition as "orphans exist"). A heartbeat step, independent of that
credential, asserts the gap since this workflow's own last successful run stays under the expected
weekly cadence, so a schedule GitHub silently stopped firing (or a chain of failures) surfaces as a
failure the next time this workflow runs at all, rather than staying invisible.

## R2 bucket layout (D2 inventory)

One bucket, `atlas-durable`, prefixes below (keeps lifecycle rule scoping simple, one bucket to
provision instead of several). Provisioning the bucket itself is a one time, out of band step (the
Cloudflare dashboard or `wrangler`), not a tofu resource: bootstrapping the very backend tofu's own
state depends on would be a chicken and egg problem for a single, cheaply hand created bucket.

| Prefix | Item (D2 inventory) | Tier |
|---|---|---|
| `pg-backup/` | CNPG barman backups (bootstrap from backup recovery) | burst only |
| `tofu-state/` | tofu state (`infra/tofu/cluster/backend.tf`) | burst only |
| `embed-cache/` | embedding cache | both, optional locally |
| `corpus/` | corpus tarballs | both, optional locally |
| `datasets/` | datasets and label JSONL | both, optional locally |
| `runs/` | run manifests, results.parquet, the runs index | both, optional locally |
| `indexes/` | chunks.parquet per index build | both, optional locally |
| `traces/` | raw OTLP traces (SP6 owned emitter) | burst only |
| `burst-reports/` | teardown burst reports | burst only |
| `tls/` | the persisted wildcard TLS secret (`wildcard-cert.yaml`) | burst only |

Retention is lifecycle rules on these prefixes (HLD section 6.4's own retention matrix for the
non infra items above); no lifecycle rule is provisioned by this task, since that requires the
bucket, and therefore real R2 credentials, to already exist. Named here as the target shape a real
provisioning step configures against, not implemented as tofu code with nothing real to apply it to.

## What Task 5 deliberately does not do

- Does not apply anything: `tofu apply`/`task burst:up` were never run against real credentials by
  this task's own implementation work (hard safety rule). `tofu validate` (no credentials, no state,
  no network call to Hetzner) is this task's own acceptance bar for the tofu module.
- No PSS/NetworkPolicy hardening (D40): the firewall half (API server/SSH restricted to the operator
  IP) is written (`infra/tofu/cluster/main.tf`); Pod Security Standards and default deny
  NetworkPolicies encoding the plane boundary are not, named here as a real, still open gap rather
  than silently left out. A future task owns it.
- No CI wrapper invoking `task burst:destroy` inside an `always()` block (HLD section 7.3's "Burst
  benchmark" lane): `.github/workflows/janitor.yml` is the one workflow file this task's own
  ownership map permits; the burst benchmark CI lane itself is future, SP6 adjacent work.
- No `atlas-ingress` wiring of `atlas-wildcard-tls` into a websecure entryPoint: named as a seam in
  `infra/charts/atlas-cert/templates/certificate.yaml`'s own comment, not implemented here.
- No real container registry for burst images: `environments/burst/values.yaml`'s own
  `registry.invalid` placeholder is unchanged (Task 5's own scope was DNS names, not the image
  pipeline). `release.yml` (SP6 task 6) is now the CI pipeline that builds, pushes, and joins real
  GHCR image digests into the release manifest, per the D37 split Task 1's own report already
  names, but it never writes into `environments/burst/values.yaml`: recording a real digest there,
  once a real burst registry exists, is still unassigned, future work.
- No burst postgres credentials: `infra/charts/scaffold/templates/secret.yaml` gates the
  `atlas-postgres-credentials` Secret on `.Values.postgres`, which no burst values layer sets, while
  burst's cnpg-cluster bootstrap, both one shot Jobs, and the backend Deployment all reference that
  Secret by name. A real `task burst:up` would wedge on it; today every real burst path is
  unreachable behind `burst-up.sh`'s fail closed stops, which name this gap. Wiring a real burst
  `secrets.enc.yaml` (SOPS, the same mechanism local already proves) is a hard prerequisite,
  together with the indexes restore below, for ever removing those stops.
- No burst indexes restore from R2: `environments/burst/values.yaml`'s `indexes.hostPath` still
  points at the same k3d only local path `environments/local/values.yaml` uses, which does not exist
  on a real Hetzner node. Implementing this means restoring `indexes/` from R2 onto the node or a PVC
  before the backend release installs; `docs/runbooks/corpus-bump.skeleton.md` names the future, not
  yet assigned, sub project that owns the retrieval index build pipeline and this wiring. Until it
  exists, `infra/scripts/burst-up.sh` stops with a worded message right after the credential gate,
  before any `tofu` invocation, so a real `task burst:up` can never bring up a backend mounting a
  nonexistent path.
- The R2 tofu backend's S3 compatibility flags (`use_path_style`, `skip_requesting_account_id`, and
  the rest of `backend.tf`'s generated block) were written from documented S3 compatible backend
  conventions, not verified against a live R2 bucket (no credentials available or permitted during
  this task). The first real `task burst:up` may need one or two additional backend flags; treat this
  as a real, not merely theoretical, risk worth attention before the very first live apply.
