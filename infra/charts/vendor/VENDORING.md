# Vendored charts: re vendoring recipe

`infra/charts/vendor/` holds third party Helm charts fetched once and committed, not added as a
live Helm repository dependency (`helmfile template` needs zero network at render time, this repo's
hermetic `task test` contract; see `infra/charts/vendor/cloudnative-pg/Chart.yaml`'s own comment).
This file is the reproducible recipe for re pulling and re verifying each one, so the 19k line
vendored CRD file is checkable against upstream instead of being archaeology.

## `cloudnative-pg` (the CNPG operator chart)

```bash
helm repo add cnpg https://cloudnative-pg.github.io/charts   # idempotent if already added
helm repo update cnpg
helm pull cnpg/cloudnative-pg --version 0.28.1
shasum -a 256 cloudnative-pg-0.28.1.tgz
```

Expected digest of the pulled tarball, verified against this exact command sequence when the chart
was first vendored (SP5 task 2):

```
26c0abd3b68db82b6d797c5db8818b26b5a46c3b7c43668de8a6cc22818c9f44  cloudnative-pg-0.28.1.tgz
```

A mismatch means either the tarball changed upstream (re review before trusting it) or something
went wrong locally (stale `helm repo update`, a proxy rewriting content, etc.) -- do not commit a
re vendor whose tarball digest does not match this line without first finding out why.

### What was removed after pulling, and why

`helm pull --untar` reproduces the chart exactly as published; three things were then removed from
the committed copy, none of which change what the chart renders in this repo (verified: nothing in
`templates/*.yaml` references any of them via `.Subcharts` or `.Files.Get`, grep checked):

- **`Chart.lock`** and the **`dependencies:`** block in `Chart.yaml` (an aliased `cluster` subchart
  from `https://cloudnative-pg.github.io/grafana-dashboards`, condition
  `monitoring.grafanaDashboard.create`, default `false`): resolving it needs a live `helm repo add`/
  `helm dependency build` against an external repo, which breaks the zero network render guarantee.
- **`charts/cluster/`**: the untarred copy of that same optional subchart (a Grafana dashboard
  ConfigMap generator), unused now that the dependency is gone.
- **`monitoring/`**: static reference assets (a dashboard JSON, a Prometheus metrics YAML) not
  loaded by any template via `.Files.Get` (grep verified); dead weight once the dependency above is
  gone.

To re vendor a newer version: run the three commands above with the new `--version`, verify the
tarball digest independently (update the expected digest in this file to match, with a one line note
of the version bump), replace `infra/charts/vendor/cloudnative-pg/` with the freshly pulled and
untarred contents, then reapply the same three removals (or re check whether the upstream chart
still ships that optional dependency at all before removing it blindly).

## Adding a new vendored chart

Same recipe shape: `helm repo add <name> <url>` (idempotent), `helm repo update <name>`, `helm pull
<name>/<chart> --version <pinned>`, record the tarball's `shasum -a 256` in this file next to the
exact command used to produce it, untar into `infra/charts/vendor/<chart>/`, document any
post pull modifications (removed files, stripped dependencies) and why, the same way the section
above does.
