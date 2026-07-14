"""SP6 task 3 parked the otel-collector/phoenix image digests in `infra/environments/base/
values.yaml` for whichever later task (Task 5, per the SP5 digest's own open decision 11) claimed
the SP5 reserved helmfile slot and wrote the actual chart content. Task 5 is that task:
`infra/charts/otel-collector` and `infra/charts/phoenix` now read this EXACT field
(`.Values.observability.otelCollector.image`/`.phoenix.image`, see either chart's own values.yaml
comment) rather than declaring a second, independently pinned digest for the same image, so this
file's own parity checks against docker-compose.yml stay the single source of truth for both the
compose profile and the two helm releases. `test_infra_manifests.py` owns the actual render
assertions (the ConfigMap content, the digest reaching the Deployment, and so on); this file stays a
direct, hermetic YAML read of the parked field itself.
"""
from __future__ import annotations

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE_VALUES_PATH = ROOT / "infra" / "environments" / "base" / "values.yaml"
COMPOSE_PATH = ROOT / "docker-compose.yml"

_DIGEST_SHAPE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _base_values() -> dict:
    return yaml.safe_load(BASE_VALUES_PATH.read_text())


def test_observability_block_is_present_with_both_images():
    values = _base_values()
    assert "observability" in values
    assert set(values["observability"]) == {"otelCollector", "phoenix"}


def test_both_images_are_digest_shaped():
    values = _base_values()["observability"]
    for component in ("otelCollector", "phoenix"):
        digest = values[component]["image"]["digest"]
        assert _DIGEST_SHAPE.match(digest), f"{component} digest is not shape sha256:<64 hex>: {digest!r}"
        assert values[component]["image"]["repository"]  # non empty


def test_parked_digests_match_docker_compose_exactly():
    """Parity: the same image, pinned to the same digest, in both places (D3's "one set of charts"
    principle applied across compose and helm too, the same way the TEI images already match between
    docker-compose.yml and infra/environments/base/values.yaml's own tei.image block)."""
    values = _base_values()["observability"]
    compose_text = COMPOSE_PATH.read_text()

    otel_repo = values["otelCollector"]["image"]["repository"]
    otel_digest = values["otelCollector"]["image"]["digest"]
    assert f"{otel_repo}@{otel_digest}" in compose_text

    phoenix_repo = values["phoenix"]["image"]["repository"]
    phoenix_digest = values["phoenix"]["image"]["digest"]
    assert f"{phoenix_repo}@{phoenix_digest}" in compose_text


def test_otel_collector_and_phoenix_releases_now_claim_the_parked_digests():
    """The "park, not claim" boundary's own deliberate flip (SP5 digest open decision 11: SP6 task 3
    parked digests, SP6 task 5 writes the actual release content -- this file's own docstring, and
    the prior version of this exact test, named the "not yet claimed" state explicitly so this
    change would be visible and reviewed, not a silent scope creep). Both charts now exist and both
    releases are wired into infra/helmfile.yaml; test_infra_manifests.py's own render tests prove the
    parked digest actually reaches each Deployment."""
    helmfile_text = (ROOT / "infra" / "helmfile.yaml").read_text()
    assert "name: otel-collector" in helmfile_text
    assert "name: phoenix" in helmfile_text
    assert (ROOT / "infra" / "charts" / "otel-collector" / "Chart.yaml").is_file()
    assert (ROOT / "infra" / "charts" / "phoenix" / "Chart.yaml").is_file()


def test_both_charts_read_the_parked_field_rather_than_a_second_pinned_digest():
    """No drift possible: both charts' own values.yaml declare the identical
    `observability.otelCollector`/`observability.phoenix` field path this file already parity checks
    above, used only as the bare `helm template` (outside helmfile) fallback -- helmfile always
    overrides it with THIS file's own real value at render time (infra/helmfile.yaml's `values:`
    list, environments/base/values.yaml first)."""
    otel_chart_values = yaml.safe_load(
        (ROOT / "infra" / "charts" / "otel-collector" / "values.yaml").read_text()
    )
    assert "otelCollector" in otel_chart_values["observability"]

    phoenix_chart_values = yaml.safe_load((ROOT / "infra" / "charts" / "phoenix" / "values.yaml").read_text())
    assert "phoenix" in phoenix_chart_values["observability"]
