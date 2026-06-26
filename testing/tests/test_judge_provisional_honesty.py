"""KAPPA HONESTY, machine enforced (SP8 task 3, binding global constraint). This repo's own
documented prior failure: a kappa reported as 0.29 in Taskfile/judge prose did not match the 0.21
actually sitting in the committed artifact. `judge.provisional`'s two numbers (registry truth
agreement, judge vs judge kappa) are exactly the shape of claim that lesson warns against, so this
file proves, by static inspection and by runtime behaviour, not by trusting a docstring, that
neither one can ever be compared against `AUTOMATION_BAR` / `quality.gate.gate_on_lower_bound` (the
0.6 deployment bar D15 reserves for real human gold, `judge.calibration.CalibrationReport.licensed`).
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
from datetime import datetime, timezone
from unittest.mock import patch

from judge import provisional
from judge.contract import JudgeContract
from judge.provisional import (
    JudgeVsJudgeAgreement,
    ProvisionalCalibrationArtifact,
    RegistryTruthAgreement,
    judge_vs_judge_kappa,
    manufactured_cases,
    provisional_calibration_artifact,
    registry_truth_agreement,
)
from quality.gate import gate_on_lower_bound

_CONTRACT_A = JudgeContract("openai:gpt-5.4-nano", "groundedness-v1", "tmpl-a")
_CONTRACT_B = JudgeContract("anthropic:claude-haiku-4-5-20251001", "groundedness-v1", "tmpl-b")
_CLOCK = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)


def _perfect_artifact() -> ProvisionalCalibrationArtifact:
    """A provisional reading where BOTH numbers would clear the 0.6 deployment bar if anyone
    (wrongly) compared them against it: registry truth agreement 1.0, judge vs judge kappa 1.0.
    This is the exact temptation the honesty rule guards against, constructed deliberately."""
    cases = manufactured_cases()
    labels = [c.ground_truth for c in cases]  # a judge that gets every manufactured case right
    rt = registry_truth_agreement(_CONTRACT_A, cases, labels, generated_at=_CLOCK)
    jvj = judge_vs_judge_kappa(
        _CONTRACT_A, _CONTRACT_B, [c.case_id for c in cases], labels, labels, generated_at=_CLOCK
    )
    return provisional_calibration_artifact(rt, jvj, generated_at=_CLOCK)


# ---- static: the provisional module's own CODE never references the deployment gate machinery -----
# (the module docstring's KAPPA HONESTY paragraph legitimately NAMES AUTOMATION_BAR/licensed in
# prose, to explain why they are absent; this walks the AST, not a raw substring search, so it
# checks real imports/name/attribute references, never a documentation mention, for exactly that
# reason.)


def test_provisional_module_code_never_references_the_deployment_gate_machinery():
    tree = ast.parse(inspect.getsource(provisional))
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    referenced |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            referenced |= {alias.asname or alias.name for alias in node.names}
    forbidden = {"AUTOMATION_BAR", "gate_on_lower_bound", "GateVerdict", "GateDecision", "licensed"}
    hit = forbidden & referenced
    assert not hit, (
        f"judge.provisional's CODE must never reference {hit}: a provisional number must have no "
        "code path that could compare it against the deployment bar"
    )


# ---- structural: none of the three dataclasses expose a licensed/gate shaped attribute at all ------


def test_registry_truth_agreement_has_no_licensing_shaped_attribute():
    names = {f.name for f in dataclasses.fields(RegistryTruthAgreement)}
    assert "licensed" not in names and "gate_decision" not in names
    rt = _perfect_artifact().registry_truth
    assert not hasattr(rt, "licensed")
    assert not hasattr(rt, "gate_decision")


def test_judge_vs_judge_agreement_has_no_licensing_shaped_attribute():
    names = {f.name for f in dataclasses.fields(JudgeVsJudgeAgreement)}
    assert "licensed" not in names and "gate_decision" not in names
    jvj = _perfect_artifact().judge_vs_judge
    assert not hasattr(jvj, "licensed")
    assert not hasattr(jvj, "gate_decision")


def test_provisional_calibration_artifact_has_no_licensing_shaped_attribute():
    names = {f.name for f in dataclasses.fields(ProvisionalCalibrationArtifact)}
    assert "licensed" not in names and "gate_decision" not in names
    artifact = _perfect_artifact()
    assert not hasattr(artifact, "licensed")
    assert not hasattr(artifact, "gate_decision")


# ---- runtime: a narrower, HONEST check that sits on top of the AST guard above ----------------------
#
# This spy patches `quality.gate.gate_on_lower_bound` at its definition site, the module attribute
# where the real function lives. That only intercepts a call reached through a live reference to
# that module at call time, the shape `import quality.gate` then `quality.gate.gate_on_lower_bound(x)`.
# Proven red first in a disposable worktree: tampering judge.provisional to call the gate that way,
# from inside registry_truth_agreement (a function this test actually exercises through
# _perfect_artifact), makes this spy raise "Called 1 times" while it is active. It bites.
#
# It does NOT intercept the shape `from quality.gate import gate_on_lower_bound` then a bare call:
# that import binds an independent name in judge.provisional's own namespace once, at import time,
# and patching the origin module's attribute afterward never reaches an already bound local name.
# Proven red first in the same worktree, same tamper site: that form runs the real gate end to end
# while this spy still reports zero calls, so a claim of "never calls the gate" from this spy alone
# would be false for that case.
#
# `test_provisional_module_code_never_references_the_deployment_gate_machinery` above is the one
# guard that catches both shapes, because it reads judge.provisional's own source for the literal
# name instead of intercepting execution at run time; it is the real enforcement here. This spy adds
# a second, narrower signal on the module reference shape, run against the real provisional functions
# through a perfect calibration reading, and is not a substitute for the AST test above.


def test_a_perfect_provisional_reading_never_reaches_the_gate_through_a_live_module_reference():
    with patch("quality.gate.gate_on_lower_bound", wraps=gate_on_lower_bound) as spy:
        artifact = _perfect_artifact()
        artifact.render()
    spy.assert_not_called()
    # confirm this really was the tempting case: both numbers WOULD clear 0.6 if compared.
    assert artifact.registry_truth.agreement == 1.0
    assert artifact.judge_vs_judge.kappa == 1.0


# ---- the rendered artifact says so explicitly, AND labels each number by its real source -----------


def test_render_states_neither_number_licenses_deployment():
    text = _perfect_artifact().render().lower()
    assert "does not license" in text or "never licenses" in text
    assert "human gold" in text
    assert "0.6" in text


def test_render_labels_each_number_by_its_own_actual_source():
    text = _perfect_artifact().render()
    assert RegistryTruthAgreement.SOURCE in text
    assert JudgeVsJudgeAgreement.SOURCE in text
    assert RegistryTruthAgreement.SOURCE != JudgeVsJudgeAgreement.SOURCE
