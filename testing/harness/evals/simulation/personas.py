"""The persona roster: a customer is a disposition and a goal, expressed well enough that a model
plays it across many turns without breaking character. Atlas's roster picks itself from the support
queue. Each persona is a generator the operator lane points at the agent; the mind-changer is the one
the hermetic gate exercises as a recorded fixture.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    disposition: str
    goal: str


PERSONAS: list[Persona] = [
    Persona(
        "mind-changer",
        "reverses direction mid-conversation to see whether the agent keeps up",
        "switch to the faster plan, after considering and rejecting a cheaper one",
    ),
    Persona(
        "confused",
        "describes the symptom, never the name of the plan",
        "find out why the connection dies every evening",
    ),
    Persona(
        "impatient",
        "wants the change made now and treats every clarifying question as an obstacle",
        "get the plan changed without being asked to confirm twice",
    ),
    Persona(
        "bargain-hunter",
        "circles the same discount from new angles, politely",
        "talk the agent into a price it should not give",
    ),
    Persona(
        "non-native-speaker",
        "phrasing is correct but unidiomatic, the kind synthetic data never produces",
        "understand what the bill actually covers",
    ),
]
