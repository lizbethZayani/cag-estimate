"""Synthetic 20-turn conversations for the memory stress test.

Three profiles probe different memory behaviours:

* ``growing``: coherent requirements added every turn (pure accumulation).
* ``pivot``: turn 5 switches the frontend from React to Flutter.
* ``contradiction``: the budget is 30k EUR at turn 3 and 80k EUR at turn 8.

Every profile opens with the project name ("Nimbus") so the runner can check that
the oldest fact survives eviction from the sliding window. ``fact_introduced`` and
``fact_field`` label the fact a turn adds (None when it only adds free requirements).
"""

from dataclasses import dataclass, replace

TRACKED_PROJECT = "Nimbus"
TURN_COUNTS = (1, 3, 6, 10, 20)
TOTAL_TURNS = 20

_OPENING = (
    f"We want to build {TRACKED_PROJECT}, a customer portal where clients follow their orders "
    "and open support tickets."
)

# Requirements 2..20 shared by every profile unless a profile overrides a turn.
_REQUIREMENTS = (
    "Customers sign in with email and password, and can reset the password by email.",
    "Add a dashboard that lists the active orders with their current status.",
    "Each order has a detail page with a timeline of status changes.",
    "Support tickets can be opened from any order and include file attachments.",
    "Agents need an internal inbox to answer tickets and reassign them to a colleague.",
    "Send an email notification whenever an order changes status.",
    "Customers can download invoices as PDF files from the order detail page.",
    "We need role based access: customer, agent and administrator.",
    "Administrators manage users and can deactivate accounts.",
    "The portal must be available in English and Spanish.",
    "Add full text search across orders and tickets.",
    "Provide a weekly report of open tickets by agent for the support manager.",
    "Integrate with our existing ERP through its REST API to read orders.",
    "All pages must meet WCAG AA accessibility and work on mobile browsers.",
    "Add an audit log of administrator actions that cannot be edited.",
    "Customers can rate a resolved ticket from one to five stars.",
    "Set up automated backups and a staging environment for acceptance tests.",
    "Plan a short pilot with ten customers before the general launch.",
    "Prepare user documentation and a recorded onboarding video for agents.",
)


@dataclass(frozen=True)
class ScenarioTurn:
    turn_index: int
    transcript: str
    fact_introduced: str | None = None
    fact_field: str | None = None

    def __post_init__(self) -> None:
        if len(self.transcript) < 20:
            raise ValueError("A scenario turn needs at least 20 characters of transcript")


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: tuple[ScenarioTurn, ...]

    @property
    def tracked_fact(self) -> str:
        """The turn-1 fact the memory metric looks for in every later snapshot."""
        fact = self.turns[0].fact_introduced
        if fact is None:
            raise ValueError("The first turn must introduce a fact")
        return fact


def _build(name: str, overrides: dict[int, ScenarioTurn]) -> Scenario:
    texts = (_OPENING, *_REQUIREMENTS)
    turns = tuple(
        overrides.get(index, ScenarioTurn(index, text)) for index, text in enumerate(texts, 1)
    )
    first = replace(turns[0], fact_introduced=TRACKED_PROJECT, fact_field="project_name")
    return Scenario(name, (first, *turns[1:]))


def _growing() -> Scenario:
    return _build("growing", {})


def _pivot() -> Scenario:
    overrides = {
        2: ScenarioTurn(
            2,
            "The frontend will be a React single page application with Tailwind for styling.",
            "React",
            "frontend",
        ),
        5: ScenarioTurn(
            5,
            "Change of plan: we are dropping React and building the frontend as a Flutter app "
            "for web and mobile instead.",
            "Flutter",
            "frontend",
        ),
    }
    return _build("pivot", overrides)


def _contradiction() -> Scenario:
    overrides = {
        3: ScenarioTurn(
            3,
            "The total budget for the whole project is 30k EUR and it is a hard limit.",
            "30k EUR",
            "budget",
        ),
        8: ScenarioTurn(
            8,
            "Correction: finance approved more money, the total budget is now 80k EUR.",
            "80k EUR",
            "budget",
        ),
    }
    return _build("contradiction", overrides)


_BUILDERS = {"growing": _growing, "pivot": _pivot, "contradiction": _contradiction}
SCENARIO_NAMES = tuple(_BUILDERS)


def get_scenario(name: str) -> Scenario:
    """The full 20-turn scenario called ``name`` (KeyError when unknown)."""
    return _BUILDERS[name]()


def slice_scenario(scenario: Scenario, turns: int) -> Scenario:
    """The first ``turns`` turns (all of them when the scenario is shorter)."""
    if turns < 1:
        raise ValueError("turns must be >= 1")
    return Scenario(scenario.name, scenario.turns[:turns])
