"""Synthetic railway-rescheduling requirements with multi-label constraint-family gold (P2-H2b).

Why this exists: P2-H2b asks whether a typed model can *drive* retrieval in front of an MILP
generator, i.e. read a raw requirement ("no overtaking at Riesa, keep 3 minutes headway") and
select the constraint-family templates the model will need. The literature catalogue behind
P2-H2a is private and small, so this benchmark is fully synthetic and publishable: every statement
is generated from a seed, and its gold label set is known by construction.

Design choices:

- **Label space from lp2graph.** The families are read from ``lp2graph.core.model.
  ConstraintDomainClass`` at runtime (minus ``unclassified``), so the benchmark uses the exact
  identifiers of the Paper-1 representation; the template table is checked against the enum.
- **Negation (NevIR-style).** Two kinds of negated sentences: a *prohibition* still needs its
  family ("No overtaking at X" is precedence/ordering, "Trains may not be cancelled" fixes the
  cancellation decisions), while a *waiver* must not trigger the family it mentions ("headways
  need not be modelled"). Lexical routers see the same words in both and cannot tell them apart.
- **Distractors.** Background sentences with irrelevant numbers and clock times.
- **Held-out paraphrases.** Each family's templates are split: the ``dev`` pool informed the
  family descriptions and is used for threshold tuning; ``test`` statements are built only from
  the remaining templates, so no description or tuning ever saw a test paraphrase.

Gold labels encode the template author's intent; a few statements are arguably multi-family
beyond their gold set (e.g. a delay definition that also feeds the objective). This label noise
is identical for every arm.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Literal, get_args

__all__ = [
    "DISTRACTORS",
    "FAMILY_SPECS",
    "DescriptionStyle",
    "FamilySpec",
    "Statement",
    "constraint_families",
    "family_descriptions",
    "generate",
    "keyword_route",
    "template_texts",
]

Split = Literal["dev", "test"]

#: Number of templates per family and kind that go to the dev pool (the rest are held out).
DEV_POS, DEV_FLIP, DEV_WAIVER = 2, 1, 1


def constraint_families() -> tuple[str, ...]:
    """Constraint families of lp2graph's closed ``ConstraintDomainClass`` enum, in enum order.

    ``unclassified`` is dropped: it is a fallback bucket, not something a requirement can ask for.
    """
    from lp2graph.core.model import ConstraintDomainClass

    return tuple(f for f in get_args(ConstraintDomainClass) if f != "unclassified")


@dataclass(frozen=True)
class FamilySpec:
    """Descriptions, keyword list and sentence templates of one constraint family.

    ``description`` speaks the modeller's language (what the constraint does); ``cue`` says what a
    requirement asks for, phrased to complete "Does the statement ask for ...?". Which of the two
    an arm sees is selected on the dev split only.
    ``pos``: plain requirements; ``flip``: prohibitions that still need the family;
    ``waiver``: sentences stating the family is *not* needed (gold excludes it).
    """

    description: str
    cue: str
    keywords: tuple[str, ...]
    pos: tuple[str, ...]
    flip: tuple[str, ...]
    waiver: tuple[str, ...]

    def pool(self, kind: str, split: Split) -> list[tuple[int, str]]:
        """(index, template) pairs of one kind in one split."""
        items = list(enumerate(getattr(self, kind)))
        n_dev = {"pos": DEV_POS, "flip": DEV_FLIP, "waiver": DEV_WAIVER}[kind]
        return items[:n_dev] if split == "dev" else items[n_dev:]


# Descriptions are written from the family names and generic OR knowledge. They deliberately do
# not quote any template (checked in tests), so the test paraphrases are unseen.
FAMILY_SPECS: dict[str, FamilySpec] = {
    "assignment_covering": FamilySpec(
        description=(
            "Assignment and covering: every element of one set is matched to elements of another,"
            " exactly once or at least once. Examples: each train or trip receives one platform,"
            " track, route alternative, vehicle or crew; every task or service is covered."
        ),
        cue=(
            "every train, trip or duty to receive exactly one (or at least one) resource such as"
            " a platform, route, vehicle or driver"
        ),
        keywords=("assign", "allocat", "cover", "exactly one", "single track", "operated by"),
        pos=(
            "Every train arriving at {station} must be given exactly one platform.",
            "Each trip in the new timetable has to be operated by one of the available units.",
            "All duties of the evening shift must be covered by a driver.",
            "Train {train} needs exactly one of its alternative routes through the {station}"
            " interlocking.",
            "Assign every service on line {line} to one of the {n} available train sets.",
            "Each departing train should be allocated to a single track in the yard at {station}.",
        ),
        flip=(
            "No trip may be left without a driver.",
            "No train may be given more than one platform at {station}.",
            "No duty of the night shift may remain uncovered.",
        ),
        waiver=(
            "Crew assignment is out of scope for this request and need not be modelled.",
            "You do not have to decide which unit runs which trip.",
            "Platforms need not be allocated; the station manager will handle that separately.",
        ),
    ),
    "flow_conservation": FamilySpec(
        description=(
            "Flow conservation: at every node and time, what flows in equals what flows out or"
            " stays. Examples: rolling-stock or vehicle balance at stations and depots, circulation"
            " of units over the day, passenger flow through transfer nodes."
        ),
        cue=(
            "vehicles, units or passengers to be balanced, so that what arrives at a place either"
            " leaves again or stays there"
        ),
        keywords=("balance", "conserv", "circulation", "parked", "appear", "vanish", "disappear"),
        pos=(
            "At the end of the day, each depot must hold the same number of units as in the"
            " morning.",
            "Every train set that arrives at {station} either departs again or stays parked"
            " there overnight.",
            "Passengers who leave one train at {station} must continue on another train or exit"
            " the station.",
            "Rolling stock cannot appear out of nowhere: the units leaving {depot} equal those that"
            " arrived there plus the ones already parked.",
            "Keep the circulation balanced so that the {n} units end up where tomorrow's timetable"
            " expects them.",
            "Track how many carriages are in {station} over time; each arrival adds to the stock"
            " and each departure removes from it.",
        ),
        flip=(
            "Units must not appear or disappear at any station: arrivals and departures have to"
            " balance.",
            "No train set may be lost from the circulation between {station} and {station2}.",
            "Passengers arriving at {station} must not vanish from the model; each one transfers"
            " or leaves the station.",
        ),
        waiver=(
            "The overnight position of the rolling stock does not matter and does not have to"
            " balance.",
            "We do not need to track how many units are parked at {depot}.",
            "Passenger flows through {station} can be ignored in this study.",
        ),
    ),
    "capacity_resource": FamilySpec(
        description=(
            "Capacity of a shared resource: the number of trains, units or staff using a resource"
            " at the same time is limited. Examples: tracks in a station, siding length, parking"
            " positions in a depot, available drivers or vehicles."
        ),
        cue=("a limit on how many trains, units or staff may use a resource at the same time"),
        keywords=("capacity", "at most", "no more than", "space for", "only", "at once", "limit"),
        pos=(
            "At most {n} trains can stand in {station} at the same time.",
            "The siding at {station} holds only two trains.",
            "The depot at {depot} has space for {n} units only.",
            "Only {n} drivers are available for the replacement services.",
            "Platform {platform} is short and can take only one train at a time.",
            "No more than {n} train sets may be in use during the peak hour.",
        ),
        flip=(
            "No more than {n} trains may wait at {station} at once.",
            "Platform {platform} must never hold more than one train.",
            "The depot at {depot} must not receive more than {n} units.",
        ),
        waiver=(
            "Platform capacity at {station} is not an issue today and need not be checked.",
            "There is no limit on the number of drivers we can use.",
            "Parking space at {depot} does not need to be considered.",
        ),
    ),
    "precedence_ordering": FamilySpec(
        description=(
            "Precedence and ordering: the sequence of trains or events. Examples: which train uses"
            " a shared track or junction first, keeping or changing the planned sequence, rules"
            " about overtaking, one event taking place before another."
        ),
        cue=(
            "a particular order between trains or events, such as which train goes first or a"
            " rule about overtaking"
        ),
        keywords=("order", "sequence", "overtak", "first", "before train", "follow", "pass "),
        pos=(
            "Train {train} must pass {station} before train {train2}.",
            "Keep the planned order of trains on the line between {station} and {station2}.",
            "Decide which of the two trains uses the junction at {station} first.",
            "The freight train should follow the regional train {train} into {station}, not the"
            " other way round.",
            "Trains leaving {depot} must enter the main line in the order of their numbers.",
            "The sequence of trains through the single-track section may be changed if that"
            " reduces delay.",
        ),
        flip=(
            "No overtaking at {station}.",
            "Train {train2} must not overtake train {train} before {station}.",
            "The freight train may not pass the regional train {train} between {station} and"
            " {station2}.",
        ),
        waiver=(
            "The order in which trains pass {station} is not relevant for this problem.",
            "Which train goes first at the junction does not need to be decided here.",
            "The train sequence on the line to {station2} can be ignored.",
        ),
    ),
    "headway_separation": FamilySpec(
        description=(
            "Headway and separation: two consecutive trains on the same track, block section or"
            " platform keep a minimum distance in time or space, for example a buffer or"
            " signalling headway."
        ),
        cue=("a minimum time or distance between two consecutive trains on the same track"),
        keywords=(
            "headway",
            "apart",
            "spacing",
            "spaced",
            "buffer",
            "block section",
            "between two",
        ),
        pos=(
            "Keep at least {min} minutes between two trains on the same track.",
            "A minimum headway of {min} minutes applies on the line to {station}.",
            "Following trains must be spaced by at least one free block section.",
            "After a train leaves platform {platform}, the next one may enter only {min} minutes"
            " later.",
            "Two trains running in the same direction need a buffer of {min} minutes between them.",
            "Respect the signalling distance between successive trains from {station} to"
            " {station2}.",
        ),
        flip=(
            "No train may follow another closer than {min} minutes.",
            "Trains must not run less than {min} minutes apart on the section to {station}.",
            "Consecutive trains must never be less than one block section apart.",
        ),
        waiver=(
            "Headways do not need to be modelled on the depot tracks.",
            "Minimum spacing between trains can be ignored on the freight line.",
            "Buffer times between consecutive trains are not required in this study.",
        ),
    ),
    "timing_window": FamilySpec(
        description=(
            "Time windows: an event time must fall inside an interval. Examples: no departure"
            " ahead of the timetable, arrival before a deadline, entry to a line only within a"
            " slot, an upper limit on the delay of an event."
        ),
        cue=(
            "an event to happen not before, not after, or within a given time, or a limit on the"
            " delay of an event"
        ),
        keywords=("arrive", "deadline", "earlier", "later than", "window", "timetabled", "by "),
        pos=(
            "Train {train} has to arrive at {station} by {hour}:{mm}.",
            "The freight train can only enter the line between {hour}:00 and {hour2}:00.",
            "Train {train} should leave {station} within {min} minutes of its scheduled time.",
            "The track possession ends at {hour}:{mm}, so work trains must be clear of the line"
            " by then.",
            "The delay of any train at its final stop may be at most {min} minutes.",
            "The connecting bus leaves at {hour}:{mm}; train {train} has to reach {station}"
            " before that.",
        ),
        flip=(
            "No train may depart earlier than timetabled.",
            "Train {train} must not arrive at {station} later than {hour}:{mm}.",
            "The freight train may not enter the line before {hour}:00.",
        ),
        waiver=(
            "There is no deadline for the empty stock move to {depot}.",
            "Early departures are acceptable today; trains need not wait for their scheduled time.",
            "Arrival times at {station} are not bound to any time window.",
        ),
    ),
    "coupling_linking_definition": FamilySpec(
        description=(
            "Coupling, linking and definition: one variable is defined from others, or decisions"
            " are logically linked. Examples: delay defined as realised minus planned time,"
            " cancelling a train cancels its remaining stops, keeping a connection makes the"
            " connecting train wait, a route usable only if its switch is set."
        ),
        cue=(
            "a logical link between decisions (if one thing happens, another must follow) or a"
            " quantity defined from other quantities"
        ),
        keywords=("if ", "whenever", "unless", "as well", "minus", "missed", "connection", "means"),
        pos=(
            "If train {train} is cancelled, all of its later stops are cancelled as well.",
            "The delay of each train is its actual arrival minus its planned arrival at {station}.",
            "When the connection at {station} is maintained, train {train2} must wait for train"
            " {train}.",
            "A train can only use the diversion via {station2} if the diversion is opened.",
            "Count a passenger connection as missed whenever the feeder train arrives too late"
            " for the transfer.",
            "Short-turning train {train} at {station} means it also skips the stops beyond"
            " {station}.",
        ),
        flip=(
            "A cancelled train must not serve any of its remaining stops.",
            "If the connection at {station} is kept, train {train2} must not leave before train"
            " {train} has arrived.",
            "A train cannot run on the diversion via {station2} unless the diversion is opened.",
        ),
        waiver=(
            "Connections at {station} do not need to be kept; transfers can be ignored.",
            "Delay does not have to be computed as a separate quantity.",
            "There is no need to link the cancellation of a train to its return trip.",
        ),
    ),
    "periodic_modulo_pesp": FamilySpec(
        description=(
            "Periodic timetabling (PESP): events repeat with a fixed cycle time and time"
            " differences are taken modulo the period. Examples: a 30- or 60-minute clock-face"
            " pattern, cyclic departure minutes, a symmetric periodic node."
        ),
        cue=(
            "a timetable pattern that repeats with a fixed period, such as every 30 or 60 minutes"
        ),
        keywords=("period", "cyclic", "cycle", "repeat", "clock-face", "hourly", "rhythm"),
        pos=(
            "The rescheduled timetable must still repeat every {period} minutes.",
            "Keep the clock-face pattern: trains on line {line} leave at the same minute every"
            " hour.",
            "The hourly cycle at {station} has to be preserved in the new plan.",
            "Plan a cyclic timetable with a period of {period} minutes for the replacement"
            " service.",
            "Departure minutes must be identical in every period of the day.",
            "Maintain the periodic node structure at {station} with a {period}-minute cycle.",
        ),
        flip=(
            "The {period}-minute rhythm must not be broken by the new timetable.",
            "Departure minutes may not differ from one hour to the next.",
            "No departure may leave the clock-face pattern of line {line}.",
        ),
        waiver=(
            "The replacement plan does not have to follow the hourly pattern.",
            "Periodicity can be dropped for the rest of the day.",
            "The timetable does not need to repeat every {period} minutes.",
        ),
    ),
    "subtour_connectivity": FamilySpec(
        description=(
            "Subtour elimination and connectivity: tours, rotations or paths form one connected"
            " chain. Examples: a vehicle rotation or crew duty begins and ends at its home depot"
            " without isolated cycles, a train's path is a gap-free sequence of track sections."
        ),
        cue=(
            "tours, vehicle rotations, crew duties or paths to form one connected chain or closed"
            " cycle"
        ),
        keywords=("rotation", "return", "closed", "connected", "continuous", "loop", "chain"),
        pos=(
            "Every vehicle rotation must start and end at {depot}.",
            "Each crew duty has to form one continuous sequence of trips beginning and ending at"
            " the home base.",
            "The rerouted path of train {train} must be a connected sequence of track sections"
            " from {station} to {station2}.",
            "Build the circulation so that each unit follows a single closed cycle through the"
            " network.",
            "Driver shifts have to return to {station} at the end.",
            "A locomotive's trips have to link up into one unbroken chain.",
        ),
        flip=(
            "No rotation may form a separate loop that never returns to {depot}.",
            "A crew duty must not break into disconnected pieces.",
            "The diverted path of train {train} must not have gaps between track sections.",
        ),
        waiver=(
            "Vehicle rotations do not need to close at the depot today.",
            "We do not care whether crews return to their home base.",
            "It does not matter whether a unit's trips link up into a closed cycle.",
        ),
    ),
    "variable_bound_fix": FamilySpec(
        description=(
            "Variable bounds and fixings: particular decisions are settled in advance. Examples:"
            " events that already happened keep their times, a given train keeps its planned"
            " platform or route, an option is forbidden or forced so its decision variable is"
            " fixed to 0 or 1."
        ),
        cue=(
            "a specific decision to be settled in advance: kept at its current or planned value,"
            " forbidden, or forced"
        ),
        keywords=("fixed", "fix ", "freeze", "keep its", "keeps its", "already", "cancel"),
        pos=(
            "Train {train} has already left {station}; its departure time is fixed.",
            "Train {train} keeps its planned platform {platform} at {station}.",
            "The international train {train} must run on its planned route.",
            "Train {train} is to be cancelled; that decision is already taken.",
            "Freeze the timetable of all trains that have already started their journey.",
            "The route via {station2} is closed, so that option is fixed to unused.",
        ),
        flip=(
            "Trains may not be cancelled.",
            "Train {train} must not be diverted from its planned route.",
            "No train may change its platform at {station}.",
        ),
        waiver=(
            "Nothing is fixed in advance; every train remains open to rescheduling.",
            "No decision has been taken yet for train {train}; all its options are still open.",
            "None of the current decisions has to be frozen.",
        ),
    ),
    "objective_defining": FamilySpec(
        description=(
            "Objective-defining constraints: auxiliary constraints that measure the quantity being"
            " optimised. Examples: an upper envelope of all delays when the maximum delay is"
            " minimised, the lateness above a tolerance that is penalised, a count of late trains"
            " that enters the objective."
        ),
        cue=(
            "something to be minimised or penalised, such as the largest delay or the number of"
            " late trains"
        ),
        keywords=("minimi", "maximum", "largest", "worst", "penalis", "objective", "small as"),
        pos=(
            "Minimise the largest delay of any train.",
            "Only delay beyond {min} minutes should be penalised in the objective.",
            "The goal is to minimise the number of trains arriving more than {min} minutes late.",
            "Keep the worst delay at {station} as small as possible.",
            "Penalise each minute that train {train} runs behind schedule at its final stop, with"
            " double weight for the express.",
            "Reduce the maximum passenger delay across all trains to a minimum.",
        ),
        flip=(
            "The worst delay must not be larger than necessary; minimise it.",
            "Do not let any train's delay exceed what is unavoidable: minimise the maximum delay.",
            "Trains arriving more than {min} minutes late should not go unpenalised.",
        ),
        waiver=(
            "We do not need to optimise anything in particular; any feasible plan is fine.",
            "The size of the worst delay does not matter here.",
            "Delays do not need to be penalised in this run.",
        ),
    ),
}

#: Background sentences with irrelevant numbers; they never add a family.
DISTRACTORS: tuple[str, ...] = (
    "The line from {station} to {station2} is {km} km long.",
    "Station {station} was opened in {year}.",
    "The dispatcher on duty has {n} years of experience.",
    "Outside it is {temp} degrees and dry.",
    "About {pax} passengers use {station} on a typical weekday.",
    "The incident was reported at {hour}:{mm} by the driver of train {train}.",
    "Train {train} is formed of {coaches} coaches today.",
    "The request was filed under ticket number {ticket}.",
    "The last timetable change took place in December {year}.",
)

_STATIONS = (
    "Riesa",
    "Coswig",
    "Meissen",
    "Pirna",
    "Freiberg",
    "Doebeln",
    "Grossenhain",
    "Bischofswerda",
    "Radebeul Ost",
    "Dresden-Neustadt",
)
_DEPOTS = ("Dresden-Altstadt depot", "Leipzig West depot", "Chemnitz depot", "Riesa yard")
_TRAINS = ("IC 2045", "RE 50", "RB 31", "S 1", "ICE 1537", "RE 3", "RB 71", "IC 2443")
_LINES = ("S1", "RE 50", "RB 31", "S3", "RE 18")


DescriptionStyle = Literal["modelling", "requirement"]


def family_descriptions(style: DescriptionStyle = "modelling") -> dict[str, str]:
    """Family id -> description (``modelling``) or requirement cue (``requirement``), enum order."""
    attr = "description" if style == "modelling" else "cue"
    return {f: getattr(FAMILY_SPECS[f], attr) for f in constraint_families()}


def _check_specs() -> tuple[str, ...]:
    fams = constraint_families()
    missing = sorted(set(fams) - set(FAMILY_SPECS))
    extra = sorted(set(FAMILY_SPECS) - set(fams))
    if missing or extra:
        raise ValueError(
            f"template table out of sync with lp2graph ConstraintDomainClass: "
            f"missing={missing}, not in enum={extra}"
        )
    return fams


def _slots(rng: random.Random) -> dict[str, str]:
    st = rng.sample(_STATIONS, 2)
    tr = rng.sample(_TRAINS, 2)
    hour = rng.randint(5, 21)
    return {
        "station": st[0],
        "station2": st[1],
        "train": tr[0],
        "train2": tr[1],
        "depot": rng.choice(_DEPOTS),
        "line": rng.choice(_LINES),
        "n": str(rng.randint(2, 6)),
        "min": str(rng.randint(2, 10)),
        "platform": str(rng.randint(1, 8)),
        "hour": f"{hour:02d}",
        "hour2": f"{hour + 2:02d}",
        "mm": f"{rng.randrange(0, 60, 5):02d}",
        "period": str(rng.choice((30, 60, 120))),
        "km": str(rng.randint(8, 140)),
        "year": str(rng.randint(1848, 2019)),
        "temp": str(rng.randint(-8, 31)),
        "pax": f"{rng.randint(2, 60) * 500:,}",
        "ticket": str(rng.randint(10000, 99999)),
        "coaches": str(rng.randint(3, 12)),
    }


@dataclass(frozen=True)
class Statement:
    """One requirement statement with its gold constraint families."""

    sid: str
    split: str
    text: str
    gold: tuple[str, ...]
    flipped: tuple[str, ...]
    waived: tuple[str, ...]
    n_distractors: int
    template_ids: tuple[str, ...]

    @property
    def negated(self) -> bool:
        """Whether the statement contains a prohibition or a waiver (the negation subset)."""
        return bool(self.flipped or self.waived)

    def to_dict(self) -> dict[str, object]:
        return {
            "sid": self.sid,
            "split": self.split,
            "text": self.text,
            "gold": list(self.gold),
            "flipped": list(self.flipped),
            "waived": list(self.waived),
            "n_distractors": self.n_distractors,
            "template_ids": list(self.template_ids),
        }


def generate(
    n: int,
    seed: int = 20260928,
    split: Split = "test",
    p_flip: float = 0.2,
    p_waiver: float = 0.3,
) -> list[Statement]:
    """``n`` seeded statements from one template split.

    Each statement has 1-3 gold families (weights 0.4/0.4/0.2), each family phrased either plainly
    or (with ``p_flip``, if the split has one) as a prohibition; with ``p_waiver`` one waiver of a
    non-gold family is added; 0-2 distractors are mixed in and the sentence order is shuffled.
    """
    fams = _check_specs()
    rng = random.Random(f"d4r-h2b:{seed}:{split}")
    out: list[Statement] = []
    for i in range(n):
        k = rng.choices((1, 2, 3), weights=(0.4, 0.4, 0.2))[0]
        gold = sorted(rng.sample(fams, k), key=fams.index)
        sentences: list[str] = []
        tids: list[str] = []
        flipped: list[str] = []
        for f in gold:
            spec = FAMILY_SPECS[f]
            flips = spec.pool("flip", split)
            if flips and rng.random() < p_flip:
                idx, tpl = rng.choice(flips)
                flipped.append(f)
                tids.append(f"{f}:flip:{idx}")
            else:
                idx, tpl = rng.choice(spec.pool("pos", split))
                tids.append(f"{f}:pos:{idx}")
            sentences.append(tpl.format(**_slots(rng)))
        waived: list[str] = []
        if rng.random() < p_waiver:
            others = [f for f in fams if f not in gold]
            w = rng.choice(others)
            wpool = FAMILY_SPECS[w].pool("waiver", split)
            if wpool:
                idx, tpl = rng.choice(wpool)
                waived.append(w)
                tids.append(f"{w}:waiver:{idx}")
                sentences.append(tpl.format(**_slots(rng)))
        n_dis = rng.choices((0, 1, 2), weights=(0.4, 0.4, 0.2))[0]
        for j in rng.sample(range(len(DISTRACTORS)), n_dis):
            sentences.append(DISTRACTORS[j].format(**_slots(rng)))
            tids.append(f"distractor:{j}")
        rng.shuffle(sentences)
        out.append(
            Statement(
                sid=f"{split}-{i:03d}",
                split=split,
                text=" ".join(sentences),
                gold=tuple(gold),
                flipped=tuple(flipped),
                waived=tuple(waived),
                n_distractors=n_dis,
                template_ids=tuple(tids),
            )
        )
    return out


def keyword_route(text: str) -> tuple[str, ...]:
    """Families whose keyword list hits ``text`` (case-insensitive substring match).

    A deliberately naive, negation-blind lexical router. Its keyword lists were written by the
    same author as the templates, so it is an optimistic lexical baseline.
    """
    low = text.lower()
    fams = constraint_families()
    return tuple(f for f in fams if any(kw in low for kw in FAMILY_SPECS[f].keywords))


def template_texts(split: Split) -> list[str]:
    """All family templates of one split with slots blanked (for leakage checks)."""
    out = []
    for spec in FAMILY_SPECS.values():
        for kind in ("pos", "flip", "waiver"):
            out += [re.sub(r"\{[a-z0-9]+\}", "", t) for _, t in spec.pool(kind, split)]
    return out
