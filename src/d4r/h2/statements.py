"""Seeded generator of noisy railway-rescheduling problem statements, with gold labels and optimum.

Why this exists: a relevance filter in front of an MILP generator (the "driver of RAG" use of Jev,
docs/hypotheses.md P2-H2) is only useful if it keeps *every* sentence the model needs. Dropping one
release time silently changes the model, so recall of data-bearing sentences is a hard floor, and
precision only buys prompt tokens. Real dispatching messages are noisy in a specific way: they
mix the data with other text that also carries numbers (unit numbers, passenger counts, tickets,
years, platforms, staff numbers) and with *near misses* that use model vocabulary for another line
or another day (a delay on the neighbouring line, a closure planned for next week, yesterday's
running time, a cancelled train, tomorrow's speed restriction), and sometimes a value is corrected
later in the same message, which makes the first statement of it obsolete (kind ``superseded``,
not needed) and the correction needed. The numbered distractors defeat a "keep what has a number"
filter; the near misses and superseded values test whether a filter reads meaning, not keywords. raiLParchitect's noisy cases motivate
the setting; the generator here is synthetic, English and publishable.

Problem family (one statement = one instance):

- a single-track corridor of 2-3 block sections, all trains run in the same direction through all
  sections in order and may wait (and overtake) at the intermediate stations;
- 2-4 trains with a release time (earliest entry into the first section), a running time per
  section, a due time at the exit of the last section and a priority weight (default 1);
- a headway: after a train leaves a section, the next train may enter it ``headway`` minutes later;
- optional "X may not overtake Y" rules (Y runs ahead of X on every section), a train fault
  (extra running time on one section) or a section blockage (no entry before a given minute);
- objective: minimise the total priority-weighted tardiness at the end of the corridor.

Every sentence carries a gold ``needed`` label and a ``kind``; every statement carries its exact
numeric ``instance`` (a plain dict). :func:`reference_optimum` computes the exact optimum by brute
force over the train order on every section (with a fixed order per section the earliest-start
schedule minimises every completion time, so the minimum over all orders is the MILP optimum), and
:func:`reference_milp` builds the same model as a :class:`~d4r.engines.milp.MilpSpec` so that the
two can be cross-checked offline with HiGHS. Station, train and staff names are fictional.
"""

from __future__ import annotations

import itertools
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from d4r.engines.milp import MilpSpec, Objective, Row, Var

__all__ = [
    "DATA_KINDS",
    "FRAME_KINDS",
    "NOISE_KINDS",
    "Reference",
    "Sentence",
    "Statement",
    "effective_run",
    "make_statement",
    "make_statements",
    "reference_milp",
    "reference_optimum",
    "schedule_for_orders",
    "weighted_tardiness",
]

#: Sentence kinds that carry model data (a dropped one changes the instance).
DATA_KINDS = frozenset(
    {
        "release",
        "due",
        "timing",
        "runtime",
        "headway",
        "priority",
        "no_overtake",
        "fault",
        "block",
        "correction",
    }
)
#: Sentence kinds that define the problem structure (line, occupancy, overtaking, objective).
FRAME_KINDS = frozenset({"frame"})
#: Sentence kinds that are not needed.
NOISE_KINDS = frozenset({"distractor", "near_miss", "chatter", "superseded"})

_DIGIT = re.compile(r"\d")

# Fictional corridors (4 stations each; a 2-section corridor uses the first three).
_CORRIDORS: tuple[tuple[str, ...], ...] = (
    ("Kleinwalde", "Tannberg", "Moorbach", "Eschenhain"),
    ("Lindenau", "Hohenfels", "Steinbrück", "Wiesental"),
    ("Rotenbach", "Grünau", "Feldheim", "Ahornstadt"),
    ("Birkfeld", "Kiefernau", "Weidenbach", "Ostermark"),
    ("Talheim", "Brunnwald", "Erlenhof", "Sandau"),
    ("Quellental", "Rabenstein", "Heidegrund", "Falkenried"),
)
_OTHER_LINES: tuple[tuple[str, str], ...] = (
    ("Nordhafen", "Ulmenfeld"),
    ("Mühlberg", "Seedorf"),
    ("Fichtenau", "Kranichsee"),
    ("Buchholz", "Lerchenfeld"),
)
_ONWARD = ("Großstadt Hauptbahnhof", "Hafenstadt", "Bergstadt", "Messestadt")

#: (category, prefix, number range, running-time range per section in minutes, long name)
_CATEGORIES: tuple[tuple[str, str, tuple[int, int], tuple[int, int], str], ...] = (
    ("ic", "IC", (2000, 2999), (5, 10), "intercity"),
    ("ec", "EC", (100, 399), (5, 10), "international express"),
    ("re", "RE", (4000, 4999), (6, 12), "regional express"),
    ("rb", "RB", (16000, 16999), (8, 14), "stopping regional train"),
    ("gz", "GZ", (44000, 49999), (10, 18), "freight train"),
)
_NUMBER_WORDS = {2: "two", 3: "three", 4: "four"}
_STAFF = (
    "Anna Weber",
    "Jonas Keller",
    "Petra Novak",
    "Mehmet Aydin",
    "Laura Schmidt",
    "Felix Braun",
)
_COLOURS = ("red", "silver", "green", "blue-and-white", "yellow")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")


@dataclass(frozen=True)
class Sentence:
    """One sentence of a statement with its gold label."""

    sid: str
    text: str
    kind: str
    needed: bool

    @property
    def has_number(self) -> bool:
        """Whether the sentence contains a digit (what a regex filter keys on)."""
        return bool(_DIGIT.search(self.text))

    @property
    def data_bearing(self) -> bool:
        return self.kind in DATA_KINDS


@dataclass(frozen=True)
class Statement:
    """A noisy problem statement: labelled sentences in reading order plus the exact instance."""

    stmt_id: str
    sentences: tuple[Sentence, ...]
    instance: dict[str, Any]

    def ids(self) -> list[str]:
        return [s.sid for s in self.sentences]

    def needed_ids(self) -> frozenset[str]:
        return frozenset(s.sid for s in self.sentences if s.needed)

    def data_ids(self) -> frozenset[str]:
        return frozenset(s.sid for s in self.sentences if s.data_bearing)

    def trap_ids(self) -> frozenset[str]:
        """Not-needed sentences that contain a number (the numeracy-trap subset)."""
        return frozenset(s.sid for s in self.sentences if not s.needed and s.has_number)

    def text(self, keep: Iterable[str] | None = None) -> str:
        """The statement as one paragraph, optionally only the kept sentences (original order)."""
        k = None if keep is None else set(keep)
        return " ".join(s.text for s in self.sentences if k is None or s.sid in k)


# --------------------------------------------------------------------------- instance
def _join(items: Sequence[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _make_instance(rng: random.Random) -> dict[str, Any]:
    n = rng.choice((2, 3, 3, 4, 4))
    k = rng.choice((2, 3))
    stations = list(rng.choice(_CORRIDORS)[: k + 1])
    sections = [f"{a}–{b}" for a, b in itertools.pairwise(stations)]
    cats = [rng.choice(_CATEGORIES) for _ in range(n)]
    names: list[str] = []
    trains: list[dict[str, Any]] = []
    for cat, prefix, (lo_n, hi_n), (lo_p, hi_p), long_name in cats:
        name = f"{prefix} {rng.randint(lo_n, hi_n)}"
        while name in names:
            name = f"{prefix} {rng.randint(lo_n, hi_n)}"
        names.append(name)
        run = [rng.randint(lo_p, hi_p) for _ in range(k)]
        release = rng.randint(0, 12)
        due = release + sum(run) + rng.randint(0, 10)
        trains.append(
            {
                "name": name,
                "category": cat,
                "long_name": long_name,
                "release": release,
                "run": run,
                "due": due,
                "weight": 1,
            }
        )
    for t in rng.sample(trains, rng.randint(1, min(2, n - 1))):
        t["weight"] = rng.choice((2, 3, 4))
    no_overtake: list[dict[str, str]] = []
    if rng.random() < 0.45:
        behind, ahead = rng.sample(names, 2)
        no_overtake.append({"behind": behind, "ahead": ahead})
    incident = rng.choices(("fault", "block", "none"), weights=(0.4, 0.3, 0.3))[0]
    faults: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    if incident == "fault":
        t = rng.choice(trains)
        faults.append(
            {"train": t["name"], "section": sections[rng.randrange(k)], "extra": rng.randint(3, 9)}
        )
    elif incident == "block":
        blocks.append({"section": sections[rng.randrange(k)], "until": rng.randint(8, 30)})
    corrections: list[dict[str, Any]] = []
    if rng.random() < 0.4:
        t = rng.choice(trains)
        if rng.random() < 0.5:
            old = t["release"] + rng.choice((-4, -3, -2, 2, 3, 4))
            corrections.append({"train": t["name"], "field": "release", "old": max(0, old)})
            if corrections[-1]["old"] == t["release"]:
                corrections[-1]["old"] = t["release"] + 3
        else:
            j = rng.randrange(k)
            old = max(1, t["run"][j] + rng.choice((-3, -2, 2, 3)))
            corrections.append({"train": t["name"], "field": "runtime", "section": j, "old": old})
    return {
        "stations": stations,
        "sections": sections,
        "trains": trains,
        "headway": rng.choice((2, 3, 4)),
        "no_overtake": no_overtake,
        "faults": faults,
        "blocks": blocks,
        "corrections": corrections,
        "objective": "min sum weight * max(0, exit_last_section - due)",
    }


# --------------------------------------------------------------------------- sentences
def _runtime_text(rng: random.Random, name: str, run: Sequence[int], secs: Sequence[str]) -> str:
    if rng.random() < 0.5:
        parts = [f"{p} minutes for {s}" for p, s in zip(run, secs, strict=True)]
        return f"{name} needs {_join(parts)}."
    parts = [f"{s} {p} min" for p, s in zip(run, secs, strict=True)]
    return f"Running times of {name}: {', '.join(parts)}."


def _core_sentences(rng: random.Random, inst: dict[str, Any]) -> list[tuple[str, str]]:
    """(kind, text) of every core sentence; the first one is the line description.

    Kinds in :data:`NOISE_KINDS` (a ``superseded`` sentence that a later correction replaces) are
    not needed; every other kind is.
    """
    st, secs, trains = inst["stations"], inst["sections"], inst["trains"]
    a, z, k = st[0], st[-1], len(secs)
    out: list[tuple[str, str]] = []
    out.append(
        (
            "frame",
            rng.choice(
                (
                    f"You are dispatching the single-track line from {a} to {z}, which has"
                    f" {_NUMBER_WORDS[k]} block sections in running order: {_join(secs)}.",
                    f"The task concerns the single-track line {a}–{z}; in running order its"
                    f" {_NUMBER_WORDS[k]} block sections are {_join(secs)}.",
                )
            ),
        )
    )
    out.append(
        (
            "frame",
            rng.choice(
                (
                    "Each section can hold only one train at a time, and every train runs from"
                    f" {a} towards {z} through all sections in that order.",
                    f"All trains travel in the direction of {z} and pass every section in order;"
                    " a section is occupied by at most one train at a time.",
                )
            ),
        )
    )
    mids = st[1:-1]
    where = (
        f"At the intermediate station {mids[0]}"
        if len(mids) == 1
        else f"At the intermediate stations {_join(mids)}"
    )
    out.append(
        (
            "frame",
            f"{where} a train may wait in a passing loop, so trains can overtake each other there"
            " unless a rule forbids it.",
        )
    )
    out.append(
        (
            "frame",
            rng.choice(
                (
                    "The aim is to minimise the total weighted tardiness: the minutes by which"
                    " each train leaves the last section after its due time, multiplied by the"
                    " train's priority weight, summed over all trains.",
                    "Dispatching should minimise the sum over all trains of priority weight times"
                    " tardiness, where tardiness is how many minutes after its due time a train"
                    " leaves the line (zero if it is on time).",
                )
            ),
        )
    )
    split_timing = len(trains) <= 3 and rng.random() < 0.5
    fix = {c["train"]: c for c in inst.get("corrections", [])}
    for t in trains:
        name, r, d = t["name"], t["release"], t["due"]
        c = fix.get(name)
        r_fix = c is not None and c["field"] == "release"
        r_old = c["old"] if r_fix and c is not None else r
        if split_timing:
            release_tpl = rng.choice(
                (
                    "{name} will be ready to enter {sec} at minute {r}.",
                    "{name} can enter the line at {a} at minute {r} at the earliest.",
                )
            )
            text = release_tpl.format(name=name, sec=secs[0], a=a, r=r_old)
            if r_fix:
                out.append(("superseded", text))
                out.append(
                    (
                        "correction",
                        f"Correction to the earlier message: {name} can enter the line at minute"
                        f" {r}, not at minute {r_old}.",
                    )
                )
            else:
                out.append(("release", text))
            out.append(
                (
                    "due",
                    rng.choice(
                        (
                            f"{name} is due to leave the line at {z} by minute {d}.",
                            f"The timetable has {name} leaving {secs[-1]} at minute {d};"
                            " any later is tardy.",
                        )
                    ),
                )
            )
        else:
            text = (
                f"{name} can enter the line at {a} at minute {r_old} and is due at {z} by"
                f" minute {d}."
            )
            if r_fix:
                out.append(("superseded", text))
                out.append(
                    (
                        "correction",
                        f"Correction to the earlier message: {name} can enter the line at {a} at"
                        f" minute {r} (not {r_old}) and is still due at {z} by minute {d}.",
                    )
                )
            else:
                out.append(("timing", text))
        if c is not None and c["field"] == "runtime":
            old_run = list(t["run"])
            old_run[c["section"]] = c["old"]
            out.append(("superseded", _runtime_text(rng, name, old_run, secs)))
            parts = [f"{p} minutes for {s}" for p, s in zip(t["run"], secs, strict=True)]
            out.append(
                (
                    "correction",
                    f"Correction: the running times of {name} are {_join(parts)} (the earlier"
                    f" figure for {secs[c['section']]} was wrong).",
                )
            )
        else:
            out.append(("runtime", _runtime_text(rng, name, t["run"], secs)))
    h = inst["headway"]
    out.append(
        (
            "headway",
            rng.choice(
                (
                    "Between one train leaving a section and the next train entering the same"
                    f" section, at least {h} minutes must pass.",
                    f"On every section the minimum headway is {h} minutes, counted from the exit"
                    " of one train to the entry of the next.",
                )
            ),
        )
    )
    for t in trains:
        if t["weight"] != 1:
            out.append(
                (
                    "priority",
                    rng.choice(
                        (
                            f"{t['name']} has priority weight {t['weight']}.",
                            f"{t['name']} is a {t['long_name']} and counts with weight"
                            f" {t['weight']} in the objective.",
                        )
                    ),
                )
            )
    out.append(
        (
            "priority",
            rng.choice(
                (
                    "Every train without a stated weight has priority weight 1.",
                    "All other trains count with weight 1.",
                )
            ),
        )
    )
    for rule in inst["no_overtake"]:
        x, y = rule["behind"], rule["ahead"]
        out.append(
            (
                "no_overtake",
                rng.choice(
                    (
                        f"{x} may not overtake {y}: {y} must run ahead of {x} on every section.",
                        f"{y} must stay ahead of {x} on all sections, because they share a crew.",
                    )
                ),
            )
        )
    for f in inst["faults"]:
        out.append(
            (
                "fault",
                rng.choice(
                    (
                        f"Because of a door fault, {f['train']} will need {f['extra']} extra"
                        f" minutes on {f['section']}.",
                        f"{f['train']} is running with a traction defect and needs {f['extra']}"
                        f" minutes more than usual on {f['section']}.",
                    )
                ),
            )
        )
    for b in inst["blocks"]:
        out.append(
            (
                "block",
                rng.choice(
                    (
                        f"{b['section']} is blocked by a broken-down freight train until minute"
                        f" {b['until']}; no train may enter it earlier.",
                        f"A signal failure keeps {b['section']} closed until minute {b['until']}.",
                    )
                ),
            )
        )
    return out


def _other_train(rng: random.Random, taken: Sequence[str]) -> str:
    _, prefix, (lo, hi), _, _ = rng.choice(_CATEGORIES)
    name = f"{prefix} {rng.randint(lo, hi)}"
    while name in taken:
        name = f"{prefix} {rng.randint(lo, hi)}"
    return name


def _noise_pools(
    rng: random.Random, inst: dict[str, Any]
) -> tuple[list[str], list[str], list[str]]:
    """Candidate (near-miss, distractor, chatter) sentences for one instance."""
    st, secs = inst["stations"], inst["sections"]
    names = [t["name"] for t in inst["trains"]]
    a, z = st[0], st[-1]
    p, q = rng.choice(_OTHER_LINES)

    def tr() -> str:
        return rng.choice(names)

    def sec() -> str:
        return rng.choice(secs)

    near = [
        f"On the neighbouring line from {p} to {q}, {_other_train(rng, names)} is running"
        f" {rng.randint(4, 35)} minutes late.",
        f"{sec()} will be closed for track work next {rng.choice(_WEEKDAYS)} for"
        f" {rng.randint(60, 240)} minutes.",
        f"Before the resignalling in {rng.randint(1995, 2019)}, the headway on this line was"
        f" {inst['headway'] + rng.randint(2, 5)} minutes.",
        f"Yesterday {tr()} needed {rng.randint(15, 30)} minutes for {sec()} because of a signal"
        " failure.",
        f"Last week a freight train broke down on {sec()} and blocked it for"
        f" {rng.randint(40, 180)} minutes.",
        f"After {z}, {tr()} continues to {rng.choice(_ONWARD)}, where it is due at minute"
        f" {rng.randint(60, 140)}.",
        f"On the {p}–{q} line, {_other_train(rng, names)} has priority weight"
        f" {rng.randint(2, 5)} tonight.",
        f"The line speed on {sec()} is {rng.choice((60, 80, 100, 120))} km/h.",
        f"{_other_train(rng, names)} has been cancelled today, so its planned entry into"
        f" {secs[0]} at minute {rng.randint(0, 15)} no longer applies.",
        f"Tomorrow {tr()} will need {rng.randint(12, 25)} minutes for {sec()} because of a"
        " temporary speed restriction.",
    ]
    distractor = [
        f"{tr()} is formed of unit {rng.randint(1400, 1499)} {rng.randint(100, 999)} in the"
        f" {rng.choice(_COLOURS)} livery.",
        f"About {rng.randint(40, 420)} passengers are on board {tr()} today.",
        f"A passenger on {tr()} reported ticket {rng.choice('KLMNPR')}{rng.randint(100000, 999999)}"
        " as lost.",
        f"{rng.choice(st)} station opened in {rng.randint(1860, 1925)}.",
        f"At {z}, {tr()} will arrive at platform {rng.randint(1, 6)}.",
        f"Dispatcher {rng.choice(_STAFF)} (staff number {rng.randint(1000, 9999)}) took over the"
        f" desk at {rng.randint(5, 9):02d}:{rng.choice((0, 15, 30, 45)):02d}.",
        f"The line from {a} to {z} is {rng.randint(18, 64)} kilometres long.",
        f"The signal box at {rng.choice(st)} dates from {rng.randint(1905, 1985)} and has"
        f" {rng.randint(12, 60)} levers.",
    ]
    chatter = [
        "The weather along the line is dry and calm.",
        f"Passengers on {tr()} have asked for more frequent announcements.",
        "The coffee machine in the control centre is out of order again.",
        f"The catering trolley on {tr()} has run out of sandwiches.",
    ]
    return near, distractor, chatter


def make_statement(index: int, seed: int = 2026) -> Statement:
    """Statement number ``index`` of the seeded family ``seed`` (deterministic)."""
    rng = random.Random(f"h2-filtering:{seed}:{index}")
    inst = _make_instance(rng)
    core = _core_sentences(rng, inst)
    n_core = len(core)
    total = rng.randint(max(10, n_core + 3), max(min(25, n_core + 10), n_core + 3))
    n_noise = total - n_core
    near_pool, dist_pool, chat_pool = _noise_pools(rng, inst)
    n_near = rng.randint(1, max(1, n_noise // 2))
    n_chat = rng.randint(0, max(0, min(2, n_noise - n_near - 1)))
    n_dist = n_noise - n_near - n_chat
    noise = (
        [("near_miss", s) for s in rng.sample(near_pool, min(n_near, len(near_pool)))]
        + [("distractor", s) for s in rng.sample(dist_pool, min(n_dist, len(dist_pool)))]
        + [("chatter", s) for s in rng.sample(chat_pool, min(n_chat, len(chat_pool)))]
    )
    body = [(kind, text, kind not in NOISE_KINDS) for kind, text in core[1:]] + [
        (kind, text, False) for kind, text in noise
    ]
    rng.shuffle(body)
    kinds = [b[0] for b in body]
    if "superseded" in kinds and kinds.index("correction") < kinds.index("superseded"):
        i, j = kinds.index("correction"), kinds.index("superseded")
        body[i], body[j] = body[j], body[i]  # a correction follows what it corrects
    ordered = [(core[0][0], core[0][1], True), *body]
    sentences = tuple(
        Sentence(sid=f"s{i + 1:02d}", text=text, kind=kind, needed=need)
        for i, (kind, text, need) in enumerate(ordered)
    )
    return Statement(stmt_id=f"h2f-{seed}-{index:03d}", sentences=sentences, instance=inst)


def make_statements(n: int = 60, seed: int = 2026) -> list[Statement]:
    """The first ``n`` statements of family ``seed``."""
    return [make_statement(i, seed) for i in range(n)]


# --------------------------------------------------------------------------- reference
@dataclass(frozen=True)
class Reference:
    """Exact optimum of an instance: objective, one optimal order per section, and its schedule."""

    objective: int
    orders: tuple[tuple[str, ...], ...]
    schedule: dict[str, list[tuple[int, int]]]


def effective_run(inst: dict[str, Any]) -> dict[str, list[int]]:
    """Running time per train and section after faults."""
    idx = {s: i for i, s in enumerate(inst["sections"])}
    run = {t["name"]: list(t["run"]) for t in inst["trains"]}
    for f in inst["faults"]:
        run[f["train"]][idx[f["section"]]] += int(f["extra"])
    return run


def schedule_for_orders(
    inst: dict[str, Any], orders: Sequence[Sequence[str]]
) -> dict[str, list[tuple[int, int]]]:
    """Earliest-start (entry, exit) per train and section for a fixed train order per section."""
    run = effective_run(inst)
    release = {t["name"]: int(t["release"]) for t in inst["trains"]}
    until = {b["section"]: int(b["until"]) for b in inst["blocks"]}
    h = int(inst["headway"])
    sched: dict[str, list[tuple[int, int]]] = {n: [] for n in release}
    for k, sec in enumerate(inst["sections"]):
        prev_exit: int | None = None
        for name in orders[k]:
            ready = release[name] if k == 0 else sched[name][k - 1][1]
            start = max(ready, until.get(sec, 0))
            if prev_exit is not None:
                start = max(start, prev_exit + h)
            end = start + run[name][k]
            sched[name].append((start, end))
            prev_exit = end
    return sched


def weighted_tardiness(inst: dict[str, Any], sched: dict[str, list[tuple[int, int]]]) -> int:
    return sum(
        int(t["weight"]) * max(0, sched[t["name"]][-1][1] - int(t["due"])) for t in inst["trains"]
    )


def _allowed_orders(inst: dict[str, Any]) -> list[tuple[str, ...]]:
    names = [t["name"] for t in inst["trains"]]
    rules = [(r["ahead"], r["behind"]) for r in inst["no_overtake"]]
    out = []
    for perm in itertools.permutations(names):
        pos = {n: i for i, n in enumerate(perm)}
        if all(pos[a] < pos[b] for a, b in rules):
            out.append(perm)
    return out


def reference_optimum(inst: dict[str, Any]) -> Reference:
    """Exact optimum by enumerating one train order per section (first optimum in lex order)."""
    perms = _allowed_orders(inst)
    best: Reference | None = None
    for orders in itertools.product(perms, repeat=len(inst["sections"])):
        sched = schedule_for_orders(inst, orders)
        obj = weighted_tardiness(inst, sched)
        if best is None or obj < best.objective:
            best = Reference(objective=obj, orders=tuple(orders), schedule=sched)
    assert best is not None
    return best


def reference_milp(inst: dict[str, Any]) -> MilpSpec:
    """The instance as a big-M MILP (the model the generator is asked to write)."""
    run = effective_run(inst)
    trains = inst["trains"]
    secs = inst["sections"]
    h = float(inst["headway"])
    until = {b["section"]: float(b["until"]) for b in inst["blocks"]}
    big_m = float(
        max(int(t["release"]) for t in trains)
        + max([0, *(int(b["until"]) for b in inst["blocks"])])
        + sum(sum(r) for r in run.values())
        + h * len(trains) * len(secs)
        + max(int(t["due"]) for t in trains)
    )
    idx = {t["name"]: i for i, t in enumerate(trains)}
    variables: list[Var] = []
    rows: list[Row] = []
    for i, t in enumerate(trains):
        for k, sec in enumerate(secs):
            lb = float(t["release"]) if k == 0 else 0.0
            lb = max(lb, until.get(sec, 0.0))
            variables.append(Var(name=f"s_{i}_{k}", type="continuous", lb=lb))
        variables.append(Var(name=f"tard_{i}", type="continuous", lb=0.0))
        for k in range(1, len(secs)):
            rows.append(
                Row(
                    name=f"route_{i}_{k}",
                    terms={f"s_{i}_{k}": 1.0, f"s_{i}_{k - 1}": -1.0},
                    sense=">=",
                    rhs=float(run[t["name"]][k - 1]),
                )
            )
        last = len(secs) - 1
        rows.append(
            Row(
                name=f"tard_def_{i}",
                terms={f"tard_{i}": 1.0, f"s_{i}_{last}": -1.0},
                sense=">=",
                rhs=float(run[t["name"]][last] - int(t["due"])),
            )
        )
    fixed = {(idx[r["ahead"]], idx[r["behind"]]) for r in inst["no_overtake"]}
    for i, j in itertools.combinations(range(len(trains)), 2):
        pi, pj = run[trains[i]["name"]], run[trains[j]["name"]]
        for k in range(len(secs)):
            y = f"y_{i}_{j}_{k}"  # 1: train i before train j on section k
            lb, ub = (1.0, 1.0) if (i, j) in fixed else (0.0, 1.0)
            if (j, i) in fixed:
                lb, ub = 0.0, 0.0
            variables.append(Var(name=y, type="binary", lb=lb, ub=ub))
            # i before j: s_j >= s_i + p_i + h - M (1 - y)
            rows.append(
                Row(
                    name=f"ord_{i}_{j}_{k}_a",
                    terms={f"s_{j}_{k}": 1.0, f"s_{i}_{k}": -1.0, y: -big_m},
                    sense=">=",
                    rhs=pi[k] + h - big_m,
                )
            )
            # j before i: s_i >= s_j + p_j + h - M y
            rows.append(
                Row(
                    name=f"ord_{i}_{j}_{k}_b",
                    terms={f"s_{i}_{k}": 1.0, f"s_{j}_{k}": -1.0, y: big_m},
                    sense=">=",
                    rhs=pj[k] + h,
                )
            )
    obj = {f"tard_{i}": float(t["weight"]) for i, t in enumerate(trains)}
    return MilpSpec(
        variables=variables, constraints=rows, objective=Objective(sense="min", terms=obj)
    )
