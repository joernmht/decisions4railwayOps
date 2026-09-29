# Dispatching rules derived from DB InfraGO Ril 420.02

Generated from `src/d4r/rules/ril420.json` by `d4r.rules.rulebook_markdown()`; do not edit
by hand.

**Source.** DB InfraGO AG, Richtlinie 420.02 Betriebsleitstellen - Zusammenarbeit mit Eisenbahnverkehrsunternehmen, part of the Infrastrukturnutzungsbedingungen (INB) 2026, INB 2026 (Bekanntgabe 14, valid from 14.12.2025; correction to 15.04.2026). Cite as `dbinfrago2026ril420`.

**Status.** Derived, paraphrased rules for research use. No text of the regulation is reproduced; every entry cites module, section, paragraph and item. Consult the original for binding wording. The regulation itself is not part of this repository.

## Semantics

- **S1** Where a rule applies 'in principle' (German: grundsaetzlich), it holds as a rule; exceptions are admissible, but a deviation in a justified individual case requires weighing the interests of the parties involved. — Ril 42002 (preface), section 1, paragraph (5) (valid from 2025-12-14)

## Objectives

| id | regime | paraphrase | formal reading | source |
|---|---|---|---|---|
| O0 | overall | The overarching aim of running trains is maximal operational quality, so that overall punctuality of all trains is high. | minimise the total (unweighted) deviation of all trains from their timetable | Ril 420.0201, section 1, paragraph (2) (valid from 2021-12-12) |
| O1 | regular dispatching (the planned operating programme is feasible) | Keep, or restore, the punctuality of all trains. | minimise total delay of all trains | Ril 420.0201, section 1, paragraph (3) a) (valid from 2021-12-12) |
| O2 | disturbed dispatching (the planned programme is not feasible, or only with restrictions) | Use the capacity of lines and nodes as fully as possible and return to regular operation as quickly as possible. | maximise throughput (trains arrived) and minimise time until the disturbance is absorbed | Ril 420.0201, section 1, paragraph (3) b) (valid from 2021-12-12) |
| O3 | overall | The timetable is the production target: all dispatching aims at keeping to, or returning to, the timetable; control centres watch operations continuously and act anticipatorily to reduce delays and to avoid their transfer to other trains. | minimise knock-on (secondary) delay | Ril 420.0200, section 2, paragraph (1), (4) (valid from 2024-12-15) |

## Priority rules (order of trains)

| id | paraphrase | classes | modelled | source |
|---|---|---|---|---|
| P1 | Urgent relief trains (sent to clear the consequences of an incident or to restore restricted infrastructure) go before all other trains. | urgent_relief | yes | Ril 420.0201, section 1, paragraph (3), item rule 1 (valid from 2021-12-12) |
| P2 | Long-distance passenger trains of very high priority (market segment 'Express') go before all other trains except urgent relief trains. | passenger_express | yes | Ril 420.0201, section 1, paragraph (3), item rule 2 (valid from 2021-12-12) |
| P3 | Freight trains of very high priority (market segment 'Express') go before all other trains except urgent relief trains and very-high-priority long-distance passenger trains. | freight_express | yes | Ril 420.0201, section 1, paragraph (3), item rule 3 (valid from 2021-12-12) |
| P4 | Freight trains of high priority (market segment 'Fast') go before other freight trains, except those of very high priority. | freight_fast | yes | Ril 420.0201, section 1, paragraph (3), item rule 4 (valid from 2021-12-12) |
| P5 | All trains not named in rules 1 to 4 are of equal standing among themselves. | passenger_other, freight_other | yes | Ril 420.0201, section 1, paragraph (3), item rule 5 (valid from 2021-12-12) |
| P6 | Among trains of equal standing, the faster train goes first in principle, where speed means the travel speed: the average speed between locations at which the order of trains can be changed, including all planned running and dwell times. Different travel speeds cause headway conflicts and additional delay. | higher travel speed first | yes | Ril 420.0201, section 1, paragraph (3), item rule 6 incl. note (valid from 2021-12-12) |
| P7 | On lines dedicated to particular services, trains providing those services go before other trains, except urgent relief trains. |  | no: Flatland has no dedicated lines. | Ril 420.0201, section 1, paragraph (3), item rule 7 (valid from 2021-12-12) |

## Constraints and process rules

| id | paraphrase | formal reading | source |
|---|---|---|---|
| C1 | A train running ahead of its schedule shall in principle not delay other trains. | if a train is early, the order must not add delay to any punctual or late train | Ril 420.0201, section 1, paragraph (3), after rule 7 (valid from 2021-12-12) |
| C2 | Deviations from the dispatching rules are decided by the network coordinator of the control centre. | governance: rule exceptions require an escalation to a designated decision maker | Ril 420.0201, section 1, paragraph (3), final sentence (valid from 2021-12-12) |
| C3 | A train loses its claim to high or very high priority (rules 2 to 4) if it deviates from its planned train characteristics in a way that harms keeping its running times; it is then dispatched under rules 5 and 6. | priority rank depends on compliance with planned characteristics | Ril 420.0201, section 1, paragraph (4) (valid from 2021-12-12) |
| C4 | Dispatchers of the control centre may instruct signallers within their functional competence but must not interfere with the signallers' safety-related duties. | separation of dispatching (decide order and timing) from interlocking/signalling (execute safely) | Ril 420.0200, section 2, paragraph (5) (valid from 2024-12-15) |
| C5 | The train dispatcher works with computer-aided systems but has no access to route setting; the dispatcher's decisions are implemented by signallers. | the dispatching engine outputs decisions, never control actions | Ril 42002 glossary, entry Zugdisponent (valid from 2025-12-14) |
| C6 | Control centres aim to agree measures with the railway undertakings, but in a conflict the infrastructure manager's control centres coordinate and decide (final decision). | final decision authority rests with the infrastructure manager | Ril 420.0200, section 1, paragraph (2) and section 4 note (valid from 2024-12-15) |
| C7 | When infrastructure becomes unavailable during operation, the area dispatcher may propose the full or partial cancellation of a train; the railway undertaking decides. | cancellation is not a unilateral dispatching action | Ril 420.0212, section 2, paragraph (1)-(2) (valid from 2024-12-15) |
| C8 | Operational rerouting is always caused by the network (not by customer wish); the area dispatcher arranges it after consulting the affected railway undertakings, checks that the diversion route is admissible for the train's characteristics, and the train keeps its number. | reroute options must be admissible for the train class | Ril 420.0211, section 1, paragraph (1)-(3) (valid from 2021-12-12) |
| C9 | On short-notice infrastructure restrictions, dispatchers may hold back trains without passengers on operational tracks (main through tracks and other main tracks such as overtaking or crossing tracks), in agreement with the railway undertaking; held trains are distinguished into tailback trains (network-caused) and parked trains (other causes). | long holds on operational tracks apply to trains without passengers | Ril 420.0209, section 1, paragraph (1)-(3) (valid from 2026-04-15) |
| C10 | If a passenger train runs in two parts on a double-track line, the second part uses the running and dwell times of the first; the original train should run first where possible. | not modelled | Ril 420.0213, section 2, paragraph (1) (valid from 2022-12-11) |

## Parameters

| id | name | value | paraphrase | source |
|---|---|---:|---|---|
| K1 | `train_ready_notice_passenger_min` | 3 min | A passenger train is normally reported ready at the latest 3 minutes before its scheduled departure. | Ril 420.0200, section 3, paragraph (2) (valid from 2024-12-15) |
| K2 | `train_ready_notice_freight_min` | 5 min | A freight train is normally reported ready at the latest 5 minutes before its scheduled departure. | Ril 420.0200, section 3, paragraph (2) (valid from 2024-12-15) |
| K3 | `timetable_void_after_delay_h` | 20 h | A train path's timetable in principle loses validity at a delay of 20 hours or more; such a train is held and needs a new timetable. | Ril 420.0214, section 1-2, paragraph (1) (valid from 2023-12-10) |
| K4 | `max_early_running_h` | 3 h | A train is in principle not taken over, not allowed to depart, and held at the next suitable station if it would run more than 3 hours ahead of schedule. | Ril 420.0214, section 1 and 3, paragraph (1)-(2) (valid from 2023-12-10) |
| K5 | `max_clearing_of_operational_tracks_h` | 48 h | Operational tracks used for unplanned parking must be cleared promptly, within at most 48 hours. | Ril 420.0209, section 1, paragraph (6) (valid from 2026-04-15) |
| K6 | `short_notice_replacement_train_h` | 1 h | If a replacement train or second part is ordered one hour or less before departure because of a disturbance, the control centre (not the timetable department) inserts it. | Ril 420.0213, section 1, paragraph (1) (valid from 2022-12-11) |
| K7 | `relief_train_traction_status_min` | 15 min | The network coordinator obtains the expected traction time of an urgent relief train at the latest 15 minutes after commissioning it. | Ril 420.0280, section 2, paragraph (3) (valid from 2022-12-11) |

## Catalogue of dispatching measures

Catalogue of dispatching measures to be checked case by case. — Ril 420.0200, section 4, paragraph (1), (3), (8) (valid from 2024-12-15)

- *general*: relieve lines and nodes of reduced capacity by excluding or reducing train paths; interrupt construction and maintenance works; provide replacement traction and staff; change vehicle and staff rosters; join trains; order additional traction and staff
- *with railway undertaking*: evacuation; prepared relief measures; cancel or run through coaches; special connection waiting-time rules; replacement connections, extra stops; early termination and turning of trains; coordination of replacement measures on line closures
- *freight*: offer trains between infrastructure managers outside the planned procedure; hold back freight trains stating reason and expected duration; park trains; clear stations when trains are not removed on time; large-scale rerouting

Coverage in the Flatland contest:

- hold back or wait: meet: HOLD
- retime release: depart: WAIT
- reorder: meet decisions choose who goes first
- reroute: not yet
- cancel: not modelled (railway undertaking decides, C7)

## How the lab uses these rules

- `rule-ril420` applies P1–P6 to the decision card (hold/wait for an oncoming train that goes
  first; ignore a train broken down longer than the hold cap, the kind of justified exception
  that S1 admits). P4 is read as ranking 'Fast' freight above other *freight* only; against a
  regional passenger train both are equal-standing and P6 (speed) decides.
- `jev-ril`, `deepseek-*-ril` receive the paraphrased rules (`rules_text()`) with the card.
- The rollout oracle scores O0 (unweighted delay of all trains). Mixed-traffic game sets also
  score a priority-weighted delay (weights in `SERVICE_WEIGHT`, an assumption of this study,
  not part of Ril 420) to measure the price of rule adherence under both views.
- C4/C5 (dispatchers decide, signallers execute; no access to route setting) are the
  regulatory counterpart of the lab's interlocking/dispatcher split (ADR-0003).
