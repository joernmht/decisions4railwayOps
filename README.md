# decisions4railwayOps

Experimental space for applying **typed decision models** to railway operations science.
The first engine under test is **Jev**, TypeSafe AI's "System One" model: given a *state*
(a disposition situation, a timetable conflict, an operator log) and a set of typed questions
(yes/no `Noul`, one-of-N `Choice`, ordinal `Score`), it returns calibrated probabilities and a
confidence value instead of free text. Code stays in control; the model makes narrow decisions.

Status: empty lab, nothing to cite yet.

## Setup

```sh
pip install typesafe-sdk          # Python >= 3.11
cp .env.example .env              # then put your key in .env (never commit it)
```

The SDK reads `TYPESAFE_API_KEY` from the environment. `.env` is git-ignored.

## AI disclaimer

This repository was developed with substantial assistance from AI coding tools
(primarily Anthropic's Claude). Code, documentation and results have been
reviewed by the author, who takes full responsibility for the content.
