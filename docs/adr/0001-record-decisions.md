# 1. Record architecture decisions as ADRs

- Status: accepted
- Date: 2026-09-28
- Deciders: Jörn Maurischat

## Context and Problem Statement

This is an open research lab whose results will feed a paper. Reviewers and later readers need to
know *why* the harness looks the way it does (which engines, which state encoding, which metrics),
not only what it does.

## Decision Outcome

Architecture decisions are recorded in `docs/adr/` in MADR format. Dated experimental observations
(what we ran, what we saw) go into `docs/lab-notebook.md`, which is append-only and newest-first.
