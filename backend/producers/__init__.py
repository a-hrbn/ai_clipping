"""Clip-pipeline Producers behind a thin Conductor.

Each Producer turns its inputs into one pipeline output behind a small,
injectable interface (see docs/design/clip-pipeline-producers.md and CONTEXT.md
for the locked design and vocabulary). Producers are pure: they never read
global settings or touch job state — their leaf operations are injected.
"""
