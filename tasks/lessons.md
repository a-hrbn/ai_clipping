# Lessons

- Do not equate proxy distribution metrics like views or engagement with intrinsic clip quality. Treat them as noisy supervision that must be combined with exposure context, creator baseline, and stronger outcome signals.
- When the user asks for a simpler feedback workflow, optimize for the minimum number of stable inputs instead of preserving analytics-heavy schemas.
- In training collection pipelines, transient network failures must pause and retry the current item instead of being downgraded to skipped/unavailable examples.
- Keep infrastructure extensions reversible and scoped; if the user later rejects an architectural add-on, revert only that slice without disturbing the surrounding workflow.
- When fixing Codex skill installs, verify both discovery and the real user invocation path; a correct `~/.codex/skills/.../SKILL.md` install does not prove the client will accept a manually typed `/trigger` in the UI.
- Do not leave downloaded SDKs or local toolchains inside the project tree; move them to a user-level path before relying on repo-wide tooling or graph scans.
