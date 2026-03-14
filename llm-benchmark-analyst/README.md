# llm-benchmark-analyst

This package is intentionally portable.

## Files for different runtimes
- `SKILL.md`: primary AgentSkills/OpenAI/OpenClaw entrypoint
- `AGENTS.md`: compact portable rules for coding agents that read AGENTS.md
- `CLAUDE.md`: compact portable rules for Claude Code style workflows
- `PORTABLE_PROMPT.md`: vendor-neutral instructions that can be pasted into Cursor, Windsurf, Roo, Cline, OpenHands, or other coding tools

## Important behavior
- benchmark scope is restricted to `references/benchmark-source.md`
- search is domain-first, then benchmark shortlist, then website retrieval
- reports always include benchmark purpose, variant, time point, and data-defect warnings
