# Portable benchmark-analysis instructions

Use this instruction pack when researching model benchmark performance.

1. Only use benchmarks listed in `references/benchmark-source.md`.
2. Start with exact model version and evaluation time point before searching scores.
3. Route through `references/core-dimensions.md` first, then shortlist benchmarks, then crawl websites.
4. Follow `references/search-playbook.md` for overlap expansion, source priority, comparison anchors, and image-only leaderboard handling.
5. Apply `references/data-defect-warnings.md` inline for every affected benchmark.
6. Use `references/report-template.md` to structure the final report.
7. Do not average incompatible benchmark metrics. Preserve exact score units, variants, and sub-scores.
8. For code or agentic coding, compare with the latest available Claude Opus, Claude Sonnet, and GPT-family model.
9. For multimodal, compare with the latest available Gemini-family model.
10. For intelligence or reasoning, compare with the latest available GPT-family model.
11. If available, compare with the model's immediate predecessor.
12. If a leaderboard is image-only, use multimodal inspection and clearly mark the row as image-extracted.
13. Default to the user's language.
