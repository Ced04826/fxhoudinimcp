<!--
Title: <type>[scope]: <description>, for example
  fix(lops): list_usd_prims treats the root as a path, not a string prefix
It becomes the squash commit and the changelog line. Types: feat, fix, docs,
refactor, perf, test, build, ci, chore, style, revert. See CONTRIBUTING.md.
-->

## What this changes

<!-- What was wrong or missing, and what the reader will see now. -->

## How it was verified

<!-- Tick what you ran. Unit tests need no Houdini. -->

- [ ] `ruff check . && ruff format --check .`
- [ ] `pytest tests -m "not integration" -q`
- [ ] `python tools/gen_prompt_vocab.py --check`
- [ ] Integration suite, on Houdini <!-- version --> / <!-- OS -->
- [ ] Generated files regenerated with their `tools/` script, not by hand

Closes #
