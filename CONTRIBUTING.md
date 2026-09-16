# Contributing

Thanks for taking the time. This page is the short version; the
[Development section of the README](README.md#development) is the long one and
stays the source of truth for the tooling.

## Setup

```shell
pip install -e ".[dev]"
```

Unit tests mock `hou` and run anywhere, on any OS, with no Houdini installed.
Only the `integration` marker needs a licence seat.

## Before you push

CI runs exactly these, so running them locally costs one minute and saves a
round trip:

```shell
ruff check .
ruff format --check .
pytest tests -m "not integration" -q
python tools/gen_prompt_vocab.py --check
```

If you have Houdini, the live suite is worth it for anything touching a
handler:

```shell
python tests/run_integration.py
```

On a machine with Red Giant / Maxon Universe installed, every `hython` run
needs `HOUDINI_DISABLE_OPENFX_DEFAULT_PATH=1` or `hou` crashes on import. That
is a Universe/Houdini conflict, not this repo.

## What a good change looks like

- **One subject per pull request.** Six small ones review faster than one large one.
- **A new or changed handler carries a test.** `tests/` mirrors the module names; the integration suite in `tests/integration/` covers the live path.
- **Never hand-edit generated content.** The node tables in `prompts/markdown/` come from `tools/prompt_vocab.json` through `gen_prompt_vocab.py`; anything between `BEGIN`/`END` markers is written by a generator in `tools/`. Edit the input, re-run the generator, commit both.
- **Never edit `CHANGELOG.md`.** It is generated from merged pull requests by `auto-changelog`.
- **Node names are evidence, not memory.** `tools/node_versions.json` records which builds were sampled and what they had. If you add a node name, run `python tools/gen_node_versions.py` on your Houdini so your build merges into the shared table. One installed version is enough.

## Commit and pull request titles

This project follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>[optional scope][!]: <description>
```

Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`,
`chore`, `style`, `revert`. The scope is the area touched, usually the handler
module: `lops`, `hda`, `parameters`, `pdg`, `bridge`.

Pull requests are squash-merged, so **the title becomes the commit message and
the changelog line**. Say what the change does for the user, in the present
tense, and keep the description specific enough to be read a year later out of
context:

> `fix(lops): list_usd_prims treats the root as a path, not a string prefix`
>
> `feat(parameters): link_parameters writes chs() for a String destination`

Not `fix: linking bug`. The generated changelog strips the type and scope back
off, so a vague description leaves a vague changelog line with nothing to
recover it from.

A breaking change takes a `!` before the colon, or a `BREAKING CHANGE:` footer
in the body. Either one makes the changelog mark the entry.

## Reporting a bug

Open an issue with the Houdini version, your OS, and the MCP client. The output
of the `get_houdini_connection_status` tool answers most of it in one paste.
