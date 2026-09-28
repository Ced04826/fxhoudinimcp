"""Graph intelligence handlers for FXHoudini-MCP.

Senior-artist tooling: atomic network building with upfront validation,
whole-network verification, version-exact node documentation, and cook
profiling. These are the commands that let a client plan a whole graph,
prove the plan against the running Houdini before mutating anything,
and then look at the evidence afterwards.
"""

from __future__ import annotations

# Built-in
import contextlib
import json
import os
import re
import tempfile
import time
import zipfile
from difflib import get_close_matches
from typing import Any

# Third-party
import hou

# Internal
from fxhoudinimcp_server.config import (
    layout_if_enabled,
    place_new_node,
    place_new_nodes,
    update_mode_warning,
)
from fxhoudinimcp_server.dispatcher import register_handler
from fxhoudinimcp_server.errors import readable_message
from fxhoudinimcp_server.handlers.node_handlers import (
    _find_input,
    _indirect_input_item,
    _input_table,
    _resolve_input_index,
)
from fxhoudinimcp_server.handlers.parameter_handlers import (
    broken_references,
    suggest_parms,
    template_labels,
)
from fxhoudinimcp_server.handlers.state_receipt_helpers import (
    check_parm_value,
    clip,
    parse_input_entries,
    resolve_input_policy,
    terminal_names,
)
from fxhoudinimcp_server.outputs import license_error

###### Helpers

# Folder types whose contents are instance templates rather than parameters.
# MultiparmBlock is the common one; the scroll and tab variants behave the same
# way for naming purposes.
_MULTIPARM_FOLDERS = tuple(
    getattr(hou.folderType, name)
    for name in ("MultiparmBlock", "ScrollingMultiparmBlock", "TabbedMultiparmBlock")
    if hasattr(hou.folderType, name)
)


def _resolve_node_type(category: hou.NodeTypeCategory, type_name: str):
    """Resolve *type_name* in *category* the way createNode would.

    hou.preferredNodeType maps an unversioned name to the version that
    createNode actually instantiates (e.g. copytopoints -> ::2.0).
    """
    with contextlib.suppress(Exception):
        preferred = hou.preferredNodeType(f"{category.name()}/{type_name}")
        if preferred is not None:
            return preferred
    types = category.nodeTypes()
    if type_name in types:
        return types[type_name]
    from fxhoudinimcp_server.handlers.hda_handlers import _version_key

    prefix = type_name + "::"
    versioned = [key for key in types if key.startswith(prefix)]
    if versioned:
        # Numeric order, not string order: ::10.0 is newer than ::2.0.
        return types[max(versioned, key=lambda key: _version_key(key[len(prefix) :]))]
    return None


def _instance_patterns(node_type) -> list[re.Pattern]:
    """Regexes matching the live names of a type's multiparm instances.

    A fresh probe node has zero instances, so `usept0` and `pt0x` on an Add
    SOP were reported as non-existent and a session fell back to a wrangle
    to make three points. The template names carry `#` where the instance
    number goes; a live name is that with digits, plus an optional vector
    component suffix.
    """
    patterns: list[re.Pattern] = []

    def walk(templates) -> None:
        for template in templates:
            if "#" in template.name():
                body = re.escape(template.name()).replace(r"\#", r"\d+")
                patterns.append(re.compile(rf"^{body}[xyzwrgba]?$"))
            if template.type() == hou.parmTemplateType.Folder:
                walk(template.parmTemplates())

    walk(node_type.parmTemplateGroup().entries())
    return patterns


def _is_instance_parm(name: str, patterns: list[re.Pattern]) -> bool:
    return any(p.match(name) for p in patterns)


def _multiparm_start(folder) -> int | None:
    """The first instance number of a multiparm block, from its template.

    Read off the folder's ``multistartoffset`` tag (the Add SOP's points start
    at 0). When the tag is absent this returns 1, which is ASSUMED to be
    Houdini's default start, not read from anything. A tag that is not an
    integer gives None: unknown, rather than a guess.
    """
    tags: Any = None
    with contextlib.suppress(Exception):
        tags = folder.tags()
    raw = tags.get("multistartoffset") if isinstance(tags, dict) else None
    if raw is None:
        return 1
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def _multiparm_blocks(node_type) -> tuple[list[dict[str, Any]], list[re.Pattern]]:
    """(outermost multiparm blocks, nested count patterns) of a node type.

    Each block has its ``count_parm`` (the folder's own name), its ``start``
    and ``patterns`` matching every instance parm inside it, with the first
    ``#`` -- this block's instance number -- captured. A block nested inside
    another has a count parm that is itself an instance name (``num#``); those
    are only returned as patterns, for ordering: which ``#`` of a nested name
    is whose instance number is not checked here. Never raises; a type whose
    templates cannot be read has no blocks.
    """
    blocks: list[dict[str, Any]] = []
    nested: list[re.Pattern] = []

    def pattern(name: str, capture: bool) -> re.Pattern:
        body = re.escape(name)
        if capture:
            body = body.replace(r"\#", r"(\d+)", 1)
        body = body.replace(r"\#", r"\d+")
        return re.compile(rf"^{body}[xyzwrgba]?$")

    def inside(folder) -> list[re.Pattern]:
        found: list[re.Pattern] = []
        for child in folder.parmTemplates():
            if "#" in child.name():
                found.append(pattern(child.name(), capture=True))
            if child.type() == hou.parmTemplateType.Folder:
                if child.folderType() in _MULTIPARM_FOLDERS and "#" in child.name():
                    nested.append(pattern(child.name(), capture=False))
                found += inside(child)
        return found

    def walk(templates) -> None:
        for template in templates:
            if template.type() != hou.parmTemplateType.Folder:
                continue
            if template.folderType() in _MULTIPARM_FOLDERS and "#" not in template.name():
                blocks.append(
                    {
                        "count_parm": template.name(),
                        "start": _multiparm_start(template),
                        "patterns": inside(template),
                    }
                )
            else:
                walk(template.parmTemplates())

    try:
        walk(node_type.parmTemplateGroup().entries())
    except Exception:  # noqa: BLE001 - no blocks known is the safe answer
        return [], []
    return blocks, nested


def _count_parms_first(parms: dict, blocks: tuple) -> list[tuple[str, Any]]:
    """*parms* in spec order, except that multiparm counts come first.

    An instance parm (``pt3``) set before its count (``points``) does not exist
    yet, so the write failed or went nowhere. Outer counts, then nested counts,
    then everything else; spec order within each group.
    """
    outer, nested = blocks
    counts = {block["count_parm"] for block in outer}

    def rank(item: tuple) -> int:
        if item[0] in counts:
            return 0
        return 1 if any(p.match(item[0]) for p in nested) else 2

    return sorted(parms.items(), key=rank)


def _count_value(value: Any) -> int | None:
    """A multiparm count the spec sets, or None when it is not a plain number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            value = int(value.strip())
    return max(0, value) if isinstance(value, int) else None


def _instance_range_errors(label: str, parms: dict, blocks: tuple) -> list[str]:
    """Instance parms whose number is outside the count the same spec sets.

    Only checked when the spec sets the block's count: without it, how many
    instances the node will have is not known here.
    """
    errors: list[str] = []
    for block in blocks[0]:
        if block["count_parm"] not in parms or block["start"] is None:
            continue
        count = _count_value(parms[block["count_parm"]])
        if count is None:
            continue
        start = block["start"]
        for name in parms:
            for candidate in block["patterns"]:
                match = candidate.match(name)
                if match is None:
                    continue
                number = int(match.group(1))
                if not start <= number <= start + count - 1:
                    valid = f"{start}..{start + count - 1}" if count else "none"
                    errors.append(
                        f"node {label}: parm '{name}' is instance {number} of multiparm "
                        f"'{block['count_parm']}', which this spec sets to {count} "
                        f"instance(s) starting at {start} (valid: {valid})"
                    )
                break
    return errors


def _is_dynamic_menu(probe: hou.Node, parm: hou.Parm) -> bool:
    """A menu whose items depend on state a fresh probe cannot have.

    The VOP ``signature`` menu lists only the signature the node currently
    has ("default" on a fresh mtlxmultiply) while ``set()`` accepts every
    signature the shader defines, so validating "color3" against that list
    refused a valid build. Such a menu is left to Houdini.
    """
    try:
        return parm.name() == "signature" and probe.type().category().name() == "Vop"
    except Exception:
        return False


def _connectors_of(node: hou.Node) -> dict[str, list[dict[str, Any]]]:
    """Input and output connectors of a live node, in order, with their names.

    A VOP's inputs are addressed by name (`texcoord`), and their order is not
    stable across versions; a SOP's are `input1`, `input2` behind labels such
    as "Spine curve". Both the index and the name are reported so a caller can
    wire by whichever the verb at hand accepts.
    """
    outputs: list[dict[str, Any]] = []
    names: list[str] = []
    labels: list[str] = []
    with contextlib.suppress(Exception):
        names = list(node.outputNames())
    with contextlib.suppress(Exception):
        labels = list(node.outputLabels())
    for index, name in enumerate(names):
        entry: dict[str, Any] = {"index": index, "name": name}
        if index < len(labels):
            entry["label"] = labels[index]
        outputs.append(entry)
    return {"inputs": _input_table(node), "outputs": outputs}


def _probe_connectors(scratch: hou.Node, node_type, parms: dict | None = None) -> dict:
    """Connectors of *node_type* created inside *scratch* with *parms* applied.

    A VOP's `signature`, a multiparm count or a variadic HDA decides what a
    node exposes, so build_network probes a spec that names an input AND sets
    parms the way the build will configure it: a dry run that passes then
    wires the same connectors the build resolves.
    """
    with hou.undos.disabler():
        probe = scratch.createNode(node_type.name())
        try:
            for name, value in (parms or {}).items():
                with contextlib.suppress(Exception):
                    _apply_parm(probe, name, value)
            return _connectors_of(probe)
        finally:
            with contextlib.suppress(Exception):
                probe.destroy()


def _instance_menu_errors(scratch: hou.Node, node_type, parms: dict, patterns: list) -> list[str]:
    """Menu errors on multiparm instance parms (a wedge's type1), found by probing.

    A fresh probe has no instances, so these menus went unchecked: a dry run
    passed "type1": "int" and the build then rolled back on "Invalid menu
    item". The probe gets the spec's other parms first, which creates the
    instances, then each instance value is checked against its real menu.
    """
    wanted = {
        name: value
        for name, value in parms.items()
        if _is_instance_parm(name, patterns)
        and isinstance(value, (str, int))
        and not isinstance(value, bool)
    }
    if node_type is None or not wanted:
        return []
    problems = []
    with hou.undos.disabler():
        probe = scratch.createNode(node_type.name())
        try:
            for name, value in parms.items():
                if name not in wanted and _expression_value(value) is None:
                    with contextlib.suppress(Exception):
                        _apply_parm(probe, name, value)
            for name, value in wanted.items():
                parm = probe.parm(name)
                if parm is None or not _is_strict_menu(parm.parmTemplate()):
                    continue
                with contextlib.suppress(Exception):
                    items = list(parm.menuItems())
                    if problem := _menu_error(name, value, items):
                        # These tokens are often bare numbers ('0'..'5'); the
                        # labels are what says which one means "Integer".
                        labels = dict(zip(items, parm.menuLabels(), strict=False))
                        problems.append(f"{problem} Labels: {labels}")
        finally:
            with contextlib.suppress(Exception):
                probe.destroy()
    return problems


# OBJ-level container to probe each category's types in, where more than one
# exists (geo and sopnet both hold SOPs). Any other category is found through
# hou.NodeType.childTypeCategory(), so Cop2, Shop, VopNet and whatever SideFX
# adds next are covered without an entry here.
_PREFERRED_CONTAINERS = {
    "Sop": "geo",
    "Object": "subnet",
    "Vop": "matnet",
    "Lop": "lopnet",
    "Dop": "dopnet",
    "Cop": "copnet",
    "Chop": "chopnet",
    "Top": "topnet",
    "Driver": "ropnet",
}


def _container_for(category_name: str) -> str | None:
    """An OBJ-level type whose children are of *category_name*."""
    object_types = hou.objNodeTypeCategory().nodeTypes()
    preferred = _PREFERRED_CONTAINERS.get(category_name)
    if preferred in object_types:
        return preferred
    fallback = None
    for type_name in sorted(object_types):
        with contextlib.suppress(Exception):
            node_type = object_types[type_name]
            child = node_type.childTypeCategory()
            if child is None or child.name() != category_name:
                continue
            if not node_type.hidden():
                return type_name
            fallback = fallback or type_name
    return fallback


# Connectors per type for the session: probing creates a node, and a card is
# read again and again. Keyed on the HDA definition's modification time too,
# so a reinstalled asset is probed afresh. Each entry is (connectors,
# generated menus): one probe answers both.
_CONNECTOR_CACHE: dict[tuple, tuple[dict, dict]] = {}


def _definition_stamp(node_type) -> Any:
    with contextlib.suppress(Exception):
        definition = node_type.definition()
        if definition is not None:
            return (definition.libraryFilePath(), definition.modificationTime())
    return None


def _is_strict_menu(template) -> bool:
    """A menu that rejects arbitrary text: an int menu or a "normal" string one."""
    return template.type() == hou.parmTemplateType.Menu or (
        template.type() == hou.parmTemplateType.String
        and template.menuType() == hou.menuType.Normal
    )


def _generated_menus_of(node: hou.Node) -> dict[str, dict[str, Any]]:
    """Menus a live node computes that its type's templates do not carry.

    `filemerge::2.0` promotes `loadtype` from an inner `file1`, and its items
    come from the script `opmenu -l -a file1 loadtype`. The type's template
    answers menuItems() with an empty tuple, so the card showed a Menu with no
    items, and a session took its tokens off another node's card. A live parm
    runs the script and answers with all seven.

    Returns {parm name: {"generator": script, "items": [...], "labels": [...]}};
    "items" is absent when the live parm had none to offer either.
    """
    generated: dict[str, dict[str, Any]] = {}
    with contextlib.suppress(Exception):
        for parm in node.parms():
            with contextlib.suppress(Exception):
                template = parm.parmTemplate()
                if list(template.menuItems()):
                    continue  # a static menu: the template already has it
                if not _is_strict_menu(template):
                    # A group picker or a wrangle's snippet list: suggestions,
                    # not tokens, and whole VEX snippets doubled the card.
                    continue
                script = ""
                with contextlib.suppress(Exception):
                    script = template.itemGeneratorScript() or ""
                if not script:
                    continue
                entry: dict[str, Any] = {"generator": script}
                items = list(parm.menuItems())
                if items:
                    entry["items"] = items
                    with contextlib.suppress(Exception):
                        entry["labels"] = list(parm.menuLabels())
                generated[parm.name()] = entry
    return generated


def _connectors_for_type(
    category_name: str,
    node_type,
    generated_menus: dict | None = None,
) -> tuple[dict | None, str | None]:
    """Connector names/labels of *node_type*, probed on a throwaway instance.

    Returns (connectors, None), or (None, why) when there was nothing to probe
    in or the probe failed: a card is never refused over its connectors, and
    never reports "no inputs" when it did not look. When *generated_menus* is
    given, it is filled from the same probe with the menus a script computes
    (see _generated_menus_of): one probe, two questions. The probe runs with
    undo disabled and without the type's creation scripts (a documentation
    read must not run an asset's OnCreated); it still marks the scene
    modified, as build_network's dry run does, once per type per session.
    """
    key = (category_name, node_type.name(), _definition_stamp(node_type))
    if key in _CONNECTOR_CACHE:
        connectors, menus = _CONNECTOR_CACHE[key]
        if generated_menus is not None:
            generated_menus.update(menus)
        return connectors, None
    container = _container_for(category_name)
    if container is None:
        return None, f"no network under /obj holds {category_name} nodes to probe in"
    root = hou.node("/obj")
    if root is None:
        return None, "/obj does not exist"
    scratch = None
    try:
        with hou.undos.disabler():
            scratch = root.createNode(container, "fxhoudinimcp_card_probe")
            probe = scratch.createNode(node_type.name(), run_init_scripts=False)
            menus = _generated_menus_of(probe)
            connectors = _connectors_of(probe)
    except Exception as exc:
        return None, f"probing {node_type.name()} failed: {readable_message(exc)}"
    finally:
        with contextlib.suppress(Exception):
            if scratch is not None:
                with hou.undos.disabler():
                    scratch.destroy()
    _CONNECTOR_CACHE[key] = (connectors, menus)
    if generated_menus is not None:
        generated_menus.update(menus)
    return connectors, None


class _TypeKnowledge(tuple):
    """What one probe of a node type teaches the validation pass.

    A tuple, so `parm_names, tuple_names, menus, instance_patterns, connectors
    = ...` unpacks it. The per-parameter shapes and the factory expressions
    ride along as attributes because they come off the same probe, and
    instantiating a second node to learn them would double what validating a
    type costs the user's scene.
    """

    def __new__(
        cls,
        parm_names,
        tuple_names,
        menus,
        instance_patterns,
        connectors,
        shapes,
        expressions=None,
        scene_menus=False,
    ):
        self = super().__new__(cls, (parm_names, tuple_names, menus, instance_patterns, connectors))
        self.shapes = shapes
        self.expressions = expressions if expressions is not None else {}
        self.scene_menus = scene_menus
        return self


def _parm_names_for_type(scratch: hou.Node, node_type) -> _TypeKnowledge:
    """Instantiate a type once to learn its parm names, tuple names, menus,
    multiparm instance patterns, connectors, per-parameter shapes and factory
    expressions.

    The third element maps a strict-menu parm name to its token list. Only
    menus that reject arbitrary text are recorded (int menus and "normal"
    string menus); a free-text field with a suggestion menu is not a menu.
    The fourth is what `_instance_patterns` returns, the fifth the probe's
    connectors (the same probe answers both questions, and probing is not
    free). Creation scripts run here on purpose: the build that follows runs
    them, so the names validated are the names the built node has.

    Shapes (`.shapes`) are learned here because names alone caught a
    misspelling and nothing else, so a spec that named a real three-float
    parameter and handed it two numbers validated cleanly and then failed
    during the build -- after other nodes had been created. The component count
    and template type are knowable from the probe, so they are checked here;
    the template type is also how a string aimed at a numeric parm is caught
    before the build starts.

    `.expressions` maps each parm and parm tuple name whose components ship
    with an expression to {component: expression}, so a dry run can name the
    literals that will not take.

    `.scene_menus` is True when a strict menu's items come from a script:
    those can follow the scene (the filecache `take` menu lists its takes), so
    _type_knowledge does not cache such a type.
    """
    scene_menus = False
    probe = scratch.createNode(node_type.name())
    try:
        connectors = _connectors_of(probe)
        expressions: dict[str, dict[str, str]] = {}
        for parm_tuple in probe.parmTuples():
            with contextlib.suppress(Exception):
                found = {
                    parm.name(): expression
                    for parm in parm_tuple
                    if (expression := _expression_of(parm)) is not None
                }
                if found:
                    expressions[parm_tuple.name()] = found
                    for component, expression in found.items():
                        expressions[component] = {component: expression}
        parm_names = {p.name() for p in probe.parms()}
        tuple_names = {pt.name() for pt in probe.parmTuples()}
        menus: dict[str, list[str]] = {}
        for parm in probe.parms():
            if _is_dynamic_menu(probe, parm):
                continue
            with contextlib.suppress(Exception):
                template = parm.parmTemplate()
                items = list(parm.menuItems())
                if items and _is_strict_menu(template):
                    menus[parm.name()] = items
                    scene_menus = scene_menus or bool(template.itemGeneratorScript())
        shapes: dict[str, dict] = {}
        for parm_tuple in probe.parmTuples():
            with contextlib.suppress(Exception):
                template = parm_tuple.parmTemplate()
                shapes[parm_tuple.name()] = {
                    "type": template.type().name(),
                    "components": len(parm_tuple),
                    "is_tuple": len(parm_tuple) > 1,
                }
        for parm in probe.parms():
            with contextlib.suppress(Exception):
                template = parm.parmTemplate()
                shapes.setdefault(
                    parm.name(),
                    {"type": template.type().name(), "components": 1, "is_tuple": False},
                )
    finally:
        with contextlib.suppress(Exception):
            probe.destroy()
    # A type whose templates cannot be read still validates on names; it just
    # loses the multiparm exemption rather than taking the whole build down.
    patterns: list[re.Pattern] = []
    with contextlib.suppress(Exception):
        patterns = _instance_patterns(node_type)
    return _TypeKnowledge(
        parm_names, tuple_names, menus, patterns, connectors, shapes, expressions, scene_menus
    )


# Learning a type's parameters means instantiating it, and instantiating it in
# the user's network runs its creation scripts and disturbs flags and
# selection. Doing that once per type per session instead of once per call is
# the difference between a quiet validation and one that churns the scene
# (probing was 90% of a dry run's time). Keyed on the type's definition
# timestamp as well, so re-saving an HDA does not leave a stale answer behind.
_TYPE_KNOWLEDGE: dict[tuple, _TypeKnowledge] = {}


def _type_cache_key(category: hou.NodeTypeCategory, node_type) -> tuple:
    stamp = None
    with contextlib.suppress(Exception):
        definition = node_type.definition()
        if definition is not None:
            stamp = definition.modificationTime()
    return (category.name(), node_type.name(), stamp)


def _type_knowledge(scratch: hou.Node, node_type, category: hou.NodeTypeCategory) -> _TypeKnowledge:
    """What one probe of *node_type* teaches, for the validation pass; cached
    unless a strict menu is script-filled (see _parm_names_for_type)."""
    key = _type_cache_key(category, node_type)
    cached = _TYPE_KNOWLEDGE.get(key)
    if cached is not None:
        return cached
    knowledge = _parm_names_for_type(scratch, node_type)
    if not getattr(knowledge, "scene_menus", False):
        _TYPE_KNOWLEDGE[key] = knowledge
    return knowledge


def _menu_error(parm_name: str, value: Any, tokens: list[str]) -> str | None:
    """Why `value` cannot be set on a strict menu parm, or None if it can.

    Houdini only says "Invalid menu item" at set time, after the whole build
    is under way, and build_network then rolls everything back. Catching it
    at validation keeps a typo in one token from costing the whole graph.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if 0 <= value < len(tokens):
            return None
        return f"parm '{parm_name}': menu index {value} is out of range (0-{len(tokens) - 1})"
    if isinstance(value, str) and value not in tokens:
        close = get_close_matches(value, tokens, n=3, cutoff=0.4)
        hint = f" Did you mean: {close}?" if close else ""
        shown = tokens[:15] + (["..."] if len(tokens) > 15 else [])
        return f"parm '{parm_name}': '{value}' is not a menu item. Items: {shown}.{hint}"
    return None


def _expression_of(parm: hou.Parm) -> str | None:
    """The expression *parm* holds, or None when it holds a plain value."""
    try:
        expression = parm.expression()
    except Exception:
        return None
    return expression or None


def _referenced_parm(parm: hou.Parm) -> str | None:
    """Path of the parm a pure `ch()` reference points at, or None.

    hou.Parm.set() writes THROUGH a bare channel reference: setting tx on a
    node whose tx reads ch("../b1/sizex") changes b1's sizex, not tx.
    """
    with contextlib.suppress(Exception):
        target = parm.getReferencedParm()
        if target is not None and target.path() != parm.path():
            return str(target.path())
    return None


# Everything a node spec may carry. An unknown key used to be dropped in
# silence, so a spec with "children" built an empty container and reported
# success.
_SPEC_KEYS = frozenset(
    {
        "type",
        "name",
        "parms",
        "expressions",
        "exprs",
        "inputs",
        "flags",
        "color",
        "comment",
        "override_expression",
    }
)

# Keys of one dict entry of a spec's `inputs` list.
_INPUT_KEYS = frozenset({"index", "input_name", "source", "source_output", "indirect_input"})

# Template types that refuse a non-numeric literal. A string aimed at one of
# these is almost always an expression the caller meant to set.
_NUMERIC_TEMPLATES = frozenset({"Int", "Float", "Toggle"})

_EXPRESSION_LANGUAGES = {"hscript": "Hscript", "python": "Python"}


_FLAG_KEYS = frozenset({"display", "render", "bypass", "template"})


def _unknown_key_error(label: str, kind: str, key: str, known: frozenset) -> str:
    """A did-you-mean refusal for a spec or input key build_network does not know."""
    close = get_close_matches(str(key), sorted(known), n=3, cutoff=0.4)
    hint = f" Did you mean: {close}?" if close else f" Known keys: {sorted(known)}."
    return f"node {label}: unknown {kind} key '{key}'.{hint}"


def _is_numeric_text(value: str) -> bool:
    """Whether a string is just a number written out ('3', '-1.5')."""
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _expression_value(value: Any) -> str | None:
    """The expression inside a `{"expr": "..."}` parm value, or None."""
    if isinstance(value, dict) and "expr" in value:
        return str(value["expr"])
    return None


def _expression_language(name: str):
    """hou.exprLanguage for "hscript" / "python"; raises on anything else."""
    key = str(name).strip().lower()
    if key not in _EXPRESSION_LANGUAGES:
        raise ValueError(f"expression language must be 'hscript' or 'python', got {name!r}")
    return getattr(hou.exprLanguage, _EXPRESSION_LANGUAGES[key])


def _spec_expressions(spec: dict) -> dict[str, str]:
    """Every expression a spec asks for, from all three spellings.

    The `expressions` block (or its alias `exprs`), plus any parm in `parms`
    given as `{"expr": "..."}`: `parms` is where a caller reaches first, and a
    bare expression string there used to fail the whole build.
    """
    found: dict[str, str] = {}
    for block in (spec.get("expressions"), spec.get("exprs")):
        for name, expression in (block or {}).items():
            found[name] = str(expression)
    for name, value in (spec.get("parms") or {}).items():
        wrapped = _expression_value(value)
        if wrapped is not None:
            found[name] = wrapped
    return found


def _apply_parm(
    node: hou.Node, name: str, value: Any, override_expression: bool = False
) -> dict[str, dict[str, str]]:
    """Set a parm or parm tuple, broadcasting scalars and coercing floats.

    Answers what the write did to expressions, per component:
    `expressions_kept` for an expression the literal did not replace (a Ray
    SOP's factory `@N.x` on dir), `written_through` for a pure ch() reference
    the value went through into the parm it reads, and `expressions_removed`
    when *override_expression* cleared them first. Nothing is evaluated: an
    eval outside a cook leaves errors on the node that the build's report
    would then pick up.
    """
    parm = node.parm(name)
    parm_tuple = node.parmTuple(name)
    if isinstance(value, (list, tuple)) or parm is None:
        components = list(parm_tuple) if parm_tuple is not None else []
    else:
        components = [parm]
    before = {p.name(): e for p in components if (e := _expression_of(p)) is not None}
    report: dict[str, dict[str, str]] = {}
    through: dict[str, str] = {}
    if before and override_expression:
        for component in components:
            if component.name() in before:
                with contextlib.suppress(Exception):
                    component.deleteAllKeyframes()
        report["expressions_removed"] = before
    elif before:
        through = {
            p.name(): target
            for p in components
            if p.name() in before and (target := _referenced_parm(p)) is not None
        }
    _set_parm_value(name, parm, parm_tuple, value)
    if before and not override_expression:
        for component in components:
            after = _expression_of(component)
            if after is None or component.name() not in before:
                continue
            if component.name() in through:
                report.setdefault("written_through", {})[component.name()] = through[
                    component.name()
                ]
            else:
                report.setdefault("expressions_kept", {})[component.name()] = after
    return report


def _set_parm_value(name: str, parm, parm_tuple, value: Any) -> None:
    """The plain write behind _apply_parm."""
    if isinstance(value, (list, tuple)):
        if parm_tuple is None:
            raise ValueError(f"'{name}' is not a parm tuple")
        if len(value) != len(parm_tuple):
            raise ValueError(f"'{name}' has {len(parm_tuple)} components, got {len(value)}")
        if all(isinstance(v, (int, float)) for v in value):
            parm_tuple.set([float(v) for v in value])
        else:
            parm_tuple.set(list(value))
    elif parm is not None:
        parm.set(value)
    elif parm_tuple is not None:
        # Scalar onto a tuple: broadcast across components.
        if isinstance(value, (int, float)):
            parm_tuple.set([float(value)] * len(parm_tuple))
        else:
            parm_tuple.set([value] * len(parm_tuple))
    else:
        raise ValueError(f"parameter '{name}' not found")


def _indirect_key(number: int) -> str:
    """How a wire from the parent subnet's input connector *number* is named.

    The connector is not a node and has no path, so the read-back and the
    request both call it this, and the two can be compared like node paths.
    """
    return f"<indirect_input {number}>"


def _indirect_label(connector: Any) -> str:
    """The indirect-input key of a live SubnetIndirectInput.

    HOM wrappers have no stable identity, so the connector is found in its
    subnet's list by path.
    """
    with contextlib.suppress(Exception):
        target = connector.path()
        for number, candidate in enumerate(connector.parent().indirectInputs()):
            if candidate.path() == target:
                return _indirect_key(number)
    return "<indirect_input ?>"


def _actual_inputs(node: hou.Node) -> list[dict[str, Any]]:
    """The connections the node really has, read back off the node.

    Not the ones the spec asked for: the two differ whenever a creation
    callback has an opinion, and the whole reason a caller asks for this is to
    find out that they differ. A wire from the parent subnet's own input
    connector is reported under its indirect-input key, not dropped.
    """
    rows: list[dict[str, Any]] = []
    try:
        connections = node.inputConnections()
    except Exception:  # noqa: BLE001 - not every node type has inputs
        return rows
    for connection in connections:
        with contextlib.suppress(Exception):
            connector = None
            with contextlib.suppress(Exception):
                connector = connection.subnetIndirectInput()
            if connector is not None:
                source, output = _indirect_label(connector), 0
            else:
                node = connection.inputNode()
                source = node.path() if node is not None else None
                output = connection.outputIndex()
            rows.append(
                {"index": connection.inputIndex(), "source": source, "source_output": output}
            )
    return rows


def _network_flags(parent: hou.Node) -> dict[str, Any]:
    """The display node, render node and selection, for putting back afterwards."""
    state: dict[str, Any] = {"display": None, "render": None, "selected": ()}
    with contextlib.suppress(Exception):
        state["display"] = parent.displayNode() if hasattr(parent, "displayNode") else None
    with contextlib.suppress(Exception):
        state["render"] = parent.renderNode() if hasattr(parent, "renderNode") else None
    with contextlib.suppress(Exception):
        state["selected"] = tuple(hou.selectedNodes())
    return state


def _restore_network_flags(state: dict[str, Any]) -> None:
    """Put back what *state* recorded, skipping anything that has since gone."""
    with contextlib.suppress(Exception):
        if state.get("display") is not None:
            state["display"].setDisplayFlag(True)
    with contextlib.suppress(Exception):
        if state.get("render") is not None:
            state["render"].setRenderFlag(True)
    # Per node: one of them having been destroyed in the meantime must not
    # cost the others their selection.
    for position, node in enumerate(state.get("selected") or ()):
        with contextlib.suppress(Exception):
            node.setSelected(True, clear_all_selected_first=position == 0)


# A node's error can be a whole VEX compile listing, and a build reports one
# per created node. Bounded per message and per node, with the count kept.
_ERROR_CHARS = 400
_ERROR_LINES = 5


def _bounded_messages(messages: list) -> tuple:
    """(the first few, clipped, total count) for a node's errors or warnings."""
    shown = [clip(str(message), _ERROR_CHARS) for message in messages[:_ERROR_LINES]]
    return shown, len(messages)


# Attributes Houdini reads with a fixed number of components. attribrandomize
# and attribcreate default to 3, so a "pscale" built with them is a vector that
# copytopoints applies as a non-uniform scale, and nothing warns.
_ATTRIB_SIZES = {"pscale": 1, "width": 1, "Alpha": 1, "N": 3, "v": 3, "Cd": 3, "up": 3}
_ATTRIB_SIZES.update(scale=3, orient=4, rot=4)


def _misshapen_attribs(node: hou.Node) -> dict[tuple[str, str], int]:
    """{("point", "pscale"): 3} for standard attributes with the wrong size.

    Reads only geometry that is already cooked: never forces a cook, so an
    undisplayed heavy branch costs nothing here.
    """
    try:
        if not hasattr(node, "geometry") or node.needsToCook():
            return {}
        geo = node.geometry()
    except Exception:
        return {}
    if geo is None:
        return {}
    found = {}
    for kind, attribs in (("point", geo.pointAttribs()), ("prim", geo.primAttribs())):
        for attrib in attribs:
            want = _ATTRIB_SIZES.get(attrib.name())
            if want is not None and attrib.size() != want:
                found[(kind, attrib.name())] = attrib.size()
    return found


def _attrib_size_warnings(node: hou.Node) -> list[str]:
    """Misshapen attributes this node introduced, not ones it inherited."""
    found = _misshapen_attribs(node)
    if not found:
        return []
    with contextlib.suppress(Exception):
        upstream = node.inputs()[0] if node.inputs() else None
        if upstream is not None:
            for key in _misshapen_attribs(upstream):
                found.pop(key, None)
    return [
        f"{kind} attribute '{name}' has {size} components where Houdini's standard "
        f"is {_ATTRIB_SIZES[name]}; set the creating node's size to match"
        for (kind, name), size in found.items()
    ]


_MESSAGE_LINES = 12


def _condensed(message: str) -> str:
    """A node message with repeated lines dropped and the rest capped.

    An HDA solver reports each failing inner node once per field it merges:
    a FLIP solver missing pscale came back as ~40 lines, the one that named
    the cause buried among copies of "Error cooking SOP: .../FLIP_DATA".
    """
    lines = list(dict.fromkeys(line.strip() for line in message.splitlines() if line.strip()))
    if len(lines) > _MESSAGE_LINES:
        lines = lines[:_MESSAGE_LINES] + [f"... {len(lines) - _MESSAGE_LINES} more distinct lines"]
    return "\n".join(lines)


def _node_report(node: hou.Node, inputs: bool = False) -> dict[str, Any]:
    """Path and type, plus errors / warnings / bypassed only when there are any.

    A clean node used to carry errors: [], warnings: [] and bypassed: false,
    which on a 40-node verify was most of the answer. Absent means none.
    Messages are condensed, then bounded per message and per node with the
    total kept in error_count / warning_count.
    """
    report: dict[str, Any] = {"path": node.path(), "type": node.type().name()}
    # A dangling channel reference cooks clean in Houdini, so it is reported
    # as an error here: otherwise a build reads healthy while it is broken.
    # ponytail: scans every parm of every node; cache per verify if large networks get slow.
    errors, error_count = _bounded_messages(
        [_condensed(e) for e in node.errors()]
        + [e for p in node.parms() for e in broken_references(p)]
    )
    warnings, warning_count = _bounded_messages(
        [_condensed(w) for w in node.warnings()] + _attrib_size_warnings(node)
    )
    if errors:
        report["errors"] = errors
    if warnings:
        report["warnings"] = warnings
    if error_count > len(errors):
        report["error_count"] = error_count
    if warning_count > len(warnings):
        report["warning_count"] = warning_count
    with contextlib.suppress(Exception):
        if node.isBypassed():
            report["bypassed"] = True
    if inputs:
        report["inputs"] = _actual_inputs(node)
    return report


# Stages bigger than this are summarised from the first prims only.
_STAGE_PRIM_CAP = 20000


def _stage_summary(node: hou.Node) -> dict[str, Any] | None:
    """Compact stage evidence for a LOP node: what a render of it would contain.

    A Solaris build used to answer geometry: null, so "is there a camera, are
    the lights in, does every gprim have a material" took three more calls.
    """
    try:
        from pxr import UsdGeom, UsdShade

        stage = node.stage()
    except Exception:
        return None
    if stage is None:
        return None
    counts: dict[str, int] = {}
    meshes = []
    for index, prim in enumerate(stage.Traverse()):
        if index >= _STAGE_PRIM_CAP:
            counts["(truncated)"] = 1
            break
        type_name = str(prim.GetTypeName()) or "(untyped)"
        counts[type_name] = counts.get(type_name, 0) + 1
        # Gprim, not Mesh: a Sphere or Cube LOP renders grey without a
        # material just the same.
        if prim.IsA(UsdGeom.Gprim):
            meshes.append(prim)
    summary: dict[str, Any] = {"prims_by_type": counts}
    lights = sum(n for t, n in counts.items() if "Light" in t)
    summary["cameras"] = counts.get("Camera", 0)
    summary["lights"] = lights
    if meshes:
        with contextlib.suppress(Exception):
            materials, _ = UsdShade.MaterialBindingAPI.ComputeBoundMaterials(
                meshes, UsdShade.Tokens.full
            )
            unbound = [
                str(m.GetPath()) for m, mat in zip(meshes, materials, strict=False) if not mat
            ]
            summary["gprims_without_material"] = len(unbound)
            if unbound:
                summary["unbound_examples"] = unbound[:5]
    return summary


_VOLUME_SUMMARY_KEYS = ("name", "resolution", "min_value", "max_value")


def _geometry_summary(node: hou.Node) -> dict[str, Any] | None:
    """Compact cooked evidence: geometry for a SOP, the stage for a LOP, the
    image layer for a Copernicus COP (whose geometry() is one quad)."""
    if hasattr(node, "stage") and not hasattr(node, "geometry"):
        return _stage_summary(node)
    is_cop = False
    with contextlib.suppress(Exception):
        is_cop = node.type().category().name() == "Cop"
    if is_cop:
        with contextlib.suppress(Exception):
            from fxhoudinimcp_server.handlers.cop_handlers import _layer_summary

            layer = node.layer()
            if layer is not None:
                return {"layer": _layer_summary(layer)}
    if hasattr(node, "tracks") and not hasattr(node, "geometry"):
        # A CHOP's evidence is its channels: that they exist, under the names
        # an export will look for, and that they move.
        with contextlib.suppress(Exception):
            from fxhoudinimcp_server.handlers.chop_handlers import _track_entry

            tracks = node.tracks()
            summary = {"channels": [_track_entry(track) for track in tracks[:12]]}
            if len(tracks) > 12:
                summary["channels_total"] = len(tracks)
            return summary
        return None
    if not hasattr(node, "geometry"):
        return None
    try:
        geo = node.geometry()
    except Exception:
        return None
    if geo is None:
        return None
    bbox = geo.boundingBox()
    summary = {
        "points": geo.intrinsicValue("pointcount"),
        "prims": geo.intrinsicValue("primitivecount"),
        "bbox_min": list(bbox.minvec()),
        "bbox_max": list(bbox.maxvec()),
        "point_attribs": [a.name() for a in geo.pointAttribs()][:30],
    }
    with contextlib.suppress(Exception):
        from fxhoudinimcp_server.handlers.geometry_handlers import prim_type_counts

        if types := prim_type_counts(geo):
            summary["prim_types"] = types
    # A heightfield or a pyro source is volumes: "7 points, attrib P" said
    # nothing about which layers exist or whether they hold anything.
    with contextlib.suppress(Exception):
        from fxhoudinimcp_server.handlers.geometry_handlers import _volume_entry

        volumes = list(geo.primsOfType(hou.primType.Volume)) + list(
            geo.primsOfType(hou.primType.VDB)
        )
        if volumes:
            summary["volumes"] = [
                {k: v for k, v in _volume_entry(prim).items() if k in _VOLUME_SUMMARY_KEYS}
                for prim in volumes[:12]
            ]
            if len(volumes) > 12:
                summary["volumes_total"] = len(volumes)
    warning = update_mode_warning()
    if warning:
        summary["warning"] = warning
    return summary


###### graph.build_network -- the "exact" input policy
#
# Two passes, deliberately separate. The first mutates: it disconnects what the
# spec did not ask for and re-connects what a callback moved, once, and lets a
# refusal propagate so the build rolls back. The second only looks: it reads
# every connection back off the nodes and compares the whole mapping -- index,
# source path, source output -- against what was requested. Nothing here
# retries, and nothing here decides a thing is wired because setInput did not
# raise.


def _names_an_input(spec: dict) -> bool:
    """Whether a spec wires any input by connector name."""
    entries = spec.get("inputs")
    return isinstance(entries, (list, tuple)) and any(
        isinstance(entry, dict) and entry.get("input_name") for entry in entries
    )


def _named_sources(entries: Any) -> list[str]:
    """The node sources a spec's inputs name, whether or not the entry parsed.

    A subnet-connector entry names none, even when it also (wrongly) carries a
    source: that entry is refused for the combination, not for the source.
    """
    if not isinstance(entries, (list, tuple)):
        return []
    sources: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            if entry.get("indirect_input") is not None:
                continue
            entry = entry.get("source")
        if isinstance(entry, str) and entry.strip():
            sources.append(entry)
    return sources


def _input_name_resolver(connectors: dict | None, max_inputs: int, type_name: Any):
    """The input_name -> index rule for one spec, as parse_input_entries takes it.

    None from the resolver when there are no connectors to look in: the type
    did not resolve, and that error is already reported.
    """

    def resolve(input_name: str) -> int | None:
        if connectors is None:
            return None
        try:
            return _find_input(connectors["inputs"], input_name, max_inputs)
        except ValueError as exc:
            raise ValueError(f"{type_name} {exc}") from None

    return resolve


def _requested_map(parsed: list) -> dict[int, tuple]:
    """{index: (source path, source output)} for one spec, as wired.

    A wire from the parent subnet's connector is keyed like the read-back
    reports it (see _indirect_key).
    """
    return {
        entry["index"]: (entry.get("source_path") or entry["source"], entry["source_output"])
        for entry in parsed
    }


def _requested_source(node: hou.Node, parsed: list, index: int) -> Any:
    """The live item input *index* of *node* should be wired from."""
    for entry in parsed:
        if entry["index"] != index:
            continue
        if entry["indirect"] is not None:
            with contextlib.suppress(Exception):
                return list(node.parent().indirectInputs())[entry["indirect"]]
            return None
        return hou.node(str(entry.get("source_path") or entry["source"]))
    return None


def _enforce_exact_inputs(nodes: list, built: list, parsed_inputs: list) -> list:
    """Make each node's inputs the ones its spec asked for. Raises on refusal."""
    enforced: list[dict[str, Any]] = []
    for spec, node, parsed in zip(nodes, built, parsed_inputs, strict=True):
        # An absent "inputs" key is not a claim about inputs, so "exact" has
        # nothing to enforce on it. An empty list is such a claim.
        if spec.get("inputs") is None:
            continue
        wanted = _requested_map(parsed)
        for connection in _actual_inputs(node):
            index = connection["index"]
            if index not in wanted:
                try:
                    node.setInput(index, None)
                except Exception as exc:
                    raise RuntimeError(
                        f"{node.path()}: input {index} is connected to "
                        f"{connection['source']} and the spec did not ask for "
                        f"it, but disconnecting it failed: {readable_message(exc)}"
                    ) from exc
                enforced.append(
                    {
                        "path": node.path(),
                        "index": index,
                        "disconnected": connection["source"],
                    }
                )
                continue
            source_path, source_output = wanted[index]
            if connection["source"] == source_path and connection["source_output"] == source_output:
                continue
            # A creation or parameter callback moved a connection this build
            # already made. One corrective pass, then the read-back decides.
            source = _requested_source(node, parsed, index)
            if source is None:
                raise RuntimeError(
                    f"{node.path()}: input {index} should come from "
                    f"{source_path}, which no longer exists"
                )
            try:
                node.setInput(index, source, source_output)
            except Exception as exc:
                raise RuntimeError(
                    f"{node.path()}: input {index} was moved to "
                    f"{connection['source']} and could not be set back to "
                    f"{source_path}: {readable_message(exc)}"
                ) from exc
            enforced.append(
                {
                    "path": node.path(),
                    "index": index,
                    "rewired_from": connection["source"],
                    "to": source_path,
                }
            )
    return enforced


def _input_mismatches(nodes: list, built: list, parsed_inputs: list) -> list:
    """Every way the wiring still differs from the spec, read off the nodes."""
    mismatches: list[dict[str, Any]] = []
    for spec, node, parsed in zip(nodes, built, parsed_inputs, strict=True):
        if spec.get("inputs") is None:
            continue
        wanted = _requested_map(parsed)
        actual = {
            row["index"]: (row["source"], row["source_output"]) for row in _actual_inputs(node)
        }
        for index, (source_path, source_output) in sorted(wanted.items()):
            if index not in actual:
                mismatches.append(
                    {"path": node.path(), "index": index, "expected": source_path, "actual": None}
                )
            elif actual[index] != (source_path, source_output):
                mismatches.append(
                    {
                        "path": node.path(),
                        "index": index,
                        "expected": source_path,
                        "expected_output": source_output,
                        "actual": actual[index][0],
                        "actual_output": actual[index][1],
                    }
                )
        for index in sorted(set(actual) - set(wanted)):
            mismatches.append(
                {
                    "path": node.path(),
                    "index": index,
                    "expected": None,
                    "actual": actual[index][0],
                }
            )
    return mismatches


# How many built nodes one call will cook and report on. Cooking is the
# expensive half of this command, and a caller that wants more than a handful
# of outputs proved is asking the wrong command.
_INSPECT_CAP = 8

# Tried in this order when a missing parent's container type is inferred from
# the specs, so a spec of types every context has (null, merge) lands in SOPs.
_INFERRED_CATEGORIES = ("Sop", "Lop", "Dop", "Cop", "Chop", "Top", "Driver", "Vop")


def _missing_parent_container(parent_path: str, nodes: Any) -> str | None:
    """The type to create a missing *parent_path* as, or None.

    Building from an empty /obj or /tasks is the usual start, and it used to
    cost a separate create_node before build_network, whose dry run otherwise
    stopped at "Parent not found" without checking a single spec. Only a
    parent directly under an existing OBJ network (any context's container)
    or TOP manager (a topnet) is created, and only when every spec type
    resolves in one context.
    """
    head, _, name = parent_path.rstrip("/").rpartition("/")
    above = hou.node(head or "/")
    if not name or above is None:
        return None
    holds = above.childTypeCategory().name() if above.childTypeCategory() else None
    if holds not in ("Object", "TopNet"):
        return None
    types = {spec.get("type") for spec in nodes or [] if isinstance(spec, dict)}
    if not types or None in types:
        return None
    categories = hou.nodeTypeCategories()
    for category_name in _INFERRED_CATEGORIES:
        category = categories.get(category_name)
        if category and all(_resolve_node_type(category, t) is not None for t in types):
            if holds == "TopNet":
                return "topnet" if category_name == "Top" else None
            return _container_for(category_name)
    return None


def _dry_run_in_scratch(
    above, container: str, name: str, nodes: list, parent_path: str, **options: Any
) -> dict:
    """Validate *nodes* inside a throwaway container named like the real one.

    Same name, so every path in the answer is the one the real build will use.
    *options* are build_network's own (input_policy, inspect_nodes).
    """
    with hou.undos.disabler():
        scratch = above.createNode(container, name)
        try:
            result = build_network(scratch.path(), nodes, dry_run=True, **options)
        finally:
            scratch.destroy()
    result["would_create_parent"] = {"path": parent_path, "type": container}
    return result


def _upstream_with(node: hou.Node, method: str, include_self: bool = False) -> hou.Node | None:
    """*node* (when include_self) or the nearest node up its first inputs having *method*."""
    current = node if include_self else (node.inputs() or (None,))[0]
    seen = set()
    while current is not None and current.path() not in seen:
        if hasattr(current, method):
            return current
        seen.add(current.path())
        current = (current.inputs() or (None,))[0]
    return None


def _build_in_new_container(
    parent_path: str, container: str, nodes: list, dry_run: bool, layout: bool, **options: Any
) -> dict:
    """build_network into a parent created for the purpose; a failed build removes it."""
    head, _, name = parent_path.rstrip("/").rpartition("/")
    above = hou.node(head or "/")
    if dry_run:
        return _dry_run_in_scratch(above, container, name, nodes, parent_path, **options)
    parent = above.createNode(container, name)
    place_new_node(parent)
    try:
        result = build_network(parent.path(), nodes, dry_run=False, layout=layout, **options)
    except Exception:
        parent.destroy()
        raise
    if result.get("success"):
        result["created_parent"] = {"path": parent.path(), "type": container}
    else:
        parent.destroy()
    return result


def build_network(
    parent_path: str,
    nodes: list,
    dry_run: bool = False,
    layout: bool = True,
    input_policy: str = "preserve",
    inspect_nodes: list | str | None = None,
    **_: Any,
) -> dict:
    """Build a whole node network atomically, with upfront validation.

    Every node type, parameter name, parameter shape, and input reference in
    the spec is validated against the running Houdini BEFORE anything is
    created — invalid specs return the full error list and mutate nothing.
    With dry_run=True only the validation runs.

    What comes back is evidence rather than assertion: the inputs each node
    really ended up with, the cook result of the nodes this build produced,
    and which node the geometry figures were read from.

    Args:
        parent_path: Network to build inside (e.g. "/obj/geo1").
        nodes: Node specs, each a dict:
            type (str, required): node type name (unversioned ok).
            name (str): node name, referenceable by later specs.
            parms (dict): parameter values; lists set whole parm tuples.
                A value written {"expr": "...", "language": "hscript" |
                "python"} is set as an expression (hscript by default). A
                string on a numeric parm is refused during validation and
                answered with that spelling. A literal on a parm that holds
                an expression (a Ray SOP's factory `@N.x` on dir) does not
                replace it: the dry run lists these in
                `expressions_in_the_way`, the build reports
                `expressions_kept`.
            expressions (dict): parm name -> expression, as a whole block
                (alias: exprs). Names are validated like those in `parms`.
            override_expression (bool): clear the expressions that literals
                in `parms` land on, so the literals take.
            inputs (list): wiring. Entries are either a source string
                (wired positionally), null (holds the position without
                connecting anything), or {"index" | "input_name", "source",
                "source_output"}; "input_name" is a connector name or
                label, as get_node_card lists them, and wins over "index".
                Sources resolve to spec node names first, then children of
                parent, then absolute paths. {"indirect_input": n} instead
                of "source" wires from connector n of the parent subnet
                itself.
            flags (dict): display/render/bypass/template booleans.
            color (list[3]) and comment (str): network annotations.
            Any other key, in a spec or in an input entry, is a validation
            error with a did-you-mean, not a request dropped in silence.
        dry_run: Validate only; never mutates the scene.
        layout: Also lay out the whole parent network afterwards, which only
            happens when FXHOUDINIMCP_AUTO_LAYOUT is enabled. It does not gate
            placement: the nodes this call creates are always positioned, each
            relative to its inputs, and nodes that already existed never move.
        input_policy: What to do about inputs the spec did not ask for.
            "preserve" (default, and what this command has always done)
            connects what the spec names and leaves everything else as
            Houdini and any creation callbacks left it. "exact" additionally,
            AFTER every creation callback has run, disconnects any input of a
            node whose "inputs" key is present that the spec did not name and
            puts back any the callbacks moved — so an environment that
            auto-wires new nodes is overruled rather than suppressed, and
            "inputs": [] really does mean no inputs. Omitting "inputs"
            entirely still means "no opinion" under both policies, so existing
            specs behave identically.
            Under "exact" the whole mapping (index, source, source output) is
            then read back off the nodes and compared with the request: a
            refusal from Houdini rolls the build back, and a mapping that
            still differs comes back as success=false with input_mismatches
            and created_paths rather than as a build that worked.
        inspect_nodes: Which of the nodes this build created to cook and
            report on. Names from the spec or paths of created nodes.
            Defaults to the build's terminal nodes — the ones nothing else
            in the build consumes. Nodes outside the build are refused
            rather than cooked, because cooking a scene this call did not
            touch is both expensive and not evidence about this call.
    """
    parent = hou.node(parent_path)
    errors: list[str] = []
    try:
        input_policy = resolve_input_policy(input_policy)
    except ValueError as exc:
        return {
            "success": False,
            "valid": False,
            "errors": [str(exc)],
            "message": str(exc),
        }
    if parent is None and (container := _missing_parent_container(parent_path, nodes)):
        return _build_in_new_container(
            parent_path,
            container,
            nodes,
            dry_run,
            layout,
            input_policy=input_policy,
            inspect_nodes=inspect_nodes,
        )
    if parent is None:
        return {
            "success": False,
            "valid": False,
            "errors": [f"Parent not found: {parent_path}"],
            # Every other command reports a single message; carry one here too so a
            # caller does not have to know that this one answers in a list.
            "message": f"Parent not found: {parent_path}",
        }
    category = parent.childTypeCategory()
    if category is None:
        return {
            "success": False,
            "valid": False,
            "errors": [f"{parent_path} cannot contain child nodes"],
        }

    if not isinstance(nodes, list) or not nodes:
        return {
            "success": False,
            "valid": False,
            "errors": ["'nodes' must be a non-empty list"],
        }

    ###### Phase 1: validate everything before touching the scene

    spec_names: list[str] = []
    existing = {child.name() for child in parent.children()}
    resolved_types: dict[str, Any] = {}

    for index, spec in enumerate(nodes):
        if not isinstance(spec, dict):
            errors.append(f"node #{index}: spec must be a dict, got {type(spec).__name__}")
            continue
        label = spec.get("name") or spec.get("type") or f"#{index}"
        # A key this verb does not know is a request it cannot honour. Dropping
        # it quietly is how a spec with "children" reported success for a
        # subnet that was built empty.
        for key in spec:
            if key not in _SPEC_KEYS:
                errors.append(_unknown_key_error(label, "spec", key, _SPEC_KEYS))
        # {"lock": true} or {"Display": true} used to vanish without a word.
        for key in spec.get("flags") or {}:
            if key not in _FLAG_KEYS:
                errors.append(_unknown_key_error(label, "flag", key, _FLAG_KEYS))
        # Only dict entries have keys: a plain source string and a null that
        # holds a position stay valid, and an "inputs" that is not a list at
        # all is parse_input_entries' error to report.
        entries = spec.get("inputs")
        for entry in entries if isinstance(entries, (list, tuple)) else []:
            if isinstance(entry, dict):
                for key in entry:
                    if key not in _INPUT_KEYS:
                        errors.append(_unknown_key_error(label, "input", key, _INPUT_KEYS))
        type_name = spec.get("type")
        if not type_name:
            errors.append(f"node {label}: missing 'type'")
            continue
        if type_name not in resolved_types:
            node_type = _resolve_node_type(category, type_name)
            if node_type is None:
                close = get_close_matches(type_name, list(category.nodeTypes()), n=3, cutoff=0.5)
                hint = f" Did you mean: {close}?" if close else ""
                errors.append(
                    f"node {label}: type '{type_name}' does not exist in {category.name()}.{hint}"
                )
                continue
            resolved_types[type_name] = node_type
        name = spec.get("name")
        if name:
            if name in spec_names:
                errors.append(f"duplicate node name in spec: '{name}'")
            if name in existing:
                errors.append(f"node '{name}' already exists under {parent_path}")
            spec_names.append(name)

    # Learn each unique type's parameters by instantiating it once (probe nodes
    # are destroyed immediately; display/render flags and the user's selection
    # are put back), so a bad parm name or a three-float parm handed two
    # numbers fails validation rather than the build. Runs even when other
    # errors exist: report everything in one pass. The same probe records
    # each parm's template type and the expressions the type ships with.
    parm_knowledge: dict[str, _TypeKnowledge] = {}
    unknown_types = [
        (type_name, node_type)
        for type_name, node_type in resolved_types.items()
        if _type_cache_key(category, node_type) not in _TYPE_KNOWLEDGE
    ]
    for type_name, node_type in resolved_types.items():
        cached = _TYPE_KNOWLEDGE.get(_type_cache_key(category, node_type))
        if cached is not None:
            parm_knowledge[type_name] = cached
    # Multiparm blocks per type, read off the templates (no probe node): the
    # build sets counts before instances, and validation checks instance
    # numbers against the count the same spec sets.
    multiparm_info: dict[str, tuple] = {}

    def multiparms_of(type_name: str) -> tuple:
        if type_name not in multiparm_info:
            multiparm_info[type_name] = _multiparm_blocks(resolved_types[type_name])
        return multiparm_info[type_name]

    # Connectors of a spec that names an input and sets parms, probed with
    # those parms applied (see _probe_connectors). Not cached: the parms
    # decide the answer.
    spec_connectors: dict[int, dict] = {}
    connector_probes = [
        index
        for index, spec in enumerate(nodes)
        if isinstance(spec, dict)
        and resolved_types.get(spec.get("type")) is not None
        and spec.get("parms")
        and _names_an_input(spec)
    ]
    if unknown_types or connector_probes:
        # A probe that stole the display flag or the selection leaves the user
        # looking at something else, and a dry run must leave no trace at all.
        before_probing = _network_flags(parent)
        try:
            for type_name, node_type in unknown_types:
                knowledge = _type_knowledge(parent, node_type, category)
                parm_knowledge[type_name] = knowledge
            for index in connector_probes:
                spec = nodes[index]
                spec_connectors[index] = _probe_connectors(
                    parent,
                    resolved_types[spec["type"]],
                    dict(_count_parms_first(spec["parms"], multiparms_of(spec["type"]))),
                )
        finally:
            _restore_network_flags(before_probing)

    # Parsed inputs are kept so the build wires exactly what validation
    # approved, rather than re-deriving it from the spec a second way.
    parsed_inputs: list[list[dict[str, Any]]] = []

    # The parent's input connectors, listed once per build rather than once
    # per entry that uses one.
    # None when the listing itself failed: _indirect_input_item then lists
    # again and reports the real HOM message instead of "not a subnet".
    listed: list = []

    def parent_connectors() -> list | None:
        if not listed:
            connectors: list | None = None
            with contextlib.suppress(Exception):
                connectors = list(parent.indirectInputs())
            listed.append(connectors)
        return listed[0]

    # Literals the spec aims at parms whose factory expression will outlive
    # them, named before anything is built.
    in_the_way: dict[str, dict[str, dict[str, str]]] = {}

    for index, spec in enumerate(nodes):
        if not isinstance(spec, dict):
            continue  # already reported above
        label = spec.get("name") or spec.get("type") or f"#{index}"
        knowledge = parm_knowledge.get(spec.get("type"))
        if knowledge and not spec.get("override_expression"):
            shipped = knowledge.expressions
            for parm_name, value in (spec.get("parms") or {}).items():
                if _expression_value(value) is None and parm_name in shipped:
                    in_the_way.setdefault(label, {})[parm_name] = shipped[parm_name]
        if knowledge:
            parm_names, tuple_names, menus, instance_patterns, _ = knowledge
            for parm_name in _spec_expressions(spec):
                if parm_name in parm_names or _is_instance_parm(parm_name, instance_patterns):
                    continue
                if parm_name in tuple_names:
                    # setExpression() is per component; the build would find
                    # no parm by the tuple's name and roll everything back.
                    components = sorted(p for p in parm_names if p.startswith(parm_name))
                    errors.append(
                        f"node {label}: '{parm_name}' is a parm tuple; an expression "
                        f"goes on each component {components}"
                    )
                    continue
                close = get_close_matches(parm_name, sorted(parm_names), n=3, cutoff=0.5)
                hint = f" Did you mean: {close}?" if close else ""
                errors.append(f"node {label}: no parm '{parm_name}' to put an expression on.{hint}")
            for parm_name, parm_value in (spec.get("parms") or {}).items():
                if isinstance(parm_value, dict):
                    # An expression wrapper; its parm name was checked above,
                    # and a shape check would refuse any dict as "not a value".
                    unknown = sorted(set(parm_value) - {"expr", "language"})
                    if "expr" not in parm_value:
                        errors.append(
                            f"node {label}: parm '{parm_name}' was given a dict without "
                            f"'expr'. Write {{\"expr\": \"ch('../x')\"}} for an expression, "
                            f"or a plain value."
                        )
                    elif unknown:
                        errors.append(
                            f"node {label}: parm '{parm_name}': unknown key(s) {unknown} "
                            f"in the expression value (known: ['expr', 'language'])"
                        )
                    elif str(parm_value.get("language", "hscript")).strip().lower() not in (
                        _EXPRESSION_LANGUAGES
                    ):
                        errors.append(
                            f"node {label}: parm '{parm_name}': expression language "
                            f"{parm_value['language']!r} is not 'hscript' or 'python'"
                        )
                    continue
                info = knowledge.shapes.get(parm_name)
                template = (info or {}).get("type")
                if parm_name in menus:
                    problem = _menu_error(parm_name, parm_value, menus[parm_name])
                    if problem:
                        errors.append(f"node {label}: {problem}")
                elif (
                    isinstance(parm_value, str)
                    and not _is_numeric_text(parm_value)
                    and template in _NUMERIC_TEMPLATES
                ):
                    # Houdini's own refusal is "Cannot set a numeric parm to a
                    # non-numeric value" plus a dump of the C++ overloads,
                    # raised mid-build, so the whole graph rolled back. Caught
                    # here instead, in terms of what the caller meant.
                    errors.append(
                        f"node {label}: parm '{parm_name}' is numeric "
                        f"({template}) and will not take the string "
                        f"{parm_value!r}. To set it as an expression, write "
                        f'"{parm_name}": {{"expr": {parm_value!r}}} or put it in the '
                        f"spec's 'expressions' block."
                    )
                if info is None:
                    # A multiparm instance has no template on a fresh probe, so
                    # neither its name nor its shape is knowable here; the name
                    # pattern is all there is to go on.
                    if _is_instance_parm(parm_name, instance_patterns):
                        continue
                    labels = dict.fromkeys(parm_names | tuple_names, "")
                    if resolved := resolved_types.get(spec.get("type")):
                        labels.update(template_labels(resolved.parmTemplateGroup().entries()))
                    close = suggest_parms(parm_name, labels)
                    hint = f" Did you mean: {close}?" if close else ""
                    errors.append(
                        f"node {label}: parm '{parm_name}' does not exist "
                        f"on {spec.get('type')}.{hint}"
                    )
                    continue
                errors += check_parm_value(
                    label, str(spec.get("type")), parm_name, parm_value, info
                )
            errors.extend(
                f"node {label}: {problem}"
                for problem in _instance_menu_errors(
                    parent,
                    resolved_types.get(spec.get("type")),
                    spec.get("parms") or {},
                    instance_patterns,
                )
            )
        if isinstance(spec.get("parms"), dict) and spec.get("type") in resolved_types:
            errors += _instance_range_errors(label, spec["parms"], multiparms_of(spec["type"]))
        node_type = resolved_types.get(spec.get("type"))
        max_inputs = node_type.maxNumInputs() if node_type else 0
        # None when the type did not resolve: that error is already reported,
        # and a connector error on top of it would only be noise.
        connectors = spec_connectors.get(index)
        if connectors is None and knowledge:
            connectors = knowledge[4]
        parsed, input_errors = parse_input_entries(
            spec.get("inputs"),
            max_inputs,
            label,
            resolve_name=_input_name_resolver(connectors, max_inputs, spec.get("type")),
        )
        errors += input_errors
        parsed_inputs.append(parsed)
        for entry in parsed:
            if entry["indirect"] is not None:
                # The parent subnet's own connector: not a node, so it has no
                # path a source string could name.
                try:
                    _indirect_input_item(parent, entry["indirect"], parent_connectors())
                except ValueError as exc:
                    errors.append(f"node {label}: {exc}")
        # Every named source, including those of entries refused above for
        # another reason: one pass reports everything wrong with a spec.
        for source in _named_sources(spec.get("inputs")):
            if source in spec_names or source in existing:
                continue
            if hou.node(source) is None:
                errors.append(
                    f"node {label}: input source '{source}' is not a spec "
                    f"node, a child of {parent_path}, or an absolute path"
                )

    if errors:
        return {"success": False, "valid": False, "errors": errors, "created": []}
    if dry_run:
        result = {
            "success": True,
            "valid": True,
            "dry_run": True,
            "validated_nodes": len(nodes),
            "validated_types": sorted(t.name() for t in resolved_types.values()),
            "input_policy": input_policy,
            "planned_connections": sum(len(entries) for entries in parsed_inputs),
            "note": (
                "shapes and names are checked against the live node types; a "
                "dry run does not cook, so it cannot predict cook errors"
            ),
        }
        if in_the_way:
            result["expressions_in_the_way"] = in_the_way
            result["warning"] = (
                f"These parms hold an expression that a literal from the spec will not "
                f"replace: {in_the_way}. Pass override_expression: true on the spec to "
                f"clear it, or set the value as an expression."
            )
        return result

    ###### Phase 2: build (atomic — any failure rolls back)

    created: dict[str, hou.Node] = {}
    # Parallel to *nodes*, because two specs can end up under one key: a spec
    # with no name is filed under whatever name Houdini generated, and zipping
    # the spec list against a dict's values assumed those never collide.
    built: list[hou.Node] = []
    enforced: list[dict[str, Any]] = []
    # A rolled-back build should leave the network as it found it. Destroying
    # the node that holds the display flag hands that flag to something else,
    # so a failed build used to change what the user was looking at.
    flags_before = _network_flags(parent)
    # Per node path: what the literals in `parms` did to expressions.
    parm_reports: dict[str, dict[str, dict[str, str]]] = {}
    flags_not_applied: dict[str, list[str]] = {}
    display_moved: dict[str, str] = {}
    try:
        for spec in nodes:
            node = parent.createNode(resolved_types[spec["type"]].name(), spec.get("name"))
            built.append(node)
            created[spec.get("name") or node.name()] = node

        for spec, node, parsed in zip(nodes, built, parsed_inputs, strict=True):
            override = bool(spec.get("override_expression"))
            spec_parms = spec.get("parms") or {}
            ordered = (
                _count_parms_first(spec_parms, multiparms_of(spec["type"])) if spec_parms else []
            )
            for parm_name, value in ordered:
                if _expression_value(value) is not None:
                    continue  # an {"expr": ...} value, set with the expressions below
                try:
                    written = _apply_parm(node, parm_name, value, override)
                except Exception as exc:
                    items = []
                    with contextlib.suppress(Exception):
                        items = list(node.parm(parm_name).menuItems())
                    shown = f" Menu items: {items[:20]}" if items else ""
                    raise RuntimeError(
                        f"{node.path()} parm '{parm_name}': {readable_message(exc)}.{shown}"
                    ) from exc
                for kind, found in written.items():
                    parm_reports.setdefault(node.path(), {}).setdefault(kind, {}).update(found)
            # "build_network cannot set expressions" was a live session's stated
            # reason for rebuilding a lantern rig in execute_python. It can, from
            # `expressions`, from `exprs`, or from an {"expr": ...} value in `parms`.
            languages = {
                name: value.get("language")
                for name, value in (spec.get("parms") or {}).items()
                if isinstance(value, dict)
            }
            for parm_name, expression in _spec_expressions(spec).items():
                parm = node.parm(parm_name)
                if parm is None:
                    raise RuntimeError(
                        f"{node.path()}: no parm '{parm_name}' to put an expression on"
                    )
                language = languages.get(parm_name)
                if language:
                    parm.setExpression(expression, language=_expression_language(language))
                else:
                    parm.setExpression(expression)
            for entry in parsed:
                if entry["input_name"]:
                    # The built node, parms applied, is the last word on which
                    # index a connector name means; the read-back compares
                    # against the index actually wired.
                    entry["index"] = _resolve_input_index(node, entry["index"], entry["input_name"])
                if entry["indirect"] is not None:
                    source = parent_connectors()[entry["indirect"]]
                    node.setInput(entry["index"], source, 0)
                    entry["source_path"] = _indirect_key(entry["indirect"])
                    continue
                source_name = entry["source"]
                source = (
                    created.get(source_name)
                    or parent.node(str(source_name))
                    or hou.node(str(source_name))
                )
                if source is None:
                    # Validation resolved this name; between then and now
                    # something removed it. setInput(index, None) would
                    # quietly disconnect instead, which is not what was asked.
                    raise RuntimeError(
                        f"{node.path()} input {entry['index']}: source "
                        f"'{source_name}' no longer exists"
                    )
                node.setInput(entry["index"], source, entry["source_output"])
                # The path, not the name: what the read-back will report.
                entry["source_path"] = source.path()
            flags = spec.get("flags") or {}
            for flag, setter in (
                ("display", "setDisplayFlag"),
                ("render", "setRenderFlag"),
                ("bypass", "bypass"),
                ("template", "setTemplateFlag"),
            ):
                if flag not in flags:
                    continue
                if hasattr(node, setter):
                    getattr(node, setter)(bool(flags[flag]))
                    continue
                # A ROP or VOP has no display/template flag: say so rather
                # than skip it as if it had been set.
                flags_not_applied.setdefault(node.path(), []).append(flag)
                if flag == "display" and flags[flag]:
                    # A usdrender_rop at the end of a stage: what the caller
                    # wants seen is what it renders, its input.
                    upstream = _upstream_with(node, setter)
                    if upstream is not None:
                        upstream.setDisplayFlag(True)
                        display_moved[node.path()] = upstream.path()
            if spec.get("color"):
                node.setColor(hou.Color(tuple(spec["color"])))
            if spec.get("comment"):
                node.setComment(spec["comment"])
                node.setGenericFlag(hou.nodeFlag.DisplayComment, True)

        # Inside the atomic block on purpose: every creation callback and
        # every parameter callback has now run, which is the moment "exact"
        # is defined to act, and a refusal here rolls the build back rather
        # than being swallowed into a receipt that claims exact wiring.
        if input_policy == "exact":
            enforced = _enforce_exact_inputs(nodes, built, parsed_inputs)
    except Exception as exc:
        # Newest first: destroying a node that still feeds another one is the
        # order most likely to be refused.
        orphans = []
        for node in reversed(built):
            path = None
            with contextlib.suppress(Exception):
                path = node.path()
            with contextlib.suppress(Exception):
                node.destroy()
            with contextlib.suppress(Exception):
                if path and hou.node(path) is not None:
                    orphans.append(path)
        _restore_network_flags(flags_before)
        failure = {
            "success": False,
            "valid": False,
            "errors": [f"build failed and was rolled back: {readable_message(exc)}"],
            "created": [],
            "rolled_back": len(built),
        }
        if orphans:
            # Saying nothing here is how a caller retries into a name clash
            # against the wreckage of the previous attempt.
            failure["orphans"] = orphans
            failure["errors"].append(
                f"rollback could not destroy {len(orphans)} node(s): {orphans}"
            )
        return failure

    # In a network that was empty, Houdini leaves the display flag on the
    # first node created: a Solaris build displayed (and reported) its lone
    # sphere instead of the finished stage. Unless a spec chose, the last
    # node that can carry the flag gets it, as an artist would set it.
    asked_for_display = any(
        isinstance(spec, dict) and "display" in (spec.get("flags") or {}) for spec in nodes
    )
    if not existing and not asked_for_display and created:
        with contextlib.suppress(Exception):
            last = _upstream_with(list(created.values())[-1], "setDisplayFlag", include_self=True)
            if last is not None:
                last.setDisplayFlag(True)

    # Placement is a floor, not a layout option: whatever `layout` says and
    # whatever the auto-layout flag says, a node THIS call created must not be
    # left stacked at the origin. Each lands relative to its inputs, in
    # creation order; nodes that already existed are never moved, so building
    # into a hand-arranged network stays safe.
    place_new_nodes(created.values())

    if layout:
        # The caller also asked for a layout of the parent, which stays gated
        # by the auto-layout flag as before.
        layout_if_enabled(parent)

    ###### Phase 2c: read the wiring back and compare it with what was asked

    mismatches = _input_mismatches(nodes, built, parsed_inputs) if input_policy == "exact" else []

    ###### Phase 3: verify — cook the nodes this build produced

    order = list(created.keys())
    edges = [
        (spec.get("name") or node.name(), entry["source"])
        for spec, node, parsed in zip(nodes, built, parsed_inputs, strict=True)
        for entry in parsed
    ]
    unresolved: list[str] = []
    if inspect_nodes is None:
        targets = [created[name] for name in terminal_names(order, edges) if name in created]
    else:
        requested = [inspect_nodes] if isinstance(inspect_nodes, str) else list(inspect_nodes)
        by_path = {node.path(): node for node in built}
        by_name = {node.name(): node for node in built}
        targets = []
        for wanted in requested:
            node = created.get(wanted) or by_path.get(str(wanted)) or by_name.get(str(wanted))
            if node is None:
                unresolved.append(str(wanted))
            else:
                targets.append(node)

    inspected: list[dict[str, Any]] = []
    cook_errors: list[dict[str, str]] = []
    for target in targets[:_INSPECT_CAP]:
        # Cook first, report second: a node's errors are only populated once it
        # has tried to do its work, and reading them beforehand would cook it
        # outside the measurement anyway.
        began = time.time()
        cook_error: str | None = None
        try:
            target.cook(force=False)
        except hou.OperationFailed as exc:
            cook_error = readable_message(exc).splitlines()[0][:300]
            cook_errors.append({"path": target.path(), "error": cook_error})
        elapsed = round((time.time() - began) * 1000, 1)
        row = _node_report(target)
        row["cook_ms"] = elapsed
        if cook_error:
            row["cook_error"] = cook_error
        geometry = _geometry_summary(target)
        if geometry is not None:
            row["geometry"] = geometry
        inspected.append(row)

    reports = [_node_report(node, inputs=True) for node in built]
    for report in reports:
        report.update(parm_reports.get(report["path"], {}))
    error_nodes = [r["path"] for r in reports if r.get("errors")]
    for row in inspected:
        if row.get("errors") and row["path"] not in error_nodes:
            error_nodes.append(row["path"])

    display = parent.displayNode() if hasattr(parent, "displayNode") else None
    queried = inspected[0] if inspected else None
    queried_geometry = queried.get("geometry") if queried else None
    if queried_geometry is not None:
        # Carried once, at the top level where callers read it; the queried
        # node's own row points there instead of repeating it.
        queried["geometry"] = "see top-level geometry"
    if queried_geometry is None and hasattr(parent, "stage"):
        # Shaders built inside a Material Library: the evidence is the stage
        # the library now authors (its Material prims), not null.
        with contextlib.suppress(Exception):
            queried_geometry = {"stage_of": parent.path(), **_stage_summary(parent)}

    # How much of the build was actually checked. "cooked": true over an empty
    # target list was the worst kind of answer -- a confident one about
    # something nobody looked at.
    omitted = max(0, len(targets) - _INSPECT_CAP)
    verification: dict[str, Any] = {
        "scope": "terminals" if inspect_nodes is None else "requested",
        "targets_requested": len(targets) + len(unresolved),
        "targets_inspected": len(inspected),
        "targets_omitted": omitted,
        "targets_unresolved": unresolved,
        "complete": bool(inspected) and not omitted and not unresolved,
    }
    if not verification["complete"]:
        verification["note"] = (
            "nothing was cooked, so nothing about cooking is known"
            if not inspected
            else "some targets were not cooked; cooked/healthy cover only the inspected ones"
        )

    # None, not false: unknown is its own answer. A caller that treats None as
    # falsy still does not get told everything was fine. "true" here means
    # every target that was asked about was cooked and was clean -- so a
    # truncated, unresolved or empty target list cannot produce one.
    if cook_errors:
        cooked: bool | None = False
    elif verification["complete"]:
        cooked = True
    else:
        cooked = None
    if error_nodes or cook_errors:
        healthy: bool | None = False
    elif verification["complete"]:
        healthy = True
    else:
        healthy = None

    wiring_ok = not mismatches
    result = {
        # The build succeeded: the nodes exist and are wired as asked. Whether
        # they cook is a separate question with a separate answer, because a
        # single "success" covering both is how a broken network gets reported
        # as a finished one. Wiring that does not match the request under
        # input_policy="exact" is a failure of what was asked for, so it is not
        # reported as a success either.
        "success": wiring_ok,
        "valid": True,
        "created": reports,
        "node_count": len(reports),
        "input_policy": input_policy,
        "queried_node": queried["path"] if queried else None,
        "geometry": queried_geometry,
        "inspected": inspected,
        "error_nodes": error_nodes,
        "cook_errors": cook_errors,
        "cooked": cooked,
        "healthy": healthy,
        "verification": verification,
        "display_node": display.path() if display is not None else None,
        # Whether the display node is even part of this build. When it is not,
        # its geometry is evidence about somebody else's network, which is why
        # it is no longer what "geometry" reports.
        "display_node_in_build": bool(display is not None and display in built),
    }
    if input_policy == "exact":
        result["exact_inputs_verified"] = wiring_ok
    if enforced:
        result["enforced_inputs"] = enforced
    if mismatches:
        # The nodes exist and are listed, so the caller can fix or remove
        # them; what they cannot do is believe the wiring is what they asked
        # for.
        result["input_mismatches"] = mismatches
        result["errors"] = [
            f"input_policy='exact' asked for wiring the network does not have "
            f"after every callback ran: {len(mismatches)} mismatch(es)"
        ]
        result["created_paths"] = [report["path"] for report in reports]
    if unresolved:
        result["inspect_unresolved"] = unresolved
        result["inspect_note"] = (
            "inspect_nodes only addresses nodes this call created; the names "
            "above were not among them and were not cooked"
        )
    if omitted:
        result["inspect_truncated"] = omitted
    if flags_not_applied:
        result["flags_not_applied"] = flags_not_applied
    if display_moved:
        result["display_set_upstream"] = display_moved
    # A literal that did not take is part of the spec this call did not
    # build, so it is said at the top level, not only inside a node report:
    # a bare set() used to leave a Ray SOP's @N.x in place and answer success.
    kept = {
        path: sorted(found["expressions_kept"])
        for path, found in parm_reports.items()
        if found.get("expressions_kept")
    }
    through = {
        path: found["written_through"]
        for path, found in parm_reports.items()
        if found.get("written_through")
    }
    warnings: list[str] = []
    if kept:
        result["expressions_kept"] = kept
        warnings.append(
            f"Literal(s) in the spec did not take, the parameter kept its expression: "
            f"{kept}. Pass override_expression: true on the spec to replace them."
        )
    if through:
        result["written_through"] = through
        warnings.append(
            f"These parameters are pure channel references, so the value was written "
            f"into the parameter each one reads, not into itself: {through}. Pass "
            f"override_expression: true on the spec to break the link instead."
        )
    if warnings:
        result["warning"] = " ".join(warnings)
    return result


###### graph.verify_network


def verify_network(parent_path: str, **_: Any) -> dict:
    """Inspect every node in a network: the 'middle-click everything' pass.

    Cooks the display node, then reports per-node errors/warnings/flags
    plus cooked-geometry evidence, so claims about a build can be checked
    against reality in one call.
    """
    parent = hou.node(parent_path)
    if parent is None:
        raise ValueError(f"Network not found: {parent_path}")

    display = parent.displayNode() if hasattr(parent, "displayNode") else None
    if display is None:
        # A copnet has no displayNode(): the child with the flag is the answer.
        display = next(
            (c for c in parent.children() if getattr(c, "isDisplayFlagSet", lambda: False)()),
            None,
        )
    if display is not None:
        with contextlib.suppress(hou.OperationFailed):
            display.cook(force=False)

    reports = []
    for child in parent.children():
        report = _node_report(child)
        if getattr(child, "isDisplayFlagSet", lambda: False)():
            report["display"] = True
        reports.append(report)

    error_nodes = [r["path"] for r in reports if r.get("errors")]
    licensing = license_error([e for r in reports for e in r.get("errors", ())])
    return {
        "parent_path": parent_path,
        "node_count": len(reports),
        "display_node": display.path() if display is not None else None,
        "geometry": _geometry_summary(display) if display is not None else None,
        "error_nodes": error_nodes,
        "healthy": not error_nodes,
        # Named apart from the rest because no scene change fixes it.
        "license_error": licensing,
        "nodes": reports,
    }


###### graph.get_node_card

_HELP_ZIP_INDEX: dict[str, str] | None = None

_CATEGORY_HELP_DIRS = {
    "Sop": ["sop"],
    "Object": ["obj"],
    "Driver": ["out"],
    "Dop": ["dop"],
    "Lop": ["lop"],
    "Cop": ["copernicus", "cop"],
    "Cop2": ["cop2"],
    "Chop": ["chop"],
    "Top": ["top"],
    "Vop": ["vex", "vop"],
}


def _help_text(node_type, category_name: str) -> str | None:
    """Houdini's own help for a node type, version-exact, headless-safe."""
    global _HELP_ZIP_INDEX
    # hou.text.expandString, not the deprecated hou.expandString.
    zip_path = os.path.join(hou.text.expandString("$HFS"), "houdini", "help", "nodes.zip")
    if _HELP_ZIP_INDEX is None:
        _HELP_ZIP_INDEX = {}
        if os.path.isfile(zip_path):
            with zipfile.ZipFile(zip_path) as zf:
                _HELP_ZIP_INDEX = {name.lower(): name for name in zf.namelist()}

    # "copytopoints::2.0" -> base "copytopoints" (version stripped);
    # "kinefx::rigpose" -> base "rigpose" (namespace stripped). Help files
    # are stored unversioned (old versions get a trailing dash).
    parts = node_type.name().split("::")
    base = parts[-1]
    if len(parts) > 1 and parts[-1][:1].isdigit():
        base = parts[-2]
    candidates = []
    for help_dir in _CATEGORY_HELP_DIRS.get(category_name, [category_name.lower()]):
        candidates += [
            f"{help_dir}/{base}.txt",
            f"{help_dir}/{base}-.txt",
        ]
    for candidate in candidates:
        actual = _HELP_ZIP_INDEX.get(candidate.lower())
        if actual:
            with zipfile.ZipFile(zip_path) as zf:
                return zf.read(actual).decode("utf-8", "replace")

    embedded = node_type.embeddedHelp()
    return embedded or None


def _menu_representation(template) -> str:
    """ "token" or "index" -- what a menu parameter stores, from the template.

    Read off the template rather than inferred from the tokens' spelling: a
    menu of "0", "1", "2" is still a token menu if the template says so, and a
    menu of words can still be stored by index.
    """
    with contextlib.suppress(Exception):
        if template.type() == hou.parmTemplateType.String:
            return "token"
    with contextlib.suppress(Exception):
        if template.menuUseToken():
            return "token"
    with contextlib.suppress(Exception):
        if template.type() == hou.parmTemplateType.Int:
            return "index"
    return "unknown"


def get_node_card(
    node_type: str,
    context: str = "Sop",
    parm_filter: str = None,
    include_help: bool = True,
    **_: Any,
) -> dict:
    """Version-exact documentation card for a node type.

    Sources everything from the running Houdini: real connector labels,
    real parameter names/defaults/menus, and the node's own shipped help
    text — so there is never a reason to guess.
    """
    categories = hou.nodeTypeCategories()
    category = categories.get(context)
    if category is None:
        raise ValueError(f"Unknown context '{context}'. Available: {sorted(categories.keys())}")
    resolved = _resolve_node_type(category, node_type)
    if resolved is None:
        close = get_close_matches(node_type, list(category.nodeTypes()), n=5, cutoff=0.4)
        raise ValueError(f"Node type '{node_type}' not found in {context}. Close matches: {close}")

    # Connectors, in order, by index AND name: the index of `texcoord` on
    # mtlximage is 3, and until now there was nowhere to read that. The same
    # throwaway probe also answers the menus a live parm computes but the
    # type's template does not carry, so it runs before the parameter walk.
    generated_menus: dict[str, dict[str, Any]] = {}
    connectors, not_probed = _connectors_for_type(context, resolved, generated_menus)

    parms: list[dict[str, Any]] = []
    _PARM_CAP = 80
    _MENU_CAP = 15
    truncated = False
    omitted: list[str] = []
    matched = 0
    for template in resolved.parmTemplateGroup().entriesWithoutFolders():
        name, label = template.name(), template.label()
        if (
            parm_filter
            and parm_filter.lower() not in name.lower()
            and parm_filter.lower() not in label.lower()
        ):
            continue
        matched += 1
        if len(parms) >= _PARM_CAP:
            # Past the cap the card still names the parameter, without its
            # details. A File Cache has 82 parameters and the cap was 80, so
            # `savebackground` was invisible and got hunted down by hand.
            truncated = True
            omitted.append(name)
            continue
        entry: dict[str, Any] = {
            "name": name,
            "label": label,
            "type": template.type().name(),
            "size": template.numComponents(),
        }
        # Hidden parameters are still settable, and skipping them silently made
        # this card answer "that parameter does not exist" about parameters that
        # do. Reported and flagged instead, so a caller can tell the difference
        # between hidden and absent.
        if template.isHidden():
            entry["hidden"] = True
        # A name containing '#' is a multiparm instance template, not a real
        # parameter name: the live parameters are source_volume1, source_volume2
        # and so on. Saying so here saves discovering it by trial and error.
        if "#" in name:
            entry["multiparm_instance"] = True
        with contextlib.suppress(Exception):
            entry["default"] = list(template.defaultValue())
        items: list[str] = []
        menu_source = "template"
        with contextlib.suppress(Exception):
            items = list(template.menuItems())
        generated = generated_menus.get(name)
        if not items and generated:
            # The template has no items because a script computes them
            # (`opmenu -l -a file1 loadtype` on filemerge::2.0). An empty Menu
            # on the card reads as "no menu", and a session then took its
            # tokens off another node's card.
            entry["menu_generator"] = generated["generator"]
            items = list(generated.get("items") or [])
            menu_source = "generator"
            labels = generated.get("labels") or []
            if labels:
                entry["menu_labels"] = list(labels)[:_MENU_CAP]
            if not items:
                entry["note"] = (
                    "This menu's items are computed by the script in "
                    "menu_generator and could not be read off a probe instance; "
                    "the menu is not empty on a real node."
                )
        if items:
            entry["menu"] = items[:_MENU_CAP]
            entry["menu_source"] = menu_source
            # Which of the two things in a menu the parameter actually
            # stores. A string parameter stores the token; an integer one
            # stores the index unless its template says otherwise. Guessing
            # from the look of the tokens is how "quads" gets written to a
            # parameter that wanted 4.
            entry["menu_value"] = _menu_representation(template)
            if len(items) > _MENU_CAP:
                # Silent truncation reads as "these are all the options",
                # which is how a caller picks a token that is not in a menu
                # it never saw the rest of.
                entry["menu_truncated"] = True
                entry["menu_count"] = len(items)
        parms.append(entry)

    # Multiparm blocks, which are the reason a parameter can be real and yet
    # findable under no name the caller can guess: the folder's own name is the
    # instance count parameter, and the contents are templates with '#' in them.
    # Recursive, because a multiparm block is almost always nested inside a
    # regular tab folder rather than sitting at the top level. Walking only the
    # top level found none on any real node type.
    multiparms: list[dict[str, Any]] = []

    def _collect_multiparms(entries) -> None:
        for folder in entries:
            if not isinstance(folder, hou.FolderParmTemplate):
                continue
            with contextlib.suppress(Exception):
                if folder.folderType() in _MULTIPARM_FOLDERS:
                    multiparms.append(
                        {
                            "count_parm": folder.name(),
                            "label": folder.label(),
                            "folder_type": folder.folderType().name(),
                            "instance_parms": [
                                child.name()
                                for child in folder.parmTemplates()
                                if "#" in child.name()
                            ][:_PARM_CAP],
                            # Where "#" starts counting: Rig Pose's first
                            # instance is group0, most nodes' is name1, and
                            # guessing wrong cost a failed set_parameters.
                            # None when the multistartoffset tag is not an
                            # integer: unknown, rather than a guess.
                            "first_index": _multiparm_start(folder),
                        }
                    )
            with contextlib.suppress(Exception):
                _collect_multiparms(folder.parmTemplates())

    _collect_multiparms(resolved.parmTemplateGroup().entries())

    help_text = _help_text(resolved, context) if include_help else None
    if help_text and len(help_text) > 5000:
        help_text = help_text[:5000] + "\n[... help truncated]"

    card = {
        "type": resolved.name(),
        "label": resolved.description(),
        "context": context,
        "min_inputs": resolved.minNumInputs(),
        "max_inputs": resolved.maxNumInputs(),
        "max_outputs": resolved.maxNumOutputs(),
        "connectors_probed": connectors is not None,
        "is_generator": resolved.minNumInputs() == 0,
        "parm_count": len(parms),
        "parms_matched": matched,
        # Capped: 274 names (5.6 KB) on a Pyro Solver card filtered to a few.
        "parms_omitted": omitted[:50],
        **({"parms_omitted_count": len(omitted)} if len(omitted) > 50 else {}),
        "parms_truncated": truncated,
        "parms": parms,
        "multiparms": multiparms,
    }
    if help_text is not None:
        card["help"] = help_text
    if connectors is not None:
        card["inputs"] = connectors["inputs"]
        card["outputs"] = connectors["outputs"]
    else:
        card["connectors_note"] = f"Connectors were not read: {not_probed}."
    return card


###### graph.find_expensive_nodes


def find_expensive_nodes(
    root_path: str = "/",
    frame: float = None,
    limit: int = 15,
    **_: Any,
) -> dict:
    """Profile cooking under *root_path* and rank nodes by cook cost.

    Records a hou.perfMon profile while force-cooking the display SOPs
    under the root, then returns the most expensive nodes — how a senior
    finds the slow node instead of guessing.
    """
    root = hou.node(root_path)
    if root is None:
        raise ValueError(f"Node not found: {root_path}")

    if frame is not None:
        hou.setFrame(frame)

    # Collect containers whose children we will force-cook. cook(force)
    # only re-cooks the node itself — cached upstream results record as
    # zero — so every node must be cooked individually inside the profile.
    containers: list[hou.Node] = []
    if hasattr(root, "displayNode") and root.displayNode() is not None:
        containers.append(root)
    else:
        for child in root.allSubChildren():
            if hasattr(child, "displayNode") and child.displayNode() is not None:
                containers.append(child)
            if len(containers) >= 25:
                break

    targets: list[hou.Node] = []
    _NODE_CAP = 300
    for container in containers:
        for child in container.children():
            targets.append(child)
            if len(targets) >= _NODE_CAP:
                break
        if len(targets) >= _NODE_CAP:
            break

    profile = hou.perfMon.startProfile("fxhoudinimcp_expensive_nodes")
    try:
        for target in targets:
            with contextlib.suppress(hou.OperationFailed, AttributeError):
                target.cook(force=True)
    finally:
        profile.stop()

    out = os.path.join(tempfile.mkdtemp(), "profile.hperf")
    profile.save(out)
    with open(out, encoding="utf-8") as fh:
        data = json.load(fh)

    rows: list[tuple[float, str]] = []

    def _walk(entry: dict, path_parts: list[str]) -> None:
        name = entry.get("name", "")
        is_real_node = (
            bool(name)
            and not name.startswith("{")
            and name
            not in (
                "Total Statistics",
                "Other",
                "Nodes",
            )
        )
        parts = path_parts + [name] if is_real_node else path_parts
        cook_ms = 0.0
        for frame_block in (entry.get("stats") or {}).values():
            for sub in frame_block.values():
                cook_ms += sub.get("Cook", 0.0)
        if is_real_node and cook_ms >= 0.5:
            rows.append((cook_ms, "/" + "/".join(parts)))
        for child in entry.get("children") or []:
            _walk(child, parts)

    _walk(data.get("stats", {}), [])
    rows.sort(reverse=True)

    return {
        "root_path": root_path,
        "cooked_nodes": len(targets),
        "top_nodes": [{"path": path, "cook_ms": round(ms, 2)} for ms, path in rows[:limit]],
        "note": (
            "cook_ms is cumulative (parents include children); compare "
            "siblings to find the real hotspot"
        ),
    }


###### Registration

register_handler("graph.build_network", build_network)
register_handler("graph.verify_network", verify_network)
register_handler("graph.get_node_card", get_node_card)
register_handler("graph.find_expensive_nodes", find_expensive_nodes)


###### graph.cook_frame_range

# A sequential solver has to be cooked in order, one frame at a time, and the
# only evidence that it is doing anything is how its output changes across those
# frames. Doing that from the client costs one round trip per frame -- ~50ms
# each before any work happens -- so a 100 frame check was 100 round trips, and
# the alternative was execute_python. Six of thirteen execute_python calls in one
# recorded session were this, every one of them explaining that no tool steps a
# SOP-level solver. step_simulation does not: it requires a DOP network, never
# cooks, and returns no measurements.
_MAX_FRAMES = 480
_FRAME_ATTRIB_CAP = 8


def _frame_stats(entry: dict[str, Any]) -> dict[str, Any]:
    """One attribute's aggregates for a frame row, per axis for a vector.

    min/max over all of P's components mixes x, y and z, so "did the liquid
    land on the ground" could not be read off it; the y range answers that.
    """
    row = {k: v for k, v in entry.items() if k in ("min", "max", "mean", "sum")}
    if entry.get("per_component"):
        row["per_axis"] = [[c["min"], c["max"]] for c in entry["per_component"]]
    return row


def _frame_measurement(
    node: hou.Node,
    attribs: list[str] | None,
    volumes: bool,
) -> dict[str, Any]:
    """What changed on this frame: counts, errors, and the asked-for aggregates."""
    row: dict[str, Any] = {}
    geo = node.geometry()
    if geo is None:
        row["points"] = row["prims"] = 0
        return row
    # Intrinsics, not len(iterPoints()): that builds a Python object per point,
    # every frame, on exactly the heavy sims this tool is for.
    row["points"] = geo.intrinsicValue("pointcount")
    row["prims"] = geo.intrinsicValue("primitivecount")
    # Counts stay flat while pieces fall or cloth drapes; the bounds are the
    # cheap proof that anything moved.
    bbox = geo.boundingBox()
    row["bbox"] = [list(bbox.minvec()), list(bbox.maxvec())]

    if attribs:
        from fxhoudinimcp_server.handlers.geometry_handlers import _get_attrib_stats

        stats = _get_attrib_stats(
            node_path=node.path(),
            attribs=attribs[:_FRAME_ATTRIB_CAP],
            attrib_class="point",
        )
        row["attribs"] = {name: _frame_stats(entry) for name, entry in stats["stats"].items()}
        if stats["missing"]:
            row["missing_attribs"] = stats["missing"]

    if volumes:
        from fxhoudinimcp_server.handlers.geometry_handlers import _get_volume_info

        info = _get_volume_info(node_path=node.path())
        row["volumes"] = [
            {
                k: v
                for k, v in entry.items()
                if k in ("name", "resolution", "active_voxels", "min_value", "max_value")
            }
            for entry in info["volumes"]
        ]
    return row


def _shape(row: dict[str, Any]) -> tuple:
    return row.get("points"), row.get("prims"), json.dumps(row.get("bbox"))


def cook_frame_range(
    node_path: str,
    start: float | None = None,
    end: float | None = None,
    step: float = 1.0,
    attribs: list[str] | str | None = None,
    volumes: bool = False,
    **_: Any,
) -> dict[str, Any]:
    """Cook a node frame by frame and report what changed on each frame.

    This is the tool for proving a simulation is doing something, and the only
    correct way to advance a sequential solver: frames are cooked in order, so a
    SOP solver, a DOP network or a plain animated chain all accumulate properly.

    The frame is left at the last one cooked, because that is what stepping a
    solver means; the caller usually wants to screenshot or read it afterwards.

    Args:
        node_path: Node to cook. Its output is what gets measured.
        start: First frame. Defaults to the playbar start.
        end: Last frame, inclusive. Defaults to the playbar end.
        step: Frame increment. 1.0 for a sequential solver -- skipping frames
            gives a solver a discontinuous time step and invalid results.
        attribs: Point attributes to aggregate per frame (min/max/mean/sum).
        volumes: Also report per-volume name, resolution and value range.
    """
    node = hou.node(node_path)
    if node is None:
        raise hou.OperationFailed(f"Node not found: {node_path}")

    playbar_start, playbar_end = hou.playbar.frameRange()
    start = float(playbar_start if start is None else start)
    end = float(playbar_end if end is None else end)
    if step <= 0:
        raise ValueError("step must be greater than 0")
    if end < start:
        raise ValueError(f"end ({end}) is before start ({start})")

    count = int((end - start) / step) + 1
    if count > _MAX_FRAMES:
        raise ValueError(
            f"{count} frames requested; the cap is {_MAX_FRAMES}. Narrow the "
            "range, or raise step if the node is not a sequential solver."
        )

    if isinstance(attribs, str):
        attribs = [attribs]

    frames: list[dict[str, Any]] = []
    total_ms = 0.0
    first_error_frame: float | None = None

    for index in range(count):
        frame = start + index * step
        hou.setFrame(frame)
        began = time.time()
        try:
            node.cook(force=False)
            cook_error = None
        except hou.OperationFailed as exc:
            # A cook failure is data, not a reason to abandon the range: a solver
            # that fails on one frame and recovers is exactly what the caller is
            # trying to see.
            cook_error = str(exc).splitlines()[0][:200]
        elapsed_ms = round((time.time() - began) * 1000, 1)
        total_ms += elapsed_ms

        row: dict[str, Any] = {"frame": frame, "cook_ms": elapsed_ms}
        if cook_error:
            row["cook_error"] = cook_error
        with contextlib.suppress(hou.OperationFailed):
            # Only when there are any: an empty pair per frame is most of a
            # 240-frame answer.
            if errors := [e.splitlines()[0][:200] for e in node.errors()]:
                row["errors"] = errors
            if warnings := [w.splitlines()[0][:200] for w in node.warnings()]:
                row["warnings"] = warnings
        if row.get("errors") and first_error_frame is None:
            first_error_frame = frame
        try:
            row.update(_frame_measurement(node, attribs, volumes))
        except hou.OperationFailed as exc:
            row["measure_error"] = str(exc).splitlines()[0][:200]
        frames.append(row)

    shown = _thin_frame_rows(frames)
    result: dict[str, Any] = {
        "node_path": node_path,
        "start": start,
        "end": end,
        "step": step,
        "frames_cooked": len(frames),
        "total_cook_ms": round(total_ms, 1),
        "mean_cook_ms": round(total_ms / len(frames), 1) if frames else 0.0,
        "first_error_frame": first_error_frame,
        # True when counts and bounds never changed: the node cooks, but over
        # this range it does nothing, which is what a sim with no gravity, a
        # pinned-everything cloth or an unconnected source looks like.
        "static": len(frames) > 1 and all(_shape(row) == _shape(frames[0]) for row in frames),
        "current_frame": hou.frame(),
        "frames": shown,
    }
    if frames:
        slowest = max(frames, key=lambda row: row["cook_ms"])
        result["slowest_frame"] = {"frame": slowest["frame"], "cook_ms": slowest["cook_ms"]}
    if len(shown) < len(frames):
        result["frames_shown"] = len(shown)
        result["frames_note"] = (
            "Rows are evenly spaced frames plus every frame with an error or "
            "warning; the totals above cover every frame. Cook a narrower range "
            "for every row."
        )
    return result


# Rows returned when a range is long: 48 frames of a crowd were 15 KB of rows
# that differed in the fourth digit, and 240 would have been 75 KB.
_FRAME_ROWS_SHOWN = 25


def _thin_frame_rows(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evenly spaced rows (first and last included) plus every row with a problem."""
    if len(frames) <= _FRAME_ROWS_SHOWN:
        return frames
    last = len(frames) - 1
    keep = {round(i * last / (_FRAME_ROWS_SHOWN - 1)) for i in range(_FRAME_ROWS_SHOWN)}
    flagged = [
        i
        for i, row in enumerate(frames)
        if row.get("errors") or row.get("cook_error") or row.get("warnings")
    ]
    keep.update(flagged[:_FRAME_ROWS_SHOWN])
    return [frames[i] for i in sorted(keep)]


register_handler("graph.cook_frame_range", cook_frame_range)


###### graph.get_cook_status


def get_cook_status(node_path: str = "/obj", **_: Any) -> dict:
    """Whether a node has cooked, how often, and whether it changes per frame.

    Read the limitation first: every command runs on Houdini's main thread, so a
    long cook BLOCKS the bridge and cannot be polled from outside while it runs.
    A recorded session ended up watching the Houdini process's CPU from
    PowerShell for exactly that reason, and no tool can change that shape. What
    this answers is the after-the-fact question -- did the thing actually recook,
    is it time dependent, did it end up in error. For genuinely asynchronous
    work, use a ROP's background execution and poll get_render_progress.

    Args:
        node_path: Node to report on.
    """
    node = hou.node(node_path)
    if node is None:
        raise hou.OperationFailed(f"Node not found: {node_path}")

    result: dict[str, Any] = {
        "node_path": node.path(),
        "type": node.type().name(),
        "frame": hou.frame(),
    }
    for key, method in (
        ("cook_count", "cookCount"),
        ("is_time_dependent", "isTimeDependent"),
    ):
        with contextlib.suppress(Exception):
            result[key] = getattr(node, method)()
    with contextlib.suppress(Exception):
        result["errors"] = [e.splitlines()[0][:200] for e in node.errors()]
        result["warnings"] = [w.splitlines()[0][:200] for w in node.warnings()]
    with contextlib.suppress(Exception):
        result["unsaved_changes"] = hou.hipFile.hasUnsavedChanges()
    with contextlib.suppress(Exception):
        result["hip_file"] = hou.hipFile.path()
    return result


register_handler("graph.get_cook_status", get_cook_status)
