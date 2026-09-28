"""Geometry (SOP) handlers for FXHoudini-MCP.

Each handler reads or modifies geometry on SOP nodes.
All functions run on the main thread via the dispatcher.
"""

from __future__ import annotations

# Built-in
import contextlib
import random
from typing import Any

# Third-party
import hou

# Internal
from fxhoudinimcp_server.config import place_new_node, update_mode_warning
from fxhoudinimcp_server.dispatcher import register_handler

###### Helpers


def _get_sop_geo(node_path: str, output_index: int = 0) -> hou.Geometry:
    """Return the cooked read-only geometry for a SOP node.

    output_index picks a secondary output: a FLIP compress node or a Vellum
    solver carries different streams on outputs 1 and 2, and "no tool reports
    per-output geometry" sent a session to execute_python for them.

    Raises:
        hou.OperationFailed: if the node doesn't exist or has no geometry.
    """
    node = hou.node(node_path)
    if node is None:
        raise hou.OperationFailed(f"Node not found: {node_path}")
    geo = node.geometry(output_index) if output_index else node.geometry()
    if geo is None:
        raise hou.OperationFailed(f"Node has no geometry: {node_path}")
    return geo


def _vec_to_list(v: Any) -> Any:
    """Convert hou.Vector2/3/4, hou.Color, etc. to a plain list."""
    if isinstance(v, (hou.Vector2, hou.Vector3, hou.Vector4, hou.Quaternion, hou.Color)):
        return list(v)
    if isinstance(v, hou.Matrix3):
        return [[v.at(r, c) for c in range(3)] for r in range(3)]
    if isinstance(v, hou.Matrix4):
        return [[v.at(r, c) for c in range(4)] for r in range(4)]
    return v


def _attrib_meta(attrib: hou.Attrib) -> dict[str, Any]:
    """Return JSON-safe metadata for an attribute."""
    return {
        "name": attrib.name(),
        "type": attrib.dataType().name(),
        "size": attrib.size(),
        "is_array": attrib.isArrayType(),
    }


def _attrib_class_obj(geo: hou.Geometry, attrib_class: str) -> Any:
    """Map a string class name to the hou.attribType enum value."""
    mapping = {
        "point": hou.attribType.Point,
        "prim": hou.attribType.Prim,
        "vertex": hou.attribType.Vertex,
        "detail": hou.attribType.Global,
        "global": hou.attribType.Global,
    }
    cls = mapping.get(attrib_class.lower())
    if cls is None:
        raise ValueError(
            f"Invalid attrib_class: {attrib_class!r}. Use one of {list(mapping.keys())}"
        )
    return cls


###### geometry.get_geometry_info


# Packed types countPrimType also counts under PackedPrim.
_PACKED_SUBTYPES = ("Agent", "PackedGeometry", "PackedFragment")


def prim_type_counts(geo: hou.Geometry) -> dict[str, int]:
    """Exact primitive counts per type, e.g. {"Polygon": 12, "Volume": 2}.

    countPrimType runs in C++ (0.4 ms for every type on 1000 agents); the old
    per-prim walk sampled 2,500 prims and scaled the counts up. PackedPrim
    counts every packed prim, agents included, so only the packed prims no
    more specific type claims are reported under it.
    """
    counts: dict[str, int] = {}
    for name in dir(hou.primType):
        value = getattr(hou.primType, name)
        if name.startswith("_") or name == "Unknown" or not isinstance(value, hou.EnumValue):
            continue
        with contextlib.suppress(Exception):
            if count := geo.countPrimType(value):
                counts[name] = count
    if "PackedPrim" in counts:
        other = counts.pop("PackedPrim") - sum(counts.get(t, 0) for t in _PACKED_SUBTYPES)
        if other > 0:
            counts["PackedPrim"] = other
    return counts


def _get_geometry_info(*, node_path: str, output_index: int = 0, **_: Any) -> dict[str, Any]:
    """Return summary information about a SOP node's geometry."""
    geo = _get_sop_geo(node_path, output_index)

    # Attribute lists per class
    attribs: dict[str, list[dict]] = {}
    for label, getter in [
        ("point", geo.pointAttribs),
        ("prim", geo.primAttribs),
        ("vertex", geo.vertexAttribs),
        ("detail", geo.globalAttribs),
    ]:
        attribs[label] = [_attrib_meta(a) for a in getter()]

    prim_types = prim_type_counts(geo)

    bbox = geo.boundingBox()

    result: dict[str, Any] = {
        "node_path": node_path,
        "num_points": geo.intrinsicValue("pointcount"),
        "num_prims": geo.intrinsicValue("primitivecount"),
        "num_vertices": geo.intrinsicValue("vertexcount"),
        "attributes": attribs,
        "bounding_box": {
            "min": list(bbox.minvec()),
            "max": list(bbox.maxvec()),
            "size": list(bbox.sizevec()),
            "center": list(bbox.center()),
        },
        "prim_type_breakdown": prim_types,
    }
    warning = update_mode_warning()
    if warning:
        result["warnings"] = [warning]
    return result


register_handler("geometry.get_geometry_info", _get_geometry_info)


###### geometry.get_points


def _get_points(
    *,
    node_path: str,
    attributes: list[str] | None = None,
    start: int = 0,
    count: int = 200,
    group: str | None = None,
) -> dict[str, Any]:
    """Read point positions and attributes with pagination.

    Uses indexed access (geo.point) so the cost scales with the page
    size, never with the total point count.
    """
    geo = _get_sop_geo(node_path)

    if attributes is None:
        attributes = ["P"]

    if group:
        pt_group = geo.findPointGroup(group)
        if pt_group is None:
            raise hou.OperationFailed(f"Point group not found: {group}")
        total = _group_size(pt_group, "pointCount", "points")
        end = min(start + count, total)
        page = _group_page(pt_group, "iterPoints", "points", start, end)
    else:
        total = geo.intrinsicValue("pointcount")
        end = min(start + count, total)
        page = [geo.point(i) for i in range(max(start, 0), end)]

    found = {name: geo.findPointAttrib(name) is not None for name in attributes}

    rows: list[dict[str, Any]] = []
    for pt in page:
        row: dict[str, Any] = {"index": pt.number()}
        for attr_name in attributes:
            row[attr_name] = _vec_to_list(pt.attribValue(attr_name)) if found[attr_name] else None
        rows.append(row)

    return {
        "node_path": node_path,
        "total_points": total,
        "start": start,
        "count": len(rows),
        "has_more": end < total,
        "points": rows,
    }


register_handler("geometry.get_points", _get_points)


###### geometry.get_prims


def _get_prims(
    *,
    node_path: str,
    attributes: list[str] | None = None,
    start: int = 0,
    count: int = 200,
    group: str | None = None,
) -> dict[str, Any]:
    """Read primitive data and attributes with pagination."""
    geo = _get_sop_geo(node_path)

    if group:
        pr_group = geo.findPrimGroup(group)
        if pr_group is None:
            raise hou.OperationFailed(f"Prim group not found: {group}")
        total = _group_size(pr_group, "primCount", "prims")
        end = min(start + count, total)
        page = _group_page(pr_group, "iterPrims", "prims", start, end)
    else:
        # Indexed access keeps the cost proportional to the page size.
        total = geo.intrinsicValue("primitivecount")
        end = min(start + count, total)
        page = [geo.prim(i) for i in range(max(start, 0), end)]

    # If no attributes specified, gather all prim attribute names
    if attributes is None:
        attributes = [a.name() for a in geo.primAttribs()]

    rows: list[dict[str, Any]] = []
    for prim in page:
        row: dict[str, Any] = {
            "index": prim.number(),
            "type": prim.type().name(),
            "num_vertices": prim.numVertices(),
        }
        for attr_name in attributes:
            attrib = geo.findPrimAttrib(attr_name)
            if attrib is None:
                row[attr_name] = None
                continue
            val = prim.attribValue(attr_name)
            row[attr_name] = _vec_to_list(val)
        rows.append(row)

    return {
        "node_path": node_path,
        "total_prims": total,
        "start": start,
        "count": len(rows),
        "has_more": end < total,
        "prims": rows,
    }


register_handler("geometry.get_prims", _get_prims)


###### geometry.get_attrib_values


def _get_attrib_values(
    *,
    node_path: str,
    attrib_name: str,
    attrib_class: str = "point",
    start: int = 0,
    count: int = 200,
) -> dict[str, Any]:
    """Read attribute values as a flat array with pagination.

    Values are element-major (e.g. for a float3 attribute with tuple_size=3,
    every 3 consecutive values belong to one element).  Use start/count to
    page through large attributes without blowing the LLM context.
    """
    geo = _get_sop_geo(node_path)

    cls = attrib_class.lower()
    finders = {
        "point": geo.findPointAttrib,
        "prim": geo.findPrimAttrib,
        "vertex": geo.findVertexAttrib,
    }
    if cls in finders:
        attrib = finders[cls](attrib_name)
        if attrib is None:
            raise hou.OperationFailed(
                f"{cls.capitalize()} attribute '{attrib_name}' not found on {node_path}"
            )
        all_values = _attrib_array(geo, cls, attrib)
    elif cls in ("detail", "global"):
        attrib = geo.findGlobalAttrib(attrib_name)
        if attrib is None:
            raise hou.OperationFailed(f"Detail attribute '{attrib_name}' not found on {node_path}")
        val = geo.attribValue(attrib_name)
        return {
            "node_path": node_path,
            "attrib_name": attrib_name,
            "attrib_class": attrib_class,
            "size": attrib.size(),
            "type": attrib.dataType().name(),
            "value": _vec_to_list(val),
        }
    else:
        raise ValueError(f"Invalid attrib_class: {attrib_class!r}")

    tuple_size = max(attrib.size(), 1)
    total_elements = len(all_values) // tuple_size
    # Clamp page to element boundaries
    start_elem = max(0, min(start, total_elements))
    end_elem = min(start_elem + max(1, count), total_elements)
    window = all_values[start_elem * tuple_size : end_elem * tuple_size]
    page = window.tolist() if hasattr(window, "tolist") else list(window)

    return {
        "node_path": node_path,
        "attrib_name": attrib_name,
        "attrib_class": attrib_class,
        "tuple_size": tuple_size,
        "type": attrib.dataType().name(),
        "total_elements": total_elements,
        "start": start_elem,
        "count": end_elem - start_elem,
        "has_more": end_elem < total_elements,
        "values": page,
    }


def _attrib_array(geo, cls: str, attrib):
    """Every value of one attribute, flat: a numpy view for numbers.

    The tuple of Python floats this used to build is 3M objects for P on a
    1M-point mesh, made to return a 200-element page. The raw buffer costs
    one copy in C and the page is sliced out of it.
    """
    name = attrib.name()
    data = attrib.dataType()
    if data in (hou.attribData.Float, hou.attribData.Int):
        kind = "Float" if data == hou.attribData.Float else "Int"
        with contextlib.suppress(Exception):
            import numpy as np

            numeric = hou.numericData.Float32 if kind == "Float" else hou.numericData.Int32
            raw = getattr(geo, f"{cls}{kind}AttribValuesAsString")(name, numeric)
            return np.frombuffer(raw, dtype=np.float32 if kind == "Float" else np.int32)
        return getattr(geo, f"{cls}{kind}AttribValues")(name)
    return getattr(geo, f"{cls}StringAttribValues")(name)


register_handler("geometry.get_attrib_values", _get_attrib_values)


###### geometry.set_detail_attrib


def _set_detail_attrib(
    *,
    node_path: str,
    attrib_name: str,
    value: Any,
) -> dict[str, Any]:
    """Set a detail (global) attribute via an appended Attribute Create SOP.

    SOP geometry cannot be edited in place from outside a node cook
    (hou.SopNode has no setGeometry), so this wires an attribcreate node
    after *node_path* and moves the display/render flags to it.
    """
    node = hou.node(node_path)
    if node is None:
        raise hou.OperationFailed(f"Node not found: {node_path}")
    if node.geometry() is None:
        raise hou.OperationFailed(f"Node has no geometry: {node_path}")

    attrib_node = node.parent().createNode("attribcreate", f"set_{attrib_name}")
    attrib_node.setInput(0, node)
    place_new_node(attrib_node)
    attrib_node.parm("numattr").set(1)
    attrib_node.parm("class1").set("detail")
    attrib_node.parm("name1").set(attrib_name)

    if isinstance(value, str):
        attrib_node.parm("type1").set("index")
        attrib_node.parm("string1").set(value)
    elif isinstance(value, bool):
        attrib_node.parm("type1").set("int")
        attrib_node.parm("value1v1").set(int(value))
    elif isinstance(value, int):
        attrib_node.parm("type1").set("int")
        attrib_node.parm("value1v1").set(value)
    elif isinstance(value, float):
        attrib_node.parm("type1").set("float")
        attrib_node.parm("value1v1").set(value)
    elif isinstance(value, (list, tuple)) and 1 <= len(value) <= 4:
        attrib_node.parm("type1").set("float")
        attrib_node.parm("size1").set(len(value))
        for index, component in enumerate(value):
            attrib_node.parm(f"value1v{index + 1}").set(float(component))
    else:
        attrib_node.destroy()
        raise ValueError(f"Unsupported value type: {type(value).__name__}")

    attrib_node.setDisplayFlag(True)
    attrib_node.setRenderFlag(True)

    # Read the value back from the cooked geometry so the result reports
    # what actually happened rather than what was requested.
    applied = attrib_node.geometry().attribValue(attrib_name)

    return {
        "node_path": node_path,
        "attrib_node_path": attrib_node.path(),
        "attrib_name": attrib_name,
        "value": _vec_to_list(applied),
        "success": True,
    }


register_handler("geometry.set_detail_attrib", _set_detail_attrib)


###### geometry.get_groups


def _get_groups(*, node_path: str) -> dict[str, Any]:
    """List all point/prim/edge groups with membership counts."""
    geo = _get_sop_geo(node_path)

    groups: dict[str, list[dict[str, Any]]] = {
        "point_groups": [],
        "prim_groups": [],
        "edge_groups": [],
    }

    for grp in geo.pointGroups():
        groups["point_groups"].append(
            {
                "name": grp.name(),
                "count": _group_size(grp, "pointCount", "points"),
            }
        )

    for grp in geo.primGroups():
        groups["prim_groups"].append(
            {
                "name": grp.name(),
                "count": _group_size(grp, "primCount", "prims"),
            }
        )

    for grp in geo.edgeGroups():
        groups["edge_groups"].append(
            {
                "name": grp.name(),
                "count": _group_size(grp, "edgeCount", "edges"),
            }
        )

    return {
        "node_path": node_path,
        **groups,
    }


def _group_page(group, iterator: str, lister: str, start: int, end: int) -> list:
    """Members start..end of a group, walking only that far into it."""
    import itertools

    try:
        return list(itertools.islice(getattr(group, iterator)(), max(start, 0), end))
    except Exception:
        return list(getattr(group, lister)()[start:end])


def _group_size(group, counter: str, lister: str) -> int:
    """A group's member count without building its members (pointCount etc.)."""
    try:
        return int(getattr(group, counter)())
    except Exception:
        return len(getattr(group, lister)())


register_handler("geometry.get_groups", _get_groups)


###### geometry.get_group_members


def _get_group_members(
    *,
    node_path: str,
    group_name: str,
    group_type: str = "point",
    start: int = 0,
    count: int = 5000,
) -> dict[str, Any]:
    """Get element indices belonging to a group, with pagination."""
    geo = _get_sop_geo(node_path)

    gt = group_type.lower()
    if gt == "point":
        grp = geo.findPointGroup(group_name)
        if grp is None:
            raise hou.OperationFailed(f"Point group '{group_name}' not found on {node_path}")
        total = _group_size(grp, "pointCount", "points")
        end = min(start + max(1, count), total)
        page = [pt.number() for pt in _group_page(grp, "iterPoints", "points", start, end)]
    elif gt == "prim":
        grp = geo.findPrimGroup(group_name)
        if grp is None:
            raise hou.OperationFailed(f"Prim group '{group_name}' not found on {node_path}")
        total = _group_size(grp, "primCount", "prims")
        end = min(start + max(1, count), total)
        page = [pr.number() for pr in _group_page(grp, "iterPrims", "prims", start, end)]
    elif gt == "edge":
        grp = geo.findEdgeGroup(group_name)
        if grp is None:
            raise hou.OperationFailed(f"Edge group '{group_name}' not found on {node_path}")
        total = _group_size(grp, "edgeCount", "edges")
        end = min(start + max(1, count), total)
        page = [
            [points[0].number(), points[1].number()]
            for points in (edge.points() for edge in grp.edges()[start:end])
        ]
    else:
        raise ValueError(f"Invalid group_type: {group_type!r}. Use 'point', 'prim', or 'edge'.")

    return {
        "node_path": node_path,
        "group_name": group_name,
        "group_type": group_type,
        "total_count": total,
        "start": start,
        "count": len(page),
        "has_more": end < total,
        "members": page,
    }


register_handler("geometry.get_group_members", _get_group_members)


###### geometry.get_bounding_box


def _get_bounding_box(*, node_path: str) -> dict[str, Any]:
    """Get axis-aligned bounding box for a SOP node's geometry."""
    geo = _get_sop_geo(node_path)
    bbox = geo.boundingBox()

    return {
        "node_path": node_path,
        "min": list(bbox.minvec()),
        "max": list(bbox.maxvec()),
        "size": list(bbox.sizevec()),
        "center": list(bbox.center()),
    }


register_handler("geometry.get_bounding_box", _get_bounding_box)


###### geometry.get_attribute_info


def _get_attribute_info(
    *,
    node_path: str,
    attrib_name: str,
    attrib_class: str = "point",
) -> dict[str, Any]:
    """Detailed attribute info: type, size, default value."""
    geo = _get_sop_geo(node_path)

    cls = attrib_class.lower()
    finders = {
        "point": geo.findPointAttrib,
        "prim": geo.findPrimAttrib,
        "vertex": geo.findVertexAttrib,
        "detail": geo.findGlobalAttrib,
        "global": geo.findGlobalAttrib,
    }
    finder = finders.get(cls)
    if finder is None:
        raise ValueError(f"Invalid attrib_class: {attrib_class!r}")

    attrib = finder(attrib_name)
    if attrib is None:
        raise hou.OperationFailed(
            f"{attrib_class.title()} attribute '{attrib_name}' not found on {node_path}"
        )

    default = attrib.defaultValue()

    return {
        "node_path": node_path,
        "attrib_name": attrib_name,
        "attrib_class": attrib_class,
        "type": attrib.dataType().name(),
        "size": attrib.size(),
        "is_array": attrib.isArrayType(),
        "default_value": _vec_to_list(default),
        "qualifier": attrib.qualifier() if hasattr(attrib, "qualifier") else None,
        "type_name": attrib.dataType().name(),
    }


register_handler("geometry.get_attribute_info", _get_attribute_info)


###### geometry.sample_geometry


def _sample_geometry(
    *,
    node_path: str,
    sample_count: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """Smart sampling: get N evenly distributed points from geometry."""
    geo = _get_sop_geo(node_path)

    total = geo.intrinsicValue("pointcount")

    if total == 0:
        return {
            "node_path": node_path,
            "total_points": 0,
            "sample_count": 0,
            "points": [],
        }

    # Determine indices to sample
    actual_count = min(sample_count, total)

    if actual_count >= total:
        # Return all points
        sampled_indices = list(range(total))
    else:
        # Evenly spaced sampling with optional seed for reproducibility
        rng = random.Random(seed)
        step = total / actual_count
        # Start with evenly spaced, then jitter slightly for variety
        sampled_indices = sorted(
            {
                min(int(i * step + rng.uniform(0, step * 0.5)), total - 1)
                for i in range(actual_count)
            }
        )

    # Gather attributes — indexed access keeps the cost O(sample_count)
    point_attrib_names = [a.name() for a in geo.pointAttribs()]

    rows: list[dict[str, Any]] = []
    for idx in sampled_indices:
        pt = geo.point(idx)
        row: dict[str, Any] = {"index": pt.number()}
        for attr_name in point_attrib_names:
            val = pt.attribValue(attr_name)
            row[attr_name] = _vec_to_list(val)
        rows.append(row)

    return {
        "node_path": node_path,
        "total_points": total,
        "sample_count": len(rows),
        "seed": seed,
        "attributes": point_attrib_names,
        "points": rows,
    }


register_handler("geometry.sample_geometry", _sample_geometry)


###### geometry.get_prim_intrinsics


def _intrinsic_row(prim, names: list[str] | None) -> dict[str, Any]:
    """One prim's intrinsics, all of them or just the ones asked for."""
    wanted = names if names is not None else list(prim.intrinsicNames())
    row: dict[str, Any] = {}
    for name in wanted:
        try:
            row[name] = _vec_to_list(prim.intrinsicValue(name))
        except Exception:
            row[name] = None
    return row


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _extremes(values: list[Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """min/max/avg over the numeric entries, and the prim each extreme is on."""
    numbers = [v for v in values if _is_number(v)]
    if not numbers:
        return None
    low = min(numbers)
    high = max(numbers)
    return {
        "min": low,
        "max": high,
        "avg": sum(numbers) / len(numbers),
        "min_prim": rows[values.index(low)]["prim_index"],
        "max_prim": rows[values.index(high)]["prim_index"],
    }


def _intrinsic_table(
    geo: hou.Geometry,
    indices: list[int],
    names: list[str] | None,
    total_prims: int,
) -> dict[str, Any]:
    """A table of intrinsics over many prims, with extremes per column.

    Finding the packed prims whose `bounds` ran away used to cost one call
    per prim: 281 calls, about 14 s at 50 ms each.
    """
    rows: list[dict[str, Any]] = []
    for index in indices:
        prim = geo.prim(index)
        row = {"prim_index": index, "prim_type": prim.type().name()}
        row.update(_intrinsic_row(prim, names))
        rows.append(row)

    columns = names if names is not None else sorted({k for r in rows for k in r} - {"prim_index"})
    stats: dict[str, Any] = {}
    for column in columns:
        values = [r.get(column) for r in rows]
        scalar = _extremes(values, rows)
        if scalar is not None:
            stats[column] = scalar
            continue
        # A vector intrinsic (`bounds` is six floats): one min/max over whole
        # lists answers nothing, so each component gets its own extremes and
        # the prim they belong to.
        vectors = [v for v in values if isinstance(v, (list, tuple))]
        if not vectors or len({len(v) for v in vectors}) != 1:
            continue
        components: list[dict[str, Any]] = []
        for component in range(len(vectors[0])):
            column_values = [v[component] if isinstance(v, (list, tuple)) else None for v in values]
            extremes = _extremes(column_values, rows)
            if extremes is not None:
                components.append({"index": component, **extremes})
        if components:
            stats[column] = {"components": components}
    # The stats cover every row; the rows returned are the first
    # _INTRINSIC_ROWS_SHOWN plus the prims holding an extreme. 2000 full rows
    # were up to 922 KB, while the question is almost always "which prims
    # are the outliers", which the stats answer.
    extreme = set()
    for entry in stats.values():
        for item in [entry, *entry.get("components", [])]:
            extreme.update(item[key] for key in ("min_prim", "max_prim") if key in item)
    shown = [
        r for i, r in enumerate(rows) if i < _INTRINSIC_ROWS_SHOWN or r["prim_index"] in extreme
    ]
    result = {
        "total_prims": total_prims,
        "prim_count": len(rows),
        "intrinsics": names,
        "prims": shown,
        "stats": stats,
    }
    if len(shown) < len(rows):
        result["rows_shown"] = len(shown)
        result["note"] = (
            f"Stats cover all {len(rows)} prims; rows shown are the first "
            f"{_INTRINSIC_ROWS_SHOWN} and the prims holding an extreme. Ask for "
            f"prim_indices to read specific rows."
        )
    return result


_INTRINSIC_ROWS_SHOWN = 100


#: Rows one batched call returns. Beyond it the caller gets a window and is
#: told there is more, rather than a reply nobody can read.
_INTRINSIC_ROW_CAP = 2000


def _get_prim_intrinsics(
    *,
    node_path: str,
    prim_index: int | None = None,
    prim_indices: list[int] | None = None,
    prim_range: list[int] | None = None,
    intrinsics: list[str] | None = None,
) -> dict[str, Any]:
    """Get intrinsic values for primitives.

    If prim_index is None and nothing else narrows the request, return a
    summary across all primitives. With prim_index, return the intrinsics of
    that one primitive.

    prim_indices or prim_range read many prims in one call and answer with a
    table (`prims`) plus `stats`: min/max/avg per intrinsic and the prim each
    extreme belongs to. `intrinsics` narrows the columns; given alone, it
    sweeps every prim for those intrinsics.
    """
    geo = _get_sop_geo(node_path)
    total_prims = geo.intrinsicValue("primitivecount")
    names = [str(n) for n in intrinsics] if intrinsics else None

    indices: list[int] | None = None
    if prim_indices is not None or prim_range is not None:
        if prim_indices is not None and prim_range is not None:
            raise hou.OperationFailed("Pass prim_indices or prim_range, not both.")
        if prim_range is not None:
            if len(prim_range) != 2:
                raise hou.OperationFailed("prim_range must be [start, end] (end inclusive).")
            start, end = int(prim_range[0]), int(prim_range[1])
            # Refuse before building the list: [0, 10**9] would allocate
            # a billion ints on Houdini's main thread just to reject them.
            if start < 0 or end >= total_prims or start > end:
                raise hou.OperationFailed(
                    f"prim_range {[start, end]} out of range "
                    f"(0..{total_prims - 1}, start <= end) on {node_path}"
                )
            indices = list(range(start, end + 1))
        else:
            indices = [int(i) for i in prim_indices]
        out_of_range = [i for i in indices if i < 0 or i >= total_prims]
        if out_of_range:
            raise hou.OperationFailed(
                f"Prim index/indices {out_of_range[:10]} out of range "
                f"(0..{total_prims - 1}) on {node_path}"
            )
    elif names and prim_index is None:
        # One or two intrinsics across every prim.
        indices = list(range(total_prims))

    if indices is not None:
        table = _intrinsic_table(geo, indices[:_INTRINSIC_ROW_CAP], names, total_prims)
        table["node_path"] = node_path
        if len(indices) > _INTRINSIC_ROW_CAP:
            table["truncated"] = True
            table["requested_count"] = len(indices)
        return table

    if prim_index is not None:
        if prim_index < 0 or prim_index >= total_prims:
            raise hou.OperationFailed(
                f"Prim index {prim_index} out of range (0..{total_prims - 1}) on {node_path}"
            )
        prim = geo.prim(prim_index)
        return {
            "node_path": node_path,
            "prim_index": prim_index,
            "prim_type": prim.type().name(),
            "intrinsics": _intrinsic_row(prim, names),
        }

    # Summary mode: aggregate intrinsics across (a sample of) the prims.
    # Reading every intrinsic of every prim is O(names * prims) HOM calls
    # — 14+ seconds on a 250k-prim mesh — so cap the statistics sample.
    if total_prims == 0:
        return {
            "node_path": node_path,
            "total_prims": 0,
            "summary": {},
        }

    _SUMMARY_SAMPLE_LIMIT = 1_000
    if total_prims <= _SUMMARY_SAMPLE_LIMIT:
        sample_prims = [geo.prim(i) for i in range(total_prims)]
        sample_note = None
    else:
        step = total_prims / _SUMMARY_SAMPLE_LIMIT
        sample_prims = [geo.prim(int(i * step)) for i in range(_SUMMARY_SAMPLE_LIMIT)]
        sample_note = f"statistics sampled from {_SUMMARY_SAMPLE_LIMIT}/{total_prims} prims"

    sample_prim = sample_prims[0]
    intrinsic_names = sample_prim.intrinsicNames()

    summary: dict[str, Any] = {}
    for name in intrinsic_names:
        try:
            first_val = sample_prim.intrinsicValue(name)
            if isinstance(first_val, (int, float)):
                # Compute min/max/avg for numeric intrinsics
                vals = [p.intrinsicValue(name) for p in sample_prims]
                summary[name] = {
                    "min": min(vals),
                    "max": max(vals),
                    "avg": sum(vals) / len(vals),
                    "sample": first_val,
                }
            elif isinstance(first_val, str):
                # Collect unique string values
                unique = set()
                for p in sample_prims:
                    unique.add(p.intrinsicValue(name))
                    if len(unique) > 20:
                        break
                summary[name] = {
                    "unique_values": sorted(unique)[:20],
                    "sample": first_val,
                }
            else:
                summary[name] = {
                    "sample": _vec_to_list(first_val),
                }
        except Exception:
            summary[name] = {"sample": None, "error": "Could not read"}

    result = {
        "node_path": node_path,
        "total_prims": total_prims,
        "intrinsic_names": list(intrinsic_names),
        "summary": summary,
    }
    if sample_note:
        result["summary_note"] = sample_note
    return result


register_handler("geometry.get_prim_intrinsics", _get_prim_intrinsics)


###### geometry.find_nearest_point


def _find_nearest_point(
    *,
    node_path: str,
    position: list[float],
    max_results: int = 1,
) -> dict[str, Any]:
    """Find nearest point(s) to a given position."""
    geo = _get_sop_geo(node_path)

    if len(position) != 3:
        raise ValueError("position must be a list of 3 floats [x, y, z]")

    total = geo.intrinsicValue("pointcount")
    if total == 0:
        return {
            "node_path": node_path,
            "position": position,
            "results": [],
        }

    k = max(max_results, 1)
    flat = geo.pointFloatAttribValues("P")

    try:
        import numpy as np

        coords = np.array(flat, dtype=np.float64).reshape(-1, 3)
        squared = ((coords - np.array(position)) ** 2).sum(axis=1)
        if k >= len(squared):
            top_indices = np.argsort(squared)
        else:
            top_indices = np.argpartition(squared, k)[:k]
            top_indices = top_indices[np.argsort(squared[top_indices])]
        top = [(float(squared[i]) ** 0.5, int(i)) for i in top_indices[:k]]
    except ImportError:
        px, py, pz = position
        distances: list[tuple[float, int]] = []
        for i in range(total):
            dx = flat[i * 3] - px
            dy = flat[i * 3 + 1] - py
            dz = flat[i * 3 + 2] - pz
            distances.append((dx * dx + dy * dy + dz * dz, i))
        distances.sort(key=lambda x: x[0])
        top = [(d**0.5, i) for d, i in distances[:k]]

    results: list[dict[str, Any]] = []
    for dist, idx in top:
        results.append(
            {
                "index": idx,
                "position": list(flat[idx * 3 : idx * 3 + 3]),
                "distance": dist,
            }
        )

    return {
        "node_path": node_path,
        "query_position": position,
        "max_results": max_results,
        "results": results,
    }


register_handler("geometry.find_nearest_point", _find_nearest_point)


###### geometry.get_attrib_stats


_STATS_ATTRIB_CAP = 12


def _numeric_components(values: Any) -> list[float]:
    """Flatten a scalar, vector or tuple attribute value to floats."""
    if isinstance(values, (int, float)):
        return [float(values)]
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return []


def _element_count(geo: hou.Geometry, cls: str, element_getter: Any) -> int:
    """Points, prims or vertices in *geo*, from intrinsics when they answer."""
    intrinsic = {"point": "pointcount", "prim": "primitivecount", "vertex": "vertexcount"}[cls]
    with contextlib.suppress(Exception):
        count = geo.intrinsicValue(intrinsic)
        if isinstance(count, int):
            return count
    return len(element_getter())


def _get_attrib_stats(
    *,
    node_path: str,
    attribs: list[str] | str | None = None,
    attrib_class: str = "point",
) -> dict[str, Any]:
    """Aggregate statistics for numeric attributes: min, max, mean, sum.

    get_geometry_info names the attributes and get_attrib_values returns every
    value, which on a 62k-point cache is unusable in a conversation. Proving a
    simulation is doing something needs the aggregate, not the values, and that
    is otherwise only reachable through execute_python.
    """
    geo = _get_sop_geo(node_path)

    cls = attrib_class.lower()
    listers = {
        "point": (geo.pointAttribs, geo.points),
        "prim": (geo.primAttribs, geo.prims),
        # uv and N usually live on vertices, and the UV range is what decides a
        # texture's tiling multiplier.
        "vertex": (geo.vertexAttribs, lambda: [v for prim in geo.prims() for v in prim.vertices()]),
        "detail": (geo.globalAttribs, None),
        "global": (geo.globalAttribs, None),
    }
    if cls not in listers:
        raise ValueError(f"Invalid attrib_class: {attrib_class!r}")
    lister, element_getter = listers[cls]

    if isinstance(attribs, str):
        attribs = [attribs]
    available = {a.name(): a for a in lister()}
    wanted = attribs or sorted(available)

    missing = [name for name in wanted if name not in available]
    wanted = [name for name in wanted if name in available][:_STATS_ATTRIB_CAP]

    if cls in ("detail", "global"):
        # A detail attribute is a single value, so "statistics" is the value.
        stats = {name: {"value": _vec_to_list(geo.attribValue(available[name]))} for name in wanted}
        return {
            "node_path": node_path,
            "attrib_class": attrib_class,
            "element_count": 1,
            "stats": stats,
            "missing": missing,
        }

    if element_getter is None:
        raise ValueError(
            f"attrib_class {attrib_class!r} has no per-element statistics; "
            "use point, prim, vertex or detail"
        )

    elements = None  # built only when a fast path is missing: vertices are costly
    stats: dict[str, Any] = {}
    for name in wanted:
        attrib = available[name]
        if attrib.isArrayType() or attrib.dataType() == hou.attribData.String:
            stats[name] = {"skipped": "not numeric"}
            continue
        size = attrib.size()
        kind = "Int" if attrib.dataType() == hou.attribData.Int else "Float"
        fast = _numpy_stats(geo, cls, kind, name, size)
        if fast is not None:
            stats[name] = fast
            continue
        # attribValues() is a single C++ call for the whole array, where a Python
        # loop over 62k elements would be thousands of times slower.
        try:
            flat = list(getattr(geo, f"{cls}{kind}AttribValues")(name))
        except (AttributeError, hou.OperationFailed):
            flat = None
        if flat is None:
            if elements is None:
                elements = element_getter()
            flat = []
            for element in elements:
                flat.extend(_numeric_components(element.attribValue(attrib)))
        if not flat:
            stats[name] = {"count": 0}
            continue
        entry: dict[str, Any] = {
            "count": len(flat) // max(size, 1),
            "size": size,
            "min": min(flat),
            "max": max(flat),
            "sum": sum(flat),
            "mean": sum(flat) / len(flat),
        }
        if size > 1:
            # Per-component ranges, because a velocity field's interesting number
            # is usually the per-axis extreme rather than the flattened one.
            entry["per_component"] = [
                {
                    "min": min(flat[i::size]),
                    "max": max(flat[i::size]),
                    "mean": sum(flat[i::size]) / len(flat[i::size]),
                }
                for i in range(size)
            ]
        stats[name] = entry

    return {
        "node_path": node_path,
        "attrib_class": attrib_class,
        "element_count": _element_count(geo, cls, element_getter),
        "stats": stats,
        "missing": missing,
        "truncated": bool(attribs is None and len(available) > _STATS_ATTRIB_CAP),
    }


def _numpy_stats(geo, cls: str, kind: str, name: str, size: int) -> dict[str, Any] | None:
    """min/max/sum/mean (and per component) from the raw buffer, or None.

    The list-of-floats path built one Python float per component and walked
    it several times: 645 ms for P on a 1M-point grid, paid again on every
    frame of cook_frame_range. The buffer goes straight into numpy instead.
    """
    try:
        import numpy as np

        if kind == "Int":
            raw = getattr(geo, f"{cls}IntAttribValuesAsString")(name, hou.numericData.Int32)
            values = np.frombuffer(raw, dtype=np.int32)
        else:
            raw = getattr(geo, f"{cls}FloatAttribValuesAsString")(name, hou.numericData.Float32)
            values = np.frombuffer(raw, dtype=np.float32)
    except Exception:
        return None
    if values.size == 0:
        return {"count": 0}
    size = max(size, 1)
    table = values.reshape(-1, size).astype(np.float64)
    cast = int if kind == "Int" else float
    entry: dict[str, Any] = {
        "count": int(table.shape[0]),
        "size": size,
        "min": cast(table.min()),
        "max": cast(table.max()),
        "sum": float(table.sum()),
        "mean": float(table.mean()),
    }
    if size > 1:
        # Per-component ranges, because a velocity field's interesting number
        # is usually the per-axis extreme rather than the flattened one.
        entry["per_component"] = [
            {"min": cast(lo), "max": cast(hi), "mean": float(mean)}
            for lo, hi, mean in zip(table.min(0), table.max(0), table.mean(0), strict=False)
        ]
    return entry


register_handler("geometry.get_attrib_stats", _get_attrib_stats)


###### geometry.get_volume_info


# Volume statistics come from intrinsics, which is the only route that works for
# both types: hou.VDB has activeVoxelCount() but no minValue()/maxValue(), and
# hou.Volume has neither. The intrinsics below are present on both.
#
# NEVER read the "voxeldata" intrinsic here: it returns every voxel value, so on
# a real sim field it is millions of floats through the bridge. The whole point
# of this handler is to answer the question without moving the data.
_VOLUME_INTRINSICS = (
    ("min_value", "volumeminvalue"),
    ("max_value", "volumemaxvalue"),
    ("mean_value", "volumeavgvalue"),
    ("active_voxels", "activevoxelcount"),
    ("voxel_size", "voxelsize"),
)


def _volume_entry(prim: Any) -> dict[str, Any]:
    """Name, resolution, voxel counts and value range for one volume prim."""
    entry: dict[str, Any] = {"prim_number": prim.number(), "kind": type(prim).__name__}
    with contextlib.suppress(Exception):
        entry["name"] = prim.attribValue("name")
    with contextlib.suppress(Exception):
        entry["resolution"] = list(prim.resolution())

    available = set()
    with contextlib.suppress(Exception):
        available = set(prim.intrinsicNames())
    for key, intrinsic in _VOLUME_INTRINSICS:
        if intrinsic not in available:
            continue
        with contextlib.suppress(Exception):
            value = prim.intrinsicValue(intrinsic)
            entry[key] = list(value) if isinstance(value, tuple) else value

    # A dense volume has no active-voxel concept, so report the grid total under
    # its own name rather than pretending the two numbers mean the same thing.
    if "active_voxels" not in entry and entry.get("resolution"):
        res = entry["resolution"]
        entry["total_voxels"] = res[0] * res[1] * res[2]

    with contextlib.suppress(Exception):
        bbox = prim.boundingBox()
        entry["bbox_min"] = list(bbox.minvec())
        entry["bbox_max"] = list(bbox.maxvec())
    return entry


def _get_volume_info(*, node_path: str, max_volumes: int = 24) -> dict[str, Any]:
    """Per-volume names, resolution, active voxels and value ranges.

    get_geometry_info reports a primitive count, which cannot distinguish a
    correctly named non-empty density field from an empty one. get_cop_vdb
    covers Copernicus; this is the SOP side of the same question.
    """
    geo = _get_sop_geo(node_path)
    # primsOfType, not a walk over geo.prims(): that built a Python object per
    # prim to find none, 983 ms on a 1M-prim mesh, and cook_frame_range asked
    # on every frame.
    try:
        volumes = sorted(
            list(geo.primsOfType(hou.primType.Volume)) + list(geo.primsOfType(hou.primType.VDB)),
            key=lambda prim: prim.number(),
        )
    except Exception:
        volumes = [
            prim
            for prim in geo.prims()
            if isinstance(prim, (hou.Volume, hou.VDB)) or type(prim).__name__ in ("Volume", "VDB")
        ]
    shown = volumes[:max_volumes]
    return {
        "node_path": node_path,
        "volume_count": len(volumes),
        "volumes": [_volume_entry(prim) for prim in shown],
        "truncated": len(volumes) > len(shown),
    }


register_handler("geometry.get_volume_info", _get_volume_info)
