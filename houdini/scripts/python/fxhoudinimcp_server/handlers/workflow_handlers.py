"""Higher-level composite workflow handlers for FXHoudini-MCP.

Provides tools that build entire node graphs in a single call --
complete simulation setups, material creation/assignment, SOP chain
building, and render configuration.
"""

from __future__ import annotations

# Built-in
from typing import Any

# Third-party
import hou

# Internal
from fxhoudinimcp_server.config import layout_if_enabled, place_new_node
from fxhoudinimcp_server.dispatcher import register_handler
from fxhoudinimcp_server.errors import readable_message
from fxhoudinimcp_server.handlers.node_handlers import _refuse_taken_name
from fxhoudinimcp_server.ui import focus_network_editor

###### Helpers


def _get_node(node_path: str) -> hou.Node:
    """Resolve a node path and raise a clear error if it does not exist."""
    node = hou.node(node_path)
    if node is None:
        raise ValueError(f"Node not found: {node_path}")
    return node


def _focus_network_editor(node: hou.Node, place_unpositioned: bool = True) -> None:
    """Best-effort: lay out, pan the editor to *node*, hide the other objects.

    The viewer follows into the new object; the source object and the rest of
    the scene would otherwise draw over the sim.
    """
    focus_network_editor(node, place_unpositioned, other_objects="hide")


def _ensure_obj_context() -> hou.Node:
    """Return the /obj context node."""
    obj = hou.node("/obj")
    if obj is None:
        raise ValueError("Cannot find /obj context")
    return obj


def _ensure_mat_context() -> hou.Node:
    """Return the /mat context node, creating it if necessary."""
    mat = hou.node("/mat")
    if mat is None:
        mat = hou.node("/").createNode("matnet", "mat")
        print("[workflow] Created /mat context")
    return mat


def _ensure_out_context() -> hou.Node:
    """Return the /out context node."""
    out = hou.node("/out")
    if out is None:
        raise ValueError("Cannot find /out context")
    return out


def _set_parm_safe(node: hou.Node, parm_name: str, value: Any) -> bool:
    """Set a parameter value if the parameter exists. Returns True on success."""
    parm = node.parm(parm_name)
    if parm is not None:
        try:
            parm.set(value)
            return True
        except Exception as e:
            print(
                f"[workflow] Warning: could not set {parm_name}={value} on {node.path()}: {readable_message(e)}"
            )
            return False
    return False


def _set_values(node: hou.Node, values: dict) -> list[str]:
    """Set parms or whole parm tuples; return the names that did not take."""
    missed = []
    for name, value in values.items():
        try:
            if isinstance(value, (list, tuple)):
                node.parmTuple(name).set(value)
            else:
                node.parm(name).set(value)
        except Exception:
            missed.append(name)
    return missed


def _create_first_available(
    parent: hou.Node, type_names: tuple[str, ...], node_name: str
) -> hou.Node:
    """Create the first of *type_names* that exists under *parent*.

    Lets a chain be written preferred-type-first without nesting try/except per
    candidate. Raises hou.OperationFailed naming the whole chain if none exist,
    which is more useful than the last candidate's own error.
    """
    for type_name in type_names:
        try:
            return parent.createNode(type_name, node_name)
        except hou.OperationFailed:
            continue
    raise hou.OperationFailed(
        f"None of these node types exist in {parent.path()}: {', '.join(type_names)}"
    )


def _stream_cache(
    geo: hou.Node, type_name: str, node_name: str, solver: hou.Node, streams: int
) -> hou.Node:
    """A solver's own I/O node, wired output-for-input, displayed and rendered.

    rbdio and vellumio carry every stream of the sim (geometry, constraints,
    proxy or collision geometry). A filecache on the first output silently
    drops the rest, which is what every setup_* used to put down. write_cache
    drives them like a filecache: they share its Save to Disk button.
    """
    cache = geo.createNode(type_name, node_name)
    for index in range(streams):
        cache.setInput(index, solver, index)
    place_new_node(cache)
    cache.setDisplayFlag(True)
    cache.setRenderFlag(True)
    return cache


def _source_status(objmerge: hou.Node, source_geo: str, what: str) -> dict[str, Any]:
    """Whether the source geometry a sim was pointed at actually exists.

    Building the network with a placeholder source is legitimate -- the path is a
    parameter, fixable later -- but reporting success: True and nothing else means a
    caller walks away with a sim that will never produce anything, and no reason to
    look. setup_pyro_sim said so; setup_flip_sim, setup_rbd_sim and
    setup_vellum_sim did not, which is the kind of inconsistency that makes a
    server's answers untrustworthy in aggregate.

    Args:
        objmerge: The Object Merge SOP holding the reference.
        source_geo: The path it was pointed at.
        what: Sim name for the description ("FLIP", "RBD", "Vellum").
    """
    found = hou.node(source_geo) is not None
    return {
        "objmerge_path": objmerge.path(),
        "source_geo": source_geo,
        "source_geo_found": found,
        "network_description": (
            f"The {what} network is wired. Source geometry is referenced via the "
            f"Object Merge SOP at {objmerge.path()} (parameter 'objpath1' = "
            f"'{source_geo}'). "
            + (
                "The source geometry was found and connected successfully."
                if found
                else f"WARNING: source geometry '{source_geo}' was NOT found, so this "
                f"sim will simulate nothing until the 'objpath1' parameter on "
                f"{objmerge.path()} points at a real SOP."
            )
        ),
    }


###### workflow.setup_pyro_sim


def _setup_pyro_sim_sop(
    geo: hou.Node,
    objmerge: hou.Node,
    substeps: int,
    all_nodes: list[str],
    res_scale: float = 1.0,
) -> dict:
    """Build a SOP-level Pyro simulation (Houdini 20+).

    Uses the modern pyrosource + pyrosolver SOP workflow.

    Args:
        geo: Parent geometry node.
        objmerge: Object Merge SOP referencing the source geometry.
        substeps: Number of solver substeps.
        all_nodes: Accumulator list for created node paths.
    """
    # -- Pyro Source: converts geometry to source volumes
    print("[workflow] Creating Pyro Source SOP")
    pyrosource = geo.createNode("pyrosource", "pyro_source1")
    pyrosource.setInput(0, objmerge, 0)
    all_nodes.append(pyrosource.path())
    # A fresh Pyro Source emits bare points (P, pscale): the solver found no
    # density, temperature or burn and simulated nothing for as long as it ran.
    # Each Initialize preset appends its attributes when its callback runs
    # (set() alone runs nothing): Smoke adds density + temperature, Burn adds
    # burn, which together are what the Pyro Solver SOP sources by default.
    initialize = pyrosource.parm("initialize")
    for preset in ("source", "sourceburn"):
        initialize.set(preset)
        initialize.pressButton()
    names = [
        pyrosource.parm(f"name{i}").eval()
        for i in range(1, pyrosource.parm("attributes").eval() + 1)
    ]

    # -- The solver sources named volumes, not points (nodes/sop/pyrosolver).
    print("[workflow] Creating Volume Rasterize Attributes SOP")
    rasterize = geo.createNode("volumerasterizeattributes", "rasterize_source1")
    rasterize.setInput(0, pyrosource, 0)
    rasterize.parm("attributes").set(" ".join(dict.fromkeys(names)))
    all_nodes.append(rasterize.path())

    # -- Pyro Solver SOP (SOP-level, Houdini 20+)
    print("[workflow] Creating Pyro Solver SOP")
    try:
        pyrosolver = geo.createNode("pyrosolver::3.0", "pyro_solver1")
    except hou.OperationFailed:
        pyrosolver = geo.createNode("pyrosolver", "pyro_solver1")
    pyrosolver.setInput(0, rasterize, 0)
    all_nodes.append(pyrosolver.path())

    # Set substeps
    _set_parm_safe(pyrosolver, "substeps", substeps)
    # res_scale was accepted and never used. 2.0 means twice the resolution:
    # half the voxel size.
    if res_scale <= 0:
        raise ValueError(f"res_scale must be positive, got {res_scale}")
    divsize = pyrosolver.parm("divsize")
    if divsize is not None and res_scale != 1.0:
        divsize.set(divsize.eval() / res_scale)

    # -- File Cache
    print("[workflow] Creating File Cache SOP")
    try:
        filecache = geo.createNode("filecache::2.0", "file_cache1")
    except hou.OperationFailed:
        filecache = geo.createNode("filecache", "file_cache1")
    filecache.setInput(0, pyrosolver, 0)
    all_nodes.append(filecache.path())

    try:
        filecache.setDisplayFlag(True)
        filecache.setRenderFlag(True)
    except Exception:
        pass

    layout_if_enabled(geo)
    _focus_network_editor(filecache)
    print("[workflow] SOP-level Pyro setup complete")

    return {
        "success": True,
        "geo_path": geo.path(),
        "solver_path": pyrosolver.path(),
        "source_path": pyrosource.path(),
        "cache_path": filecache.path(),
        "approach": "sop",
        "all_nodes": all_nodes,
    }


def _setup_pyro_sim_dop(
    geo: hou.Node,
    objmerge: hou.Node,
    substeps: int,
    all_nodes: list[str],
) -> dict:
    """Build a DOP-level Pyro simulation (fallback for older Houdini).

    Uses a DOP Network with smokeobject_sparse, volumesource and
    pyrosolver_sparse, merged into the solver's first input.

    Args:
        geo: Parent geometry node.
        objmerge: Object Merge SOP referencing the source geometry.
        substeps: Number of DOP substeps.
        all_nodes: Accumulator list for created node paths.
    """
    # -- DOP Network
    print("[workflow] Creating DOP Network inside geo")
    dopnet = geo.createNode("dopnet", "dopnet1")
    all_nodes.append(dopnet.path())
    _set_parm_safe(dopnet, "substep", substeps)

    # -- Smoke Object
    #
    # smokeobject_sparse first. Houdini 22.0 marks both smokeobject and
    # smokeconfigureobject deprecated (hou.NodeType.deprecated is True for both)
    # and the release notes say they are "scheduled to be deleted in an upcoming
    # revision", naming Smoke Object (Sparse) as the replacement. The sparse node
    # exists in every build we sample, back to 20.5.278, so preferring it costs
    # nothing on older Houdini and keeps this path alive once the old nodes go.
    print("[workflow] Creating smoke object DOP")
    smokeobj = _create_first_available(
        dopnet,
        ("smokeobject_sparse", "smokeobject", "smokeconfigureobject"),
        "smokeobject1",
    )
    all_nodes.append(smokeobj.path())

    # -- Volume Source
    #
    # volumesource, not sourcevolume: Houdini marks sourcevolume deprecated and
    # ships no help page for it, while volumesource is documented as the node
    # that "imports SOP source geometry into smoke, pyro, and FLIP simulations"
    # and is present in every build we sample.
    print("[workflow] Creating volume source DOP")
    source_vol = None
    try:
        source_vol = _create_first_available(
            dopnet, ("volumesource", "sourcevolume"), "source_volume1"
        )
        all_nodes.append(source_vol.path())
        # Point the volume source at the Object Merge SOP
        for parm_name in ("sop_path", "soppath", "geometry"):
            if _set_parm_safe(source_vol, parm_name, objmerge.path()):
                print(f"[workflow] Set volume source {parm_name} = {objmerge.path()}")
                break
    except hou.OperationFailed:
        print("[workflow] Warning: no volume source node available, skipping")

    # -- Pyro Solver
    #
    # Same reasoning as the smoke object: the 22.0 notes list "Pyro Solver DOP"
    # among the deprecated DOP pyro nodes and point at Pyro Solver (Sparse).
    # Houdini's own deprecated flag is still False on pyrosolver, so the old
    # versioned types stay as fallbacks rather than being dropped.
    print("[workflow] Creating pyro solver DOP")
    pyrosolver = _create_first_available(
        dopnet,
        ("pyrosolver_sparse", "pyrosolver::2.0", "pyrosolver"),
        "pyrosolver1",
    )
    all_nodes.append(pyrosolver.path())

    # No resize container. This used to create a gasresizefluiddynamic and wire
    # it to the solver's output, which is not a valid DOP-level connection -- it
    # is a microsolver belonging inside a solver's subnet, so the wiring always
    # failed and the node sat there doing nothing. It is also unnecessary now:
    # SideFX document the sparse smoke object as growing on its own ("the
    # container will grow as the simulated smoke and fire inside it" expands,
    # starting empty and resizing at initial sourcing).

    # -- Merge DOP to combine smoke object and volume source
    print("[workflow] Creating merge DOP and wiring solver chain")
    try:
        merge_dop = dopnet.createNode("merge", "merge1")
        merge_dop.setInput(0, smokeobj, 0)
        if source_vol is not None:
            merge_dop.setInput(1, source_vol, 0)
        pyrosolver.setInput(0, merge_dop, 0)
        all_nodes.append(merge_dop.path())
    except Exception as e:
        print(f"[workflow] Warning: merge DOP failed, wiring directly: {readable_message(e)}")
        pyrosolver.setInput(0, smokeobj, 0)

    # -- DOP Import SOP
    print("[workflow] Creating DOP Import SOP")
    dopimport = geo.createNode("dopimport", "dop_import1")
    _set_parm_safe(dopimport, "doppath", dopnet.path())
    all_nodes.append(dopimport.path())

    # -- File Cache
    print("[workflow] Creating File Cache SOP")
    try:
        filecache = geo.createNode("filecache::2.0", "file_cache1")
    except hou.OperationFailed:
        filecache = geo.createNode("filecache", "file_cache1")
    filecache.setInput(0, dopimport, 0)
    all_nodes.append(filecache.path())

    try:
        filecache.setDisplayFlag(True)
        filecache.setRenderFlag(True)
    except Exception:
        pass

    layout_if_enabled(geo)
    layout_if_enabled(dopnet)
    _focus_network_editor(filecache)
    print("[workflow] DOP-level Pyro setup complete")

    return {
        "success": True,
        "geo_path": geo.path(),
        "dop_path": dopnet.path(),
        "cache_path": filecache.path(),
        "approach": "dop",
        "all_nodes": all_nodes,
    }


def _setup_pyro_sim(
    source_geo: str = "/obj/geo1/sphere1",
    container: str = "box",
    res_scale: float = 1.0,
    substeps: int = 1,
    name: str = "pyro_sim",
    **_: Any,
) -> dict:
    """Build a complete Pyro smoke/fire simulation network.

    Tries the modern SOP-level approach first (Houdini 20+, using
    pyrosource + pyrosolver SOPs).  Falls back to the classic DOP
    approach (smokeobject_sparse + volumesource + pyrosolver_sparse DOPs) for
    older Houdini versions.

    Args:
        source_geo: Path to the source geometry SOP to drive the simulation.
        container: Container shape. Only "box" is implemented; anything else is
            rejected rather than silently ignored.
        res_scale: Resolution scale multiplier for the simulation.
        substeps: Number of DOP substeps or solver substeps.
        name: Name for the top-level geometry node.
    """
    obj = _ensure_obj_context()
    # A parameter that is accepted and ignored is worse than one that is missing:
    # it reads as functional, so asking for something else returns a box and
    # reports success. Only "box" is built, so say so.
    if container != "box":
        raise ValueError(
            f"Unsupported container '{container}'. Only 'box' is implemented; the "
            f"other shapes are not built yet."
        )

    all_nodes: list[str] = []

    # -- Create top-level geo container
    print(f"[workflow] Creating geo node '{name}' under /obj")
    geo = obj.createNode("geo", name)
    for child in geo.children():
        child.destroy()
    all_nodes.append(geo.path())

    # -- Object Merge for source geometry
    print(f"[workflow] Creating Object Merge for source: {source_geo}")
    objmerge = geo.createNode("object_merge", "source_geo")
    _set_parm_safe(objmerge, "objpath1", source_geo)
    all_nodes.append(objmerge.path())

    if hou.node(source_geo) is not None:
        print(f"[workflow] Source geometry found at {source_geo}")
    else:
        print(
            f"[workflow] Warning: source geometry '{source_geo}' not found, Object Merge created but path may need updating"
        )

    # -- Try modern SOP-level Pyro first, fall back to DOP approach
    try:
        print("[workflow] Attempting SOP-level Pyro workflow (Houdini 20+)")
        result = _setup_pyro_sim_sop(geo, objmerge, substeps, all_nodes, res_scale)
        print(f"[workflow] Pyro simulation '{name}' setup complete (SOP approach)")
    except hou.OperationFailed:
        print("[workflow] SOP-level Pyro not available, falling back to DOP approach")
        # Clean up any partially-created SOP nodes (keep geo and objmerge)
        for child in geo.children():
            if child != objmerge:
                try:
                    if child.type().name() not in ("object_merge",):
                        child.destroy()
                except Exception:
                    pass

        result = _setup_pyro_sim_dop(geo, objmerge, substeps, all_nodes)
        if res_scale != 1.0:
            result["warning"] = (
                "res_scale is applied only by the SOP-level Pyro Solver; the DOP fallback ignored it."
            )
        print(f"[workflow] Pyro simulation '{name}' setup complete (DOP approach)")

    # Augment the result with source wiring info so the AI client
    # understands the network and does not try to re-wire things.
    source_found = hou.node(source_geo) is not None
    result["objmerge_path"] = objmerge.path()
    result["source_geo"] = source_geo
    result["source_geo_found"] = source_found
    result["network_description"] = (
        f"The Pyro network is fully wired and ready to simulate. "
        f"Source geometry is referenced via the Object Merge SOP at "
        f"{objmerge.path()} (parameter 'objpath1' = '{source_geo}'). "
        + (
            "The source geometry was found and connected successfully. "
            if source_found
            else f"WARNING: source geometry '{source_geo}' was NOT found. "
            f"Update the 'objpath1' parameter on {objmerge.path()} to "
            f"point to a valid SOP path. "
        )
        + "No additional wiring is needed."
    )
    return result


###### workflow.setup_rbd_sim


def _setup_rbd_sim(
    geo_path: str = "/obj/geo1",
    ground: bool = True,
    pieces_type: str = "voronoi",
    name: str = "rbd_sim",
    **_: Any,
) -> dict:
    """Build a complete RBD (rigid body dynamics) simulation network.

    Creates a geometry node with a DOP Network, RBD solver, optional
    RBD Material Fracture, optional ground plane, and an RBD I/O cache.

    Args:
        geo_path: Path to the source geometry object.
        ground: If True, add a ground plane to the simulation.
        pieces_type: Fracture method -- "voronoi" adds a Voronoi Fracture SOP.
        name: Name for the top-level geometry node.
    """
    obj = _ensure_obj_context()
    all_nodes: list[str] = []

    # -- Step 1: Create geo container
    print(f"[workflow] Creating geo node '{name}' under /obj")
    geo = obj.createNode("geo", name)
    for child in geo.children():
        child.destroy()
    all_nodes.append(geo.path())

    # -- Step 2: Object Merge source
    print(f"[workflow] Creating Object Merge for source: {geo_path}")
    objmerge = geo.createNode("object_merge", "source_geo")
    _set_parm_safe(objmerge, "objpath1", geo_path)
    all_nodes.append(objmerge.path())
    last_sop = objmerge

    # -- Step 3: Optional fracture. rbdmaterialfracture is the modern
    # one-input choice whose outputs (geometry/constraints/proxy) feed the
    # Bullet solver directly. Plain voronoifracture REQUIRES cell points
    # on its second input — wiring it alone leaves an uncookable network.
    # Only "voronoi" ever did anything; every other string silently skipped
    # fracturing and still reported success, so a typo returned an unfractured sim
    # that looks like a working one. Not fracturing is a legitimate request, so it
    # gets a name rather than being what happens when you misspell the other one.
    valid_pieces = ("voronoi", "none")
    if pieces_type not in valid_pieces:
        raise ValueError(
            f"Invalid pieces_type '{pieces_type}'. Must be one of: {valid_pieces}. "
            f"Use 'none' to simulate the geometry unfractured."
        )

    fracture = None
    if pieces_type == "voronoi":
        try:
            print("[workflow] Creating RBD Material Fracture SOP")
            fracture = geo.createNode("rbdmaterialfracture", "fracture1")
            fracture.setInput(0, last_sop, 0)
            all_nodes.append(fracture.path())
            last_sop = fracture
        except hou.OperationFailed:
            print("[workflow] rbdmaterialfracture unavailable, using scatter + voronoifracture")
            scatter = geo.createNode("scatter", "cell_points")
            scatter.setInput(0, last_sop, 0)
            _set_parm_safe(scatter, "npts", 30)
            voronoi = geo.createNode("voronoifracture", "voronoi_fracture1")
            voronoi.setInput(0, last_sop, 0)
            voronoi.setInput(1, scatter, 0)
            all_nodes.extend([scatter.path(), voronoi.path()])
            last_sop = voronoi

    # -- Step 4: Solver. Prefer the SOP-level Bullet solver (Houdini
    # 18.5+): it simulates its input directly, applies gravity, and has a
    # built-in ground plane — no DOP wiring to get wrong.
    solver_path = None
    try:
        solver = geo.createNode("rbdbulletsolver", "rbd_solver1")
    except hou.OperationFailed:
        solver = None

    if solver is not None:
        print("[workflow] Creating SOP-level RBD Bullet Solver")
        solver.setInput(0, last_sop, 0)
        if fracture is not None and fracture.type().name().startswith("rbdmaterialfracture"):
            # Outputs line up 1:1 with the solver's first three inputs.
            solver.setInput(1, fracture, 1)
            solver.setInput(2, fracture, 2)
        _set_parm_safe(solver, "useground", 1 if ground else 0)
        all_nodes.append(solver.path())
        solver_path = solver.path()
        last_sop = solver
    else:
        # Legacy DOP fallback: the packed object must point at the SOP
        # geometry, and object + ground must merge into the solver chain.
        print("[workflow] Creating DOP Network (SOP solver unavailable)")
        dopnet = geo.createNode("dopnet", "dopnet1")
        all_nodes.append(dopnet.path())
        solver_path = dopnet.path()

        rbdobj = dopnet.createNode("rbdpackedobject", "rbdobject1")
        for parm_name in ("soppath", "sop_path"):
            if _set_parm_safe(rbdobj, parm_name, last_sop.path()):
                break
        all_nodes.append(rbdobj.path())

        rbdsolver = dopnet.createNode("rbdsolver", "rbdsolver1")
        all_nodes.append(rbdsolver.path())

        merge_dop = dopnet.createNode("merge", "merge1")
        merge_dop.setInput(0, rbdobj, 0)
        if ground:
            groundplane = dopnet.createNode("groundplane", "groundplane1")
            merge_dop.setInput(1, groundplane, 0)
            all_nodes.append(groundplane.path())
        rbdsolver.setInput(0, merge_dop, 0)
        gravity = dopnet.createNode("gravity", "gravity1")
        gravity.setInput(0, rbdsolver, 0)
        gravity.setDisplayFlag(True)
        all_nodes.append(gravity.path())

        # The dopnet's own children need their own call: geo.layoutChildren()
        # does not descend, and the origin sweep only walks direct children.
        layout_if_enabled(dopnet)

        dopimport = geo.createNode("dopimport", "dop_import1")
        _set_parm_safe(dopimport, "doppath", dopnet.path())
        # No setInput: "dopimport" resolves to dopimport::2.0, which has no
        # inputs at all and reads the sim straight out of doppath. Only the
        # version 1 node took one, so wiring it raised hou.InvalidInput and
        # killed this branch outright. The pyro and flip paths above already
        # treat the DOP Import as the head of its own chain.
        all_nodes.append(dopimport.path())
        last_sop = dopimport

    # -- Step 5: Cache. RBD I/O behind the Bullet solver, so constraints and
    # proxy geometry are cached with the pieces; the DOP fallback has one
    # stream and keeps a File Cache.
    if solver is not None:
        print("[workflow] Creating RBD I/O")
        filecache = _stream_cache(geo, "rbdio", "rbd_io1", solver, 4)
    else:
        print("[workflow] Creating File Cache SOP")
        filecache = _create_first_available(geo, ("filecache", "filecache::2.0"), "file_cache1")
        filecache.setInput(0, last_sop, 0)
        filecache.setDisplayFlag(True)
        filecache.setRenderFlag(True)
    all_nodes.append(filecache.path())

    # -- Step 6: Layout
    print("[workflow] Laying out nodes")
    layout_if_enabled(geo)
    _focus_network_editor(filecache)

    print(f"[workflow] RBD simulation '{name}' setup complete")

    return {
        "success": True,
        "geo_path": geo.path(),
        "solver_path": solver_path,
        "cache_path": filecache.path(),
        "all_nodes": all_nodes,
        **_source_status(objmerge, geo_path, "RBD"),
    }


###### workflow.setup_flip_sim


def _setup_flip_sim(
    source_geo: str = "/obj/geo1/sphere1",
    domain: str = "box",
    particle_sep: float = 0.05,
    name: str = "flip_sim",
    **_: Any,
) -> dict:
    """Build a complete FLIP fluid simulation network.

    Creates a geometry node with the SOP FLIP chain (FLIP Container, FLIP
    Boundary sourcing from the Object Merge, FLIP Solver on a ground plane)
    and a File Cache.

    Args:
        source_geo: Path to the source geometry SOP.
        domain: Domain shape. Only "box" is implemented; anything else is
            rejected rather than silently ignored.
        particle_sep: Particle separation distance for the FLIP sim.
        name: Name for the top-level geometry node.
    """
    obj = _ensure_obj_context()
    # A parameter that is accepted and ignored is worse than one that is missing:
    # it reads as functional, so asking for something else returns a box and
    # reports success. Only "box" is built, so say so.
    if domain != "box":
        raise ValueError(
            f"Unsupported domain '{domain}'. Only 'box' is implemented; the "
            f"other shapes are not built yet."
        )

    all_nodes: list[str] = []

    # -- Step 1: Create geo container
    print(f"[workflow] Creating geo node '{name}' under /obj")
    geo = obj.createNode("geo", name)
    for child in geo.children():
        child.destroy()
    all_nodes.append(geo.path())

    # -- Step 2: Object Merge for source
    print(f"[workflow] Creating Object Merge for source: {source_geo}")
    objmerge = geo.createNode("object_merge", "source_geo")
    _set_parm_safe(objmerge, "objpath1", source_geo)
    all_nodes.append(objmerge.path())

    if hou.node(source_geo) is not None:
        print(f"[workflow] Source geometry found at {source_geo}")
    else:
        print(
            f"[workflow] Warning: source geometry '{source_geo}' not found -- Object Merge created but path may need updating"
        )

    # -- Step 3: SOP FLIP, as fluid/sopminimalsetup builds it: FLIP Container,
    # then FLIP Boundary with the source on its 4th input, then FLIP Solver,
    # each passing its three streams (particles, container, collisions) on.
    # The DOP network this replaced was never wired: its source pointed at no
    # geometry and fed no solver, so every sim it built was empty.
    print("[workflow] Creating FLIP Container, Boundary and Solver SOPs")
    container = geo.createNode("flipcontainer", "flip_container1")
    _set_parm_safe(container, "particlesep", particle_sep)
    # The container deletes particles that reach its walls, so it is sized
    # around the source and stands on the ground plane the solver adds at y=0.
    # ponytail: a source below y=0 still lands on y=0; move ground_posy if needed.
    bbox = objmerge.geometry().boundingBox() if objmerge.geometry() else None
    if bbox is not None and bbox.isValid():
        width = max(4.0, 3.0 * max(bbox.sizevec()[0], bbox.sizevec()[2]))
        height = max(4.0, bbox.maxvec()[1] + 1.5)
        center = bbox.center()
        container.parmTuple("size").set((width, height, width))
        container.parmTuple("t").set((center[0], height / 2.0 - 0.5, center[2]))
    all_nodes.append(container.path())

    source = geo.createNode("flipboundary", "flip_source1")
    for index in range(3):
        source.setInput(index, container, index)
    source.setInput(3, objmerge, 0)
    all_nodes.append(source.path())

    solver = geo.createNode("flipsolver", "flip_solver1")
    for index in range(3):
        solver.setInput(index, source, index)
    _set_parm_safe(solver, "particlesep", particle_sep)
    _set_parm_safe(solver, "useground", "ground")
    all_nodes.append(solver.path())

    # -- Step 4: File Cache on the particles, the stream surfacing reads.
    print("[workflow] Creating File Cache SOP")
    filecache = _create_first_available(geo, ("filecache::2.0", "filecache"), "file_cache1")
    filecache.setInput(0, solver, 0)
    filecache.setDisplayFlag(True)
    filecache.setRenderFlag(True)
    all_nodes.append(filecache.path())

    layout_if_enabled(geo)
    _focus_network_editor(filecache)
    print(f"[workflow] FLIP simulation '{name}' setup complete")

    return {
        "success": True,
        "geo_path": geo.path(),
        "solver_path": solver.path(),
        "container_path": container.path(),
        "source_path": source.path(),
        "cache_path": filecache.path(),
        "all_nodes": all_nodes,
        **_source_status(objmerge, source_geo, "FLIP"),
    }


###### workflow.setup_vellum_sim


def _setup_vellum_sim(
    geo_path: str = "/obj/geo1",
    sim_type: str = "cloth",
    substeps: int = 5,
    name: str = "vellum_sim",
    ground: bool = True,
    **_: Any,
) -> dict:
    """Build a complete Vellum simulation network.

    Creates a geometry node with Vellum Configure (cloth/hair/grain/softbody),
    Vellum Solver, Object Merge source, and a Vellum I/O cache.

    Args:
        geo_path: Path to the source geometry object.
        sim_type: Simulation type -- "cloth", "hair", "grain", or "softbody".
        substeps: Number of solver substeps.
        name: Name for the top-level geometry node.
        ground: Turn on the solver's ground plane. Without one a draped cloth
            falls through y=0 forever, as setup_rbd_sim's pieces used not to.
    """
    obj = _ensure_obj_context()
    all_nodes: list[str] = []

    valid_types = ("cloth", "hair", "grain", "softbody")
    if sim_type not in valid_types:
        raise ValueError(f"Invalid sim_type '{sim_type}'. Must be one of: {valid_types}")

    # What the Vellum shelf's Configure tools put down. vellumhair, vellumgrain
    # and vellumsoftbody are not node types, so three of the four used to fall
    # back to a bare vellumconstraints, and "cloth" built a Vellum Drape, which
    # runs a draping sim of its own. A strut softbody is a cloth shell plus
    # struts (shelf/vellumsoftbody); grain has its own node.
    configure_chain = {
        "cloth": [("vellumconstraints", "cloth")],
        "hair": [("vellumconstraints", "hair")],
        "grain": [("vellumconstraints_grain", None)],
        "softbody": [("vellumconstraints", "cloth"), ("vellumconstraints", "struts")],
    }

    # -- Step 1: Create geo container
    print(f"[workflow] Creating geo node '{name}' under /obj")
    geo = obj.createNode("geo", name)
    for child in geo.children():
        child.destroy()
    all_nodes.append(geo.path())

    # -- Step 2: Object Merge source
    print(f"[workflow] Creating Object Merge for source: {geo_path}")
    objmerge = geo.createNode("object_merge", "source_geo")
    _set_parm_safe(objmerge, "objpath1", geo_path)
    all_nodes.append(objmerge.path())

    if hou.node(geo_path) is not None:
        print(f"[workflow] Source geometry found at {geo_path}")
    else:
        print(
            f"[workflow] Warning: source geometry '{geo_path}' not found -- Object Merge created but path may need updating"
        )

    # -- Step 3: Vellum Configure, chained geometry-to-geometry and
    # constraints-to-constraints the way the shelf chains them.
    print(f"[workflow] Creating Vellum Configure ({sim_type})")
    vellum_configure = None
    for index, (type_name, constraint) in enumerate(configure_chain[sim_type], start=1):
        node = geo.createNode(type_name, f"vellum_{sim_type}{index}")
        if constraint is not None:
            node.parm("constrainttype").set(constraint)
        node.setInput(0, vellum_configure or objmerge, 0)
        if vellum_configure is not None:
            node.setInput(1, vellum_configure, 1)
        all_nodes.append(node.path())
        vellum_configure = node

    # -- Step 4: Vellum Solver
    print("[workflow] Creating Vellum Solver SOP")
    try:
        vellum_solver = geo.createNode("vellumsolver", "vellum_solver1")
    except hou.OperationFailed:
        vellum_solver = geo.createNode("vellumsolver::2.0", "vellum_solver1")
    # Connect geometry output and constraints output
    vellum_solver.setInput(0, vellum_configure, 0)
    try:
        vellum_solver.setInput(1, vellum_configure, 1)
    except Exception:
        print("[workflow] Warning: could not connect constraints output to solver input 1")
    all_nodes.append(vellum_solver.path())

    # Set substeps
    _set_parm_safe(vellum_solver, "substeps", substeps)
    _set_parm_safe(vellum_solver, "useground", 1 if ground else 0)

    # -- Step 5: Vellum I/O, not a File Cache: a filecache on the first output
    # drops the constraints and collisions, and vellumpostprocess downstream
    # of the cache needs them.
    print("[workflow] Creating Vellum I/O")
    filecache = _stream_cache(geo, "vellumio", "vellum_io1", vellum_solver, 3)
    all_nodes.append(filecache.path())

    # -- Step 6: Layout
    print("[workflow] Laying out nodes")
    layout_if_enabled(geo)
    _focus_network_editor(filecache)

    print(f"[workflow] Vellum simulation '{name}' ({sim_type}) setup complete")

    return {
        "success": True,
        "geo_path": geo.path(),
        "solver_path": vellum_solver.path(),
        "configure_path": vellum_configure.path(),
        "cache_path": filecache.path(),
        "sim_type": sim_type,
        "all_nodes": all_nodes,
        **_source_status(objmerge, geo_path, "Vellum"),
    }


###### workflow.create_material


def _create_material(
    name: str | None = None,
    mat_type: str = "principled",
    base_color: list = None,
    roughness: float = 0.5,
    metallic: float = 0.0,
    opacity: float = 1.0,
    **_: Any,
) -> dict:
    """Create a material/shader in the /mat context.

    Supports principled shader and MaterialX standard surface.

    Args:
        name: Name for the material subnet/node.
        mat_type: Material type -- "principled" or "materialx".
        base_color: Optional [R, G, B] base color (0.0-1.0 each).
        roughness: Surface roughness (0.0 = mirror, 1.0 = diffuse).
        metallic: Metallic factor (0.0 = dielectric, 1.0 = metal).
        opacity: Opacity (0.0 = transparent, 1.0 = opaque).
    """
    mat = _ensure_mat_context()
    # A name the caller chose is theirs or an error; only the default numbers up.
    if name:
        _refuse_taken_name(mat, name)
    else:
        name = "material1"
    color = list(base_color[:3]) if base_color is not None and len(base_color) >= 3 else None

    if mat_type == "principled":
        shader = _create_first_available(mat, ("principledshader::2.0", "principledshader"), name)
        values = {"rough": roughness, "metallic": metallic, "opac": opacity}
        if color:
            values["basecolor"] = color
    elif mat_type == "materialx":
        # The bare-subnet fallback this used to take returned an empty network
        # as type "materialx" with success; a missing MaterialX is an error.
        shader = _create_first_available(mat, ("mtlxstandard_surface", "mtlxsurface"), name)
        # opacity is a colour on MaterialX (opacityr/g/b): a scalar "opacity"
        # parm does not exist, and 0.5 used to be dropped without a word.
        values = {"specular_roughness": roughness, "metalness": metallic, "opacity": [opacity] * 3}
        if color:
            values["base_color"] = color
    else:
        raise ValueError(f"Unknown mat_type '{mat_type}'. Must be 'principled' or 'materialx'.")
    skipped = _set_values(shader, values)
    shader_path = shader.path()

    layout_if_enabled(mat, place_unpositioned=True)
    # The shader was just created (through _create_first_available): place it.
    _focus_network_editor(shader, place_unpositioned=True)

    print(f"[workflow] Material '{name}' created at {shader_path}")

    result = {
        "success": not skipped,
        "material_path": shader_path,
        "shader_node_path": shader_path,
        "type": mat_type,
    }
    if skipped:
        result["not_set"] = skipped
    return result


###### workflow.assign_material


def _assign_material(
    geo_path: str,
    material_path: str,
    **_: Any,
) -> dict:
    """Assign a material to a geometry node.

    Creates a Material SOP at the end of the SOP chain inside the
    geometry node, sets the material path, and enables the display flag.

    Args:
        geo_path: Path to the geometry Object node (e.g. "/obj/geo1").
        material_path: Path to the material to assign (e.g. "/mat/material1").
    """
    print(f"[workflow] Assigning material '{material_path}' to '{geo_path}'")

    geo = _get_node(geo_path)
    # A typo in material_path used to be written into the Material SOP and
    # answered with success; the object then rendered unshaded.
    if hou.node(material_path) is None:
        raise ValueError(f"Material not found: {material_path}")

    # Determine the SOP-level parent -- if geo_path points to an Object-level
    # node we work inside it; if it already points to a SOP network, use it.
    category = geo.type().category().name()
    if category == "Object":
        sop_parent = geo
    elif category == "Sop":
        sop_parent = geo.parent()
    else:
        sop_parent = geo

    # A SOP named explicitly is where the material goes; only an object gets
    # the Material SOP after its displayed node.
    last_displayed = geo if category == "Sop" else None
    for child in () if last_displayed is not None else sop_parent.children():
        try:
            if child.isDisplayFlagSet():
                last_displayed = child
        except Exception:
            pass

    # If no display flag found, pick the last child
    if last_displayed is None:
        children = list(sop_parent.children())
        if children:
            last_displayed = children[-1]

    # -- Create Material SOP
    print("[workflow] Creating Material SOP")
    mat_sop = sop_parent.createNode("material", "material1")

    # Wire after last displayed SOP
    if last_displayed is not None:
        print(f"[workflow] Wiring Material SOP after {last_displayed.path()}")
        mat_sop.setInput(0, last_displayed, 0)

    # Set material path
    print(f"[workflow] Setting shop_materialpath1 to {material_path}")
    _set_parm_safe(mat_sop, "shop_materialpath1", material_path)

    # Set display and render flags
    try:
        mat_sop.setDisplayFlag(True)
        mat_sop.setRenderFlag(True)
    except Exception:
        pass

    layout_if_enabled(sop_parent)
    _focus_network_editor(mat_sop)

    print(f"[workflow] Material assigned: {mat_sop.path()} -> {material_path}")

    return {
        "success": True,
        "material_sop_path": mat_sop.path(),
        "material_path": material_path,
    }


###### workflow.build_sop_chain


def _build_sop_chain(
    parent_path: str = "/obj/geo1",
    steps: list = None,
    **_: Any,
) -> dict:
    """Build a sequential chain of SOPs inside a network.

    Each step dict specifies a node type and optional name/params.
    Nodes are wired sequentially (output 0 -> input 0).

    Args:
        parent_path: Path to the parent SOP network.
        steps: List of step dicts, each with keys:
               - "type" (str): Node type to create (required).
               - "name" (str): Optional node name.
               - "params" (dict): Optional parameter values to set.
    """
    if steps is None or len(steps) == 0:
        raise ValueError("steps list is required and must not be empty")
    _get_node(parent_path)
    from fxhoudinimcp_server.handlers.graph_handlers import build_network

    # A linear build_network: every type and parameter is validated before
    # anything is created, tuples ("rad", "t", "size") are set whole, and a
    # failure builds nothing. The hand-rolled loop this replaces skipped a
    # misspelt or tuple-only parameter and a failed wire with success: True.
    specs = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or not step.get("type"):
            raise ValueError(f"Step {index} is missing required 'type' key")
        name = step.get("name") or f"{step['type'].split('::')[0].split(':')[-1]}_step{index + 1}"
        spec = {"type": step["type"], "name": name, "parms": step.get("params") or {}}
        if specs:
            spec["inputs"] = [specs[-1]["name"]]
        specs.append(spec)
    specs[-1]["flags"] = {"display": True, "render": True}
    result = build_network(parent_path=parent_path, nodes=specs)
    if not result.get("success"):
        return result
    last = result["created"][-1]["path"] if result.get("created") else None
    if last:
        # build_network placed what it created; nothing else may move.
        _focus_network_editor(hou.node(last), place_unpositioned=False)
    return {
        "success": True,
        "nodes": result["created"],
        "displayed": last,
        "geometry": result.get("geometry"),
        "error_nodes": result.get("error_nodes", []),
    }


###### workflow.setup_render


def _setup_render(
    renderer: str = "karma",
    camera: str = None,
    output_path: str = "$HIP/render/output.$F4.exr",
    resolution: list = None,
    samples: int = 64,
    name: str = "render1",
    **_: Any,
) -> dict:
    """Set up a complete render configuration.

    Creates a camera (if none specified), a ROP node in /out, and
    configures output path, resolution, and sample count.

    Args:
        renderer: Renderer to use -- "karma" or "mantra".
        camera: Path to an existing camera. If None, creates one in /obj.
        output_path: Output image file path (supports Houdini variables).
        resolution: [width, height] resolution (default: [1920, 1080]).
        samples: Number of render samples.
        name: Name for the ROP node.
    """
    if resolution is None:
        resolution = [1920, 1080]

    obj = _ensure_obj_context()
    out = _ensure_out_context()
    all_nodes: list[str] = []
    # What did not take goes here and into the reply; it used to be printed
    # to Houdini's console, where the caller never saw it.
    warnings: list[str] = []

    # -- Step 1: Camera
    if camera is None:
        print("[workflow] Creating camera at /obj")
        cam = obj.createNode("cam", "render_cam")
        # Only /out is laid out below, so the camera is on its own -- without
        # this, back-to-back setup_render calls pile cameras at /obj's origin.
        place_new_node(cam)
        camera = cam.path()
        all_nodes.append(camera)

        # Set reasonable defaults
        _set_parm_safe(cam, "resx", resolution[0])
        _set_parm_safe(cam, "resy", resolution[1])
        print(f"[workflow] Camera created at {camera}")
    else:
        print(f"[workflow] Using existing camera: {camera}")
        if hou.node(camera) is None:
            warnings.append(f"camera '{camera}' does not exist; the ROP references it anyway")

    # -- Step 2: ROP node
    if renderer == "karma":
        print(f"[workflow] Creating Karma ROP '{name}' in /out")
        try:
            rop = out.createNode("karma", name)
        except hou.OperationFailed:
            try:
                rop = out.createNode("karma::2.0", name)
            except hou.OperationFailed:
                rop = out.createNode("usdrender_rop", name)
                print("[workflow] Warning: karma ROP not available, using usdrender_rop")
    elif renderer == "mantra":
        print(f"[workflow] Creating Mantra ROP '{name}' in /out")
        try:
            rop = out.createNode("ifd", name)
        except hou.OperationFailed:
            rop = out.createNode("mantra", name)
    else:
        raise ValueError(f"Unknown renderer '{renderer}'. Must be 'karma' or 'mantra'.")

    all_nodes.append(rop.path())

    # -- Step 3: Output path
    applied: dict[str, Any] = {}
    for parm_name in ("picture", "vm_picture", "outputimage", "ar_picture"):
        if _set_parm_safe(rop, parm_name, output_path):
            applied["output"] = parm_name
            break
    else:
        warnings.append("no output image parameter found on the ROP; output_path not set")

    # -- Step 4: Resolution. res_override* only counts with override_camerares
    # on, and both Karma and Mantra ship it off: the render used the camera's
    # resolution while the reply echoed the one asked for.
    if _set_parm_safe(rop, "override_camerares", 1):
        _set_parm_safe(rop, "res_overridex", resolution[0])
        _set_parm_safe(rop, "res_overridey", resolution[1])
        applied["resolution"] = "override_camerares + res_override"
        # The Karma ROP in /out also has its own resolutionx/y, and that is
        # the size it renders at: the override pair alone left a 1280x720
        # image while this reply said 320x180.
        if _set_parm_safe(rop, "resolutionx", resolution[0]):
            _set_parm_safe(rop, "resolutiony", resolution[1])
            applied["resolution"] += " + resolutionx/y"
    elif _set_parm_safe(rop, "resolutionx", resolution[0]):
        _set_parm_safe(rop, "resolutiony", resolution[1])
        applied["resolution"] = "resolutionx/y"
    else:
        warnings.append("no resolution parameter found on the ROP; resolution not set")

    # -- Step 5: Samples. Mantra splits them per axis (64 = 8 x 8).
    if _set_parm_safe(rop, "samplesperpixel", samples):
        applied["samples"] = "samplesperpixel"
    elif rop.parm("vm_samplesx") is not None:
        side = max(1, round(samples**0.5))
        _set_parm_safe(rop, "vm_samplesx", side)
        _set_parm_safe(rop, "vm_samplesy", side)
        applied["samples"] = f"vm_samplesx/y = {side}x{side}"
    else:
        warnings.append("no pixel samples parameter found on the ROP; samples not set")

    # -- Step 6: Camera path
    for parm_name in ("camera", "cam", "viewcamera"):
        if _set_parm_safe(rop, parm_name, camera):
            applied["camera"] = parm_name
            break
    else:
        warnings.append("no camera parameter found on the ROP; camera not set")

    # Layout
    layout_if_enabled(out)
    _focus_network_editor(rop)

    print(f"[workflow] Render setup '{name}' complete ({renderer})")

    return {
        "success": True,
        "rop_path": rop.path(),
        "camera_path": camera,
        "output_path": output_path,
        "renderer": renderer,
        "resolution": resolution,
        "samples": samples,
        "applied": applied,
        "all_nodes": all_nodes,
        **({"warnings": warnings} if warnings else {}),
    }


###### Registration

register_handler("workflow.setup_pyro_sim", _setup_pyro_sim)
register_handler("workflow.setup_rbd_sim", _setup_rbd_sim)
register_handler("workflow.setup_flip_sim", _setup_flip_sim)
register_handler("workflow.setup_vellum_sim", _setup_vellum_sim)
register_handler("workflow.create_material", _create_material)
register_handler("workflow.assign_material", _assign_material)
register_handler("workflow.build_sop_chain", _build_sop_chain)
register_handler("workflow.setup_render", _setup_render)
