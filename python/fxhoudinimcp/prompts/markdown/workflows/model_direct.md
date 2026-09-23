Direct polygon modeling in Houdini SOPs, revised 2026-09-11.
Task: {description}
Reference: {reference}

The current workspace modeling standard takes precedence over this short
prompt. Deliver the requested model and evidence; do not add packaging,
rendering or recovery experiments to a basic modeling task.

## Before modeling

1. Check `get_houdini_connection_status` and `get_scene_info`. If unreachable,
   report and wait; never start or close Houdini yourself.
2. Use the request and established context to determine the part, reference,
   purpose, units, budget and required features. Record reasonable assumptions
   for non-blocking gaps; do not ask again for already supplied information.
3. Work in a named subnet under the requested parent. Protect the reference
   and user edits. Query specified paths, never scan all of `/obj`.
4. Read a baseline `get_mesh_report` for the target.

## Choose the route by deliverable

For Nanite static hard surfaces where mixed topology is accepted, favor
Boolean shaping, PolyBevel for regular bevels, measured profiles for regular
bodies, and local SubD for irregular transitions or smooth surfaces. Triangles
and stable ngons are acceptable. Clean topology only where shape, bevels,
triangulation, basic shading, UVs or requested editability require it.

Use the actual 3D reference for hole positions, profiles, dimensions and
restricted projection. Do not reduce it to screenshots. Develop a difficult
unit and its join first, check it, then repeat. Unwrap each component after
its major Boolean/bevel work stabilizes; validate seams before broad copying.
Only components using SubD require a control cage and subdivision check.

## Native operations and evidence

`create_node` / `build_network` can create installed node types appropriate
to the network. Boolean and PolyBevel are native SOPs; `modeler::connect`
is an installed Modeler HDA. No separate Boolean MCP tool is required.

- Batch planned nodes in `build_network`; dry-run unfamiliar types.
- Query `get_node_card` with substring `parm_filter`; request help only when
  needed. Check menu value types and point/edge/primitive selection semantics.
- Use intent-named nodes/groups. Keep `OUT_<part>`; add `CAGE_<part>` and
  `SUBD_L1_<part>` only where applicable.
- Use `edit_points` for point edits. At meaningful stages verify actual inputs,
  parameter expressions, cook and geometry. A request echo is not read-back.
- Use `get_mesh_report` for geometry health and `compare_geometry` when the
  intended position/topology change needs proof. Do not run every check after
  every small node when existing evidence is sufficient.
- Native choices include `boolean`, `polybevel::3.0`, `polyextrude::2.0`,
  `polybridge`, `polyfill`, `polysplit::2.0`, `modeler::connect`, `fuse`,
  `dissolve::2.0`, `smooth`, `crease` and `subdivide`. Fill Quads is optional
  local cleanup, not a mandatory Boolean finishing step.
- Python can measure, inspect, organize selections and invoke native viewport
  controls when no suitable dedicated interface exists. Do not turn a model
  task into development of a general modeling algorithm.
- Put large results in files, return counts/worst samples/paths. Node counts
  and script milliseconds are not end-to-end task timing.

## Acceptance scoped to the task

1. Measure bidirectional area-sampled surface distances in a declared space.
   Record samples, seed, tolerances, mean/P95/max and coverage; check important
   holes/contact regions separately (`section_geometry`, `fit="circle"` for
   hole and boss diameters, outlines at given heights). Use the actual output surface, including
   evaluated SubD only where used. Coverage is not visual similarity.
2. Check unexpected non-manifold/degenerate/flipped faces, wrong openings and
   part connectivity. Inspect actual triangulation of ngons; legal planar
   concavity is not a failure. Do not require all quads or low pole valence on
   surfaces that will not be subdivided.
3. If UVs are requested, check finite values, area, unintended overlap,
   stretching, density and padding, assisted by a checker view. Report tiny
   overlap at texture scale instead of endlessly chasing numerical noise.
4. Align reference and result. Use front/side/oblique and wireframe views for
   proportions, depth, holes and topology: one `capture_viewport` call with
   those views, the part as target and `shading="smooth_wire"` frames each
   view on the part and says whether it is fully in frame. Basic shaded inspection may reveal
   flipped normals or obvious pinching; strict matched-highlight/material
   comparison is outside basic modeling acceptance.

Use local self-intersection diagnostics for actual doubts or an explicit
requirement. Keep repairs off and distinguish intended assembly contact from
self-intersection. Interpret PolyDoctor marks by category; nonconvex ngons
are not automatically invalid. Validate an unfamiliar diagnostic configuration
once, not by rebuilding unchanged positive-control probes every task.

After the necessary checks pass, finish. Recheck affected regions after a
change; expand verification only for a new issue or an explicit requirement.
Never claim an unperformed check passed, or make an out-of-scope check a gate.

## Saving and handoff

Save HIP when authorized by the user or project, without asking again; honor
an explicit no-save instruction. Use the agreed path or a task version and
preserve relative-resource resolution and user edits.

The existing network output is the default deliverable. Do not automatically
export CPIO, restore it into a temporary subnet, or compare every reloaded
attribute. Native caching normally does not change source node parameters.
Only an explicitly requested portable package, format conversion, or actual
fault warrants the corresponding extra checks. Numbering/storage-order
changes alone do not prove corruption. A missing HIP save is not a reason
to expand scope. Report the final node, save status and necessary evidence;
remove task probes through Houdini's node interface.

{network_housekeeping}
