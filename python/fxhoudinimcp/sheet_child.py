"""Run by a fresh hython for render_sheet: render frames once, tile them, exit.

Not imported by the server. On Houdini 22.0.368 an OpenGL ROP renders once per
hython: the second render in the same process segfaults, and a render off the
main thread (as in a serving hython) crashes creating Qt's platform. So each
sheet is one process, one render call, on its main thread.

argv[1] is a JSON object: hip (the snapshot), hip_file (the session's own
path, for $HIP), camera, start, end, step, resolution, columns, output. The
last stdout line is "__MCP_RESULT__ " + JSON.
"""

from __future__ import annotations

# Built-in
import json
import math
import os
import sys
import tempfile
import time


def _result(payload: dict) -> None:
    print("__MCP_RESULT__ " + json.dumps(payload), flush=True)


def main() -> int:
    import hou
    from PIL import Image, ImageDraw

    args = json.loads(sys.argv[1])
    hou.hipFile.load(args["hip"], suppress_save_prompt=True, ignore_load_warnings=True)
    if args.get("hip_file"):
        # The snapshot lives in the temp dir, which made $HIP point there and
        # broke every $HIP-relative cache and texture. Put the original back.
        original = args["hip_file"]
        hou.hscript(f'set -g HIP = "{os.path.dirname(original)}"')
        hou.hscript(f'set -g HIPFILE = "{original}"')
        hou.hscript(f'set -g HIPNAME = "{os.path.splitext(os.path.basename(original))[0]}"')
        hou.hscript("varchange")

    cameras = [n.path() for n in hou.node("/obj").allSubChildren() if n.type().name() == "cam"]
    camera = args.get("camera") or (cameras[0] if len(cameras) == 1 else None)
    if camera is None or hou.node(camera) is None:
        _result(
            {
                "error": "Pass camera=: "
                + ("none named" if not args.get("camera") else f"{args['camera']} not found")
                + (
                    f"; the scene has {', '.join(cameras)}."
                    if cameras
                    else "; the scene has no camera."
                ),
                "cameras": cameras,
            }
        )
        return 2

    start, end, step = args["start"], args["end"], args["step"]
    frames = list(range(int(start), int(end) + 1, int(step)))

    # In hython the OpenGL ROP draws SOP geometry as cooked on its first frame
    # (a SOP expression, even a $F File SOP, rendered the same image on every
    # frame; checked on 22.0.368), while object parms are read per frame. So
    # each frame's geometry is frozen into an object of its own, in world
    # space, shown on that frame only by an object-level display expression.
    shown = [
        obj
        for obj in hou.node("/obj").allSubChildren()
        if isinstance(obj, hou.ObjNode)
        and hasattr(obj, "displayNode")
        and obj.displayNode() is not None
        and obj.isObjectDisplayed()
    ]
    for frame in frames:
        hou.setFrame(frame)
        for index, obj in enumerate(shown):
            frozen = obj.displayNode().geometry().freeze()
            holder = hou.node("/obj").createNode("geo", f"__mcp_f{frame}_{index}")
            stash = holder.createNode("stash")
            stash.parm("stash").set(frozen)
            stash.setDisplayFlag(True)
            holder.setWorldTransform(obj.worldTransform())
            holder.parm("tdisplay").set(1)
            holder.parm("display").setExpression(f"$F == {frame}")
    for obj in shown:
        obj.setDisplayFlag(False)

    folder = tempfile.mkdtemp(prefix="fxhoudinimcp_sheet_")
    rop = hou.node("/out").createNode("opengl", "__mcp_sheet")
    rop.parm("camera").set(camera)
    rop.parm("tres").set(1)
    rop.parmTuple("res").set(tuple(args["resolution"]))
    rop.parm("picture").set(f"{folder}/frame.$F4.png".replace("\\", "/"))

    began = time.time()
    rop.render(frame_range=(start, end, step))
    seconds = round(time.time() - began, 2)

    tiles = []
    for frame in frames:
        path = os.path.join(folder, f"frame.{frame:04d}.png")
        if os.path.isfile(path):
            tiles.append((frame, Image.open(path).convert("RGB")))
    if not tiles:
        _result({"error": f"The OpenGL ROP wrote no image. ROP errors: {list(rop.errors())}"})
        return 3

    width, height = tiles[0][1].size
    columns = args.get("columns") or math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * width, rows * height), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    for index, (frame, tile) in enumerate(tiles):
        x, y = (index % columns) * width, (index // columns) * height
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y, x + 44, y + 14), fill=(0, 0, 0))
        draw.text((x + 3, y + 2), f"f{frame}", fill=(255, 255, 255))
    os.makedirs(os.path.dirname(args["output"]) or ".", exist_ok=True)
    sheet.save(args["output"])
    _result(
        {
            "output_path": args["output"],
            "frames": [frame for frame, _ in tiles],
            "missing_frames": [f for f in frames if f not in {t[0] for t in tiles}],
            "camera": camera,
            "grid": [columns, rows],
            "tile_resolution": [width, height],
            "render_seconds": seconds,
            "frame_folder": folder,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
