# :material-wrench: Tools

fxhoudinimcp exposes **215 tools** across **24 categories**. The [Technical](../technical/tools/index.md) section documents each one.

--8<-- "README.md:features"

## Worth knowing

- **Build in one call.** `build_network` validates every node type, parameter and input against the running Houdini, then builds the whole network or nothing. `dry_run=True` checks a plan and suggests corrections. It is also about ten times faster than creating nodes one by one, since each call waits for a Houdini main-thread tick.
- **One undo step per call**, however many nodes it touched.
- **Look it up, don't guess.** `get_node_card` gives a node type's real parameters and connectors for this Houdini build. `search_help` and `get_help_page` read the manual Houdini ships, workflow guides included, offline.
- **Verify, then claim.** `verify_network` reports every node's errors and cooked geometry. `capture_screenshot` and `render_sheet` show the result: `render_sheet` tiles a frame range into one image and needs no viewport.
- **Sessions.** `get_houdini_connection_status` lists every Houdini serving the plugin. `connect_houdini` switches between them, and `start_houdini` / `stop_houdini` run one of the server's own, windowless or not. A started Houdini holds a license seat.
- **Code as a last resort.** `execute_python` and `execute_hscript` exist, but the server instructions steer assistants to nodes first.
