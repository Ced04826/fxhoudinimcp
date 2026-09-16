# 截图类工具把同一张图的 base64 发了两遍

发现于 2026-09-15，排查 Claude Code 在 Houdini 会话里的 token 开销时撞到的。

**已处置（2026-09-16）**：上游 healkeiser/fxhoudinimcp 在 2026-09-12 的 PR #49（提交 218bb47）已把内嵌 JPEG 整个拿掉，四个截图类工具只返回文件路径，由代理用 Read 读图。本 fork 以 cherry-pick 方式采纳（本地提交 b608d48），784 个单元测试通过，`tools/list` 确认这四个工具不再声明 outputSchema。没有采用下文"明天怎么做"里的 `structured_output=False` 方案，因为那只去掉重复份，仍留一张低清缩略图和整套编码代码。实机回读（调一次 `render_viewport` 看回执只剩路径）待 Houdini 开启后补做。下文保留为原始分析记录。

## 现象

调用一次 `render_viewport` / `capture_screenshot`，送到模型那边的 `tool_result` 里有两块内容：

```
block[0]  image  512x358 JPEG          → 244 token   ← 模型靠这块看图
block[1]  text   75,532 chars          → ~25,000 token  ← 其中 99.5% 是 block[0] 的 base64 副本
```

把 block[1] 当 JSON 解开是这个结构：

```json
{"result": [
  {"type":"text",  "text":"{\"success\":true,\"output_path\":\"...\",\"resolution\":[1000,700],\"camera\":null,\"frame\":1.0}"},
  {"type":"image", "data":"<和 block[0] 一模一样的 base64>", "mimeType":"image/jpeg"}
]}
```

两边的 base64 片段逐字对得上。有用的只有开头那 ~380 字符元数据，其余是纯副本。

## 代价

昨天那个 Houdini 会话里三次截图：

| | 图像尺寸 | image 块 | 重复的 text 块 | 倍数 |
|---|---|---|---|---|
| capture_screenshot | 512×288 | 197 tok | 3,840 tok | 19× |
| render_viewport | 512×358 | 244 tok | 7,119 tok | 29× |
| render_viewport | 512×358 | 244 tok | 25,177 tok | **103×** |

重复块进上下文后每一轮都要重发。按 Fable 5.1 计价（缓存写 $20/MTok、读 $0.25/MTok）：首次写入 $0.72，之后累计 27.15M token 的缓存读 $6.79，**合计 $7.51，占该会话账单的 2.1%**。

第三次的重复块还吃满了 Claude Code 给 MCP 输出的 25,000 token 上限，触发 `[OUTPUT TRUNCATED]`——整个输出预算被一份重复且被截断的 base64 占光。

对照：同期跑的 SoL-ClaudeCode 网关（专门做上下文压缩的一整套机制）在 387 个请求上实测省下的上限是 $1.46。**这一处浪费是它的 5 倍。**

## 不是谁的错，是哪一层的问题

排除掉的：

- **`tools/__init__.py:16` 的 `result_with_image()` 是正确的**——它在 `json.dumps(result)` 之前就把 `image_base64` pop 掉了，产出的 TextContent 只有元数据。
- **不是 Claude Code 的通病**。扫了本机 114 个 transcript：带图像的工具结果共 124 次，94 次干净（Blender MCP / UnrealMCP / 直接 Read 图片 / VPS），30 次重复，重复的全部来自本项目的截图类工具。同一个 Houdini 会话里走其他路径拿到的 22 次图像结果也是干净的。

定位到的机制（`mcp` 2.0.0）：

```
mcp/server/mcpserver/utilities/func_metadata.py:138     result = {"result": result}
```

这些工具的返回标注是 `list[TextContent | ImageContent]`——不是 dict，所以 mcp 会为它生成 output schema，把**整个返回值**包进 `{"result": ...}` 作为 `structuredContent`，再序列化成一个 text 内容块发出去。于是 ImageContent 里的 base64 第二次进了请求。

## 明天怎么做

1. 给受影响的工具加 `structured_output=False`：

   ```python
   @mcp.tool(structured_output=False)     # tools/base.py:68 支持这个参数
   async def capture_screenshot(...) -> list[TextContent | ImageContent]:
   ```

   受影响的是所有走 `result_with_image()` 返回的工具，至少有 `viewport.py` 的 `capture_screenshot` / `capture_network_editor` 和 `rendering.py` 的 `render_viewport`。用 `grep -rn "result_with_image" python/fxhoudinimcp/tools/` 列全。

2. 验证：重启 MCP，调一次 `render_viewport`，确认 `tool_result` 里只剩 image 块加一个 ~380 字符的元数据 text 块，没有第二份 base64。

3. 顺带对齐一处：元数据报 `"resolution": [1000, 700]`，实际交付的 JPEG 是 512×358（服务端编码前降采样了）。模型如果按 1000×700 判断画面能看清多少细节会被误导，改成报实际交付尺寸。

## 复核方法

拿任意一个含截图的 transcript（`~/.claude/projects/<项目>/<session>.jsonl`），找 `tool_result` 里同时有 `type:"image"` 和一个包含 `"mimeType"` 的 `type:"text"` 块的——有就是没修好。
