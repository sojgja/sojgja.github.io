---
id: 27-playbook-tich-hop-code
title: "Playbook · Tích hợp harness (code)"
sidebar_label: "Playbook · Tích hợp harness (code)"
sidebar_position: 27
---

# Playbook · Tích hợp harness (code)

Bốn công thức triển khai được ngay, xếp từ rẻ tới mạnh: chia sẻ skill, plugin tool, harness
làm MCP server, và cầu model. Mỗi công thức có code thật, cách xác minh, và best practice.

## A · Chia sẻ skill (rẻ nhất) {#a}

Harness và Hermes dùng chung chuẩn **agentskills.io**. Đưa skill điều tra dựa trên
bằng chứng của bạn vào Hermes gần như miễn phí:

```
# copy skill của harness vào Hermes
mkdir -p ~/.hermes/skills
cp -r skills/soigia-harness-evidence-base-investigate-workflows ~/.hermes/skills/

# hoặc phát hành qua tap để tái sử dụng
hermes skills tap add <owner>/<repo-chua-skill>
hermes skills install <skill-name>
```

Kiểm tra skill đã nạp: mở `hermes`, gõ `/skills` — thấy tên skill trong danh sách.

> **Điểm mạnh.** Không cần code, không thêm schema tool, không rủi ro phiên bản. Đây là bước đầu tiên nên làm.

## B · Plugin tool gọi harness {#b}

Cho agent gọi harness như một tool. Cấu trúc plugin:

```
~/.hermes/plugins/soigia-harness/
├── plugin.yaml
└── __init__.py
```

`plugin.yaml`:

```
name: soigia-harness
version: 1.0.0
description: Chạy tác vụ qua soigia-harness (điều tra dựa trên bằng chứng)
provides_tools:
  - harness_run
  - harness_evidence_check
requires_env:
  - SOIGIA_HARNESS_DIR
```

`__init__.py` — tuân thủ luật Hermes: handler trả **chuỗi JSON**, lỗi
trả `{"error": ...}`, không raise:

```
import json, os, subprocess

HARNESS = os.environ.get("SOIGIA_HARNESS_DIR", "")

def _run(cmd: list[str], cwd: str, timeout: int = 900) -> dict:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return {"ok": p.returncode == 0, "exit_code": p.returncode,
                "output": (p.stdout or "") + (p.stderr or "")}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except OSError as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

def harness_run(args: dict, **kw) -> str:
    task = args.get("task", "")
    if not HARNESS:
        return json.dumps({"error": "SOIGIA_HARNESS_DIR not set"})
    # ví dụ: chạy điều tra một-phát qua CLI của harness
    result = _run(["python", "-m", "adapters.cli", "--task", task], cwd=HARNESS)
    return json.dumps(result)

def harness_evidence_check(args: dict, **kw) -> str:
    path = args.get("path", "")
    result = _run(["python", "06-scripts/evidence_log.py", path],
                  cwd=os.path.join(HARNESS, ".."), timeout=60)
    return json.dumps(result)

TOOLS = [
    {
        "name": "harness_run",
        "description": "Chạy một tác vụ điều tra qua soigia-harness và trả kết quả.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string", "description": "Mô tả tác vụ điều tra"}},
            "required": ["task"]},
        "handler": harness_run,
    },
    {
        "name": "harness_evidence_check",
        "description": "Chấm một evidence-log của soigia-harness (thiếu mục → lỗi).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Đường dẫn evidence-log.md"}},
            "required": ["path"]},
        "handler": harness_evidence_check,
    },
]

def register(ctx):
    for t in TOOLS:
        schema = {"name": t["name"], "description": t["description"],
                  "parameters": t["parameters"]}
        ctx.register_tool(name=t["name"], schema=schema, handler=t["handler"])
```

```
# kiểm thử trước khi tin dùng
hermes plugins doctor ~/.hermes/plugins/soigia-harness --ci
hermes plugins list
hermes plugins enable soigia-harness
```

> **Có sẵn trong repo.** Plugin này đã được dựng thật tại `soigia-hermes-plugin/` — gồm `plugin.yaml`, `__init__.py` (`register(ctx)` + 7 tool `harness_*`), `bridge.py` (chạy subprocess trong checkout harness), skill đi kèm và bộ test offline. Cài: `cp -r soigia-hermes-plugin ~/.hermes/plugins/soigia-harness`.

> **Bắt buộc.** Không import harness *trong tiến trình* Hermes. Gọi bằng subprocess để tránh xung đột dependency và vòng đời agent. Đặt `SOIGIA_HARNESS_DIR` trong `~/.hermes/.env`.

## C · Harness làm MCP server {#c}

Cách tách rời sạch nhất: harness lộ tool qua MCP (stdio), Hermes consum như tool bản địa. Server
MCP tối giản (dùng SDK `mcp`):

```
# mcp_harness_server.py
from mcp.server.fastmcp import FastMCP
import json, subprocess

mcp = FastMCP("soigia-harness")

@mcp.tool()
def run_tests(path: str = ".") -> str:
    """Chạy test của harness, trả JSON."""
    p = subprocess.run(["python", "-m", "pytest", path, "-q"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=900)
    return json.dumps({"exit_code": p.returncode, "output": p.stdout + p.stderr})

@mcp.tool()
def search_context(query: str, root: str = ".") -> str:
    """Tìm ngữ cảnh trong mã nguồn (bọc tools/search_context.py)."""
    p = subprocess.run(["python", "soigia-harness/tools/search_context.py", query],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120)
    return json.dumps({"exit_code": p.returncode, "output": p.stdout + p.stderr})

if __name__ == "__main__":
    mcp.run()
```

Khai báo trong `~/.hermes/config.yaml`:

```
mcp_servers:
  soigia-harness:
    command: "python"
    args: ["/abs/path/mcp_harness_server.py"]
```

```
hermes            # khởi động → tool MCP được khám phá
# trong chat:
/context all      # xem chi phí schema của MCP
```

> **Best practice MCP.** Chỉ lộ tool cần thiết; pin phiên bản server; không để credential trong `mcp.json`/args. Chạy `hermes security audit` để kiểm tra chuỗi cung ứng cho MCP đã ghim.

## E · Cầu model (hermes proxy) {#e}

Cho harness dùng chính model/credential của Hermes thay vì tự cấu hình lại. Hermes có proxy
OpenAI-compatible cục bộ gắn credential OAuth:

```
hermes proxy        # mở proxy OpenAI-compatible cục bộ (gắn credential)
```

Rồi trỏ harness tới proxy đó bằng adapter sẵn có:

```
from adapters.factory import make_model

# harness dùng model qua proxy của Hermes
model = make_model("openai:<model>", base_url="http://127.0.0.1:<port>/v1",
                   api_key="<proxy-key-nếu-có>")
```

| Ưu điểm | Lưu ý |
| --- | --- |
| Một nguồn credential; không nhân đôi key | Proxy cục bộ là điểm phụ thuộc — cần chạy kèm |
| Đổi provider ở Hermes là harness hưởng lợi | Đo chi phí: proxy có thể ảnh hưởng prompt cache |

## Xác minh & best practices {#xac-minh}

| Việc | Lệnh / cách |
| --- | --- |
| Kiểm plugin trước khi cài | `hermes plugins doctor <path> --ci` |
| Xem plugin/skill đã nạp | `hermes plugins list`, `/skills` |
| Đo chi phí context | `/context all`, `hermes prompt-size` |
| Audit chuỗi cung ứng | `hermes security audit` |
| Chẩn đoán chung | `hermes doctor` |

| Best practice | Vì sao |
| --- | --- |
| Bắt đầu bằng skill (A) | Rẻ, an toàn, không schema |
| Tách tiến trình + JSON | Tránh xung đột dependency/vòng đời |
| Namespace tool (`harness_*`) | Không đè built-in |
| Một bên sở hữu model config | Tránh hai nguồn sự thật |
| Chỉ lộ tool cần; đo context | Schema thừa làm agent lú & tốn token |
| Dùng chung chuẩn bằng chứng | Hai bên cùng tạo "bằng chứng" phải cùng định nghĩa |

> **Nguồn.** `developer-guide/plugins/index.md` (register(ctx), plugin.yaml, plugins doctor), `adding-tools.md` (JSON string, error shape), `user-guide/features/mcp.md`, `reference/cli-commands.md` (`hermes proxy`, `skills tap add`), API `soigia-harness/adapters/factory.py`.
