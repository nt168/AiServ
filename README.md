# AiServ

一个可以和 `pi-mono` 对接的本地推理服务，目标是对外暴露 OpenAI-compatible `/v1` 接口，内部预留 MNN 推理引擎接入点。

当前提供两种引擎模式：

- `mock`: 不依赖 MNN，先把 HTTP 协议和 `pi-mono` 对接跑通
- `mnn`: 使用外部 `mnn-llm`/脚本/命令行程序，通过 `subprocess` 调用

## 目录结构

```text
AiServ/
  app/
    engines/
      base.py
      mock_engine.py
      mnn_engine.py
    config.py
    main.py
    openai.py
    schemas.py
  config.example.json
  pyproject.toml
```

## 快速开始

1. 创建虚拟环境并安装依赖

```bash
cd /home/nt/aidev/AiServ
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. 复制配置

```bash
cp config.example.json config.json
```

3. 先用 mock 模式启动

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

4. 验证接口

```bash
curl http://127.0.0.1:8000/healthz

curl http://127.0.0.1:8000/v1/models \
  -H 'Authorization: Bearer dummy'

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer dummy' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-0.8b-mnn",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

## 切换到 MNN 模式

把 `config.json` 中的 `service.engine` 改成 `mnn`，并配置：

- `mnn.model_path`: `.mnn` 模型文件路径
- `mnn.tokenizer_path`: tokenizer 文件路径
- `mnn.backend`: `cpu` / `opencl` / `vulkan` 等
- `mnn.num_threads`: CPU 线程数
- `mnn.precision`: `fp32` / `fp16` / `bf16`
- `mnn.prompt_template`: 把 system + conversation 组装成单轮 prompt 的模板
- `mnn.command`: 外部推理命令，例如 `["python3", "/path/to/run_mnn.py"]`
- `mnn.prompt_mode`: `stdin` 或 `arg`
- `mnn.command_timeout_seconds`: 单次推理超时

## 关于 MNN 适配

当前实现走外部命令行程序。`AiServ` 会在每次请求时启动你配置的命令，并把 prompt 通过 stdin 或命令行参数传进去。

默认约定：

- stdout: 只输出模型最终文本
- stderr: 用于错误日志
- 非 0 退出码: 视为推理失败

### 示例 1：通过 stdin 传 prompt

```json
{
  "service": {
    "engine": "mnn"
  },
  "mnn": {
    "model_path": "/data/qwen/model.mnn",
    "tokenizer_path": "/data/qwen/tokenizer.json",
    "command": ["python3", "/opt/mnn/run_mnn.py"],
    "prompt_mode": "stdin"
  }
}
```

你的脚本需要能接受：

```bash
python3 /opt/mnn/run_mnn.py \
  --model /data/qwen/model.mnn \
  --tokenizer /data/qwen/tokenizer.json \
  --backend cpu \
  --threads 4 \
  --precision fp16 \
  --max-tokens 2048 \
  --temperature 0.7 \
  --top-p 0.9
```

并从 stdin 读取 prompt，最终把生成文本打印到 stdout。

### 示例 2：通过参数传 prompt

```json
{
  "service": {
    "engine": "mnn"
  },
  "mnn": {
    "model_path": "/data/qwen/model.mnn",
    "tokenizer_path": "/data/qwen/tokenizer.json",
    "command": ["python3", "/opt/mnn/run_mnn.py"],
    "prompt_mode": "arg",
    "prompt_arg": "--prompt"
  }
}
```

## 与 pi-mono 对接

在 `~/.pi/agent/models.json` 里增加一个自定义 provider：

```json
{
  "providers": {
    "mnn-local": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "qwen3.5-0.8b-mnn",
          "name": "Qwen3.5 0.8B (MNN)",
          "reasoning": false,
          "input": ["text"],
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "contextWindow": 32768,
          "maxTokens": 2048
        }
      ]
    }
  }
}
```

然后就可以：

```bash
pi --model mnn-local/qwen3.5-0.8b-mnn
```

## 已知限制

- 当前默认只保证文本对话接口
- tool calling 需要你的模型和 prompt 具备稳定的结构化输出能力，再在 `mnn_engine.py` 中补充 tool call 解析
- 当前实现是“每次请求启动一次外部进程”，适合先跑通接入；如果后续追求吞吐和延迟，可以再改成长驻 worker
