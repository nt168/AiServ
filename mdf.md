# AiServ + MNN + pi-mono 操作步骤

按这个顺序执行就行。

## 1. 安装系统依赖

```bash
sudo apt update
sudo apt install -y python3-pip python3.12-venv
```

## 2. 创建虚拟环境并安装 AiServ 依赖

```bash
cd /home/nt/aidev/AiServ
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 3. 确认配置已经是 MNN 模式

检查这个文件：
`/home/nt/aidev/AiServ/config.json`

关键几项应该是：

- `"engine": "mnn"`
- `"model_path": "/home/nt/models/Qwen35-08b_mnn/config.json"`
- `"command": ["python3", "/home/nt/aidev/AiServ/run_mnn.py"]`

## 4. 先单独验证 MNN 包装脚本

```bash
printf '你好，请用两句话介绍你自己。' | python3 /home/nt/aidev/AiServ/run_mnn.py \
  --model /home/nt/models/Qwen35-08b_mnn/config.json \
  --backend cpu \
  --threads 4 \
  --precision fp16 \
  --max-tokens 128 \
  --temperature 0.7 \
  --top-p 0.9
```

如果这里有正常中文输出，说明 MNN 这层已经通了。

## 5. 启动 AiServ

```bash
cd /home/nt/aidev/AiServ
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 6. 验证 HTTP 接口

新开一个终端：

```bash
curl http://127.0.0.1:8000/healthz
```

```bash
curl http://127.0.0.1:8000/v1/models \
  -H 'Authorization: Bearer dummy'
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer dummy' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-0.8b-mnn",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己。"}
    ]
  }'
```

## 7. 配置 pi-mono

编辑 `~/.pi/agent/models.json`，写入：

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

## 8. 在 pi 里使用

```bash
pi --model mnn-local/qwen3.5-0.8b-mnn
```

如果你想先只测试模型列表，也可以：

```bash
pi --list-models mnn-local
```

如果你执行到哪一步报错了，把那一步的输出贴出来，再继续定位。
