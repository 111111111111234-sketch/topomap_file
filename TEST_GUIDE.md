# GoatAgent New Pipeline 测试指南

## 📋 三层测试流程

### 第一步：环境准备

```bash
cd /workspace/tangyx7@xiaopeng.com
export PYTHONPATH=$PWD
```

### 第二步：启动 vLLM 服务（真实 VLM）

**在一个新的终端窗口中执行：**

```bash
conda activate vllm

vllm serve /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --dtype auto \
  --limit-mm-per-prompt '{"image": 1}' \
  --max-model-len 4096 \
  --port 8000
```

**预期输出：**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ 验证服务可用：
```bash
curl http://localhost:8000/v1/models
```

---

### 第三步：安装依赖（主终端）

```bash
conda activate base  # 或你的主环境
cd /workspace/tangyx7@xiaopeng.com

python3 -m pip install pytest networkx numpy -q
```

---

## 🔍 一层：语法检查

验证核心文件能否编译（不需要 VLM）：

```bash
cd /workspace/tangyx7@xiaopeng.com
export PYTHONPATH=$PWD

python3 -m py_compile \
  conftopo/core/confidence.py \
  conftopo/core/instruction_graph.py \
  conftopo/core/hypothesis_pool.py \
  conftopo/core/object_memory_pool.py \
  conftopo/core/dynamic_topo_map.py \
  conftopo/agents/goat_agent_new.py

echo "✅ 语法检查通过"
```

**预期结果：** 无输出或只有 `✅ 语法检查通过`

---

## 🧪 二层：单元测试（Mock 模式，不需要真实 VLM）

先跑单元测试验证核心逻辑：

```bash
cd /workspace/tangyx7@xiaopeng.com
export PYTHONPATH=$PWD

# 只跑 goat_agent_new 的测试
python3 -m pytest conftopo/tests/test_goat_agent_new.py -v

# 或者跑所有核心测试
python3 -m pytest \
  conftopo/tests/test_core.py \
  conftopo/tests/test_goat_agent_new.py \
  -v
```

---

## 🚀 三层：真实 Smoke Test（需要 VLM）

### 前置条件：
- ✅ vLLM 服务已启动（端口 8000）
- ✅ `config.py` 中 VLM 配置已设置：
  ```python
  vlm_api_base: str = "http://localhost:8000/v1"
  vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
  ```

### 运行最小 Smoke Test：

```bash
cd /workspace/tangyx7@xiaopeng.com
export PYTHONPATH=$PWD

python3 - <<'PY'
import numpy as np
from conftopo.agents.goat_agent_new import GoatAgent
from conftopo.core.instruction_graph import GoalNode, Relation

print("[1/5] 创建 GoatAgent...")
agent = GoatAgent()

print("[2/5] 设置目标...")
goal = GoalNode(
    target_object="chair",
    attributes=["red"],
    room_prior=["dining room"],
    landmarks=["table"],
    relations=[Relation("near", "table")],
)
agent.set_new_goal(goal)

print("[3/5] 模拟观测...")
obs = {
    "rgb": None,
    "rgb_embed": np.random.rand(512).astype("float32"),
    "position": np.array([0.0, 0.0, 0.0], dtype="float32"),
    "heading": 0.0,
}
agent.observe(obs)

print("[4/5] 更新记忆...")
agent.update_memory()

print("[5/5] 生成计划...")
plan = agent.plan()
out = agent.act(plan)

print("\n✅ Pipeline 通过！")
print(f"📌 Plan: {plan}")
print(f"📌 Action: {out}")
print(f"📌 Memory stats: {agent.memory_stats()}")
PY
```

**预期输出：**
```
[1/5] 创建 GoatAgent...
[2/5] 设置目标...
[3/5] 模拟观测...
[4/5] 更新记忆...
[5/5] 生成计划...

✅ Pipeline 通过！
📌 Plan: ...
📌 Action: ...
📌 Memory stats: ...
```

---

## 🎯 完整快速执行序列

### 终端1（vLLM 服务）：
```bash
conda activate vllm
vllm serve /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --dtype auto \
  --limit-mm-per-prompt '{"image": 1}' \
  --max-model-len 4096 \
  --port 8000
```

### 终端2（测试）：
```bash
cd /workspace/tangyx7@xiaopeng.com
export PYTHONPATH=$PWD

# 1. 语法检查
python3 -m py_compile conftopo/agents/goat_agent_new.py && echo "✅ 语法通过"

# 2. 单元测试
python3 -m pytest conftopo/tests/test_goat_agent_new.py -q

# 3. Smoke test（见上面的完整代码）
```

---

## 📊 测试文件说明

| 文件 | 说明 |
|------|------|
| `test_goat_agent_new.py` | GoatAgent 新 pipeline 单元测试 |
| `test_core.py` | 核心模块测试（confidence, graphs 等） |
| `test_perception_report.py` | 感知报告生成测试 |
| `test_phase2.py` | Phase 2（规划）测试 |
| `test_phase3.py` | Phase 3（动态更新）测试 |

---

## 🐛 常见问题

### 问题1：VLM 服务连接失败
```
ConnectionError: http://localhost:8000
```
**解决：** 检查 vLLM 服务是否启动
```bash
curl http://localhost:8000/v1/models
```

### 问题2：PYTHONPATH 导入错误
```
ModuleNotFoundError: No module named 'conftopo'
```
**解决：** 确保设置了 PYTHONPATH
```bash
export PYTHONPATH=/workspace/tangyx7@xiaopeng.com
```

### 问题3：缺少依赖
```
ModuleNotFoundError: No module named 'pytest'
```
**解决：** 安装依赖
```bash
python3 -m pip install pytest networkx numpy
```

---

## ✨ 成功标志

- ✅ 语法检查：无报错
- ✅ 单元测试：所有测试通过
- ✅ Smoke test：能正常生成 Plan 和 Action

---

**开始测试吧！** 按顺序执行上面的命令。
