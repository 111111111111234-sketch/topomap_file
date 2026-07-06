#!/bin/bash
# GoatAgent 完整测试流程 - 一键执行指南

echo "=========================================="
echo "🎯 GoatAgent Pipeline 完整测试"
echo "=========================================="

# ========== 第一步：环境准备 ==========
echo ""
echo "📍 第一步：激活 goat 环境并进入项目目录"
echo "----------------------------------------"
echo "执行命令："
echo "conda activate goat"
echo "cd /workspace/tangyx7@xiaopeng.com"
echo "export PYTHONPATH=\$PWD"
echo ""
echo "或者一行执行："
echo "conda activate goat && cd /workspace/tangyx7@xiaopeng.com && export PYTHONPATH=\$PWD"
echo ""
echo "✅ 完成后继续"
echo ""

# ========== 第二步：语法检查 ==========
echo "=========================================="
echo "📍 第二步：语法检查"
echo "----------------------------------------"
echo "复制以下命令执行："
echo ""
cat << 'SYNTAX'
python3 -m py_compile \
  conftopo/core/confidence.py \
  conftopo/core/instruction_graph.py \
  conftopo/core/hypothesis_pool.py \
  conftopo/core/object_memory_pool.py \
  conftopo/core/dynamic_topo_map.py \
  conftopo/agents/goat_agent_new.py && echo "✅ 语法检查通过"
SYNTAX
echo ""

# ========== 第三步：单元测试 ==========
echo "=========================================="
echo "📍 第三步：运行单元测试"
echo "----------------------------------------"
echo "复制以下命令执行："
echo ""
cat << 'UNITTEST'
python3 -m pytest conftopo/tests/test_goat_agent_new.py -v
UNITTEST
echo ""

# ========== 第四步：VLM 启动（新终端） ==========
echo "=========================================="
echo "📍 第四步：启动 VLM 服务（在新的终端窗口执行）"
echo "----------------------------------------"
echo "打开新的终端窗口，执行以下命令："
echo ""
cat << 'VLLM'
conda activate vllm && \
vllm serve /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --dtype auto \
  --limit-mm-per-prompt '{"image": 1}' \
  --max-model-len 4096 \
  --port 8000
VLLM
echo ""
echo "⏳ 等待输出："
echo "   INFO:     Uvicorn running on http://0.0.0.0:8000"
echo ""

# ========== 第五步：验证 VLM 连接 ==========
echo "=========================================="
echo "📍 第五步：验证 VLM 连接（在原来的终端执行）"
echo "----------------------------------------"
echo "复制以下命令验证 VLM 服务："
echo ""
cat << 'VERIFY'
curl http://localhost:8000/v1/models
VERIFY
echo ""
echo "✅ 预期输出包含 'Qwen/Qwen3-VL-8B-Instruct'"
echo ""

# ========== 第六步：正式 Smoke Test ==========
echo "=========================================="
echo "📍 第六步：运行正式 Smoke Test（需要 VLM 启动）"
echo "----------------------------------------"
echo "复制以下完整脚本执行："
echo ""
cat << 'SMOKETEST'
python3 - <<'PY'
import sys
import numpy as np
from conftopo.agents import GoatAgent
from conftopo.core.instruction_graph import GoalNode, Relation

print("\n" + "="*60)
print("🚀 GoatAgent Smoke Test")
print("="*60)

try:
    print("\n[1/6] 创建 GoatAgent 实例...")
    agent = GoatAgent()
    print("✅ 成功创建 Agent")
    
    print("\n[2/6] 设置目标 Goal...")
    goal = GoalNode(
        target_object="chair",
        attributes=["red"],
        room_prior=["dining room"],
        landmarks=["table"],
        relations=[Relation("near", "table")],
    )
    agent.set_new_goal(goal)
    print(f"✅ Goal 设置完成: {goal.target_object}")
    
    print("\n[3/6] 模拟观测 (Observe)...")
    obs = {
        "rgb": None,
        "rgb_embed": np.random.rand(512).astype("float32"),
        "position": np.array([0.0, 0.0, 0.0], dtype="float32"),
        "heading": 0.0,
    }
    agent.observe(obs)
    print("✅ 观测数据已输入")
    
    print("\n[4/6] 更新记忆 (Update Memory)...")
    agent.update_memory()
    print("✅ 记忆已更新")
    
    print("\n[5/6] 生成计划 (Plan)...")
    plan = agent.plan()
    print(f"✅ 计划生成完成: {plan}")
    
    print("\n[6/6] 执行动作 (Act)...")
    action_output = agent.act(plan)
    print(f"✅ 动作执行完成: {action_output}")
    
    print("\n" + "="*60)
    print("📊 Pipeline 统计信息:")
    print("="*60)
    stats = agent.memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！Pipeline 可以正常运行")
    print("="*60 + "\n")
    
except Exception as e:
    print(f"\n❌ 测试失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PY
SMOKETEST
echo ""

# ========== 总结 ==========
echo ""
echo "=========================================="
echo "📋 快速参考"
echo "=========================================="
echo ""
echo "📝 完整命令序列（复制到终端）："
echo ""
echo "# 终端1：主测试终端"
echo "conda activate goat && cd /workspace/tangyx7@xiaopeng.com && export PYTHONPATH=\$PWD"
echo ""
echo "# 验证语法"
echo "python3 -m py_compile conftopo/agents/goat_agent_new.py && echo '✅ 语法通过'"
echo ""
echo "# 运行单元测试"
echo "python3 -m pytest conftopo/tests/test_goat_agent_new.py -v"
echo ""
echo "---"
echo ""
echo "# 终端2：VLM 服务（新开）"
echo "conda activate vllm && vllm serve /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b --served-model-name Qwen/Qwen3-VL-8B-Instruct --dtype auto --limit-mm-per-prompt '{\"image\": 1}' --max-model-len 4096 --port 8000"
echo ""
echo "---"
echo ""
echo "# 终端1：验证 VLM"
echo "curl http://localhost:8000/v1/models"
echo ""
echo "# 终端1：Smoke Test（见上面的完整脚本）"
echo ""
echo "=========================================="
echo "🎯 预期结果"
echo "=========================================="
echo "✅ 语法检查：无报错"
echo "✅ 单元测试：所有 test 通过"
echo "✅ VLM 服务：正常运行在 8000 端口"
echo "✅ Smoke Test：输出 Plan/Action/Memory stats"
echo ""
echo "=========================================="
echo ""
