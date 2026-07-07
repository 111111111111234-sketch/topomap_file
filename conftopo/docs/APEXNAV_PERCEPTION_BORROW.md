# ApexNav 感知借鉴：FOV 置信度融合 + CLIP-crop STOP 闸门 + 距离门收紧

面向 `goat_agent_1.py` 栈的三处改动，全部带**行为中性兜底**并有单测覆盖。灵感来自
[ApexNav](https://github.com/Robotics-STAR-Lab/ApexNav)（RA-L'25，Target-centric
Semantic Fusion）的两条感知思想，外加一处针对本项目 VLM 距离误判的根治。

> **当前状态（重要）**：STOP 判据后续被**大幅简化为恰好三条通用条件**（goal 可见 + 语义相关 +
> `arrived` 到达）。特性 3 的深度门已**整条移除**（agent 不接收深度输入），特性 2 的 CLIP 余弦否决门
> 也**不再参与 STOP 决策**。各节保留原始设计记录，但请以每节顶部的「状态更新」为准。

---

## 特性 1：FOV / 中心加权的物体置信度融合（替换 `object_map_confidence`）

### 动机
旧的 C 轴是"每次复检等权计数"：

```python
# 旧公式
multi_view = int(node.attributes.get("multi_view_count", node.visit_count or 1) or 1)
return 0.30 * float(node.confidence) + min(0.20, 0.04 * multi_view)
```

问题：视野边缘的一瞥、3 米外的模糊检测，和居中的清晰检测计数完全一样。
ApexNav 的 `value_map2d.cpp` 用 `cos²` 的 FOV/中心置信度对每次观测加权后融合。

### 数学（每次观测）
用检测框水平中心 `bbox_center_x` 作为"在视野内的角度"代理：

- `theta = clip((cx - 0.5) / 0.5, -1, 1)`
- `w = cos(theta * pi/2) ** 2` （中心=1，边缘=0，下限 `max(w, 1e-3)`；`cx is None` → `w = 0.5`）

在节点上累积两个量（`fov_view_mass` M、`fov_value_v` V，初值 0）：

- `M_new = M_last + w` （视图质量之和：居中约 +1，边缘约 +0）
- `V_new = (w * v_now + M_last * V_last) / M_new` （检测分的 FOV 加权滑动均值，`v_now = obj confidence`）

### 新的 C 读取（有界到旧的 ~[0, 0.5]，排序尺度不变）
```python
OBJECT_FOV_MASS_K = 2.0

def object_map_confidence(node):
    M = node.attributes.get("fov_view_mass")
    V = node.attributes.get("fov_value_v")
    if M is None or V is None:            # 旧节点回退（与旧公式逐位一致）
        multi_view = int(node.attributes.get("multi_view_count", node.visit_count or 1) or 1)
        return 0.30 * float(node.confidence) + min(0.20, 0.04 * multi_view)
    m = float(M)
    return 0.30 * float(V) + min(0.20, 0.20 * (m / (m + OBJECT_FOV_MASS_K)))
```

居中多帧把佐证项推向 0.20；边缘一瞥几乎不涨。

### 落点
- [conftopo/agents/goat_agent1_planner.py](../agents/goat_agent1_planner.py)：重写
  `object_map_confidence` + 常量 `OBJECT_FOV_MASS_K`。
- [conftopo/agents/goat_agent1_memory.py](../agents/goat_agent1_memory.py)：新增
  `object_fov_weight(bbox)` 与 `accumulate_object_fov_confidence(node, *, bbox, detection_score)`；
  在 `_write_vlm_report` 每次观测后调用，并在 `record_object_verification` 的 VERIFY 成功分支追加一次佐证。

### 实测（在线 trace 验证生效）
object 节点已写入 `fov_view_mass`/`fov_value_v`，行为符合预期：
- 居中被看 17 次 → `mass≈14.7, V≈0.956`
- 边缘瞥 1 次 → `mass≈0.41`（尽管 `V≈0.92`）
- 单次未知中心 → `mass=0.5`

---

## 特性 2：CLIP-crop 与目标嵌入的余弦 —— STOP 附加闸门

> **状态更新（随特性 3 简化，并已清理死代码）**：STOP 已收敛为**恰好三条通用条件**
> （goal 可见 + 相关 + `arrived`，见特性 3），**CLIP 余弦否决门彻底移除**——`can_stop`
> 不再接收 `clip_gate` 参数，`_make_clip_stop_gate()` 与 `clip_gate` 透传链路已删除
> （`config.stop_clip_*` 仍保留供 CLIP 候选/其他用途）。下文仅作历史设计记录。

### 动机
旧的 STOP 只由 VLM 报告单方面决定。ApexNav 的 `get_object_with_itm` 在检测后
裁剪目标框，跑一次**独立** ITM 复核。本项目已有 CLIP crop encoder + goal embedding，
因此加一路独立余弦闸门，砍掉"VLM 自说自话"的误停。

> 以下为历史设计（相关代码已删除）。

### 控制层（保持"哑"，阈值不落在这里）
[conftopo/agents/goat_agent1_control.py](../agents/goat_agent1_control.py)：

- 曾有 `StopVerifier.can_stop(report, goal, clip_gate=None)`：`(False, cos)` 否决 STOP、
  `(True, cos)` 放行、`None` 行为中性；经 `StateMachine.act(...)` 透传。
- 现已移除：`can_stop` 不再有 `clip_gate` 参数。

### 智能体闭包（按 goal_type 选阈值）
曾在 [conftopo/agents/goat_agent_1.py](../agents/goat_agent_1.py) 的 `_make_clip_stop_gate()`：
按 `goal_type` 选 `stop_clip_min_image` / `stop_clip_min_text` 阈值构造闭包。该函数已删除。

### 实测（在线 trace 验证生效）
rack 停靠段 CLIP 余弦在 0.20 阈值上下抖动：step 21–31 `cos=0.195 < 0.20` → **否决了 11 帧**；
step 20/32/33 `cos≈0.209 > 0.20` → 放行。证明闸门确实改变了行为。

---

## 特性 3：STOP 近距判据 —— **纯到达**（最终形态，深度已移除）

> **状态更新**：这条判据一度收敛到"到达 + 中心深度 ≤ 1.0m"（v6 曾开 Habitat 深度传感器），
> 但 v6 实测暴露该深度阈值**结构性地不可满足**，遂将深度**整条移除**，STOP 收敛为**纯到达式**。
> 下面先记演进，再记最终形态。

### 演进
这条判据迭代了很多版：
- v1/v2：放开 `range_bin=near` → 3.14m 误停；
- 收紧到 `very_near/close` → v3 贴到 0.05m 仍报 `near`，停不下来；
- v4：保留的 `very_near/close` 捷径被 VLM 在 3.5m 误报 `very_near` 钻空子 → 远处误停；
- 去掉 range_bin，改「到达 + y2 触地」→ 挂墙/高处物体（不落地）永远触不了地；
- 「到达 + (y2 触地 或 面积够大)」→ v5 里 shelf 被 VLM 只框成画面上方 0.08 高的细带
  （y2=0.25、面积 0.03），两条都不过 → 又停不下来；
- v6：改「到达 + 中心深度 ≤ 1.0m」→ **`saw_stop=false`，rack 全程停不下来**。逐步 trace 显示：
  居中、相关、CLIP 全过、连 `arrived` 都过，唯一挂掉的永远是深度门（中心深度 ~1.7–1.83m > 1.0m）。

**v6 根因（关键）**：**GOAT 的合法成功视点本身离物体表面就有 ~1.5m**（视点带 standoff；对照 v48
成功停靠时 `goal_stop_viewpoint_geodesic=0.263m` 却 `goal_stop_object_euclidean=1.57m`）。因此
"中心深度 ≤ 1.0m"这种紧阈值**在任何合法视点上都触不了**——要满足就得冲过合法视点贴到物体 1m 内，
而 navmesh 上往往没有这种点。**绝对深度阈与视点几何天生冲突**，且任何"把阈值调到 1.8m 去凑"都是
对特定场景的特殊改造，不通用。

### 最终改动（纯到达，agent 不接收深度）
STOP 收敛为**恰好三条通用条件**（[goat_agent1_control.py](../agents/goat_agent1_control.py)
`StopVerifier.can_stop`）：

```python
# 通用 STOP 门：看得到目标 + 确认是目标 + 已到达（导航到极限）
ok = bool(report.goal_visible) and relevance >= 0.70 and bool(arrived)
```

其中唯一的"够不够近"信号是 **`arrived`（位姿停滞到达）**——尺度公平、摆放公平、无需按场景调参：

```python
ARRIVAL_STALL_WINDOW = 4     # 检查最近 4 步
ARRIVAL_STALL_EPS    = 0.30  # 窗口内水平位移 < 0.30m 视为停滞
ARRIVAL_MIN_FORWARD  = 2     # 窗口内至少 2 次前进尝试（排除原地转向）
```

同步移除的东西（**agent 已完全不接收深度输入**）：
- 删除 Habitat `depth_sensor`（[run_goat_minimal.py](../../scripts/run_goat_minimal.py) `make_sim` 只留 RGB）；
- 删除 agent 侧 `_target_center_depth()`、`self._depth`、观测 dict 的 `"depth"` 键
  （[run_goat_agent1_smoke_v2.py](../../scripts/run_goat_agent1_smoke_v2.py) 不再快照深度）；
- 删除 `object_proximity_ok` / `STOP_DEPTH_MAX` / `config.perception.stop_depth_max`。

同时也砍掉了 v6 里叠加的其余 STOP 门（水平居中门、bbox 置信门、CLIP 余弦否决门）——它们的合取正是
"条件过多、合法停靠不可达"的来源。`confirm_action` 的 hold 判断也从"深度到达"改为纯 `arrived`。

### 为什么 range_bin / bbox 几何 / 绝对深度都不作停靠依据
- `range_bin` 双向乱报：3.14m 报 `near`、0.05m 也只报 `near`、3.5m 竟报 `very_near`；
- bbox **绝对尺寸**对不同大小物体不公平（床远处填满、杯子贴脸也不大）；
- bbox **y2 触地**假设物体落地，挂墙/高处物体（电视、画、挂钟、高货架）不成立；
- **绝对深度阈**与 GOAT 视点几何冲突（合法视点 ~1.5m off，紧阈值永远触不了）；
- 而 VLM 框本身又抖又烂（特性 4）。

**`arrived`（导航已到极限）是唯一既尺度/摆放公平、又不依赖任何绝对距离标定的近距信号**，故作为
最终停靠判据。

> 遗留：深度移除后 STOP 该停能停，但 v6 trace 还暴露两个**导航接近**问题（`target_object` 常返 None
> 导致 CONFIRM 饿死、recovery 升顶把已到达的 1.61m 视点丢弃去瞎逛），这属于导航侧，另议。

---

## 特性 4：VLM bbox 鲁棒性（坐标缩放兜底 + 时间中位平滑）

### 动机
实测 VLM（Qwen3-VL）画框**不准**：静止场景下同一物体连续帧 IoU 0.44–1.0、中心 cx 在
0.5↔0.67 反复跳（±0.17）、面积 ±25%、坐标 snap 到 0.15/0.85/0.33 这种圆整值；而且 Qwen 原生
是 **0–1000** 坐标，旧 `normalize_bbox_flexible` 对 >1.05 **整帧丢弃**。

### 改动
1. **坐标缩放自适应**（[heavy_perceiver.py](../perception/heavy_perceiver.py) `normalize_bbox_flexible`）：
   若坐标最大值 >1.05，按 `>100 → /1000`、否则 `/100` 归一（覆盖 0–1000 / 0–100）；只有疑似像素
   坐标（>1000、缺图像尺寸）才拒绝。避免"模型偶尔用 0–1000 → 那帧目标凭空消失"。
2. **bbox 时间中位平滑**（[goat_agent1_control.py](../agents/goat_agent1_control.py)）：
   `StateMachine` 维护最近 `BBOX_SMOOTH_WINDOW=3` 帧目标框，`median_bbox` 取逐坐标中位数，
   `confirm_action` / `can_stop` 用平滑框算中心与 y2 触地，压掉单帧抖动。trace 记 `bbox_smoothed`。

> 注意：VLM 框只适合**粗用**（粗中心、粗触地阈值 + 多帧缓冲）。要精确框应改用
> `HeavyPerceiver`/GroundingDINO（VLM 管语义、检测器管框），列为后续。

---

## 配置（[conftopo/config.py](../config.py) `PerceptionConfig`）

```python
stop_clip_gate_enabled: bool = True
stop_clip_min_image: float = 0.55   # 图-图 余弦
stop_clip_min_text:  float = 0.18   # 图-文 余弦（历程 0.20→0.23→0.18，见下方说明）
```

> 注：`stop_depth_max` 已随特性 3 删除（agent 不再接收深度）。上述 `stop_clip_*` 阈值也已**不再
> 参与 STOP 决策**（见特性 2 状态更新），保留仅为兼容/其他候选用途；以下调参历程仅作历史。

默认保守：只在"有嵌入且明显不匹配"时否决，否则是严格 no-op。

> **`stop_clip_min_text` 调参历程**：初始 0.20 → 因 rack 在 3.14m 误停上调到 0.23 →
> 复盘发现那次误停本质是**距离**问题（`range_bin=near`），已被「到达 + y2 触地」
> 距离门独立挡住（range_bin 不再采信）；而 CLIP **图-文**余弦本身偏低，
> 真目标在 very_near 时也只有 ~0.20–0.21（smoke ls=32–37 实测 `clip_cos=0.199`），
> 0.23 反而把真目标毙了导致停不下来。故回调到 **0.18**：距离门负责挡假阳性，
> 该阈值只需卡在真样本带之下、放行合法的 very_near STOP。

---

## 测试

| 文件 | 覆盖 |
| --- | --- |
| [conftopo/tests/test_object_fov_confidence.py](../tests/test_object_fov_confidence.py) | 递推公式对照（tol 1e-12）、中心 vs 边缘、居中多帧 > 边缘、旧公式回退 ULP 一致 |
| [conftopo/tests/test_stop_clip_gate.py](../tests/test_stop_clip_gate.py) | 通用三条门（可见+相关+到达）：到达则停 / 未到达不停 / 不可见不停 / 非目标标签 missing / 偏离中心仍停 / 挂墙物体仍停 / 低 CLIP 不再否决 / bbox 缩放与中位平滑 |
| [conftopo/tests/test_active_topo_phase_d.py](../tests/test_active_topo_phase_d.py) | VERIFY 闭环（样本 range_bin 由 `near`→`close`）|

回归：`test_planner_rca_split` / `test_active_topo_phase_{c,d,e,f}` 全绿。

> 注：`goat` 环境未装 pytest，测试通过各文件的 `__main__` 直接运行验证。

---

## 运行 / 验证

VLM 服务（本机，`vllm` 环境，模型已缓存）：

```bash
conda activate vllm
unset VLLM_USE_MODELSCOPE          # 避免走被封的阿里云内网 OSS
HF_HUB_OFFLINE=1 vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --dtype auto --limit-mm-per-prompt '{"image": 1}' \
  --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8000
```

确定性 smoke（`goat` 环境）：

```bash
conda activate goat
cd /workspace/tangyx7@xiaopeng.com
python scripts/run_goat_agent1_smoke_v2.py \
  --scene 4ok3usBNeis --episode-index 0 \
  --max-goals 2 --steps-per-goal 120 \
  --perception-backend vlm \
  --vlm-api-base http://localhost:8000/v1 \
  --vlm-model Qwen/Qwen3-VL-8B-Instruct \
  --clip-model ViT-B/32 --clip-image-model RN50 \
  --deterministic \
  --output data/logs/goat_topo/agent1_apexnav_borrow_v2/trace.json
```

trace 里应能看到：object 节点的 `fov_view_mass`/`fov_value_v`，以及
`debug.stop_verifier` 的 `goal_visible`/`relevance`/`arrived`/`reason`
（深度移除后不再有 `center_depth`/`near`/`clip_ok` 等字段）。

---

## 未纳入（后续阶段）
- ApexNav 的度量栅格 / SDF / kino-A* / MINCO / MPC —— 与拓扑离散动作栈无关，不借鉴。
- 导航侧的 **FUEL TSP 巡回前沿调度**（对应上次 bed 找不到、`recovery_level` 打满的覆盖问题）
  作为独立阶段另议。
