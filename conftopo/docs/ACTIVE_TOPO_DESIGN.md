# ActiveTopo：任务条件化、置信度演化的拓扑图（A–G 分期改造）

> 本文说明 ActiveTopo 这次改造**最终想实现什么**，以及 **A–G 每一期具体改了什么、落在哪些文件**。
> 面向已经读过 [CONFTOPO_CODEBASE_GUIDE.md](CONFTOPO_CODEBASE_GUIDE.md) 的读者。

---

## 1. 最终目标

把原本"静态、全量"的拓扑图，升级成一张 **随任务动态聚焦、随观测演化置信度、能自我验证和恢复** 的活地图。

核心设计约束（贯穿所有分期，不可违背）：

- **只有一张持久图 + 一个投影视图**，不是四张物理地图。
- 持久图（`DynamicTopoMap`）只存 **环境事实**：
  - `C` = 地图置信度（map confidence）
  - `A` = 可达性 / 可访问性（reachability / accessibility）
- **任务相关性 `R`** 和每个节点的 **ActivityState**（生命周期状态）只活在 **投影视图** 里，
  **换 goal 就丢弃**，绝不写回持久节点。
- 规划器读 **投影视图**，而不是直接读全图；视图按当前 goal 决定节点的"折叠 / 验证 / 抑制 / 恢复"。

一句话：**图存事实（C/A），视图存任务态（R + 状态）；同一张图，按任务投影出不同的"活跃子图"给规划器用。**

### 节点生命周期状态（`ActivityState`，仅存在于视图）

| 状态 | 含义 | 引入分期 |
|---|---|---|
| `ACTIVE` | 当前参与规划候选 | A/B |
| `SUPPRESSED` | 规划器本就无条件跳过的节点（blocked/consumed/blocked portal） | C |
| `VERIFY` | 高相关但低置信，需再观测后才可信到能 STOP | D |
| `FOLDED` | 暂时隐藏但可恢复（低相关房间区域） | E |

---

## 2. A–G 分期改动一览

| 期 | 目标 | 主要落点 | 回归测试 |
|---|---|---|---|
| **A** | 数据契约与类型：持久图存 `C`/`A`，视图存 `R`+`ActivityState`，换 goal 丢弃 | `goat_agent1_types.py` | — |
| **B** | `ActiveTopoProjector` 投影器：每个节点标 ACTIVE 并标注 `(R,C,A)`，**仅可观测、行为中立** | `goat_agent1_active_topo.py` | — |
| **C** | **SUPPRESS**：把规划器本就跳过的节点标 SUPPRESSED，行为与全图一致但活跃节点数下降 | 投影器 + `goat_agent1_planner.py` | `test_active_topo_phase_c.py` |
| **D** | **VERIFY 闭环**：高相关低置信物体标 VERIFY，再观测更新置信度，状态机抬高 STOP 门槛并回调 | `goat_agent1_control.py` + `goat_agent1_memory.py` | `test_active_topo_phase_d.py` |
| **E** | **FOLD**：与 goal room prior 不匹配的低相关房间，其内部探索目标/低相关物体折叠出候选集 | `goat_agent1_active_topo.py` | `test_active_topo_phase_e.py` |
| **F** | **恢复阶梯**：活跃区探索耗尽时恢复等级 0→1→2 单调升级，放宽折叠去下一相关区域 | `RecoveryState`（`goat_agent1_types.py`） | `test_active_topo_phase_f.py` |
| **G** | **多模态任务相关性**：`object_task_relevance` 按 `goal.goal_type` 切换融合权重 | `goat_agent1_planner.py` | `test_planner_phase_g.py` |

演进逻辑：**A/B 搭骨架且行为中立 → C 安全裁剪 → D 让地图自我验证物体 → E 按任务折叠无关房间 → F 卡住时逐级恢复 → G 让相关性按目标模态自适应。**

---

## 3. 各期详解

### Phase A — 数据契约与类型
定义"图存事实、视图存任务态"的骨架类型（[goat_agent1_types.py](../agents/goat_agent1_types.py)）：

- `ActivityState`（Enum）：`ACTIVE / VERIFY / FOLDED / SUPPRESSED`，仅存在于视图。
- `ActiveNodeInfo`：每个节点的任务标注——`relevance (R)`、`activity_state`、以及为单遍规划方便而镜像自持久节点的 `map_confidence (C)` / `reachability (A)`、`utility`。
- `ProjectionConfig`：投影分类的阈值与开关（`suppress_enabled` / `verify_enabled` / `fold_enabled`、各相关性阈值、`verify_min_view_count` 等）。**默认是退化值（全 ACTIVE）**，后续分期逐步收紧。
- `ActiveTopoView`：goal 作用域的投影视图，事件触发重建（换 goal / 提升 / 验证结果），非每步重建。
- `RecoveryState`：per-goal 的搜索预算 / 恢复阶梯（agent 侧，绝不写节点）。

### Phase B — 投影器（行为中立）
`ActiveTopoProjector`（[goat_agent1_active_topo.py](../agents/goat_agent1_active_topo.py)）遍历持久图，为每个节点算 `(R,C,A)` 并分类。B 期把所有节点标 **ACTIVE**，只做**可观测层**：读全图的规划器行为完全不变。这样先把"投影"这条链路接通、可视化，再在后续分期收紧 `ProjectionConfig` 真正驱动折叠/验证/路由。

### Phase C — SUPPRESS（安全裁剪）
投影器（`suppress_enabled`）只把 **规划器本就无条件跳过** 的节点标为 `SUPPRESSED`：blocked / consumed 的导航节点、blocked 的 portal。因此"用视图门控候选生成"与"忽略视图"产生**完全相同**的 intent 和导航目标，但 `active_node_count` 低于全图。这是把裁剪能力先落地、且保证行为不变的一步。回归：`test_active_topo_phase_c.py`。

### Phase D — VERIFY 闭环（自我验证）
- 投影器把 **高相关、低置信**（单瞥、view count 少于 `verify_min_view_count`）的物体标为 `VERIFY`。
- `memory.record_object_verification`：验证成功→置信度↑；失败→置信度↓ 且写 `strong_negative_evidence`（走 `confidence.py`）。
- 状态机对 VERIFY 目标**抬高 STOP 门槛**，并触发成功/失败回调——这才是真正的闭环（先再观测确认，再决定停不停）。
落点：`goat_agent1_control.py` + `goat_agent1_memory.py`。回归：`test_active_topo_phase_d.py`。

### Phase E — FOLD（按任务折叠无关房间）
与 goal `room_prior` 不匹配、任务相关性低于 `relevance_fold_threshold` 的房间，把其**内部探索目标**（frontier / candidate）和**低相关物体**折叠出候选集（标 `FOLDED`）。房间节点、portal、已访问路点**保留**，骨架仍可通行。约束：**当前所在房间永不折叠**；进入恢复时放宽折叠。回归：`test_active_topo_phase_e.py`。

### Phase F — 恢复阶梯（recovery ladder）
活跃区被探索耗尽（连续 `budget` 步无搜索进展）时，恢复等级 **0 → 1 → 2 单调升级**（一个 goal 内不回退）。升级会记录耗尽区域，并通过投影器**放宽折叠**，让 agent 能到达先前被折叠 / 下一相关的区域。落点：`RecoveryState.maybe_escalate`。回归：`test_active_topo_phase_f.py`。

### Phase G — 多模态任务相关性
`object_task_relevance` 按 `goal.goal_type` 切换融合权重（[goat_agent1_planner.py](../agents/goat_agent1_planner.py) 的 `_relevance_weights` / `_object_retrieval_score`）：

- `image`：视觉为主（与旧的固定融合一致）。
- `category`：强调 label / room。
- `description`：强调属性 + 空间关系（复用 `RelationScorer`）。

回归：`test_planner_phase_g.py`。

---

## 4. 快速验证

```bash
cd /workspace/tangyx7@xiaopeng.com
PYTHONPATH=. conda run -n goat python conftopo/tests/test_active_topo_phase_c.py
PYTHONPATH=. conda run -n goat python conftopo/tests/test_active_topo_phase_d.py
PYTHONPATH=. conda run -n goat python conftopo/tests/test_active_topo_phase_e.py
PYTHONPATH=. conda run -n goat python conftopo/tests/test_active_topo_phase_f.py
PYTHONPATH=. conda run -n goat python conftopo/tests/test_planner_phase_g.py
```

---

## 5. 相关文档
- [CONFTOPO_CODEBASE_GUIDE.md](CONFTOPO_CODEBASE_GUIDE.md) — 各模块与文件职责总览
- [APEXNAV_PERCEPTION_BORROW.md](APEXNAV_PERCEPTION_BORROW.md) — 感知/停靠层的后续改造（FOV 置信度融合、CLIP 停靠闸门、到达+视觉近距判据）
- [README.md](README.md) — docs 导航索引
