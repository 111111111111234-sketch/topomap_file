# ConfTopo 文档导栏

ConfTopo（置信度感知的语义拓扑记忆导航智能体）全部设计 / 实现 / 报告文档的索引。
**不知道从哪看起，就按下面「推荐阅读顺序」走。**

> 当前在用的实现是 `goat_agent_1.py` 栈（`ConfTopoGOATAgent`）。看代码结构直接读
> [CONFTOPO_CODEBASE_GUIDE.md](CONFTOPO_CODEBASE_GUIDE.md)。

---

## 推荐阅读顺序

1. [CONFTOPO_DESIGN_SUMMARY.md](CONFTOPO_DESIGN_SUMMARY.md) — 先看"这是什么、为什么这么做"。
2. [CONFTOPO_METHOD_OVERVIEW.md](CONFTOPO_METHOD_OVERVIEW.md) — 方法思路（置信度 / 拓扑记忆的核心机制）。
3. [CONFTOPO_CODEBASE_GUIDE.md](CONFTOPO_CODEBASE_GUIDE.md) — 落到代码：每个模块 / 文件的作用。
4. 专题按需查（感知、导航、Phase 报告，见下）。

---

## 一、总览与设计（先读这些）

| 文档 | 说明 | 更新 |
| --- | --- | --- |
| [CONFTOPO_DESIGN_SUMMARY.md](CONFTOPO_DESIGN_SUMMARY.md) | 项目总体设计总结：定位、动机、整体方案 | 2026-06-23 |
| [CONFTOPO_METHOD_OVERVIEW.md](CONFTOPO_METHOD_OVERVIEW.md) | 方法思路整理：核心问题与置信度 / 拓扑记忆机制 | 2026-06-23 |
| [PLAN.md](PLAN.md) | 立项方案（Confidence-Aware Semantic-Topological Memory 全案） | 2026-06-09 |

## 二、代码与实现（改代码看这些）

| 文档 | 说明 | 更新 |
| --- | --- | --- |
| [CONFTOPO_CODEBASE_GUIDE.md](CONFTOPO_CODEBASE_GUIDE.md) | **代码库完整指南**：当前 `goat_agent_1` 栈的架构 + 逐文件作用 + 历史遗留清单 | 2026-07-06 |
| [ACTIVE_TOPO_DESIGN.md](ACTIVE_TOPO_DESIGN.md) | **ActiveTopo A–G 分期改造**：任务条件化、置信度演化拓扑图的最终目标 + 每期改动与落点 | 2026-07-06 |
| [APEXNAV_PERCEPTION_BORROW.md](APEXNAV_PERCEPTION_BORROW.md) | 最近感知改动：FOV 置信度融合 / CLIP-crop STOP 闸门 / 距离门收紧 | 2026-07-06 |
| [CONFTOPO_VLM_MULTI_AGENT_PLAN.md](CONFTOPO_VLM_MULTI_AGENT_PLAN.md) | VLM 感知与 Multi-Agent 改造计划 | 2026-06-15 |

## 三、专题：感知与导航

| 文档 | 说明 | 更新 |
| --- | --- | --- |
| [VLM_SPATIAL_ANCHOR.md](VLM_SPATIAL_ANCHOR.md) | VLM 粗空间语义锚点（把 VLM 观测挂到拓扑锚点上） | 2026-06-16 |
| [MULTIGOAL_NAVIGATION.md](MULTIGOAL_NAVIGATION.md) | 多目标导航逻辑（GOAT 多子目标切换与流程） | 2026-06-23 |
| [TOPO_NAVIGATION.md](TOPO_NAVIGATION.md) | 拓扑导航逻辑与 agent step loop | 2026-06-15 |

## 四、阶段报告：Phase 3

| 文档 | 说明 | 更新 |
| --- | --- | --- |
| [PHASE3_DYNAMIC_SEMANTIC_MEMORY.md](PHASE3_DYNAMIC_SEMANTIC_MEMORY.md) | 长距离动态语义拓扑记忆设定（近处保 object、远处折叠） | 2026-06-15 |
| [PHASE3_SPL_OPTIMIZATION_FINAL.md](PHASE3_SPL_OPTIMIZATION_FINAL.md) | SPL 优化最终报告（object anchor 保留等） | 2026-06-11 |
| [PHASE3_SPL_OPTIMIZATION_REPORT.md](PHASE3_SPL_OPTIMIZATION_REPORT.md) | SPL 优化过程报告（文件级改动明细） | 2026-06-11 |

## 五、历史 / 重复副本（一般不用看）

- [PLAN copy.md](PLAN%20copy.md) — 与 [PLAN.md](PLAN.md) 基本相同的副本。
- [PHASE3_DYNAMIC_SEMANTIC_MEMORY copy.md](PHASE3_DYNAMIC_SEMANTIC_MEMORY%20copy.md) — 上面 Phase3 记忆文档的旧副本（内容更短）。

> 提示：`*.md` 里的 `copy` 文件是历史留存副本，以非 copy 版本为准。

---

_导栏最后更新：2026-07-06_
