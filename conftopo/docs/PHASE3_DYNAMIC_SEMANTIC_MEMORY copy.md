# Phase 3 长距离动态语义拓扑记忆设定

> 本文档说明 ConfTopo 在 Phase 3 中采用的长距离动态语义拓扑记忆设定。核心原则是：近处保留细粒度 object-level 记忆，远处压缩为 landmark / room-level 语义摘要，重新靠近后再通过 heavy perception 恢复细节。

## 总体设定

本项目不维护一个固定粒度、无限增长的 TopoMap，而是维护一个动态层级语义拓扑图。智能体在探索过程中，根据距离、时间、任务相关性和置信度，动态调整语义节点的粒度：

```text
近处：object-level 细粒度记忆
中距离：landmark-level 导航锚点记忆
远距离 / 长时间未访问：room-level / region-level 空间摘要
```

这个设定的目标是让地图在当前可行动区域保持足够细，在远距离区域只保留对长期导航有用的高层语义，从而避免 object / landmark 观测历史无限增长。

## 1. 近处保留 object-level 细节

当智能体靠近某一区域，或者该区域与当前目标强相关时，TopoMap 保留细粒度 object 信息。例如：

```text
rack, chair, table, sink, cabinet, sofa, bed
```

object 节点用于目标确认和精确 grounding。当前实现中，object 节点可以保存：

```text
bbox_observations
detection_scores
multi_view_count
observed_at waypoint
view_heading
first_seen_step / last_seen_step
room_context / room_contexts
confidence
granularity
```

这一层可以回答：

```text
目标 rack 在哪个 waypoint 附近？
它被看到过几次？
最高和平均检测置信度是多少？
最近一次在哪个视角看到？
它属于哪个 room context？
```

当前 Phase 3 已接入 object-level heavy perception：RGB 图像按需触发 GroundingDINO，检测结果通过 `upsert_object_observation()` 写入 `DynamicTopoMap`，并基于 label、位置、room context 和多视角一致性进行合并。

## 2. 中距离压缩为 landmark-level

当 object 逐渐远离智能体，或者长时间没有被重新观测，但仍然有导航价值时，系统不再保留完整 bbox 历史，而是将其降为 landmark-level 记忆。

例如：

```text
rack / cabinet / table / door
```

可以从 object-level 降为：

```text
semantic landmark / navigation anchor
```

这不是说物体真实类别变成了 landmark，而是说该节点的细节被压缩，只保留 landmark 级别的导航语义。当前实现中，object 节点在中远距离会保留原始 `NodeType.OBJECT`，但 `attributes["granularity"]` 会被标记为 `"landmark"`，并触发 history 压缩。

中距离压缩后保留的信息包括：

```text
粗略位置
语义标签
少量代表观测
置信度摘要
所属 room context
附近 waypoint / observed_at 关系
```

这一层的作用是：

```text
作为导航参照物
辅助回忆某个区域
帮助语言关系对齐
降低长期记忆成本
```

## 3. 远距离进一步聚合为 room-level / region-level

当节点距离很远、长时间未访问、置信度低，或者与当前任务无关时，系统进一步将 object / landmark 信息压缩为 room-level 或 region-level 摘要。

理想设定中，近处原本有：

```text
chair, table, sofa, TV, lamp, cabinet
```

远处可以压缩成：

```text
living room area:
  contains: chair / table / sofa / TV
```

当前代码已经实现了远距离低置信 object 的 `room_level` 粒度标记和观测历史压缩：

```text
granularity = "room_level"
history_compression_reason = "far_low_confidence"
```

当前实现会复用 `NodeType.ROOM` 表示 room / region summary：远距离低置信 object 或远距离 landmark 会被非破坏性地汇入 nearby / same-room summary。原 object / landmark 节点不会被删除，而是压缩 history，并在 summary 中记录 `contains_labels`、`contains_node_ids`、`summary_observations`、`source_granularities` 和 `last_summary_update_step`。

## 4. 重新靠近时恢复细节

动态 TopoMap 的目标不是单向压缩，而是 coarse-to-fine 可恢复：

```text
agent 靠近粗粒度区域
-> 触发 heavy perception
-> 重新检测 object / landmark
-> 与旧 summary 匹配
-> 恢复或新建 object-level 节点
-> 更新 confidence 和 observation history
```

当前实现已经具备重新靠近时按需触发 heavy perception 的基础能力，包括：

```text
heavy_goal_warmup_steps
heavy_interval
heavy_goal_sim_threshold
heavy_on_frontier
heavy_low_object_confidence
```

当前实现已经支持基于 coarse summary 的恢复触发：当 agent 靠近 `summary_type="room_region"` 的 ROOM 节点时，会以 `coarse_summary_context` 作为 reason 触发 heavy perception。若检测到的 object label 命中 summary 中的 `contains_labels`，则新建或合并的 object 节点会记录：

```text
recovered_from_summary = true
summary_node_id
recovered_step
```

## 5. 动态更新依据

每个语义节点可以根据动态分数决定当前粒度：

```text
detail_score =
  距离近
+ 最近被观测
+ 当前任务相关
+ 置信度高
+ 多视角一致
- 距离远
- 长时间未访问
- 低置信度
- 重复节点过多
```

然后根据分数决定：

```text
detail_score 高：
  保持 object-level

detail_score 中：
  降为 landmark-level

detail_score 低：
  聚合到 room-level / region-level
```

可以概括成：

```text
near + relevant + confident -> fine memory
far + stale + low confidence -> coarse memory
```

当前代码中的 `adaptive_granularity()` 使用距离和 confidence 做初步粒度管理：

```text
near object -> granularity = "object"
mid/far object -> granularity = "landmark"
far low-confidence object -> granularity = "room_level"
far landmark -> compress landmark history
```

## 6. object / landmark / room 的角色

```text
object：具体目标，粒度最细
例如 rack, chair, sink, table
用于目标确认和精确 grounding

landmark：导航锚点，中等粒度
例如 doorway, corner, stairs, fireplace, cabinet
用于定位、参照和区域连接

room：空间区域，粒度最粗
例如 kitchen, bedroom, bathroom, living room
用于长期空间组织和远距离摘要
```

在本设定中，它们不是固定不变的静态层级，而是可以随着时间和距离发生粒度变化：

```text
object -> landmark-level summary -> room-level summary
landmark -> room-level / region-level summary
room -> 保留区域级长期记忆
```

当前 Phase 3 已实现 object / landmark 节点的粒度标记、history 压缩、room / region summary 聚合，以及靠近 summary 后通过 heavy perception 恢复 object-level 细节的基础机制。

## 当前实现状态

| 能力 | 状态 | 证据 |
|------|------|------|
| CLIP 轻量语义感知 | 已实现 | `LightPerceiver` 每步运行 |
| GroundingDINO object-level heavy perception | 已实现 | `HeavyPerceiver` + `GroundingDINOBackend` |
| 常用 object vocabulary 检测 | 已实现 | `DEFAULT_HEAVY_OBJECT_VOCABULARY` |
| object 检测写入 DynamicTopoMap | 已实现 | `upsert_object_observation()` |
| object 多视角合并 | 已实现 | `multi_view_count` / `bbox_observations` |
| confidence 多因子更新 | 已实现 | `compute_semantic_confidence()` |
| object 远距离 history 压缩 | 已实现 | `history_compressed` + `mid_or_far_object` / `far_low_confidence` |
| landmark 远距离 history 压缩 | 已实现 | `history_compressed` + `far_landmark` |
| 显式 room / region summary 节点聚合 | 已实现 | `summary_type="room_region"` + `contains_labels` |
| 从 summary 恢复 object-level 子节点 | 已实现（基础版） | `coarse_summary_context` + `recovered_from_summary` |

## 字段对齐

当前实验 trace 中可以观察到以下字段：

```text
granularity
history_compressed
history_compression_reason
history_original_observation_count
history_kept_observation_count
history_best_confidence
history_mean_confidence
bbox_observations
multi_view_count
room_context
last_seen_step
summary_type
contains_labels
contains_node_ids
summary_observations
source_granularities
last_summary_update_step
recovered_from_summary
summary_node_id
recovered_step
```

常见压缩原因：

```text
far_landmark
mid_or_far_object
far_low_confidence
```

这些字段说明当前系统已经具备动态压缩机制的初步效果：远距离 object / landmark 不再保留完整历史，而是保留少量代表观测和摘要统计。

## 验证方式

相关测试位于：

```text
conftopo/tests/test_phase3.py
```

重点测试包括：

```text
test_far_object_history_is_compressed
test_far_landmark_history_is_compressed
test_far_object_is_added_to_room_summary
test_far_landmark_is_added_to_room_summary
test_summary_observations_are_bounded
test_goat_agent_heavy_triggers_near_coarse_summary
test_heavy_detection_marks_recovered_from_summary
test_upsert_object_observation_merges_multiview
test_goat_agent_heavy_trigger_and_cooldown
```

也可以从生成的 trace 中检查：

```text
history_compressed = true
granularity = object / landmark / room_level
history_compression_reason = far_landmark / mid_or_far_object / far_low_confidence
summary_type = room_region
recovered_from_summary = true
```

## 最终总结

本项目采用动态层级语义拓扑图来支持长距离探索和长期记忆。智能体附近和当前任务相关区域保留细粒度 object-level 记忆，包括 bbox、多视角观测、置信度和观测历史；随着节点距离增大、长时间未访问或任务相关性降低，object 节点会被压缩为 landmark-level 导航锚点，landmark 进一步压缩为 room-level 或 region-level 空间摘要。这样，TopoMap 在近处具有丰富细节，在远处只保留大的空间语义结构。当智能体重新靠近某个粗粒度区域时，再触发 heavy perception 更新或恢复 object / landmark 细节，从而实现“近处细、远处粗、可恢复”的长距离动态记忆管理。

一句话概括：

```text
动态 TopoMap 不是无限保存所有细节，而是根据距离、时间和任务相关性动态调整语义粒度：近处展开 object，远处压缩成 landmark / room，重新靠近时再恢复细节。
```
