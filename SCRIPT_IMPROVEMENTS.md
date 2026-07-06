# 脚本改进总结：双层语义拓扑记忆支持

## 背景
你更新了 `conftopo/core/dynamic_topo_map.py` 和 `conftopo/agents/goat_agent_1.py`，构建了一个**双层语义拓扑记忆**系统：
- **结构层**：维护 room / object / landmark / portal 等语义知识
- **导航层**：维护 waypoint / frontier / candidate / approach 等可执行路径
- **跨层链接**：绑定结构意图到导航目标的映射

## 脚本改进内容

### 1. 增强 `snapshot_topo` 函数（run_goat_topo_trace.py）
**改进**：从混合捕获改为分层捕获拓扑图

**前**：所有节点混合在一个列表中
```json
{
  "nodes": [...],
  "edges": [...]
}
```

**后**：分离结构层、导航层和跨层链接
```json
{
  "nodes": [...],          // 向后兼容
  "edges": [...],          // 向后兼容
  "structure_layer": {
    "nodes": [房间、物体、地标、传送点],
    "node_count": number
  },
  "navigation_layer": {
    "nodes": [waypoint、frontier、candidate、approach],
    "node_count": number
  },
  "cross_layer": {
    "edges": [结构→导航的映射],
    "edge_count": number
  },
  "layers_summary": {
    "structure_layer_nodes": count,
    "navigation_layer_nodes": count,
    "cross_layer_links": count,
    ...
  }
}
```

### 2. 改进 `_memory_stats` 统计（run_goat_agent1_smoke_v2.py）
**改进**：按层级分类，显示双层架构的工作状态

**前**：混合统计
```python
{
  "total_nodes": 50,
  "visited_waypoints": 10,
  "objects": 5,
  ...
}
```

**后**：分层统计
```python
{
  "total_nodes": 50,
  "structure_layer": {
    "total": 20,
    "rooms": 3,
    "objects": 5,
    "landmarks": 8,
    "portals": 4
  },
  "navigation_layer": {
    "total": 30,
    "visited_waypoints": 10,
    "frontiers": 8,
    "candidate_waypoints": 8,
    "approach_waypoints": 4
  },
  "cross_layer_links": 12,
  "hypotheses": 2,
  "object_memory_pool": 3
}
```

### 3. 添加 `_cross_layer_transition` 函数（run_goat_agent1_smoke_v2.py）
**功能**：捕获每一步的"结构意图 → 导航目标"转换过程

追踪内容：
```python
{
  "structure_intent": {
    "type": "room_search" | "object_anchor" | ...,
    "label": "bedroom",
    "structure_node_id": "n_123",
    "reason": "found_target"
  },
  "navigation_target": "wp_45",  // 对应的导航waypoint
  "target_node_info": {
    "type": "waypoint_visited",
    "label": "approach_point",
    "confidence": 0.85
  },
  "anchor_waypoint": "wp_40"     // 当前锚点
}
```

### 4. 增强 `_summarize_trace` 统计（run_goat_agent1_smoke_v2.py）
**改进**：添加双层架构的工作统计

```python
"smoke_summary": {
  ...
  "dual_layer_stats": {
    "cross_layer_activations": 45,  // 结构意图被激活次数
    "navigation_layer_updates": 80,  // 导航层更新次数
    "structure_layer_proposals": 60  // 结构层提案数
  }
}
```

## 使用场景

### 调试双层架构工作流
查看 trace.json 中的每一步：
```json
{
  "step": 5,
  "cross_layer_transition": {
    "structure_intent": {
      "type": "object_anchor",
      "label": "bed"
    },
    "navigation_target": "wp_12",
    "target_node_info": {
      "type": "waypoint_approach",
      "confidence": 0.92
    }
  },
  "topo": {
    "layers_summary": {
      "structure_layer_nodes": 8,
      "navigation_layer_nodes": 15,
      "cross_layer_links": 7
    }
  }
}
```

### 分析内存使用
```python
final_memory = {
  "structure_layer": {
    "rooms": 4,      # 发现的房间数
    "objects": 12,   # 检测到的物体数
    "landmarks": 18  # 地标数
  },
  "navigation_layer": {
    "visited_waypoints": 35,
    "frontiers": 8,
    "approach_waypoints": 6
  }
}
```

## 运行命令
```bash
python scripts/run_goat_agent1_smoke_v2.py \
  --scene 4ok3usBNeis --episode-index 0 \
  --max-goals 3 --steps-per-goal 120 \
  --perception-backend vlm \
  --vlm-api-base http://192.168.134.154:8000/v1 \
  --vlm-model Qwen/Qwen3-VL-8B-Instruct \
  --output data/logs/goat_topo/agent1_smoke_4ok3usBNeis_ep0_v17/trace.json \
  --viz --viz-stride 2 --fps 6
```

脚本现在会自动：
✅ 分层捕获拓扑图结构  
✅ 记录层级间的转换过程  
✅ 统计双层架构的工作状态  
✅ 生成包含双层信息的完整 trace.json  
✅ 支持可视化时查看层级信息

## 后续改进建议

1. **可视化增强**：在 `visualize_goat_topo_trace.py` 中添加分层可视化
2. **层级同步指标**：添加结构-导航对齐度的度量
3. **失败分析**：追踪层级转换失败的原因
4. **性能监控**：统计层级查询和转换的时间成本
