"""DAG node definition for MedVerse parallel decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from sglang.srt.medverse.section_types import MedicalSectionType


class NodeStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PRUNED = "PRUNED"


@dataclass
class DAGNode:
    node_id: int
    section_type: MedicalSectionType
    token_ids: list[int] = field(default_factory=list)
    parent_ids: list[int] = field(default_factory=list)
    child_ids: list[int] = field(default_factory=list)
    position_offset: int = 0
    depth: int = 0
    confidence_score: float = 1.0
    kv_cache_ref: Optional[int] = None
    status: NodeStatus = NodeStatus.PENDING
    output_token_ids: list[int] = field(default_factory=list)
    output_text: str = ""
    finished: bool = False

    def all_parents_completed(self, node_map: dict[int, "DAGNode"]) -> bool:
        return all(
            node_map[pid].status == NodeStatus.COMPLETED
            for pid in self.parent_ids
            if pid in node_map
        )

    def is_root(self) -> bool:
        return len(self.parent_ids) == 0

    def is_merge_node(self) -> bool:
        return len(self.parent_ids) > 1

    def is_leaf(self) -> bool:
        return len(self.child_ids) == 0


@dataclass
class DAG:
    nodes: dict[int, DAGNode] = field(default_factory=dict)
    root_id: int = 0

    def add_node(self, node: DAGNode) -> None:
        self.nodes[node.node_id] = node

    def get_ready_nodes(self) -> list[DAGNode]:
        return [
            n for n in self.nodes.values()
            if n.status == NodeStatus.PENDING and n.all_parents_completed(self.nodes)
        ]

    def get_children(self, node: DAGNode) -> list[DAGNode]:
        return [self.nodes[cid] for cid in node.child_ids if cid in self.nodes]

    def get_parents(self, node: DAGNode) -> list[DAGNode]:
        return [self.nodes[pid] for pid in node.parent_ids if pid in self.nodes]

    def get_ancestors(self, node_id: int) -> set[int]:
        visited: set[int] = set()
        queue = list(self.nodes[node_id].parent_ids)
        while queue:
            pid = queue.pop()
            if pid in visited:
                continue
            visited.add(pid)
            queue.extend(self.nodes[pid].parent_ids)
        return visited

    def topological_order(self) -> list[DAGNode]:
        in_degree: dict[int, int] = {nid: len(n.parent_ids) for nid, n in self.nodes.items()}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result: list[DAGNode] = []
        while queue:
            nid = queue.pop(0)
            result.append(self.nodes[nid])
            for cid in self.nodes[nid].child_ids:
                in_degree[cid] -= 1
                if in_degree[cid] == 0:
                    queue.append(cid)
        return result

    def validate(self) -> None:
        order = self.topological_order()
        if len(order) != len(self.nodes):
            raise ValueError("DAG contains cycles or disconnected nodes")
