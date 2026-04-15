"""Adaptive position index computation for DAG-structured parallel decoding."""

from __future__ import annotations

from sglang.srt.medverse.dag.dag_node import DAG, DAGNode


class PositionManager:
    """Compute adaptive position indices for all nodes in a DAG."""

    def compute(self, dag: DAG) -> dict[int, list[int]]:
        node_position_ranges: dict[int, tuple[int, int]] = {}
        for node in dag.topological_order():
            start = self._compute_start(node, dag, node_position_ranges)
            end = start + max(len(node.token_ids), 1)
            node_position_ranges[node.node_id] = (start, end)
        return {
            nid: list(range(start, end))
            for nid, (start, end) in node_position_ranges.items()
        }

    def get_position_offset(self, node: DAGNode, dag: DAG) -> int:
        node_position_ranges: dict[int, tuple[int, int]] = {}
        for n in dag.topological_order():
            start = self._compute_start(n, dag, node_position_ranges)
            end = start + max(len(n.token_ids), 1)
            node_position_ranges[n.node_id] = (start, end)
            if n.node_id == node.node_id:
                return start
        return 0

    def _compute_start(
        self,
        node: DAGNode,
        dag: DAG,
        ranges: dict[int, tuple[int, int]],
    ) -> int:
        if not node.parent_ids:
            return 0
        parent_ends = [ranges[pid][1] for pid in node.parent_ids if pid in ranges]
        if not parent_ends:
            return 0
        return max(parent_ends)

    def update_node_offsets(self, dag: DAG) -> None:
        """In-place update of DAGNode.position_offset for every node."""
        position_map = self.compute(dag)
        for node_id, positions in position_map.items():
            dag.nodes[node_id].position_offset = positions[0] if positions else 0
