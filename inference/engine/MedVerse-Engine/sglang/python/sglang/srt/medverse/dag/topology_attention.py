"""Topology-aware attention mask generation for MedVerse DAG decoding."""

from __future__ import annotations

import torch

from sglang.srt.medverse.dag.dag_node import DAG, DAGNode, NodeStatus


class TopologyAttentionBuilder:
    """Build attention masks that respect the DAG reasoning topology."""

    def build_mask_sparse(
        self,
        dag: DAG,
        node_token_ranges: dict[int, tuple[int, int]],
        total_tokens: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Vectorized mask construction using node-to-token range mappings."""
        ancestors: dict[int, set[int]] = {
            nid: dag.get_ancestors(nid) for nid in dag.nodes
        }
        completed_siblings = self._completed_siblings(dag)

        mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)

        for node_i_id, (i_start, i_end) in node_token_ranges.items():
            ancestors_i = ancestors.get(node_i_id, set())
            siblings_i = completed_siblings.get(node_i_id, set())

            for node_j_id, (j_start, j_end) in node_token_ranges.items():
                if node_j_id == node_i_id:
                    size_i = i_end - i_start
                    for row_offset in range(size_i):
                        mask[i_start + row_offset, i_start : i_start + row_offset + 1] = True
                elif node_j_id in ancestors_i:
                    mask[i_start:i_end, j_start:j_end] = True
                elif node_j_id in siblings_i:
                    mask[i_start:i_end, j_start:j_end] = True

        return mask

    def _completed_siblings(self, dag: DAG) -> dict[int, set[int]]:
        result: dict[int, set[int]] = {}
        for node in dag.nodes.values():
            siblings: set[int] = set()
            for parent in dag.get_parents(node):
                for sibling_id in parent.child_ids:
                    if sibling_id != node.node_id:
                        sibling = dag.nodes.get(sibling_id)
                        if sibling and sibling.status == NodeStatus.COMPLETED:
                            siblings.add(sibling_id)
            result[node.node_id] = siblings
        return result
