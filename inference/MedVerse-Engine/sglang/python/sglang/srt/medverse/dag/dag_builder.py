"""DAG construction from parsed medical sections.

MedVerse pre-analyzes the input structure to build the reasoning DAG,
unlike Multiverse which forks at runtime on FINISH_MATCHED_GOAL_TOKEN.
"""

from __future__ import annotations

from sglang.srt.medverse.dag.dag_node import DAG, DAGNode, NodeStatus
from sglang.srt.medverse.section_types import MedicalSection, MedicalSectionType
from sglang.srt.medverse.utils.config import DAGConfig
from sglang.srt.medverse.utils.constants import MAX_DAG_DEPTH


class DAGBuilder:
    """Build a reasoning DAG from a list of parsed MedicalSection objects."""

    def build(
        self,
        sections: list[MedicalSection],
        token_ids_map: dict[MedicalSectionType, list[int]] | None = None,
        config: DAGConfig | None = None,
    ) -> DAG:
        cfg = config or DAGConfig()
        token_ids_map = token_ids_map or {}
        dag = DAG()
        node_id_counter = 0

        root_sections = [s for s in sections if s.section_type == MedicalSectionType.HPI]
        parallel_sections = [s for s in sections if s.section_type.is_independent]
        merge_sections = [
            s for s in sections
            if s.section_type.is_evidence_dependent and not s.section_type.is_terminal
        ]
        terminal_sections = [s for s in sections if s.section_type.is_terminal]

        root_section_type = (
            root_sections[0].section_type if root_sections else MedicalSectionType.HPI
        )
        root_node = DAGNode(
            node_id=node_id_counter,
            section_type=root_section_type,
            token_ids=token_ids_map.get(root_section_type, []),
            parent_ids=[],
            child_ids=[],
            position_offset=0,
            depth=0,
        )
        dag.add_node(root_node)
        node_id_counter += 1

        max_depth = min(cfg.max_depth, MAX_DAG_DEPTH)

        sibling_node_ids: list[int] = []
        for section in parallel_sections[: cfg.max_parallel_branches]:
            depth = 1
            if depth > max_depth:
                break
            node = DAGNode(
                node_id=node_id_counter,
                section_type=section.section_type,
                token_ids=token_ids_map.get(section.section_type, []),
                parent_ids=[root_node.node_id],
                child_ids=[],
                position_offset=0,
                depth=depth,
            )
            dag.add_node(node)
            root_node.child_ids.append(node.node_id)
            sibling_node_ids.append(node.node_id)
            node_id_counter += 1

        merge_parent_ids = sibling_node_ids if sibling_node_ids else [root_node.node_id]

        merge_node_ids: list[int] = []
        for section in merge_sections:
            depth = 2
            if depth > max_depth:
                break
            node = DAGNode(
                node_id=node_id_counter,
                section_type=section.section_type,
                token_ids=token_ids_map.get(section.section_type, []),
                parent_ids=list(merge_parent_ids),
                child_ids=[],
                position_offset=0,
                depth=depth,
            )
            dag.add_node(node)
            for pid in merge_parent_ids:
                dag.nodes[pid].child_ids.append(node.node_id)
            merge_node_ids.append(node.node_id)
            node_id_counter += 1

        terminal_parent_ids = merge_node_ids if merge_node_ids else merge_parent_ids
        for section in terminal_sections:
            depth = (
                max(dag.nodes[pid].depth for pid in terminal_parent_ids if pid in dag.nodes) + 1
                if terminal_parent_ids
                else 1
            )
            if depth > max_depth:
                break
            node = DAGNode(
                node_id=node_id_counter,
                section_type=section.section_type,
                token_ids=token_ids_map.get(section.section_type, []),
                parent_ids=list(terminal_parent_ids),
                child_ids=[],
                position_offset=0,
                depth=depth,
            )
            dag.add_node(node)
            for pid in terminal_parent_ids:
                dag.nodes[pid].child_ids.append(node.node_id)
            node_id_counter += 1

        dag.validate()
        return dag
