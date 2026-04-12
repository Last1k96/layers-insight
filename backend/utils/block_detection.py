"""Detect repeated transformer blocks in large neural network graphs."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class BlockInfo:
    index: int
    node_ids: list[str] = field(default_factory=list)


@dataclass
class BlockStructure:
    blocks: dict[int, BlockInfo] = field(default_factory=dict)
    infra_node_ids: list[str] = field(default_factory=list)
    block_of: dict[str, int] = field(default_factory=dict)


BLOCK_PATTERNS = [
    re.compile(r'/layers\.(\d+)/'),
    re.compile(r'/blocks\.(\d+)/'),
    re.compile(r'/decoder\.layer\.(\d+)/'),
    re.compile(r'/encoder\.layer\.(\d+)/'),
    re.compile(r'(?:^|/)h\.(\d+)/'),
]

KV_PATTERNS = [
    re.compile(r'past_key_values\.(\d+)\.'),
    re.compile(r'present\.(\d+)\.'),
]


def detect_blocks(
    nodes: list[dict],
    edges: list[dict],
    max_absorption_rounds: int = 20,
) -> BlockStructure:
    """Detect transformer blocks via name regex + iterative neighbor absorption.

    Returns a BlockStructure with every node assigned to either a block or
    the infra set.
    """
    preds: dict[str, set[str]] = defaultdict(set)
    succs: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        preds[e['target']].add(e['source'])
        succs[e['source']].add(e['target'])

    block_of: dict[str, int] = {}

    # Phase 1: regex scan on node names
    for n in nodes:
        name = n.get('name', '')
        for pat in BLOCK_PATTERNS:
            m = pat.search(name)
            if m:
                block_of[n['id']] = int(m.group(1))
                break

    # Phase 2: assign KV-cache / present nodes by index
    for n in nodes:
        if n['id'] in block_of:
            continue
        name = n.get('name', '')
        for pat in KV_PATTERNS:
            m = pat.search(name)
            if m:
                block_of[n['id']] = int(m.group(1))
                break

    # Phase 3: iterative absorption — if ALL neighbors of a node belong to
    # the same block, absorb it into that block
    for _ in range(max_absorption_rounds):
        absorbed = 0
        for n in nodes:
            nid = n['id']
            if nid in block_of:
                continue
            neighbor_blocks: set[int] = set()
            for p in preds[nid]:
                if p in block_of:
                    neighbor_blocks.add(block_of[p])
            for s in succs[nid]:
                if s in block_of:
                    neighbor_blocks.add(block_of[s])
            if len(neighbor_blocks) == 1:
                block_of[nid] = next(iter(neighbor_blocks))
                absorbed += 1
        if absorbed == 0:
            break

    # Build result
    blocks: dict[int, BlockInfo] = {}
    for nid, bi in block_of.items():
        if bi not in blocks:
            blocks[bi] = BlockInfo(index=bi)
        blocks[bi].node_ids.append(nid)

    infra_ids = [n['id'] for n in nodes if n['id'] not in block_of]

    return BlockStructure(blocks=blocks, infra_node_ids=infra_ids, block_of=block_of)


def extract_block_subgraph(
    block: BlockInfo,
    nodes: list[dict],
    edges: list[dict],
    block_of: dict[str, int],
) -> tuple[list[dict], list[dict]]:
    """Extract the subgraph for a single block (nodes + intra-block edges)."""
    member_set = set(block.node_ids)
    sub_nodes = [n for n in nodes if n['id'] in member_set]
    sub_edges = [
        e for e in edges
        if e['source'] in member_set and e['target'] in member_set
    ]
    return sub_nodes, sub_edges


def get_block_signature(
    block: BlockInfo,
    nodes: list[dict],
    edges: list[dict],
    block_of: dict[str, int],
) -> tuple:
    """Compute a structural signature for a block.

    Two blocks with the same signature have isomorphic internal structure
    and can share a single layout (just offset by Y).
    """
    sub_nodes, sub_edges = extract_block_subgraph(block, nodes, edges, block_of)

    # Sort nodes by type for a canonical ordering
    type_counts = defaultdict(int)
    for n in sub_nodes:
        type_counts[n.get('type', '?')] += 1

    # Edge pattern: count of (source_type, target_type) pairs
    node_type = {n['id']: n.get('type', '?') for n in sub_nodes}
    edge_pattern = defaultdict(int)
    for e in sub_edges:
        st = node_type.get(e['source'], '?')
        tt = node_type.get(e['target'], '?')
        edge_pattern[(st, tt)] += 1

    return (
        tuple(sorted(type_counts.items())),
        tuple(sorted(edge_pattern.items())),
        len(sub_nodes),
        len(sub_edges),
    )
