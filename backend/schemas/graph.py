"""Graph-related schemas."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class NodeInput(BaseModel):
    """Describes one input port of a node."""
    name: str  # source node/constant name
    port: int = 0  # target port index
    shape: Optional[list] = None
    element_type: Optional[str] = None
    is_const: bool = False  # True when source is a filtered constant/weight-prep chain


class GraphNode(BaseModel):
    """A node in the computational graph."""
    id: str
    name: str
    type: str
    shape: Optional[list[int]] = None
    element_type: Optional[str] = None
    category: str = "Other"
    color: str = "#78909C"
    attributes: dict[str, Any] = {}
    inputs: list[NodeInput] = []
    # Layout positions (set after ELK layout)
    x: float = 0.0
    y: float = 0.0
    # Node dimensions for layout/rendering consistency
    width: float = 0.0
    height: float = 0.0


class GraphEdge(BaseModel):
    """An edge in the computational graph."""
    source: str
    target: str
    source_port: int = 0
    target_port: int = 0
    waypoints: Optional[list[dict[str, float]]] = None


class GraphData(BaseModel):
    """Complete graph with nodes and edges."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]
