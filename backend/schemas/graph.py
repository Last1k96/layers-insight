"""Graph-related schemas."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


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
    # Layout positions (set after ELK layout)
    x: float = 0.0
    y: float = 0.0


class GraphEdge(BaseModel):
    """An edge in the computational graph."""
    source: str
    target: str
    source_port: int = 0
    target_port: int = 0


class GraphData(BaseModel):
    """Complete graph with nodes and edges."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]
