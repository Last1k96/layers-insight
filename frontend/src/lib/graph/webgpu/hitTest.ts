/**
 * Spatial grid for fast CPU-based node hit testing.
 * Uses axis-aligned bounding boxes (AABBs) in graph space.
 */

interface NodeRect {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

export class SpatialGrid {
  private cellSize: number;
  private cells = new Map<string, NodeRect[]>();
  private allNodes: NodeRect[] = [];

  constructor(cellSize = 200) {
    this.cellSize = cellSize;
  }

  /** Build the grid from node positions and sizes */
  build(nodes: { id: string; x: number; y: number; width: number; height: number }[]): void {
    this.cells.clear();
    this.allNodes = [];

    for (const n of nodes) {
      const rect: NodeRect = { id: n.id, x: n.x, y: n.y, w: n.width, h: n.height };
      this.allNodes.push(rect);

      const minCellX = Math.floor(n.x / this.cellSize);
      const minCellY = Math.floor(n.y / this.cellSize);
      const maxCellX = Math.floor((n.x + n.width) / this.cellSize);
      const maxCellY = Math.floor((n.y + n.height) / this.cellSize);

      for (let cx = minCellX; cx <= maxCellX; cx++) {
        for (let cy = minCellY; cy <= maxCellY; cy++) {
          const key = `${cx},${cy}`;
          let cell = this.cells.get(key);
          if (!cell) {
            cell = [];
            this.cells.set(key, cell);
          }
          cell.push(rect);
        }
      }
    }
  }

  /** Find the node at graph-space point (x, y), or null */
  query(x: number, y: number): string | null {
    const cx = Math.floor(x / this.cellSize);
    const cy = Math.floor(y / this.cellSize);
    const key = `${cx},${cy}`;
    const cell = this.cells.get(key);
    if (!cell) return null;

    // Check all nodes in this cell (reverse order = top-most first)
    for (let i = cell.length - 1; i >= 0; i--) {
      const r = cell[i];
      if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) {
        return r.id;
      }
    }
    return null;
  }
}
