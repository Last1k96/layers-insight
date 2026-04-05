/**
 * Spatial index for edge proximity queries.
 * Stores tessellated edge polylines in a grid for fast closest-edge lookup.
 */
import type { GraphEdge, GraphNode } from '../../stores/types';
import { evaluateEdgeCurve } from './edgesPipeline';

interface EdgePolyline {
  edgeIndex: number;
  points: { x: number; y: number }[];
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export class EdgeHitIndex {
  private cellSize: number;
  private cells = new Map<string, EdgePolyline[]>();

  constructor(cellSize = 400) {
    this.cellSize = cellSize;
  }

  build(
    edges: GraphEdge[],
    nodes: GraphNode[],
    nodeSize: (id: string) => { width: number; height: number },
  ): void {
    this.cells.clear();

    const nodeMap = new Map<string, GraphNode>();
    for (const n of nodes) nodeMap.set(n.id, n);

    for (let i = 0; i < edges.length; i++) {
      const points = evaluateEdgeCurve(edges[i], nodeMap, nodeSize);
      if (points.length < 2) continue;

      // Compute bounding box
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const p of points) {
        if (p.x < minX) minX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.x > maxX) maxX = p.x;
        if (p.y > maxY) maxY = p.y;
      }

      const poly: EdgePolyline = { edgeIndex: i, points, minX, minY, maxX, maxY };

      // Insert into grid cells that the AABB overlaps
      const cMinX = Math.floor(minX / this.cellSize);
      const cMinY = Math.floor(minY / this.cellSize);
      const cMaxX = Math.floor(maxX / this.cellSize);
      const cMaxY = Math.floor(maxY / this.cellSize);

      for (let cx = cMinX; cx <= cMaxX; cx++) {
        for (let cy = cMinY; cy <= cMaxY; cy++) {
          const key = `${cx},${cy}`;
          let cell = this.cells.get(key);
          if (!cell) {
            cell = [];
            this.cells.set(key, cell);
          }
          cell.push(poly);
        }
      }
    }
  }

  /** Find the closest edge to (gx, gy) within radius, or null. */
  query(gx: number, gy: number, radius: number): { edgeIndex: number; dist: number } | null {
    const cs = this.cellSize;
    const cx = Math.floor(gx / cs);
    const cy = Math.floor(gy / cs);

    let bestDist = radius;
    let bestIndex = -1;
    const seen = new Set<number>();

    // Check the 3x3 neighborhood of cells
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const cell = this.cells.get(`${cx + dx},${cy + dy}`);
        if (!cell) continue;

        for (const poly of cell) {
          if (seen.has(poly.edgeIndex)) continue;
          seen.add(poly.edgeIndex);

          // Quick AABB reject
          if (gx < poly.minX - radius || gx > poly.maxX + radius ||
              gy < poly.minY - radius || gy > poly.maxY + radius) continue;

          // Point-to-polyline distance
          const d = pointToPolylineDist(gx, gy, poly.points);
          if (d < bestDist) {
            bestDist = d;
            bestIndex = poly.edgeIndex;
          }
        }
      }
    }

    return bestIndex >= 0 ? { edgeIndex: bestIndex, dist: bestDist } : null;
  }
}

function pointToPolylineDist(px: number, py: number, pts: { x: number; y: number }[]): number {
  let minDist = Infinity;
  for (let i = 0; i < pts.length - 1; i++) {
    const d = pointToSegmentDist(px, py, pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y);
    if (d < minDist) minDist = d;
  }
  return minDist;
}

function pointToSegmentDist(
  px: number, py: number,
  ax: number, ay: number,
  bx: number, by: number,
): number {
  const dx = bx - ax;
  const dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq < 0.0001) {
    const ex = px - ax, ey = py - ay;
    return Math.sqrt(ex * ex + ey * ey);
  }
  let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
  if (t < 0) t = 0;
  else if (t > 1) t = 1;
  const cx = ax + t * dx - px;
  const cy = ay + t * dy - py;
  return Math.sqrt(cx * cx + cy * cy);
}
