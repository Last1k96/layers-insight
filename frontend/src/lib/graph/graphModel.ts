/**
 * Lightweight directed graph model replacing graphology.
 * Provides the same API surface used by consumers.
 */

export interface NodeAttributes {
  x: number;
  y: number;
  label: string;
  color: string;
  opType: string;
  nodeName: string;
  category: string;
  shape?: (number | string)[];
  elementType?: string;
  attributes: Record<string, any>;
  [key: string]: any;
}

export class GraphModel {
  private nodeAttrs = new Map<string, NodeAttributes>();
  private inAdj = new Map<string, string[]>();
  private outAdj = new Map<string, string[]>();

  addNode(id: string, attrs: NodeAttributes): void {
    this.nodeAttrs.set(id, attrs);
    if (!this.inAdj.has(id)) this.inAdj.set(id, []);
    if (!this.outAdj.has(id)) this.outAdj.set(id, []);
  }

  addEdge(source: string, target: string): void {
    this.outAdj.get(source)?.push(target);
    this.inAdj.get(target)?.push(source);
  }

  hasNode(id: string): boolean {
    return this.nodeAttrs.has(id);
  }

  getNodeAttributes(id: string): NodeAttributes {
    const attrs = this.nodeAttrs.get(id);
    if (!attrs) throw new Error(`Node not found: ${id}`);
    return attrs;
  }

  nodes(): string[] {
    return Array.from(this.nodeAttrs.keys());
  }

  inNeighbors(id: string): string[] {
    return this.inAdj.get(id) ?? [];
  }

  outNeighbors(id: string): string[] {
    return this.outAdj.get(id) ?? [];
  }

  size(): number {
    return this.nodeAttrs.size;
  }
}
