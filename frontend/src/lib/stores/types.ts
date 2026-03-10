export interface SessionInfo {
  id: string;
  model_path: string;
  model_name: string;
  created_at: string;
  main_device: string;
  ref_device: string;
  task_count: number;
  success_count: number;
  failed_count: number;
}

export interface SessionConfig {
  ov_path?: string;
  model_path: string;
  input_path?: string;
  main_device: string;
  ref_device: string;
  input_precision: string;
  input_layout: string;
}

export interface SessionDetail {
  id: string;
  config: SessionConfig;
  info: SessionInfo;
  tasks: InferenceTask[];
}

export interface GraphNode {
  id: string;
  name: string;
  type: string;
  shape?: (number | string)[];
  element_type?: string;
  category: string;
  color: string;
  attributes: Record<string, any>;
  x: number;
  y: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  source_port: number;
  target_port: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface DeviceResult {
  device: string;
  output_shapes: number[][];
  dtype?: string;
  min_val?: number;
  max_val?: number;
  mean_val?: number;
  std_val?: number;
}

export interface AccuracyMetrics {
  mse: number;
  max_abs_diff: number;
  cosine_similarity: number;
}

export type TaskStatus = 'waiting' | 'executing' | 'success' | 'failed';

export interface InferenceTask {
  task_id: string;
  session_id: string;
  node_id: string;
  node_name: string;
  node_type: string;
  status: TaskStatus;
  stage?: string;
  error_detail?: string;
  main_result?: DeviceResult;
  ref_result?: DeviceResult;
  metrics?: AccuracyMetrics;
}

export interface TaskStatusMessage {
  type: 'task_status';
  task_id: string;
  node_id: string;
  node_name: string;
  status: TaskStatus;
  stage?: string;
  error_detail?: string;
  metrics?: AccuracyMetrics;
  main_result?: DeviceResult;
  ref_result?: DeviceResult;
}
