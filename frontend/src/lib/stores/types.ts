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
  folder_size: number;
  sub_sessions: any[];
}

export interface InputConfig {
  name: string;
  shape: number[];
  element_type: string;
  data_type: string;
  source: 'random' | 'file';
  path?: string;
  layout: string;
}

export interface ModelInputInfo {
  name: string;
  shape: number[];
  element_type: string;
}

export interface OvValidationResult {
  valid: boolean;
  devices: string[];
  error: string | null;
}

export interface AppDefaults {
  ov_path?: string;
  model_path?: string;
  input_path?: string;
  cli_inputs?: string[];
  main_device: string;
  ref_device: string;
}

export interface SessionConfig {
  ov_path?: string;
  model_path: string;
  input_path?: string;
  main_device: string;
  ref_device: string;
  input_precision: string;
  input_layout: string;
  inputs?: InputConfig[];
}

export interface SessionDetail {
  id: string;
  config: SessionConfig;
  info: SessionInfo;
  tasks: InferenceTask[];
}

export interface NodeInput {
  name: string;
  port: number;
  shape?: (number | string)[];
  element_type?: string;
  is_const: boolean;
  const_node_name?: string;
}

export interface ConstantData {
  name: string;
  shape: number[];
  dtype: string;
  total_elements: number;
  truncated: boolean;
  stats: { min: number; max: number; mean: number; std: number };
  data: number[];
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
  inputs?: NodeInput[];
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  source_port: number;
  target_port: number;
  waypoints?: { x: number; y: number }[];
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
  sub_session_id?: string | null;
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
  sub_session_id?: string;
}

export interface SubSessionInfo {
  id: string;
  parent_id: string;
  cut_type: 'output' | 'input';
  cut_node: string;
  grayed_nodes: string[];
  ancestor_cuts?: { cut_node: string; cut_type: string }[];
  created_at: string;
  task_count: number;
  success_count: number;
  failed_count: number;
}

export interface SubSessionCreatedMessage {
  type: 'sub_session_created';
  sub_session_id: string;
  parent_sub_session_id?: string;
  cut_type: string;
  cut_node: string;
  grayed_nodes: string[];
  ancestor_cuts?: { cut_node: string; cut_type: string }[];
}

export interface TensorMeta {
  shape: number[];
  dtype: string;
  size_bytes: number;
  min: number;
  max: number;
  mean: number;
  std: number;
}
