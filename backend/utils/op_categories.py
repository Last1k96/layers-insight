"""OpenVINO operation type to category and color mapping."""
from __future__ import annotations

# Category definitions with colors
CATEGORIES: dict[str, str] = {
    "Convolution": "#335588",    # rgb(51,85,136) — Netron "layer"
    "Normalization": "#335544",  # rgb(51,85,68)  — Netron "normalization"
    "Activation": "#702921",     # rgb(112,41,33) — Netron "activation"
    "Pooling": "#335533",        # rgb(51,85,51)  — Netron "pool"
    "Elementwise": "#335544",    # rgb(51,85,68)  — Netron "transform"
    "MatMul": "#335588",         # rgb(51,85,136) — Netron "layer"
    "DataMovement": "#6C4F47",   # rgb(108,79,71) — Netron "shape"
    "Quantization": "#502800",   # rgb(80,40,0)   — Netron "quantization"
    "Reduce": "#454770",         # rgb(69,71,112) — Netron "dropout"
    "Parameter": "#eeeeee",      # Netron "constant" — light gray, dark text
    "Attention": "#783C00",      # rgb(120,60,0)  — Netron "attention"
    "Tensor": "#59423B",         # rgb(89,66,59)  — Netron "tensor"
    "Other": "#333333",          # Default dark charcoal
}

# Op type to category mapping
OP_CATEGORY_MAP: dict[str, str] = {
    # Convolution
    "Convolution": "Convolution",
    "GroupConvolution": "Convolution",
    "DeformableConvolution": "Convolution",
    "ConvolutionBackpropData": "Convolution",
    "GroupConvolutionBackpropData": "Convolution",
    # Normalization
    "BatchNormInference": "Normalization",
    "MVN": "Normalization",
    "NormalizeL2": "Normalization",
    "LRN": "Normalization",
    "GroupNormalization": "Normalization",
    # Activation
    "Relu": "Activation",
    "Sigmoid": "Activation",
    "Tanh": "Activation",
    "Clamp": "Activation",
    "Elu": "Activation",
    "Swish": "Activation",
    "PRelu": "Activation",
    "Mish": "Activation",
    "SoftMax": "Activation",
    "Softmax": "Activation",
    "Gelu": "Activation",
    "HSigmoid": "Activation",
    "HSwish": "Activation",
    "SoftPlus": "Activation",
    "Log": "Activation",
    "Exp": "Activation",
    "Abs": "Activation",
    "Ceiling": "Activation",
    "Floor": "Activation",
    "Round": "Activation",
    "Negative": "Activation",
    "Sign": "Activation",
    "Sin": "Activation",
    "Cos": "Activation",
    "Sqrt": "Activation",
    "Selu": "Activation",
    # Pooling
    "MaxPool": "Pooling",
    "AvgPool": "Pooling",
    "AdaptiveAvgPool": "Pooling",
    "AdaptiveMaxPool": "Pooling",
    # Elementwise
    "Add": "Elementwise",
    "Multiply": "Elementwise",
    "Subtract": "Elementwise",
    "Divide": "Elementwise",
    "Maximum": "Elementwise",
    "Minimum": "Elementwise",
    "Power": "Elementwise",
    "Equal": "Elementwise",
    "Greater": "Elementwise",
    "GreaterEqual": "Elementwise",
    "Less": "Elementwise",
    "LessEqual": "Elementwise",
    "NotEqual": "Elementwise",
    "LogicalAnd": "Elementwise",
    "LogicalOr": "Elementwise",
    "LogicalNot": "Elementwise",
    "FloorMod": "Elementwise",
    "Mod": "Elementwise",
    # MatMul/FC
    "MatMul": "MatMul",
    "FullyConnected": "MatMul",
    # Data Movement
    "Reshape": "DataMovement",
    "Transpose": "DataMovement",
    "Concat": "DataMovement",
    "Split": "DataMovement",
    "VariadicSplit": "DataMovement",
    "StridedSlice": "DataMovement",
    "Slice": "DataMovement",
    "Gather": "DataMovement",
    "GatherElements": "DataMovement",
    "GatherND": "DataMovement",
    "ScatterUpdate": "DataMovement",
    "ScatterNDUpdate": "DataMovement",
    "ScatterElementsUpdate": "DataMovement",
    "Squeeze": "DataMovement",
    "Unsqueeze": "DataMovement",
    "ShapeOf": "DataMovement",
    "Convert": "DataMovement",
    "ConvertLike": "DataMovement",
    "Broadcast": "DataMovement",
    "Tile": "DataMovement",
    "Pad": "DataMovement",
    "Interpolate": "DataMovement",
    "ROIPooling": "DataMovement",
    "PSROIPooling": "DataMovement",
    "ROIAlign": "DataMovement",
    "DepthToSpace": "DataMovement",
    "SpaceToDepth": "DataMovement",
    "BatchToSpace": "DataMovement",
    "SpaceToBatch": "DataMovement",
    "ReverseSequence": "DataMovement",
    "ShuffleChannels": "DataMovement",
    "Select": "DataMovement",
    "TopK": "DataMovement",
    "NonMaxSuppression": "DataMovement",
    "Range": "DataMovement",
    "Einsum": "DataMovement",
    "Eye": "DataMovement",
    # Quantization
    "FakeQuantize": "Quantization",
    "Quantize": "Quantization",
    "Dequantize": "Quantization",
    # Reduce
    "ReduceMean": "Reduce",
    "ReduceSum": "Reduce",
    "ReduceMax": "Reduce",
    "ReduceMin": "Reduce",
    "ReduceProd": "Reduce",
    "ReduceL1": "Reduce",
    "ReduceL2": "Reduce",
    "ReduceLogicalAnd": "Reduce",
    "ReduceLogicalOr": "Reduce",
    # Parameter/Result/Constant
    "Parameter": "Parameter",
    "Result": "Parameter",
    "Constant": "Parameter",
    "ReadValue": "Parameter",
    "Assign": "Parameter",
}


def get_op_category(op_type: str) -> str:
    """Get category name for an op type."""
    return OP_CATEGORY_MAP.get(op_type, "Other")


def get_op_color(op_type: str) -> str:
    """Get hex color for an op type."""
    category = get_op_category(op_type)
    return CATEGORIES[category]
