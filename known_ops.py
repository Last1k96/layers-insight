# known_ops.py

# A single array of unique dark-theme colors:
COLOR_PALETTE = [
    "#303030",  # 0
    "#4CAF50",  # 1
    "#FF9800",  # 2
    "#9C27B0",  # 3
    "#00BCD4",  # 4
    "#E91E63",  # 5
    "#FFC107",  # 6
    "#477F73",  # 7 Green
    "#FF5722"   # 8
]

OPENVINO_OP_COLORS_DARK = {
    #
    # 1) I/O & Data
    #
    "Parameter": COLOR_PALETTE[0],
    "Result":    COLOR_PALETTE[0],
    "Const":  COLOR_PALETTE[0],
    "ReadValue": COLOR_PALETTE[0],
    "Assign":    COLOR_PALETTE[0],
    "FakeQuantize":        COLOR_PALETTE[0],
    "Convert":             COLOR_PALETTE[0],
    "ConvertLike":         COLOR_PALETTE[0],
    "ConvertPromoteTypes": COLOR_PALETTE[0],
    "Pad":                 COLOR_PALETTE[0],

    #
    # 2) Convolution & Deconvolution
    #
    "Convolution":                   COLOR_PALETTE[1],
    "ConvolutionBackpropData":      COLOR_PALETTE[1],
    "GroupConvolution":             COLOR_PALETTE[1],
    "GroupConvolutionBackpropData": COLOR_PALETTE[1],
    "BinaryConvolution":            COLOR_PALETTE[1],
    "DeformableConvolution":        COLOR_PALETTE[1],
    "DeformablePSROIPooling":       COLOR_PALETTE[1],
    "Col2Im":                       COLOR_PALETTE[1],

    #
    # 3) Pooling
    #
    "MaxPool":         COLOR_PALETTE[0],
    "AvgPool":         COLOR_PALETTE[0],
    "AdaptiveAvgPool": COLOR_PALETTE[0],
    "AdaptiveMaxPool": COLOR_PALETTE[0],

    #
    # 4) Activations
    #
    "Clamp":      COLOR_PALETTE[3],
    "Elu":        COLOR_PALETTE[3],
    "Gelu":       COLOR_PALETTE[3],
    "HardSigmoid":COLOR_PALETTE[3],
    "HSigmoid":   COLOR_PALETTE[3],
    "HSwish":     COLOR_PALETTE[3],
    "Mish":       COLOR_PALETTE[3],
    "PReLU":      COLOR_PALETTE[3],
    "Relu":       COLOR_PALETTE[3],
    "ReLU":       COLOR_PALETTE[3],
    "Selu":       COLOR_PALETTE[3],
    "Sigmoid":    COLOR_PALETTE[3],
    "SoftSign":   COLOR_PALETTE[3],
    "Swish":      COLOR_PALETTE[3],
    "Tan":        COLOR_PALETTE[3],
    "Tanh":       COLOR_PALETTE[3],

    #
    # 5) Normalization & BatchNorm
    #
    "BatchNormInference": COLOR_PALETTE[4],
    "LRN":                COLOR_PALETTE[4],
    "MVN":                COLOR_PALETTE[4],
    "GroupNormalization": COLOR_PALETTE[4],

    #
    # 6) Arithmetic (Eltwise, Bitwise, etc.)
    #
    "Add":              COLOR_PALETTE[0],
    "Subtract":         COLOR_PALETTE[0],
    "Multiply":         COLOR_PALETTE[0],
    "Divide":           COLOR_PALETTE[0],
    "Mod":              COLOR_PALETTE[0],
    "FloorMod":         COLOR_PALETTE[0],
    "Power":            COLOR_PALETTE[0],
    "SquaredDifference":COLOR_PALETTE[0],
    "BitwiseAnd":       COLOR_PALETTE[0],
    "BitwiseOr":        COLOR_PALETTE[0],
    "BitwiseXor":       COLOR_PALETTE[0],
    "BitwiseNot":       COLOR_PALETTE[0],
    "BitwiseLeftShift": COLOR_PALETTE[0],
    "BitwiseRightShift":COLOR_PALETTE[0],

    #
    # 7) Comparison
    #
    "Equal":        COLOR_PALETTE[0],
    "NotEqual":     COLOR_PALETTE[0],
    "Greater":      COLOR_PALETTE[0],
    "GreaterEqual": COLOR_PALETTE[0],
    "Less":         COLOR_PALETTE[0],
    "LessEqual":    COLOR_PALETTE[0],

    #
    # 8) Logical
    #
    "LogicalAnd": COLOR_PALETTE[0],
    "LogicalNot": COLOR_PALETTE[0],
    "LogicalOr":  COLOR_PALETTE[0],
    "LogicalXor": COLOR_PALETTE[0],

    #
    # 9) Unary Ops (Abs, Ceil, Floor, Erf, Exp, etc.)
    #
    "Abs":     COLOR_PALETTE[0],
    "Acos":    COLOR_PALETTE[0],
    "Acosh":   COLOR_PALETTE[0],
    "Asin":    COLOR_PALETTE[0],
    "Asinh":   COLOR_PALETTE[0],
    "Atan":    COLOR_PALETTE[0],
    "Atanh":   COLOR_PALETTE[0],
    "Ceiling": COLOR_PALETTE[0],
    "Cos":     COLOR_PALETTE[0],
    "Cosh":    COLOR_PALETTE[0],
    "Exp":     COLOR_PALETTE[0],
    "Erf":     COLOR_PALETTE[0],
    "Floor":   COLOR_PALETTE[0],
    "Log":     COLOR_PALETTE[0],
    "Negative":COLOR_PALETTE[0],
    "Round":   COLOR_PALETTE[0],
    "Sign":    COLOR_PALETTE[0],
    "Sin":     COLOR_PALETTE[0],
    "Sinh":    COLOR_PALETTE[0],
    "Sqrt":    COLOR_PALETTE[0],

    #
    # 10) RNN ops
    #
    "GRUCell":     COLOR_PALETTE[6],
    "LSTMCell":    COLOR_PALETTE[6],
    "RNNCell":     COLOR_PALETTE[6],
    "GRUSequence": COLOR_PALETTE[6],
    "LSTMSequence":COLOR_PALETTE[6],
    "RNNSequence": COLOR_PALETTE[6],

    #
    # 11) MatMul & Similar
    #
    "MatMul":                    COLOR_PALETTE[7],
    "Einsum":                    COLOR_PALETTE[7],
    "ScaledDotProductAttention": COLOR_PALETTE[7],

    #
    # 12) Softmax / LogSoftmax
    #
    "SoftMax":    COLOR_PALETTE[7],
    "LogSoftmax": COLOR_PALETTE[7],

    #
    # 13) Transform / Shape Ops
    #
    "Concat":               COLOR_PALETTE[0],
    "Reshape":              COLOR_PALETTE[0],
    "StridedSlice":         COLOR_PALETTE[0],
    "Squeeze":              COLOR_PALETTE[0],
    "Unsqueeze":            COLOR_PALETTE[0],

    "ShuffleChannels":      COLOR_PALETTE[7],
    "DepthToSpace":         COLOR_PALETTE[7],
    "SpaceToDepth":         COLOR_PALETTE[7],
    "VariadicSplit":        COLOR_PALETTE[7],
    "Transpose":            COLOR_PALETTE[7],
    "Gather":               COLOR_PALETTE[7],
    "GatherElements":       COLOR_PALETTE[7],
    "GatherND":             COLOR_PALETTE[7],
    "GatherTree":           COLOR_PALETTE[7],
    "ScatterElementsUpdate":COLOR_PALETTE[7],
    "ScatterNDUpdate":      COLOR_PALETTE[7],
    "ScatterUpdate":        COLOR_PALETTE[7],
    "Split":                COLOR_PALETTE[7],
    "Roll":                 COLOR_PALETTE[7],
    "Slice":                COLOR_PALETTE[7],
    "SliceScatter":         COLOR_PALETTE[7],
    "Broadcast":            COLOR_PALETTE[7],
    "Tile":                 COLOR_PALETTE[7],

    #
    # 14) Other / Misc (Rotated NMS, DetectionOutput, etc.)
    #
    # Using index 8 for these
    "If":                COLOR_PALETTE[8],
    "Loop":              COLOR_PALETTE[8],
    "TensorIterator":    COLOR_PALETTE[8],
    "DetectionOutput":   COLOR_PALETTE[8],
    "ExperimentalDetectronDetectionOutput_6":           COLOR_PALETTE[8],
    "ExperimentalDetectronGenerateProposalsSingleImage_6": COLOR_PALETTE[8],
    "ExperimentalDetectronROIFeatureExtractor_6":       COLOR_PALETTE[8],
    "ExperimentalDetectronTopKROIs_6":                  COLOR_PALETTE[8],
    "ExperimentalDetectronPriorGridGenerator_6":        COLOR_PALETTE[8],
    "NMSRotated":          COLOR_PALETTE[8],
    "NonMaxSuppression":   COLOR_PALETTE[8],
    "MulticlassNMS":       COLOR_PALETTE[8],
    "MatrixNMS":           COLOR_PALETTE[8],
    "RandomUniform":       COLOR_PALETTE[8],
    "Multinomial":         COLOR_PALETTE[8],
    "GridSample":          COLOR_PALETTE[8],
    "SpaceToBatch":        COLOR_PALETTE[8],
    "BatchToSpace":        COLOR_PALETTE[8],
    "DFT":                 COLOR_PALETTE[8],
    "IDFT":                COLOR_PALETTE[8],
    "RDFT":                COLOR_PALETTE[8],
    "IRDFT":               COLOR_PALETTE[8],
    "STFT":                COLOR_PALETTE[8],
    "ROIAlign":            COLOR_PALETTE[8],
    "ROIAlignRotated":     COLOR_PALETTE[8],
    "ROIPooling":          COLOR_PALETTE[8],
    "Proposal":            COLOR_PALETTE[8],
    "GenerateProposals":   COLOR_PALETTE[8],
    "ExtractImagePatches": COLOR_PALETTE[8],
    "I420toBGR":           COLOR_PALETTE[8],
    "I420toRGB":           COLOR_PALETTE[8],
    "NV12toBGR":           COLOR_PALETTE[8],
    "NV12toRGB":           COLOR_PALETTE[8],
    "IsInf":               COLOR_PALETTE[8],
    "IsNaN":               COLOR_PALETTE[8],
    "NonZero":             COLOR_PALETTE[8],
    "PSROIPooling":        COLOR_PALETTE[8],
    "PriorBox":            COLOR_PALETTE[8],
    "PriorBoxClustered":   COLOR_PALETTE[8],
    "CTCGreedyDecoder":    COLOR_PALETTE[8],
    "CTCGreedyDecoderSeqLen": COLOR_PALETTE[8],
    "CTCLoss":             COLOR_PALETTE[8],
    "EmbeddingBagOffsetsSum":  COLOR_PALETTE[8],
    "EmbeddingBagPackedSum":   COLOR_PALETTE[8],
    "EmbeddingBagOffsets":     COLOR_PALETTE[8],
    "EmbeddingBagPacked":      COLOR_PALETTE[8],
    "Eye":               COLOR_PALETTE[8],
    "Inverse":           COLOR_PALETTE[8],
    "FakeConvert":       COLOR_PALETTE[8],
    "StringTensorPack":   COLOR_PALETTE[8],
    "StringTensorUnpack": COLOR_PALETTE[8],
}
