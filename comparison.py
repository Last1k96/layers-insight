import numpy as np
from typing import Tuple, List, Union


###############################################################################
# 1. Numerical / Tensor Similarity Functions
###############################################################################

def mean_absolute_error(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Mean Absolute Error (MAE) between two tensors x and y.
    MAE = sum(|x_i - y_i|) / N
    """
    return np.mean(np.abs(x - y))


def mean_squared_error(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE) between x and y.
    MSE = sum((x_i - y_i)^2) / N
    """
    return np.mean((x - y) ** 2)


def root_mean_squared_error(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) between x and y.
    RMSE = sqrt( MSE )
    """
    return np.sqrt(mean_squared_error(x, y))


def max_absolute_difference(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the maximum absolute difference (L∞ norm of x - y).
    max_diff = max(|x_i - y_i|)
    """
    return np.max(np.abs(x - y))


def relative_error(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    """
    Computes the average relative error between x and y.
    Rel Error = sum(|x_i - y_i| / (|x_i| + eps)) / N
    """
    return np.mean(np.abs(x - y) / (np.abs(x) + eps))


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors (or flattened tensors).
    Similarity = (x . y) / (||x|| * ||y||)
    Returns 1.0 if perfectly aligned, -1.0 if perfectly opposite, 0 if orthogonal.
    """
    # Flatten to 1D if needed
    x_flat = x.flatten()
    y_flat = y.flatten()
    dot = np.dot(x_flat, y_flat)
    norm_x = np.linalg.norm(x_flat)
    norm_y = np.linalg.norm(y_flat)
    if norm_x < 1e-12 or norm_y < 1e-12:
        return 0.0
    return dot / (norm_x * norm_y)


def psnr(x: np.ndarray, y: np.ndarray, max_val: float = 1.0) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between x and y in decibels.
    Typically used in image comparison.
    PSNR = 20 * log10(max_val) - 10 * log10(MSE)
    If x and y are images, max_val is often 1.0 or 255.
    """
    mse_value = mean_squared_error(x, y)
    if mse_value < 1e-12:
        return float('inf')
    return 20 * np.log10(max_val) - 10 * np.log10(mse_value)


def ssim(x: np.ndarray, y: np.ndarray,
         C1: float = 0.01 ** 2,
         C2: float = 0.03 ** 2) -> float:
    """
    Computes the Structural Similarity Index Measure (SSIM) between two images/tensors.
    This is a simplified version that operates on a single channel.
    For multi-channel images, you may average SSIM across channels.
    """
    # Make sure x, y are float64 for numerical stability
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    return numerator / denominator


###############################################################################
# 2. Classification Accuracy Functions
###############################################################################

def top1_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes the Top-1 accuracy given predictions and ground-truth labels.

    Args:
        predictions: shape [N, num_classes] with prediction scores or probabilities
        labels: shape [N] with ground-truth class indices
    Returns:
        accuracy (float): ratio of samples that have the highest predicted score
                          matching the ground-truth label
    """
    pred_classes = np.argmax(predictions, axis=1)
    correct = (pred_classes == labels).sum()
    return correct / labels.shape[0]


def topk_accuracy(predictions: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """
    Computes the Top-k accuracy.

    Args:
        predictions: shape [N, num_classes]
        labels: shape [N]
        k: how many top predictions to consider
    Returns:
        accuracy (float)
    """
    # Get the indices of the top-k predictions for each sample
    topk_preds = np.argsort(-predictions, axis=1)[:, :k]
    # Check if label is among the top-k predicted classes
    correct = 0
    for i in range(labels.shape[0]):
        if labels[i] in topk_preds[i]:
            correct += 1
    return correct / labels.shape[0]


def confusion_matrix(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Computes a confusion matrix for classification.

    Args:
        predictions: shape [N], predicted class indices
        labels: shape [N], ground-truth class indices
        num_classes: total number of classes
    Returns:
        conf_mat: a num_classes x num_classes matrix
    """
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(predictions, labels):
        conf_mat[t, p] += 1
    return conf_mat


def precision_recall_f1(conf_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a confusion matrix, returns (precision, recall, f1) for each class.

    Args:
        conf_mat: shape [num_classes, num_classes]
    Returns:
        (precision, recall, f1) each of shape [num_classes]
    """
    num_classes = conf_mat.shape[0]
    precision = np.zeros(num_classes, dtype=float)
    recall = np.zeros(num_classes, dtype=float)
    f1 = np.zeros(num_classes, dtype=float)

    for c in range(num_classes):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp

        precision[c] = tp / (tp + fp + 1e-8)
        recall[c] = tp / (tp + fn + 1e-8)
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c] + 1e-8)

    return precision, recall, f1


###############################################################################
# 3. Object Detection Overlap Functions
###############################################################################

def iou(boxA: Union[List[float], np.ndarray],
        boxB: Union[List[float], np.ndarray]) -> float:
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA: [x1, y1, x2, y2] or [left, top, right, bottom]
        boxB: same format
    Returns:
        iou: Intersection over Union (0 <= iou <= 1)
    """
    # Convert to np.array
    boxA = np.array(boxA, dtype=float)
    boxB = np.array(boxB, dtype=float)

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea < 1e-12:
        return 0.0

    return interArea / unionArea


###############################################################################
# 4. Example Usage & Tips
###############################################################################

if __name__ == "__main__":
    # Example usage for numerical metrics:
    x = np.array([[[[-0.4665752]], [[0.46655202]]]], dtype=np.float32)
    y = np.array([[[[-0.4665712]], [[0.46654803]]]], dtype=np.float32)

    print("MAE:", mean_absolute_error(x, y))
    print("MSE:", mean_squared_error(x, y))
    print("RMSE:", root_mean_squared_error(x, y))
    print("Max Diff:", max_absolute_difference(x, y))
    print("Relative Error:", relative_error(x, y))
    print("Cosine Similarity:", cosine_similarity(x, y))
    print("PSNR:", psnr(x, y, max_val=1.0))
    print("SSIM:", ssim(x, y))

    # Classification example:
    predictions = np.array([
        [0.1, 0.7, 0.2],  # predicted scores for classes 0,1,2
        [0.8, 0.1, 0.1],
        [0.2, 0.5, 0.3]
    ])
    labels = np.array([1, 0, 1])

    top1 = top1_accuracy(predictions, labels)
    top5 = topk_accuracy(predictions, labels, k=2)  # top-2 for demonstration
    print("Top-1 Accuracy:", top1)
    print("Top-2 Accuracy:", top5)

    # Confusion matrix:
    pred_classes = np.argmax(predictions, axis=1)
    conf_mat = confusion_matrix(pred_classes, labels, num_classes=3)
    print("Confusion Matrix:\n", conf_mat)
    prec, rec, f1_vals = precision_recall_f1(conf_mat)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Scores:", f1_vals)

    # Bounding box IoU:
    box1 = [100, 100, 200, 200]
    box2 = [150, 150, 250, 250]
    print("IoU:", iou(box1, box2))
