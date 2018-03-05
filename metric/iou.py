import numpy as np
from metric import metric
from metric.multilabelconfusionmatrix import MultiLabelConfusionMatrix


class IoU(metric.Metric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Keyword arguments:
    - k (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    """

    def __init__(self, k, normalized=False):
        self.conf_metric = MultiLabelConfusionMatrix(
            k, normalized=normalized)

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """
        Computes the intersection over union for K classes.

        Keyword arguments:
        - predicted (tensor or numpy.ndarray): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (tensor or numpy.ndarray): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.

        """
        self.conf_metric.add(predicted, target)

    def value(self):
        """
        Returns:
            Per class IoU and mean IoU. The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive
        iou = true_positive / (true_positive + false_positive + false_negative)
        return iou, np.nanmean(iou)
