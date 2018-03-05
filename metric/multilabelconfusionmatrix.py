import torch
import numpy as np
from metric import metric
from metric.confusionmatrix import ConfusionMatrix


class MultiLabelConfusionMatrix(metric.Metric):
    """
    The MultiLabelConfusionMeter constructs a confusion matrix for multi-class
    multi-label classification problems.

    Keyword arguments:
    - k (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.

    """

    def __init__(self, k, normalized=False):
        self.confusion = ConfusionMatrix(k, normalized)
        self.normalized = normalized
        self.k = k

    def reset(self):
        self.confusion.reset()

    def add(self, predicted, target):
        """
        Computes the confusion matrix of (K, K) size where K is no. of classes.

        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.

        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        assert np.ndim(target) == 3 or np.ndim(target) == 4, \
            'targets must be of dimension (N, H, W) or (N, K, H, W)'
        assert np.ndim(predicted) == 3 or np.ndim(predicted) == 4, \
            'predicted must be of dimension (N, H, W) or (N, K, H, W)'
        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(target) == 4:
            target = np.argmax(target, 1)

        if np.ndim(predicted) == 4:
            predicted = np.argmax(predicted, 1)

        self.confusion.add(predicted.flatten(), target.flatten())

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        return self.confusion.value()
