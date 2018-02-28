import torchvision
import numpy as np
import matplotlib.pyplot as plt


def imshow_batch(images, labels):
    """Displays two grids of images. The top grid displays ``images``
    and the bottom grid ``labels``

    Keyword arguments:
    - images (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)
    - labels (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)

    """

    print(images.size(), labels.size())
    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy()
    labels = torchvision.utils.make_grid(labels).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))

    plt.show()


def enet_weighing(dataset, c=1.02):
    """Computes class weights as described in the ENet paper:

    w_class = 1 / (ln(c + p_class)),

    where c is usually 1.02 and p_class is the propensity score of that
    class:

        propensity_score = freq_class / total_pixels.

    References: https://arxiv.org/abs/1606.02147

    """
    freq = 0
    total = 0
    for _, label in dataset:
        label = label.cpu().numpy()

        # Flatten image
        flat_label = np.reshape(label, -1)

        # Sum up the class frequencies and pixel counts for each label
        freq += np.bincount(flat_label)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = freq / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights
