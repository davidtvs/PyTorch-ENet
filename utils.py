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


def enet_weighing(dataset, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:

        w_class = 1 / (ln(c + p_class)),

    where c is usually 1.02 and p_class is the propensity score of that
    class:

        propensity_score = freq_class / total_pixels.

    References: https://arxiv.org/abs/1606.02147

    Keyword arguments:
    - dataset (``Dataset``): A ``Dataset`` instance containing the labels
    whose weights are going to be computed.
    - num_classes (``int``): The number of classes
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.

    """
    class_count = 0
    total = 0
    for _, label in dataset:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = np.reshape(label, -1)

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def median_freq_balancing(dataset, num_classes):
    """Computes class weights using median frequency balancing as described
    in https://arxiv.org/abs/1411.4734:

        w_class = median_freq / freq_class,

    where freq_class is the number of pixels of a given class divided by
    the total number of pixels in images where that class is present, and
    median_freq is the median of freq_class.

    Keyword arguments:
    - dataset (``Dataset``): A ``Dataset`` instance containing the labels
    whose weights are going to be computed.
    - num_classes (``int``): The number of classes

    """
    class_count = 0
    total = 0
    for _, label in dataset:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = np.reshape(label, -1)

        # Sum up the class frequencies
        bincount = np.bincount(flat_label, minlength=num_classes)

        # Create of mask of classes that exist in the label
        mask = bincount > 0
        # Multiply the mask by the pixel count. The resulting array has
        # one element for each class. The value is either 0 (if the class
        # does not exist in the label) or equal to the pixel count (if
        # the class exists in the label)
        total += mask * flat_label.size

        # Sum up the number of pixels found for each class
        class_count += bincount

    # Compute the frequency and its median
    freq = class_count / total
    med = np.median(freq)

    return med / freq


def save(model, name, save_dir):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The model to save.
    - name (``string``): The saved model name.
    - save_dir (``string``): The directory where the model is saved

    """
    if not os.path.isdir(save_dir):
        raise RuntimeError(
            "The directory \"{0}\" doesn't exist.".format(save_dir))

    path = os.path.join(save_dir, name)
    torch.save(model.state_dict(), path)
