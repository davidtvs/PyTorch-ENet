import os
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch
import torch.utils.data as data
from torchvision.transforms import ToPILImage


class LabelPILToTensor(object):
    """Converts a CamVid label ``PIL Image`` to a ``torch.LongTensor``.

    Code adapted from: http://pytorch.org/docs/master/torchvision/transforms.html?highlight=totensor

    """

    def __call__(self, pic):
        """Performs the conversion from a CamVid label ``PIL Image`` to a
        ``torch.LongTensor``.

        Keyword arguments:
        - pic (``PIL.Image``): the image to convert to ``torch.LongTensor``

        Returns:
        A ``torch.LongTensor``.

        """
        if not isinstance(pic, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(
                type(pic)))

        # handle numpy array
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.long()

        # Convert PIL image to ByteTensor
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        # Reshape tensor
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        # Convert to long and squeeze the channels
        return img.transpose(0, 1).transpose(0,
                                             2).contiguous().long().squeeze_()


class LabelTensorToPIL(object):
    """Converts a ``torch.LongTensor`` to a ``PIL image``.

    The input is a ``torch.LongTensor`` where each pixel's value identifies the
    class. The default encoding is the following:

    pixel value : class name : class color
    0: sky : (128, 128, 128)
    1: building : (128, 0, 0)
    2: pole : (192, 192, 128)
    3: road_marking : (255, 69, 0)
    4: road : (128, 64, 128)
    5: pavement : (60, 40, 222)
    6: tree : (128, 128, 0)
    7: signSymbol : (192, 128, 128)
    8: fence : (64, 64, 128)
    9: car : (64, 0, 128)
    10: pedestrian : (64, 64, 0)
    11: bicyclist : (0, 128, 192)
    12: unlabelled : (0, 0, 0)

    """

    def get_econding(self):
        """Builds the default encoding for pixel value, class name, and
        class color.

        Returns:
            An ``OrderedDict`` which stores the default encoding between pixel
            value, class names and class colors. The dictionary can be
            customized and passed to ``__call__`` through
            ``encoding``.

        """
        # Build the relationship between class number, name and color. The
        # dictionary is ordered so we can assume that the index of the
        # (key,value) pair is the same as the class number
        label_to_color = OrderedDict([('sky', (128, 128, 128)),
                              ('building', (128, 0, 0)),
                              ('pole', (192, 192, 128)),
                              ('road_marking', (255, 69, 0)),
                              ('road', (128, 64, 128)),
                              ('pavement', (60, 40, 222)),
                              ('tree', (128, 128, 0)),
                              ('sign_symbol', (192, 128, 128)),
                              ('fence', (64, 64, 128)),
                              ('car', (64, 0, 128)),
                              ('pedestrian', (64, 64, 0)),
                              ('bicyclist', (0, 128, 192)),
                              ('unlabelled', (0, 0, 0))])

        return label_to_color

    def __call__(self, label_tensor, encoding=None):
        """Performs the conversion from ``torch.LongTensor`` to a ``PIL image``

        Keyword arguments:
        - label_tensor (``torch.LongTensor``): the label tensor to Convert
        - encoding (``OrderedDict``, optional): An ``OrderedDict`` containing
        the correspondence between pixel value, class names and class colors.
        Default: None (the encoding returned by ``get_econding`` is used).

        Returns:
        A ``PIL.Image``.

        """
        # Check if label_tensor is a LongTensor
        if not isinstance(label_tensor, torch.LongTensor):
            raise TypeError("label_tensor should be torch.LongTensor. Got {}"
                            .format(type(label_tensor)))

        # If a custom label to color dictionary is not supplied use the default
        # otherwise, check encoding and if valid, use it
        if encoding is None:
            label_to_color = self.get_econding()
        else:
            if not isinstance(encoding, OrderedDict):
                raise TypeError("encoding should be an OrderedDict. Got {}"
                                .format(type(encoding)))
            else:
                label_to_color = encoding

        # label_tensor might be an image without a channel dimension, in this
        # case unsqueeze it
        if len(label_tensor.size()) == 2:
            label_tensor.unsqueeze_(0)

        color_tensor = torch.Tensor(3, label_tensor.size(1),
                                    label_tensor.size(2))

        for index, (class_name, color) in enumerate(label_to_color.items()):
            # Get a mask of elements equal to index
            mask = torch.eq(label_tensor, index).squeeze_()
            # Fill color_tensor with corresponding colors
            for channel, color_value in enumerate(color):
                color_tensor[channel].masked_fill_(mask, color_value)

        return ToPILImage()(color_tensor)


def get_files(folder, extension_filter):
    """Helper function that returns the list of files in a specified folder
    with a specified extension.

    Keyword arguments:
    - folder (``string``): The path to a folder.
    - extension_filter (``string``): The desired file extension.

    """
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    files = []

    # Iterate over the PNG files in the train directory and store the filepaths
    for file in os.listdir(folder):
        if file.endswith(extension_filter):
            files.append(os.path.join(folder, file))

    return files


def default_loader(data_path, label_path):
    """The default dataset loader function.

    Keyword arguments:
    - data_path (``string``): The filepath to the image.
    - label_path (``string``): The filepath to the ground-truth image.

    Returns the image and the label as PIL images.

    """
    data = Image.open(data_path)
    label = Image.open(label_path)

    return data, label


class CamVidDataset(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.


    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: ``ToLongTensor``.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """
    # Training dataset root folders
    train_folder = 'train'
    train_lbl_folder = 'trainannot'

    # Validation dataset root folders
    val_folder = 'val'
    val_lbl_folder = 'valannot'

    # Test dataset root folders
    test_folder = 'test'
    test_lbl_folder = 'testannot'

    def __init__(self,
                 root_dir,
                 mode='train',
                 transform=None,
                 label_transform=LabelPILToTensor(),
                 loader=default_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader

        if self.mode == 'train':
            # Get the training data and labels filepaths
            self.train_data = get_files(
                os.path.join(root_dir, self.train_folder), '.png')
            self.train_labels = get_files(
                os.path.join(root_dir, self.train_lbl_folder), '.png')
        elif self.mode == 'val':
            # Get the validation data and labels filepaths
            self.val_data = get_files(
                os.path.join(root_dir, self.val_folder), '.png')
            self.val_labels = get_files(
                os.path.join(root_dir, self.val_lbl_folder), '.png')
        elif self.mode == 'test':
            # Get the test data and labels filepaths
            self.test_data = get_files(
                os.path.join(root_dir, self.test_folder), '.png')
            self.test_labels = get_files(
                os.path.join(root_dir, self.test_lbl_folder), '.png')
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """
        if self.mode == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
        elif self.mode == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
        elif self.mode == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.val_data)
        elif self.mode == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
