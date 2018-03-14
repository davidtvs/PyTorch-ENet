from argparse import ArgumentParser


def get_arguments():
    """Defines command-line arguments, and parses them.

    """
    parser = ArgumentParser()

    # Execution mode
    parser.add_argument(
        "--mode",
        "-m",
        choices=['train', 'test', 'full'],
        default='train',
        help="train: performs training and validation; test: tests the model "
        "found in \"--save_dir\" with name \"--name\" on \"--dataset\"; "
        "full: combines train and test modes. Default: train")

    # Hyperparameters
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=10,
        help="The batch size. Default: 10")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs. Default: 300")
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=5e-4,
        help="The learning rate. Default: 5e-4")
    parser.add_argument(
        "--weight_decay",
        "-wd",
        type=float,
        default=2e-4,
        help="L2 regularization factor. Default: 2e-4")

    # Dataset
    parser.add_argument(
        "--dataset",
        choices=['camvid', 'cityscapes'],
        default='camvid',
        help="Dataset to use. Default: camvid")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/CamVid",
        help="Path to the root directory of the selected dataset. "
        "Default: data/CamVid")
    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="The image height. Default: 360")
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="The image height. Default: 480")
    parser.add_argument(
        "--weighing",
        choices=['enet', 'mfb', 'none'],
        default='ENet',
        help="The class weighing technique to apply to the dataset. "
        "Default: enet")
    parser.add_argument(
        "--ignore_unlabelled",
        type=bool,
        default=True,
        help="If True, the unlabelled class weight is ignored (set to 0); "
        "otherwise, it's kept as computed. Default: True")

    # Settings
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of subprocesses to use for data loading. Default: 8")
    parser.add_argument(
        "--print_step",
        type=bool,
        default=False,
        help="True to print step loss. Default: False")
    parser.add_argument(
        "--cuda",
        type=bool,
        default=True,
        help="True to use CUDA (GPU). Default: True")

    # Storage settings
    parser.add_argument(
        "--name",
        type=str,
        default='ENet',
        help="Name given to the model when saving.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default='save',
        help="The directory where models are saved.")

    return parser.parse_args()
