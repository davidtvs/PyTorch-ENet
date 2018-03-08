from argparse import ArgumentParser


def get_arguments():
    """Defines command-line arguments, and parses them.

    """
    parser = ArgumentParser()

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
        choices=['CamVid'],
        default='CamVid',
        help="Dataset to use. Default: CamVid")
    parser.add_argument(
        "--num_classes",
        "-c",
        type=int,
        default=12,
        help="Number of classes to segment. Default: 12")
    parser.add_argument(
        "--weighing",
        choices=['ENet', 'MFB'],
        default='ENet',
        help=
        "The class weighing technique to apply to the dataset. Default: ENet")
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
