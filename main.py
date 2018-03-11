import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.functional as F
import torchvision.transforms as transforms

import data as dataset
import transforms as ext_transforms
from models.enet import ENet
from train import Train
from val import Validation
from test import Test
from metric.iou import IoU
from args import get_arguments
import utils

# Run only if this module is being run directly
if __name__ == '__main__':
    # Get the arguments
    args = get_arguments()

    use_cuda = args.cuda and torch.cuda.is_available()

    print(">>>> Model initialization")

    # Fail fast if the dataset directory to save doesn't exist
    assert os.path.isdir(
        args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
            args.dataset_dir)

    # Fail fast if the saving directory doesn't exist
    assert os.path.isdir(
        args.save_dir), "The directory \"{0}\" doesn't exist.".format(
            args.save_dir)

    print("\nSelected dataset:", args.dataset)
    print("\nDataset directory:", args.dataset_dir)
    print("\nSave directory:", args.save_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        ext_transforms.PILToLongTensor()
    ])

    # Get selected dataset
    if args.dataset.lower() == 'camvid':
        # Load the training set as tensors
        trainset = dataset.CamVid(
            args.dataset_dir,
            transform=image_transform,
            label_transform=ext_transforms.PILToLongTensor())

        # Load the validation set as tensors
        valset = dataset.CamVid(
            args.dataset_dir,
            mode='val',
            transform=image_transform,
            label_transform=label_transform)

        # Load the test set as tensors
        testset = dataset.CamVid(
            args.dataset_dir,
            mode='test',
            transform=image_transform,
            label_transform=label_transform)

        # Remove the road_marking class as it's merged with the road class in
        # the dataset used by the ENet authors
        encoding = trainset.color_encoding
        _ = encoding.pop('road_marking')

    elif args.dataset.lower() == 'cityscapes':
        # Load the training set as tensors
        trainset = dataset.Cityscapes(
            args.dataset_dir,
            transform=image_transform,
            label_transform=label_transform)

        # Load the validation set as tensors
        valset = dataset.Cityscapes(
            args.dataset_dir,
            mode='val',
            transform=image_transform,
            label_transform=label_transform)

        # Load the test set as tensors
        testset = dataset.Cityscapes(
            args.dataset_dir,
            mode='test',
            transform=image_transform,
            label_transform=label_transform)

        # Get encoding between pixel valus in label images and RGB colors
        encoding = trainset.color_encoding
    else:
        raise RuntimeError("\"{0}\" is not a supported dataset.".format(
            args.dataset))

    # Get number of classes to predict
    num_classes = len(encoding)

    # Print information for debugging
    print("\nNumber of classes to predict:", num_classes)
    print("\nTrain dataset size:", len(trainset))
    print("\nValidation dataset size:", len(valset))
    print("\nTest dataset size:", len(testset))

    # Dataloaders
    trainloader = data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)
    valloader = data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)
    testloader = data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    # Initialize the label to PIL trasnform
    to_pil = ext_transforms.LongTensorToPIL()

    # Display a minibatch to make sure all is ok
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Convert the single channel label to RGB in tensor form
    # 1. F.unbind removes the 0-dimension of "labels" and returns a tuple of all
    # slices along that dimension
    # 2. to_pil converts the single channel tensor image to an RGB PIL image,
    # using the specified color encoding
    # 3. The color image is converted to Tensor
    # The result is a tuple of RGB tensor images
    tensor_labels = [
        transforms.ToTensor()(to_pil(tensor, encoding))
        for tensor in F.unbind(labels)
    ]
    color_labels = F.stack(tensor_labels)
    print("\nImage size:", images.size())
    print("\nLabel size:", labels.size())
    print("\nClass-color encoding:", encoding)
    print("\nClose the figure window to continue...")
    utils.imshow_batch(images, color_labels)

    # Intialize ENet
    net = ENet(num_classes)

    # Check if the network architecture is correct
    print(net)

    # Get class weights from the selected weighing technique
    print("\nWeighing technique:", args.weighing)
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")
    class_weights = 0
    if args.weighing == 'ENet':
        class_weights = utils.enet_weighing(trainset, num_classes)
    elif args.Weighing == 'MFB':
        class_weights = utils.median_freq_balancing(trainset, num_classes)
    else:
        raise RuntimeError(
            "\"{0}\" is not a valid choice for class weighing.".format(
                args.weighing))

    class_weights = torch.from_numpy(class_weights).float()

    # Handle unlabelled class
    if args.ignore_unlabelled:
        if args.dataset.lower() == 'camvid':
            class_weights[-1] = 0
        elif args.dataset.lower() == 'cityscapes':
            class_weights[0] = 0

    print("\nClass weights:", class_weights)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ENet authors used mini-batch gradient descent
    optimizer = optim.Adam(
        net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # Evaluation metrics
    metrics = IoU(num_classes)

    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    # Start Training
    print()
    train = Train(net, trainloader, optimizer, criterion, use_cuda)
    val = Validation(net, valloader, criterion, metrics, use_cuda)
    for epoch in range(args.epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        epoch_loss = train.run_epoch(args.print_step)

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f}".format(
            epoch, epoch_loss))

        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, (iou, miou) = val.run_epoch(args.print_step)

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(epoch, epoch_loss, miou))

            # Print per class IoU on last epoch
            if epoch + 1 == args.epochs:
                for key, class_iou in zip(encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))

    # Test the trained model on the test set
    test = Test(net, testloader, criterion, metrics, use_cuda)

    print("\n>>>> Running test dataset")

    loss, (iou, miou) = test.run_epoch(args.print_step)
    class_iou = dict(zip(encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(
        epoch_loss, miou))

    # Print per class IoU
    for key, class_iou in zip(encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

    # Save the model in the given directory with the given name
    print("\n>>>> Saving model")
    utils.save(net, args.name, args.save_dir)
