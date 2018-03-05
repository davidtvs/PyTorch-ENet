import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.functional as F
import torchvision.transforms as transforms

from data.camvid import CamVidDataset, LabelTensorToPIL
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

    # Fail fast if the specified directory to save doesn't exist
    if not os.path.isdir(args.save_dir):
        raise RuntimeError("The directory \"{0}\" doesn't exist.".format(
            args.save_dir))

    use_cuda = args.cuda and torch.cuda.is_available()

    # Folder where datasets are placed
    data_folder = 'data'
    # Build path to dataset
    dataset_path = os.path.join(data_folder, args.dataset)
    print("Selected dataset: ", dataset_path)

    if args.dataset == 'CamVid':
        # Load the training set as tensors
        trainset = CamVidDataset(dataset_path, transform=transforms.ToTensor())
        # Split it into minibatches, shuffle, and set the no. of workers
        trainloader = data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers)

        # Load the validation set as tensors
        valset = CamVidDataset(
            dataset_path, mode='val', transform=transforms.ToTensor())
        # Split it into minibatches, shuffle, and set the no. of workers
        valloader = data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers)

        # Load the test set as tensors
        testset = CamVidDataset(
            dataset_path, mode='test', transform=transforms.ToTensor())
        # Split it into minibatches, shuffle, and set the no. of workers
        testloader = data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers)
    else:
        raise RuntimeError("\"{0}\" is not a supported dataset.".format(
            args.dataset))

    # Initialize the label to PIL class
    to_pil = LabelTensorToPIL()

    # Remove the road_marking class as it's merged with the road class in the
    # dataset used by the ENet authors
    encoding = to_pil.get_econding()
    _ = encoding.pop('road_marking')

    # Display a minibatch to make sure all is ok
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Convert the single channel label to RGB
    labels_list = [transforms.ToTensor()(to_pil(t)) for t in F.unbind(labels)]
    color_labels = torch.functional.stack(labels_list)
    print(">>>> Close the figure window to continue...")
    utils.imshow_batch(images, color_labels)

    # Intialize ENet
    net = ENet(args.num_classes)

    # Check if the network architecture is correct
    print(net)

    # Get class weights from the selected weighing technique
    class_weights = 0
    if args.weighing == 'ENet':
        class_weights = utils.enet_weighing(trainset, args.num_classes)
    elif args.Weighing == 'MFB':
        class_weights = utils.median_freq_balancing(trainset, args.num_classes)
    else:
        raise RuntimeError(
            "\"{0}\" is not a valid choice for class weighing.".format(
                args.weighing))

    class_weights = torch.from_numpy(class_weights).float()

    # Handle unlabelled class
    if args.ignore_unlabelled:
        class_weights[-1] = 0

    print("Weighing technique: ", args.weighing)
    print("Class weights: ", class_weights)

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
    metrics = IoU(args.num_classes)

    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    # Start Training
    train = Train(net, trainloader, optimizer, criterion, use_cuda)
    val = Validation(net, valloader, criterion, metrics, use_cuda)
    for epoch in range(args.epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        epoch_loss = train.run_epoch()

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f}".format(
            epoch, epoch_loss))

        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, (iou, miou) = val.run_epoch()

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(epoch, epoch_loss, miou))

            # Print per class IoU on last epoch
            if epoch + 1 == args.epochs:
                for key, class_iou in zip(encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))

    # Test the trained model on the test set
    test = Test(net, testloader, criterion, metrics, use_cuda)

    print("\n>>>> Running test dataset")

    loss, (iou, miou) = test.run_epoch()
    class_iou = dict(zip(encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(
        epoch_loss, miou))

    # Print per class IoU
    for key, class_iou in zip(encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

    # Save the model in the given directory with the given name
    utils.save(net, args.name, args.save_dir)
