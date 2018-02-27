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
from metrics.iou import IoU
from utils import imshow_batch

# TODO: Convert variables below into command-line arguments using argparse
num_workers = 2
use_cuda = True
num_classes = 12
batch_size = 4
learning_rate = 5e-4
momentum = 0.9
weight_decay = 2e-4
num_epochs = 300
use_cuda = True and torch.cuda.is_available()

# Load the training set as tensors
trainset = CamVidDataset('data/CamVid/', transform=transforms.ToTensor())
# Split it into minibatches of 4, shuffle, and set the no. of workers
trainloader = data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Load the validation set as tensors
valset = CamVidDataset(
    'data/CamVid/', mode='val', transform=transforms.ToTensor())
# Split it into minibatches of 4, shuffle, and set the no. of workers
valloader = data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

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
imshow_batch(images, color_labels)

# Intialize ENet
net = ENet(num_classes)

# Check if the network architecture is correct
print(net)

# We are going to use the CrossEntropyLoss loss function as it's most
# frequentely used in classification problems with multiple classes which
# fits the problem. This criterion  combines LogSoftMax and NLLLoss.
criterion = nn.CrossEntropyLoss()

# ENet authors used mini-batch gradient descent
optimizer = optim.SGD(
    net.parameters(),
    lr=learning_rate,
    momentum=momentum,
    weight_decay=weight_decay)

# Evaluation metrics
metrics = IoU(num_classes)

if use_cuda:
    net = net.cuda()
    criterion = criterion.cuda()

# Start Training
train = Train(net, trainloader, optimizer, criterion, use_cuda)
val = Validation(net, valloader, criterion, metrics, use_cuda)
for epoch in range(num_epochs):
    print("\n>>>> [Epoch: %d] Training" % epoch)

    epoch_loss = train.run_epoch()

    print(">>>> [Epoch: %d] Avg. loss: %.4f" % (epoch, epoch_loss))

    if (epoch + 1) % 10 == 0:
        print(">>>> [Epoch: %d] Validation" % epoch)

        loss, iou, miou = val.run_epoch()

        print(">>>> [Epoch: %d] Avg. loss: %.4f | Mean IoU: %.4f" %
              (epoch, epoch_loss, miou))

print("Finished training!")
