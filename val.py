from torch.autograd import Variable


class Validation():
    def __init__(self, model, data_loader, criterion, metrics, use_cuda):
        self.model = model
        self.data_loader = data_loader
        self.metrics = metrics
        self.criterion = criterion
        self.use_cuda = use_cuda

    def run_epoch(self, iteration_loss=False):
        epoch_loss = 0.0
        self.metrics.reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs, labels = batch_data

            # Wrap them in a Varaible
            inputs, labels = Variable(inputs), Variable(labels)
            if self.use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Forward propagation
            outputs = self.model(inputs)

            # Loss computation
            loss = self.criterion(outputs, labels)

            # Keep track of loss for current epoch
            epoch_loss += loss.data[0]

            # Keep track of evaluation metrics
            self.metrics.add(outputs.data.cpu(), labels.data.cpu())

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.data[0]))

        iou, miou = self.metrics.value()
        self.metrics.reset()

        return epoch_loss / len(self.data_loader), iou, miou
