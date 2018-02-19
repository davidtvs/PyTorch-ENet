from torch.autograd import Variable


class Train():
    def __init__(self, model, data_loader, optim, criterion, use_cuda):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.use_cuda = use_cuda

    def run_epoch(self):
        epoch_loss = 0.0
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

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.data[0]

            print("[Step: %d] Iteration loss: %.4f" % (step, loss.data[0]))

        return epoch_loss / len(self.data_loader)
