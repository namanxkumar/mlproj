import argparse
from os import truncate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time

#Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) #in [?, 28, 28, 1] | out [?, 26, 26, 32]
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # out [?, 24, 24, 64]
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) #out [?, 12, 12, 64]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim = 1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #Initialize optimizer with zero grad
        optimizer.zero_grad()
        #Forward step through one mini batch
        output = model(data)
        #Calc loss
        loss = F.nll_loss(output, target)
        #Calculate backprop parameter gradients using chain rule
        loss.backward()
        #Optimize parameters based on backprop calculations
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                batch_idx*len(data), #number of data points iterated through
                len(train_loader.dataset), #total data points
                100. * batch_idx / len(train_loader), #percentage done
                loss.item() #loss value
            ))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim = 1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct/len(test_loader.dataset)
    ))

def main():
    #general environment options
    parser = argparse.ArgumentParser(description="Pytorch MNIST")
    # Train batch size
    parser.add_argument('--train-batch-size', type = int, default = 64, metavar = 'N', help = 'input batch size for training (default: 64)')
    # Test batch size
    parser.add_argument('--test-batch-size', type = int, default = 1000, metavar = 'N', help = 'input batch size for testing (default: 1000)')
    # Epochs
    parser.add_argument('--epochs', type = int, default = 14, metavar = 'N', help = 'number of epochs to train (default: 14)')
    # Learning rate
    parser.add_argument('--lr', type = float, default = 1.0, metavar = 'LR', help = 'learning rate (default: 1.0)')
    # Gamma
    parser.add_argument('--gamma', type = float, default = 0.7, metavar = 'M', help = 'Learning rate step gamma (default: 0.7)')
    # Cuda?
    parser.add_argument('--no-cuda', action = 'store_true', default = False, help = 'disables CUDA training')
    # Dry-run?
    parser.add_argument('--dry-run', action='store_true', default = False, help = 'quickly check a single pass')
    # Random seed
    parser.add_argument('--seed', type = int, default = 1, metavar = 'S', help = 'random seed (default: 1)')
    # Training log interval
    parser.add_argument('--log-interval', type = int, default = 10, metavar = 'N', help = 'number of batches to wait before logging status (default: 10)')
    # Save model?
    parser.add_argument('--save-model', action='store_true', default = False, help = 'saves current model')

    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed) #set random seed

    device = torch.device('cuda' if use_cuda else 'cpu') #set processing device

    #initialize kwargs

    train_kwargs = {'batch_size': args.train_batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    #set data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    #download/load dataset
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    #Initialize model
    model = Net().to(device)
    print(model)

    #Initialize optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    #Initialize scheduler to decay LR
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    #Iterate model training and testing
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step() #Decay learning rate
    end = time.time()

    print('Total training time: {} seconds'.format(end-start))

    #Save model
    if args.save_model:
        torch.save(model.state_dict(), 'mnist_cnn.pt')

if __name__ == '__main__':
    main()