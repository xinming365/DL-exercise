# Author : xinming
# Time : 2020/06/27

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#Create fully connected network
# Base class for all neural network modules. Your models should also subclass this class.
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = NN(input_size= 784, num_classes=10)
x = torch.randn(64, 784)
print(model(x).shape)

# Set  device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size=64
num_epochs=1
learning_rate=0.001
# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape
        data = data.reshape(data.shape[0], -1)

        #forward
        scores=model(data)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step
        optimizer.step()


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct=0
    num_samples=0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x=x.to(device=device)
            y=y.to(device=device)
            x=x.reshape(x.shape[0],-1)

            scores=model(x)
            _, predictions = scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)
        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) *100:.2f}")
    model.train()
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)



