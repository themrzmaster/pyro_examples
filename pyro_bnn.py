"""
Simple BNN trained using variational inference.

"""


import pyro
import pyro.contrib.gp as gp
import torch.nn.functional as F
import pyro.infer as infer
import torch
from torch import nn
import pyro.distributions as dist
from pyro.distributions import Normal
from torch.distributions.categorical import Categorical
import torchvision.datasets as dset
import torchvision.transforms as transforms
from pyro.contrib.examples.util import get_data_loader, get_data_directory
import pyro.optim as optim
from pyro.nn import PyroModule, PyroSample

class BNN(PyroModule):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(BNN, self).__init__()
        self.fc1 = PyroModule[nn.Linear](input_size, hidden_size)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_size, input_size]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 10.).expand([hidden_size]).to_event(1))
        self.out = PyroModule[nn.Linear](hidden_size, output_size)
        self.out.weight = PyroSample(dist.Normal(0., 1.).expand([output_size, hidden_size]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., 10.).expand([output_size]).to_event(1))
        
    def forward(self, x, y_data=None):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        self.lhat = F.log_softmax(output)
        obs = pyro.sample("obs", dist.Categorical(logits=self.lhat), obs=y_data)
        return obs

train_loader = torch.utils.data.DataLoader(
        dset.MNIST('mnist-data/', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),])),
        batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        dset.MNIST('mnist-data/', train=False, transform=transforms.Compose([transforms.ToTensor(),])
                       ),
        batch_size=128, shuffle=True)


model2 = BNN(28*28, 1024, 10)
from pyro.infer.autoguide import AutoDiagonalNormal
guide2 = AutoDiagonalNormal(model2)
optima = optim.Adam({"lr": 0.01})
svi = infer.SVI(model2, guide2, optima, loss=infer.Trace_ELBO())

num_iterations = 5
loss = 0

for j in range(num_iterations):
    loss = 0
    for batch_id, data in enumerate(train_loader):
        # calculate the loss and take a gradient step
        data, label = data[0].view(-1,28*28), data[1]
        loss += svi.step(data, label)
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = loss / normalizer_train
    
    print("Epoch ", j, " Loss ", total_epoch_loss_train)

    num_samples = 10

from pyro.infer import Predictive
predictive = Predictive(model2, guide=guide2, num_samples=128,
                        return_sites=("out.weight", "obs", "_RETURN", "lhat"))

import numpy as np
total = 0
acertos = 0
for j, data in enumerate(test_loader):
    images, labels = data
    predicted = predictive(images.view(-1,28*28))
    acertos += torch.sum(labels == predicted['obs']).numpy()
    total += len(labels)
    
acc = acertos/total
print(acc)
