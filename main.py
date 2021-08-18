import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import grad

import methods
import line_search
import vec_prod
from parser import build_parser
from model import NeuralNet
from utils import log

args = build_parser()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 784
num_classes = 10

train_dataset = torchvision.datasets.MNIST(root='../../data',
                                                train=True,
                                                transform=transforms.ToTensor(),
                                                download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                                train=False,
                                                transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False)

model = NeuralNet(input_size, args.hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = getattr(methods, args.algo)(args, model, criterion)

if args.line_search==True:
    searcher = getattr(line_search, args.search_algo)(args, model, criterion)


total_step = len(train_loader)
for epoch in range(args.epochs):
    total_loss = 0
    total_step_rate = 0
    step_adj = args.max_step_len
    for i, (images, labels) in enumerate(train_loader):  
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        if args.line_search==True:
            dir = searcher.direction(images, labels)
            step_adj = searcher.optimize(images, labels, dir, args.max_step_len, args.step_coeff, args.step_iter)
            total_step_rate += step_adj

        model, loss = optimizer.optimize(images, labels, args.alpha, epoch)
        total_loss += loss
    args.alpha *= step_adj / args.max_step_len
    print(args.alpha)
    log(args, epoch, total_loss / i+1, total_step_rate / i+1)
        

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

