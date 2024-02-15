import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


batch_size = 50


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #print(m)
        nn.init.xavier_normal_(m.weight)
    elif classname.find('Linear') != -1:
        #print(m)
        nn.init.xavier_normal_(m.weight)
    # elif classname.find('BatchNorm') != -1:
    #     nn.init.xavier_uniform_(m.weight)


k = 82
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=2),
            #nn.init.xavier_uniform_(nn.Conv2d(3, 16, kernel_size=3, padding=2).weight, double gain = 1.0)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))     #pooling with stride=1 produces too many parameeters
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, k, kernel_size=3, padding=2),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.MaxPool2d(2))                                              #73% with 4 epochs, maybe u shld add another mid-layer
        self.fc = nn.Linear(5*5*k, 10)                #74% 16 32 110 2 2 3    333
        self.dropout = nn.Dropout(p=0.1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        #print(x.shape)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.dropout(out)
        out = self.fc(out)
        return self.logsoftmax(out)

def evaluate_model_q1():
    #Image Preprocessing
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.2434, 0.2615)),
    ])


    # CIFAR-10 Dataset
    test_dataset = dsets.CIFAR10(root='./data/',
                                  train=False,
                                  transform=test_transform,
                                  download=True)

    # Data Loader (Input Pipeline)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    model = CNN()
    model.load_state_dict(torch.load('model_q1.pkl',map_location=lambda storage, loc: storage))
    model.eval()

    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Error percentage of the model on the 10000 test images: %.3f %%' % (100-(100 * correct / total)))
    return 100-(100 * correct / total)

evaluate_model_q1()
