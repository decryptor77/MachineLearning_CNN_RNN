import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Hyper Parameters
num_epochs = 25
batch_size = 50
learning_rate = 0.001



#Image Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])



# CIFAR-10 Dataset
train_dataset_original = dsets.CIFAR10(root='./data/',
                               train=True,
                               transform=transform,
                               download=True)
train_dataset_aug = dsets.CIFAR10(root='./data/',
                               train=True,
                               transform=train_transform,
                               download=True)
train_dataset = torch.utils.data.ConcatDataset([train_dataset_original, train_dataset_aug])
test_dataset = dsets.CIFAR10(root='./data/',
                              train=False,
                              transform=test_transform,
                              download=True)



# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


def error_plot(train_error, test_error):
    plt.xticks(np.arange(0, 101, 1))
    plt.yticks(np.arange(0, 101, 1))
    plt.figure(dpi=125)
    #plt.plot(acc)
    plt.plot(train_error, label="Train")
    plt.plot(test_error, label="Test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Train & Test Error q1")
    plt.ylim(0,101)
    plt.show
    plt.savefig("plot_error_q1.png")

def loss_plot(train_loss, test_loss):
    plt.xticks(np.arange(0, 101, 1))
    plt.yticks(np.arange(0, 101, 1))
    plt.figure(dpi=125)
    #plt.plot(acc)
    plt.plot(train_loss, label="Train")
    plt.plot(test_loss, label="Test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Train & Test Loss q1")
    plt.ylim(0,2)
    plt.show
    plt.savefig("plot_loss_q1.png")

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


def train_model_q1():
    cnn = CNN()
    cnn.apply(weights_init)

    if torch.cuda.is_available():
        cnn = cnn.cuda()

    # convert all the weights tensors to cuda()
    # Loss and Optimizer

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))


    train_error = []
    test_error = []
    train_loss = []
    test_loss = []


    for epoch in range(num_epochs):
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            outputs = cnn(images)

            _, predicted_train = torch.max(outputs.data, 1)  #error_clac
            total_train += labels.size(0)  #error_clac
            correct_train += (predicted_train == labels).sum()  #error_clac

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1,
                         len(train_dataset) // batch_size, loss.data))

        train_error.append(100 - (100 * correct_train / total_train))  #error_clac
        train_loss.append(loss.data)                                    #loss_calc
        cnn.eval()
        if epoch >= 0:                                                  #error_calc
            correct = 0
            total = 0
            for images, labels in test_loader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = cnn(images)
                loss_t = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            test_loss.append(loss_t.data)
            test_error.append(100-(100 * correct / total))                #error_calc
            print('Test Accuracy of the model on the 10000 test images: %.3f %%' % (100 * correct / total))
        cnn.train()

    cnn.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %.3f %%' % (100 * correct / total))



    # Plot
    error_plot(train_error, test_error)
    loss_plot(train_loss, test_loss)


    # Save the Trained Model
    torch.save(cnn.state_dict(), 'model_q1.pkl')
    return cnn.state_dict()

train_model_q1()