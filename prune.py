import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from core.model.lenet import LeNet
from torch.utils.data import DataLoader

"""
	Step 1:
    -------
		Implement the pruning algorithm to remove weights which are smaller than the given threshold (prune_model)
	Step 2:
    -------
		As you may know after pruning, the accuracy drops a lot. To recover the accuracy, we need to do the fine-tuning.
		It means, we need to retrain the network for few epochs. Use prune_model method which you have implemented in
		step 1 and then fine-tune the network to regain the accuracy drop (prune_model_finetune)

    *** You need to install torch 0.3.1 on ubuntu
"""


def get_mnist_dataloaders(root, batch_size):
    """
    This function should return a pair of torch.utils.data.DataLoader.
    The first element is the training loader, the second is the test loader.
    Those loaders should have been created from the MNIST dataset available in torchvision.

    Train_Loader:
        - Resize to 32x32
        - Randomly do a horizontal flip
        - Normalize using mean 0.1307 and std 0.3081

    Test_Loader:
        - Resize to 32x32
        - Normalize using mean 0.1307 and std 0.3081

    :param root: Folder where the dataset will be downloaded into.
    :param batch_size: Number of samples in each mini-batch
    :return: tuple of train_loader and test_loader
    """
    # download and transform train dataset
    train_loader = DataLoader(datasets.MNIST(root,download=True,train=True,
                                                transform=transforms.Compose([
                                                                    transforms.Resize((32,32)), # resize the  image to fit the model
                                                                    transforms.RandomHorizontalFlip(), # Performs a random horizontal flip
                                                                    transforms.ToTensor(), # convert image to PyTorch tensor
                                                                    transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                                    ])),
                                                batch_size=batch_size,
                                                shuffle=True)

    # download and transform test dataset
    test_loader = DataLoader(datasets.MNIST(root,download=True,train=False,
                                                transform=transforms.Compose([
                                                                    transforms.Resize((32,32)), # resize the image to fit the model
                                                                    transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                                    transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                                    ])),
                                                batch_size=batch_size,
                                                shuffle=True)

    return (train_loader, test_loader)


def get_accuracy_top1(model, data_loader):
    """
    This function should return the top1 accuracy in percentage of the model on the given data loader.
    :param model: LeNet object
    :param data_loader: torch.utils.data.DataLoader
    :return: float
    """
    if torch.cuda.is_available():
        model.cuda()
    total = 0
    correct = 0
    for inputs,targets in data_loader:
        if torch.cuda.is_available():
            inputs , targets = inputs.cuda(),targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        outputs = model(inputs)
        _,predicted = torch.max(outputs.data,1)
        total+=targets.size(0)
        correct+=(predicted==targets.data).sum()
    return (100*float(correct)/total)

def prune_model(model, threshold):
    """
    This function should set the model's weight to 0 if their absolutes values are lower or equal to the given threshold.
    :param model: LeNet object
    :param threshold: float
    """
    for params in model.parameters():
        cond = torch.abs(params.data)<=threshold
        params.data[cond]=0

def prune_model_finetune(model, train_loader, test_loader, threshold):
    """
    This function should first set the model's weight to 0 if their absolutes values are lower or equal to the given threshold.
    Then, it should finetune the model by making sure that the weights that were set to zero remain zero after the gradient descent steps.
    :param model: LeNet object
    :param train_loader: training set torch.utils.data.DataLoader
    :param test_loader: testing set torch.utils.data.DataLoader
    :param threshold: float
    """
    # setting the indexes to be pruned to zeros based on threshold
    prune_model(model,threshold)
    cond_dict={}
    for index,params in enumerate(model.parameters()):
        cond = torch.abs(params.data)==threshold
        cond_dict[index] = cond

    # Fine tuning the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    if torch.cuda.is_available():
        model.cuda()

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            if torch.cuda.is_available():
                inputs , labels = inputs.cuda(),labels.cuda()
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            # finding the gradient
            loss = criterion(outputs, labels)
            loss.backward()

            # ensuring that the model doesn't update parameters which are already
            for index,params in enumerate(model.parameters()):
                cond = cond_dict[index]
                params.grad.data[cond]=0

            # optimization step
            optimizer.step()





if __name__ == '__main__':
    # get the model
    model = torch.load('lenet.pth')
    # setting the parameters
    root ='data/'
    batch_size = 32
    threshold = 0.2

    # get the train loader and test loader
    train_loader, test_loader =  get_mnist_dataloaders(root, batch_size)

    # check the performance before pruning
    print("Training Accuracy:",get_accuracy_top1(model,train_loader))
    print("Testing Accuracy:",get_accuracy_top1(model,test_loader))

    # Naive pruning
    prune_model(model,threshold)
    print("Naive Pruning Done.....")
    # check the performance after naive pruning
    print("Training Accuracy after naive pruning:",get_accuracy_top1(model,train_loader))
    print("Testing Accuracy after naive pruning:",get_accuracy_top1(model,test_loader))

    # Finetuning the pruned model
    prune_model_finetune(model,train_loader,test_loader,threshold)
    # check the performance after finetuning
    print("Training Accuracy after finetuning:",get_accuracy_top1(model,train_loader))
    print("Testing Accuracy after finetuning:",get_accuracy_top1(model,test_loader))
