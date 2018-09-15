import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from core.model.lenet import LeNet
from torch.utils.data import DataLoader
import argparse

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
class prune_network:
    def __init__(self,model,args):
        """
        Initializes the class and sets up hyper parameters
        """
        self.model = model     # model to be worked upon
        self.root = args.root  # location to save the dataset
        self.batch_size = args.batch_size # batch size for training and testing
        self.verbose = args.verbose # decides the verbosity of the entire process
        self.get_mnist_dataloaders() # creates the dataloader
        self.use_cuda = torch.cuda.is_available()  # flag to decide if or not to use cuda
        self.fine_tune = not args.do_not_fine_tune # flag for fine tuning
        self.threshold = args.threshold
        self.epochs = args.epochs # number of epochs the network has to be trained for
        if self.use_cuda:
            self.model.cuda() # convert model from a cpu based model to a gpu based model

        # other parameters currently fixed can be later given to argmax if needed
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters())

        # if traing from scratch is enabled
        if args.train:
            self.train(self.epochs)

    def get_mnist_dataloaders(self):
        """
        Function Creates 2 data loaders one for train and other for test

        Train_Loader:
        ---------------
            - Resize to 32x32
            - Randomly do a horizontal flip
            - Normalize using mean 0.1307 and std 0.3081

        Test_Loader:
        -------------
            - Resize to 32x32
            - Normalize using mean 0.1307 and std 0.3081

        Class Parameters:
        -----------------
        - root       : location where the data is saved [string]
        - batch_size : batch_size [int]
        - verbose    : prints out all the details

        """
        # download and transform train dataset
        self.train_loader = DataLoader(datasets.MNIST(self.root,download=True,train=True,
                                                    transform=transforms.Compose([
                                                                        transforms.Resize((32,32)), # resize the  image to fit the model
                                                                        transforms.RandomHorizontalFlip(), # Performs a random horizontal flip
                                                                        transforms.ToTensor(), # convert image to PyTorch tensor
                                                                        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                                        ])),
                                                    batch_size=self.batch_size,
                                                    shuffle=True)
        if self.verbose:
            print("Done Creating Train Loader")
        # download and transform test dataset
        self.test_loader = DataLoader(datasets.MNIST(self.root,download=True,train=False,
                                                    transform=transforms.Compose([
                                                                        transforms.Resize((32,32)), # resize the image to fit the model
                                                                        transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                                        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                                        ])),
                                                    batch_size=self.batch_size,
                                                    shuffle=True)
        if self.verbose:
            print("Done Creating Test Loader")

    def get_accuracy_top1(self,data_loader):
        """
        Accuracy in percentage of the model on the given data loader.

        Parameters:
        -----------
        data_loader: the current data loader used [torch.utils.data.DataLoader]

        Class Parameters:
        -----------------
        model : the model being pruned [LeNet Object]

        Returns:
        -------
        Accuracy : returns the Accuracy [float]

        """

        total = 0
        correct = 0
        for inputs,targets in data_loader:
            if self.use_cuda:
                inputs , targets = inputs.cuda(),targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs = self.model(inputs)
            _,predicted = torch.max(outputs.data,1)
            total+=targets.size(0)
            correct+=(predicted==targets.data).sum()
        accuracy = (100*float(correct)/total)
        return accuracy

    def prune_model(self):
        """
        This function sets the model's weight to 0 if their absolutes values are lower or equal to the given threshold.
        Saves a prune mask that can be used in fine tuning stage (delets if not needed)
        Finds the number of parameters that got pruned

        Class Parameters:
        -----------------
        model     : Model being pruned [LeNet Object]
        threshold : value of the threshold for pruning [float]
        fine_tune : decides wether  to keep the masks or not based on this [bool]

        """
        self.prune_mask={}
        self.pruned_params = 0
        for index,params in enumerate(self.model.parameters()):
            # mask is found using the condition that abs parameter is less than threshold
            mask = torch.abs(params.data)<=self.threshold
            self.pruned_params += mask.sum()
            # all parameters satisfying the condition is set to 0
            params.data[mask]=0
            # a dictionary of masks are created for finetuning
            self.prune_mask[index]=mask

        if( not self.fine_tune): # if fine tune is not going to be done deleting mask to save space
            del self.prune_mask

    def train(self,epochs=20,use_mask=False):
        """
        Train function trains the Model using the train_loader

        Parameters :
        -------------
        epochs   : number of epochs to be trained for
        use_mask : if true uses prune mask to mask updates of certain parameters to ensure they don't change

        Class Parameters:
        -----------------
        train_loader : the train loader [DataLoader]
        verbose      : prints progress of training [bool]
        model        : Model being trained [LeNet Object]
        """

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.train_loader):
                # get the inputs
                inputs, labels = data
                if self.use_cuda:
                    inputs , labels = inputs.cuda(),labels.cuda()
                inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward
                outputs = self.model(inputs)
                # finding the gradient
                loss = self.criterion(outputs, labels)
                loss.backward()
                if use_mask:
                    # ensuring that the model doesn't update parameters which are already
                    for index,params in enumerate(self.model.parameters()):
                        mask = self.prune_mask[index]
                        params.grad.data[mask]=0
                # optimization step
                self.optimizer.step()

                running_loss += loss.cpu().data.numpy()[0]
                if self.verbose:
                    if i % 100 == 99:    # print every 100 mini-batches
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                        running_loss = 0.0

    def prune_model_finetune(self):
        """
        This function first sets the model's weight to 0 if their absolutes values are lower or equal to the given threshold.
        Then, it finetunes the model by making sure that the weights that were set to zero remain zero after the gradient descent steps.
        Class Parameters used:
        ----------------------
        model     : Model being pruned [LeNet object]
        threshold : Threshold to be pruned to [float]
        """
        # setting the indexes to be pruned to zeros based on threshold
        self.prune_model()
        self.train(epochs=3,use_mask=True) # after detailed study i feel 3 is a good enough number of epochs



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",help="if set starts training from scartch",
                            action = "store_true")
    parser.add_argument("-b","--batch_size",default = 32,help="Set the batch_size")
    parser.add_argument("-t","--threshold",default = 0.2,help="Set the threshold for pruning")
    parser.add_argument("--verbose", action="store_true",
                            help="increase output verbosity")
    parser.add_argument("-df","--do_not_fine_tune",default = False,action="store_true",
                        help="Set to not run fine tuning after pruning")
    parser.add_argument("-r","--root",default="data/",help="location where the dataset will be downloaded")
    parser.add_argument("-ploc","--pretrained_model_location",default="lenet.pth",help="Set the location of the pretrained_model")
    parser.add_argument("--epochs",type=int,default=20,help="Number for epochs for training")
    return parser.parse_args()




if __name__ == '__main__':
    args=get_args()
    # get the model
    if args.train:
        model = LeNet()
        if args.verbose:
            print "Created the model"
    else:
        model = torch.load(args.pretrained_model_location)
        if args.verbose:
            print "loaded the pretrained model weights"

    prune_net = prune_network(model,args)

    # check the performance before pruning
    print "Training Accuracy:",prune_net.get_accuracy_top1(prune_net.train_loader)
    print "Testing Accuracy:",prune_net.get_accuracy_top1(prune_net.test_loader)

    # Naive pruning
    prune_net.prune_model()
    if args.verbose:
        print "Naive Pruning Done....."
    print "Number of Parameters pruned : ",prune_net.pruned_params
    # check the performance after naive pruning
    print "Training Accuracy after naive pruning:",prune_net.get_accuracy_top1(prune_net.train_loader)
    print "Testing Accuracy after naive pruning:",prune_net.get_accuracy_top1(prune_net.test_loader)

    # Finetuning the pruned model
    prune_net.prune_model_finetune()
    # check the performance after finetuning
    print "Training Accuracy after finetuning:",prune_net.get_accuracy_top1(prune_net.train_loader)
    print "Testing Accuracy after finetuning:",prune_net.get_accuracy_top1(prune_net.test_loader)
