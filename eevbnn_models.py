
#################################################################
#                                                               #
# Master's thesis : Binarized Neural Networks                   #
# eevbnn models for the script                                  #
#                                                               #
# Author : Benoit Ronval                                        #
#                                                               #
#################################################################


## IMPORT

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from eevbnn.satenv import SolveResult
from eevbnn.eval_bin import ModelVerifier, init_argparser
from eevbnn.net_bin import BinLinear, BinConv2d, BinConv2dPos, BinLinearPos, InputQuantizer, MnistMLP, TernaryWeightWithMaskFn, SeqBinModelHelper, Binarize01Act, BatchNormStatsCallbak, setattr_inplace
from eevbnn.utils import Flatten, ModelHelper, torch_as_npy
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle
import time



#======================================================================================================================================
## CONSTANTS


DATASET_NAMES       = ["MNIST", "CIFAR10", "Fashion"]
INPUT_OUTPUT_SIZES  = [(28*28, 10), (32*32*3, 10), (28*28, 10)]
MNIST_CLASSES       = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
CIFAR_CLASSES       = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
FASHION_CLASSES     = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MODEL_TYPES         = [1, 2, 3, 4]
FILEPATH            = "/home/ben/Documents/Masters_thesis/eevbnn/"

device = "cuda" if torch.cuda.is_available() else "cpu"
w_binarizer = TernaryWeightWithMaskFn  ## necessary for the bin layers in eevbnn


#======================================================================================================================================
## MODELS ARCHITECTURES


class Model1(SeqBinModelHelper, nn.Module, ModelHelper):
    def __init__(self, input_size, output_size, quant_value=0.1):
        """
        Creates a MLP with following architecture\\
        InputQuantizer\\
        Flatten\\
        linear (input_size, 512)\\
        batchNorm + act\\
        linear (512, 512)\\
        bacthNorm + act\\
        linear (512, 100)\\
        bacthNorm + act\\
        linear (512, output_size)\\
        bacthNorm
        """
        super().__init__()
        self.features = nn.Sequential(
            InputQuantizer(quant_value),
            Flatten(),
            BinLinearPos(w_binarizer, input_size, 512, rounding=False),
            setattr_inplace(BatchNormStatsCallbak(self, 512), 'bias_regularizer_coeff', 0),
            Binarize01Act(),

            BinLinearPos(w_binarizer, 512, 512, rounding=False),
            BatchNormStatsCallbak(self, 512),
            Binarize01Act(),

            BinLinearPos(w_binarizer, 512, 100, rounding=False),
            BatchNormStatsCallbak(self, 100),
            Binarize01Act(),

            BinLinearPos(w_binarizer, 100, output_size, rounding=False),
            setattr_inplace(BatchNormStatsCallbak(self, output_size, use_scalar_scale=True), 'bias_regularizer_coeff', 0)
        )

    def forward(self, x):
        logits = self.features(x)
        return logits


class Model2(SeqBinModelHelper, nn.Module, ModelHelper):
    def __init__(self, input_size, output_size, quant_value=0.1):
        """
        Create a MLP model with following architecture\\
        InputQuantizer\\
        Flatten\\
        linear (input_size, 50)\\
        bacthNorm + act\\
        linear (50, 50)\\
        batchNorm + act\\
        linear (50, output_size)\\
        batchNorm
        """
        super().__init__()
        self.features = nn.Sequential(
            InputQuantizer(quant_value),
            Flatten(),
            BinLinearPos(w_binarizer, input_size, 50, rounding=False),
            setattr_inplace(BatchNormStatsCallbak(self, 50), "bias_regularizer_coeff", 0),
            Binarize01Act(),

            BinLinearPos(w_binarizer, 50, 50, rounding=False),
            BatchNormStatsCallbak(self, 50),
            Binarize01Act(),

            BinLinearPos(w_binarizer, 50, output_size, rounding=False),
            setattr_inplace(BatchNormStatsCallbak(self, output_size, use_scalar_scale=True), "bias_regularizer_coeff", 0)
        )

    def forward(self, x):
        logits = self.features(x)
        return logits


class Model3(SeqBinModelHelper, nn.Module, ModelHelper):
    def __init__(self, input_size, output_size, quant_value=0.1):
        """ 
        Create a model with the following architecture\\
        InputQuantizer\\
        Flatten\\
        linear (input_size, 50)\\
        bacthNorm + act\\
        linear (50, 200)\\
        batchNorm + act\\
        linear (200, 400)\\
        batchNorm + act\\
        linear (400, output_size)\\
        batchNorm
        """
        super().__init__()
        self.features = nn.Sequential(
            InputQuantizer(quant_value),
            Flatten(),
            BinLinearPos(w_binarizer, input_size, 50, rounding=False),
            setattr_inplace(BatchNormStatsCallbak(self, 50), "bias_regularizer_coeff", 0),
            Binarize01Act(),

            BinLinearPos(w_binarizer, 50, 200, rounding=False),
            BatchNormStatsCallbak(self, 200),
            Binarize01Act(),

            BinLinearPos(w_binarizer, 200, 400, rounding=False),
            BatchNormStatsCallbak(self, 400),
            Binarize01Act(),

            BinLinearPos(w_binarizer, 400, output_size, rounding=False),
            setattr_inplace(BatchNormStatsCallbak(self, output_size, use_scalar_scale=True), "bias_regularizer_coeff", 0)
        )

    def forward(self, x):
        logits = self.features(x)
        return logits


class Model4(SeqBinModelHelper, nn.Module, ModelHelper):
    def __init__(self, in_channel, output_size, mult, quant_value=0.1):
        """ 
        Create a model with the following architecture:\\
        TODO
        """
        super().__init__()
        self.features = nn.Sequential(
            InputQuantizer(quant_value),
            BinConv2dPos(w_binarizer, in_channel, 32, kernel_size=4, stride=2, padding=1, rounding=False), 
            setattr_inplace(BatchNormStatsCallbak(self, 32), "bias_regularizer_coeff", 0),
            Binarize01Act(),

            BinConv2dPos(w_binarizer, 32, 64, kernel_size=4, stride=2, padding=1),
            BatchNormStatsCallbak(self, 64),
            Binarize01Act(),

            Flatten(),
            BinLinearPos(w_binarizer, 64*mult**2, 512),
            BatchNormStatsCallbak(self, 512),
            Binarize01Act(),

            BinLinearPos(w_binarizer, 512, 256),
            BatchNormStatsCallbak(self, 256),
            Binarize01Act(),

            BinLinearPos(w_binarizer, 256, output_size),
            setattr_inplace(BatchNormStatsCallbak(self, output_size, use_scalar_scale=True), "bias_regularizer_coeff", 0)
        )


class ModelQuant(SeqBinModelHelper, nn.Module, ModelHelper):
    def __init__(self, quant_value=0.1):
        super().__init__()
        self.features = nn.Sequential(
            InputQuantizer(quant_value)
        )

    def forward(self, x):
        logits = self.features(x)
        return logits



#======================================================================================================================================
## DATA FUNCTIONS


def load_data(name="MNIST"):
    """
    Load the given dataset
    
        Parameters:
                name (str) : the name of the dataset to load. Either MNIST, CIFAR10 or Fashion
                             by default : MNIST
        
        Returns:
                (training_data, test_data) for the wanted dataset
    """
    training_data, test_data = None, None
    if name == "MNIST":
        training_data = datasets.MNIST(
                            root="data",
                            train=True,
                            download=True,
                            transform=ToTensor()
                        )
        test_data = datasets.MNIST(
                        root="data",
                        train=False,
                        download=True,
                        transform=ToTensor()
                    )
    elif name == "CIFAR10":
        training_data = datasets.CIFAR10(
                            root="data",
                            train=True,
                            download=False,   # to download, pass to True
                            transform=ToTensor()
                        )
        test_data = datasets.CIFAR10(
                        root="data",
                        train=False,
                        download=False,     # to download, pass to True
                        transform=ToTensor()
                    )
    elif name == "Fashion":
        training_data = datasets.FashionMNIST(
                            root="data",
                            train=True,
                            download=True,
                            transform=ToTensor()
                        )
        test_data = datasets.FashionMNIST(
                        root="data",
                        train=False,
                        download=True,
                        transform=ToTensor()
                    )
    
    if training_data is None or test_data is None:
        print(f"Error when loading the data\nNo matching found for dataset {name}")
    return training_data, test_data


def get_data(name, batch_size=64):
    """
    Returns train and test dataloader objects from the dataset with the given name
    """
    training_data, test_data = load_data(name=name)
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


def get_classes(dataset_name : str, with_uncertain=False):
    """
    Returns the classes for the given dataset "dataset_name"\\
    dataset_name is either MNIST, CIFAR10 or Fashion (for FashionMNIST)
    """

    if dataset_name == "MNIST":
        if with_uncertain:
            lst = MNIST_CLASSES.copy()
            lst.extend(["adv"])
            return lst
        return MNIST_CLASSES
    elif dataset_name == "CIFAR10":
        if with_uncertain:
            lst = CIFAR_CLASSES.copy()
            lst.extend(["adv"])
            return lst
        return CIFAR_CLASSES
    elif dataset_name == "Fashion":
        if with_uncertain:
            lst = FASHION_CLASSES.copy()
            lst.extend(["adv"])
            return lst
        return FASHION_CLASSES
    print(f"ERROR get_classes : invalid name : {dataset_name}")
    return -1



#======================================================================================================================================
## TRAIN AND TEST LOOPS


def train(dataloader, model, loss_fn, optimizer, verbose=True):
    """
    Train a model with the given dataset (dataloader) using the given loss function and optimizer
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss : {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, verbose=True):
    """
    Test a model with the given dataset (dataloader) using the given loss function\\
    Returns (accuracy, test_loss)
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if verbose:
        print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%   Avg loss: {test_loss:>8f}")
    return correct, test_loss


def train_test_model(model, train_dataloader, test_dataloader, epochs=5, save=True, filename="my_model.pth", verbose=True):
    """
    Train and test a model with given train/test_dataloader\\
    By default use CrossEntropyLoss as the loss function and Adam (lr=1e-3) as optimizer\\
    
    Return the trained and tested model

    Parameters :
        - model : the model to train
        - train_dataloader : the train dataset as a DataLoader object
        - test_dataloader : the test dataset as a DataLoader object
        - epochs : (default=5) the number of epochs to train the model
        - save : (default=True) wether to save the trained model or not
        - filename : (default=my_model.pth) the name for the saved model file
        - verbose : wether to show information during training and testing or not

    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(epochs):
        if verbose:
            print(f"Epoch {t+1}\n------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, verbose=verbose)
        test(test_dataloader, model, loss_fn, verbose=verbose)

    if save:
        filename = f"{FILEPATH}/saved_models_eevbnn/{filename}"
        torch.save(model.state_dict(), filename)
        if verbose:
            print("model saved as : " + filename)
    return model



#=====================================================================================================================================
## LOAD FUNCTIONS


def load_model(model_type : int, dataset_name : str, quant_value=None, show_test_acc=True):
    """
    Get a model with the given type (1, 2, 3, 4) for the given dataset

    Returns the loaded model

    Parameters:
        - model_type    : the type of the model (1, 2, 3 or 4)
        - dataset_name  : the name of the dataset on which the model has been trained (MNIST, CIFAR10 or Fashion)
        - quant_value   : (default=None) the value for input quantization used for the model. If None load quant=0.1
        - show_test_acc : (default=True) wether to test the model accuracy when loaded
    """
    if dataset_name not in DATASET_NAMES:
        print(f"ERROR load model function\n Invalid dataset name : {dataset_name}")
        return
    filename = f"{FILEPATH}/saved_models_eevbnn/model{model_type}_{dataset_name}.pth"
    if quant_value != None:
        filename = f"{FILEPATH}/saved_models_eevbnn/models_1_quant/model{model_type}_{dataset_name}_{quant_value}.pth"
    data_idx = DATASET_NAMES.index(dataset_name)
    input_size = INPUT_OUTPUT_SIZES[data_idx][0]
    output_size = INPUT_OUTPUT_SIZES[data_idx][1]
    model = None
    if model_type == 1:
        model = Model1(input_size, output_size)
    elif model_type == 2:
        model = Model2(input_size, output_size)
    elif model_type == 3:
        model = Model3(input_size, output_size)
    elif model_type == 4:
        input_size = 1
        mult = 7
        if dataset_name == "CIFAR10":
            input_size = 3
            mult = 8
        model = Model4(input_size, output_size, mult)
    elif model_type == 5:
        raise NotImplementedError()
    else:
        print(f"ERROR load_model : incorrect type : {model_type}")
    model.load_state_dict(torch.load(filename))

    if show_test_acc:
        model.eval()
        test(get_data(dataset_name)[1], model, nn.CrossEntropyLoss())

    return model


def load_all_models():
    """
    Loads all the models saved and returns them in a list
    """
    res = []
    for t in MODEL_TYPES:
        for d in DATASET_NAMES:
            res.append(load_model(t, d, show_test_acc=False))
    return res


def get_eval_models(models):
    res = []
    for m in models:
        res.append(m.cvt_to_eval())
    return res



#=====================================================================================================================================
## STATS MODELS


def get_stat_model(model_type : int, dataset_name : str, quant_value=None):
    """
    Gets the accuracy and the sparisty of the given model

    Prints the information

    Parameters:
        - model_type : the type of the model (1, 2, 3 or 4)
        - dataset_name : the name of the dataset (MNIST, CIFAR10 or Fashion)
    """
    print(f"Stats for model {model_type} on {dataset_name} dataset")
    model = load_model(model_type, dataset_name, show_test_acc=True, quant_value=quant_value)
    spar_layers, n_zeros, n_params = model.get_sparsity_stat()
    print(f"Sparisty of model (# zeros / # weights):\n {n_zeros} / {n_params} = {(n_zeros / n_params) * 100:.2f}%")













