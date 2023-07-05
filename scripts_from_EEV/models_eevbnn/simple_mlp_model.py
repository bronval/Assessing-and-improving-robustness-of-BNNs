
# implement a simple mlp model, train + test it and verify it

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from eevbnn.satenv import SolveResult
from eevbnn.eval_bin import ModelVerifier, init_argparser
from eevbnn.net_bin import BinLinear, BinConv2d, BinLinearPos, InputQuantizer, MnistMLP, TernaryWeightWithMaskFn, SeqBinModelHelper, Binarize01Act, BatchNormStatsCallbak, setattr_inplace
from eevbnn.utils import Flatten, ModelHelper, torch_as_npy
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle


#--------------------- get the data MNIST

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

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader  = DataLoader(test_data, batch_size=batch_size)

### just to check the shapes
# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break



# ------------------- build model

device = "cuda" if torch.cuda.is_available() else 'cpu'

w_binarizer = TernaryWeightWithMaskFn     # necessary for the layers


class SimpleMLP(SeqBinModelHelper, nn.Module, ModelHelper):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            InputQuantizer(0.1),
            Flatten(),
            BinLinearPos(w_binarizer, 28*28, 512, rounding=False),
            setattr_inplace(BatchNormStatsCallbak(self, 512), 'bias_regularizer_coeff', 0),
            Binarize01Act(),

            BinLinearPos(w_binarizer, 512, 512, rounding=False),
            BatchNormStatsCallbak(self, 512),
            Binarize01Act(),

            BinLinearPos(w_binarizer, 512, 100, rounding=False),
            BatchNormStatsCallbak(self, 100),
            Binarize01Act(),

            BinLinearPos(w_binarizer, 100, 10, rounding=False),
            setattr_inplace(BatchNormStatsCallbak(self, 10, use_scalar_scale=True), 'bias_regularizer_coeff', 0)
        )

    def forward(self, x):
        logits = self.features(x)
        return logits

model = SimpleMLP().to(device)
print(model, "\n------------------------------------------------------------------\n")


#------------------------- train and test model

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
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
    correct   /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}% Avg loss: {test_loss:>8f} \n")


def train_test_model(save=True, epochs=5, model=model, name="simple_mlp_model.pth"):
    """
    Simple function to run train and test on model with given number of epochs\\
    If save is True, saves the model in file with the given name (in .pth)
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(epochs):
        print(f"Epoch {t+1}\n---------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    if save:
        torch.save(model.state_dict(), "saved_models_eevbnn/" + name)

    return model


# model = train_test_model(epochs=3)
model = SimpleMLP()
model.load_state_dict(torch.load("simple_mlp_model.pth"))



#---------------------------- output weights of BNN


# names = []
# params = []
# for name, param in model.named_parameters():
#     print(name)
#     names.append(name)
#     params.append(param)

# with open("simple_mlp_model_weights_names.pickle", "wb") as file:
#     pickle.dump(names, file)

# with open("simple_mlp_model_weights.pickle", "wb") as file:
#     pickle.dump(params, file)
    


weights_tf = None
with open("weights_tf.pickle", "rb") as file:
    weights_tf = pickle.load(file)

print(weights_tf.shape)






#--------------------------- nice functions for visualization

classes = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]


### some code just to test the model
# model.eval()
# for  i in range(10):
#     x, y = test_data[i][0], test_data[i][1]
#     with torch.no_grad():
#         pred = model(x)
#         predicted, actual = classes[pred[0].argmax(0)], classes[y]
#         print(f"Predicted: '{predicted}', Actual : '{actual}'")


def show_picture(data, index, adv=False, img=None, eps=None, clsfy=None):
    """
    Function to save the given picture\\
    data is the dataset, index the index of the image in data\\
    If adv is True, data and index are ignored and img must contain the data of the adversarial image computed with the eps value
    """
    if img is None:
        img, label = data[index]
    plt.imshow(img.squeeze(), cmap='gray')
    file_name = ""
    if adv:
        plt.title(f"classified as : {clsfy} (adv, eps={eps})")
        file_name = "imgs_mnist_dataset/mnist_image_" + str(index) + "_adv_" + str(eps) + ".png"
    else:
        plt.title(f"classified as : {clsfy}")
        file_name = "imgs_mnist_dataset/mnist_image_" + str(index) + ".png"
    plt.savefig(file_name)
    # plt.show()

# show_picture(test_data, 5)




# ------------------------------- verify model

eps = 0.08

parser = argparse.ArgumentParser(
        description='evaluate robustness of 0/1 binaried network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
init_argparser(parser)
parser.add_argument('-e', '--eps', type=float, default=eps,    # classical value : 0.08
                    help='adv input eps on 0-1 scale')          # nb : modif from file required=True -> default=0.08
parser.add_argument('--model', default=model)
args = parser.parse_args()

model_eval = model.cvt_to_eval()
# test_model_eval = test_model.cvt_to_eval()


verifier = ModelVerifier(args, [model_eval])



# nb : avoid 0, 
idxs = [1, 5, 42, 50]
idxs = []
for idx in idxs:
    for inputs, labels in test_dataloader:

        model.eval()
        with torch.no_grad():
            pred = model(inputs[idx])
            predicted, actual = classes[pred[0].argmax(0)], classes[test_data[idx][1]]
            print(f"predicted : {predicted}\t actual : {actual} (idx : {idx})")

        inputs = torch_as_npy(inputs)
        labels = torch_as_npy(labels)

        check_result, (time_build, time_verif), inp_adv, clsfy_scores = verifier.check(inputs[idx], labels[idx])
        print(f"Result : {check_result}\n build time : {time_build:.3f}\t verif time : {time_verif:.3f}")
        clsfy_adv = classes[clsfy_scores[0][0].argmax(0)]
        
        if check_result == SolveResult.SAT:
            show_picture(test_data, idx, adv=False, clsfy=predicted)
            show_picture(test_data, idx, adv=True, img=inp_adv, eps=eps, clsfy=clsfy_adv)
            print("images saved")

        break




### use the code below to go through all the data
### WARNING some inputs take a lot of time

# idx = 1
# for inputs, labels in test_dataloader:
#     inputs = torch_as_npy(inputs)
#     labels = torch_as_npy(labels)
#     for i in range(1, inputs.shape[0]):
#         i = 42
#         # verifier.check_and_save(idx, inputs[i], labels[i])

#         check_result, (time_build, time_verif), inp_adv = verifier.check(inputs[i], labels[i])
#         print(f"Result : {check_result}\n build time : {time_build:.3f}\t verif time : {time_verif:.3f}")
        
#         if check_result == SolveResult.SAT:
#             show_picture(test_data, i, adv=False)
#             show_picture(test_data, i, adv=True, img=inp_adv)
#             print("images saved")

#         # print(verifier.log)

#         break
#     idx += 1
#     break





print("DONE !")







