
import torch
from torch import nn 
from torch.utils.data import DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
from eevbnn_models import load_data, FILEPATH, test
from eevbnn_augmentation import Data, evaluate_robustness_acc



device = "cuda" if torch.cuda.is_available() else "cpu"


class StandardModel(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),

            nn.Linear(100, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.features(x)
        return logits
    

def train_standard(model, train_dataloader, test_dataloader, epochs=5, verbose=True):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    size = len(train_dataloader.dataset)

    for t in range(epochs):
        if verbose:
            print(f"Epoch {t} " + "---"*10)
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

            if verbose:
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss : {loss:>7f} [{current:>5d}/{size:>5d}]")
        
        # --
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(test_dataloader)
        correct /= len(test_dataloader.dataset)
        if verbose:
            print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%   Avg loss: {test_loss:>8f}")

    return model



def robust_loop_standard(dataset_name, input_size, output_size, epochs=5, epochs_training=5,
                         n_imgs_created=500, recreate_model=True, first_sample_limit=3000,
                         batch_size=64, verbose=True):
    train_set, test_set = load_data(dataset_name)
    dtrain = Data(train_set, limit=first_sample_limit)
    dtest = Data(test_set, limit=first_sample_limit)
    model = None

    with open(f"{FILEPATH}/saved_generated_data/gendata_40000_eps0.5_mS1_mnist_fixed.pickle", "rb") as file:
        adv_imgs = pickle.load(file)

    with open(f"{FILEPATH}/saved_generated_data/gendata_10000_eps0.5_mS1_TEST_fixed.pickle", "rb") as file:
        adv_imgs_test = pickle.load(file)

    test_values = []
    robust_values = []

    for idx_robust in range(epochs):
        train_dataloader = DataLoader(dtrain, batch_size=batch_size)
        test_dataloader = DataLoader(dtest, batch_size=batch_size)

        if recreate_model or idx_robust == 0:
            model = StandardModel(input_size, output_size)

        model = train_standard(model, train_dataloader, test_dataloader, epochs=epochs_training, verbose=False)
        test_acc, _ = test(test_dataloader, model, nn.CrossEntropyLoss(), verbose=False)
        test_acc *= 100
        robust_acc = evaluate_robustness_acc(model, adv_imgs_test, dataset_name="MNIST")

        test_values.append(test_acc)
        robust_values.append(robust_acc)

        if verbose:
            print(f"Robust Epoch [{idx_robust} / {epochs}]" + "---"*20)
            print(f"  Test accuracy   : {test_acc:.3f}")
            print(f"  Robust accuracy : {robust_acc:.3f}")

        ## augment data with adv img
        if idx_robust != epochs-1:
            start = idx_robust * n_imgs_created
            end = (idx_robust + 1) * n_imgs_created
            new_img = adv_imgs[start:end]

            dtrain.augment(new_img, shuffle=True)

            if verbose:
                print(f"Added data to train ({n_imgs_created}) set. New sizes :\n  train size : {len(dtrain)}\n  test  size : {len(dtest)}")
                
    return test_values, robust_values











if __name__ == "__main__":
    
    ## simple training of the model
    # train_set, test_set = load_data("MNIST")
    # train_dataloader = DataLoader(train_set, batch_size=64)
    # test_dataloader = DataLoader(test_set, batch_size=64)

    # model = StandardModel(28*28, 10).to(device)
    # train_standard(model, train_dataloader, test_dataloader, epochs=25, verbose=True)


    ## robust train loop

    epochs = 25

    test_vals, rob_vals = robust_loop_standard("MNIST", 28*28, 10, epochs=epochs, epochs_training=15,
                                               n_imgs_created=500, recreate_model=True, first_sample_limit=3000,
                                               batch_size=64, verbose=True)
    
    with open(f"{FILEPATH}/saved_graphs/16_robust_values_standardModel.pickle", "wb") as file:
        pickle.dump(rob_vals, file)
    with open(f"{FILEPATH}/saved_graphs/16_test_values_standardModel.pickle", "wb") as file:
        pickle.dump(test_vals, file)

    x = [i for i in range(epochs)]
    plt.plot(x, rob_vals, label="adversarial")
    plt.plot(x, test_vals, label="original")
    plt.xlabel("robust epochs")
    plt.ylabel("accuracy [%]")
    plt.legend()
    plt.savefig(f"{FILEPATH}/saved_graphs/16_robust_Loop_standardModel.pdf")

    

    



        




















