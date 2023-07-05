
#################################################################
#                                                               #
# Master's thesis : Binarized Neural Networks                   #
# data augmentation using eevbnn                                #
#                                                               #
# Author : Benoit Ronval                                        #
#                                                               #
#################################################################


## Import

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from eevbnn.satenv import SolveResult
from eevbnn.eval_bin import ModelVerifier, init_argparser
from eevbnn.net_bin import BinLinear, BinConv2d, BinConv2dPos, BinLinearPos, InputQuantizer, MnistMLP, TernaryWeightWithMaskFn, SeqBinModelHelper, Binarize01Act, BatchNormStatsCallbak, setattr_inplace
from eevbnn.utils import Flatten, ModelHelper, torch_as_npy
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle
import time
import random
from scipy.special import softmax

from eevbnn_figures import *
from eevbnn_models import *
from eevbnn_verif import *



#======================================================================================================================================
## Generation of data



### DATASET CLASS ######################################################################

class Data(Dataset):

    def __init__(self, data, limit=None):
        self.data = []
        if limit is None:
            limit = len(data)

        for i in range(limit):
            d = data[i]
            self.data.append((d[0], d[1]))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        return sample


    def augment(self, new_data, shuffle=False, seed=0):
        """
        Add the new_data to the current data
        
        new_data must be in the format [(tensor, int)]
        """
        current_shape = self.data[0][0].shape

        for d in new_data:
            if d[0].shape != current_shape:
                print(f"Error, bad shape in new_data. Should be {current_shape} and is {d[0].shape}")
        
        self.data.extend(new_data)

        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)



### ROBUSTNESS EVALUATION ######################################################################


def evaluate_robustness_search(model, dataset_name, eps=0.5, n_imgs=100, imgs=None):
    """
    Evaluate robustness of a model.
    For n_imgs images, computes the percentage of SAT cases (number of times we found an adversarial image)
    Uses the provided eps value.
    If search reaches time out, consider it as robust (UNSAT)
    A model is totally robust if the ratio = 0, not robust if ratio = 1
    
    Parameters:
        - model : the model whose robustness is evaluated
        - dataset_name : the name of the dataset used
        - eps : the epsilon value used when searching adversarial images
        - n_imgs : the number of images to consider to compute the SAT ratio
        - imgs : the images to use for the verification, format : [(image, label)]. Uses random indexes if None
    """
    model = model.cvt_to_eval()
    count = 0
    args = get_args(eps, model)
    classes = get_classes(dataset_name)
    _, test_set = load_data(dataset_name)

    img, true_label = None, None

    if imgs is None:
        imgs = []
        for _ in range(n_imgs):
            idx = random.randint(0, len(test_set)-1)
            img, true_label = test_set[idx]
            imgs.append((img, true_label))
            
    for img, true_label in imgs:
        check_result, adv_img = search_adv_image(model, img, true_label, classes, args)
        if check_result == SolveResult.SAT:
            count += 1

    return count / len(imgs)



def evaluate_robustness_acc(model, imgs, dataset_name="MNIST", with_uncertain=False):
    """
    Evaluate the robustness of a model based on its accuracy on adversarial images

    Parameters:
        - model : the model we want to evaluate
        - imgs : the (adversarial) images used to compute the accuracy, format : [(image, label)]
    """
    score = 0
    classes = get_classes(dataset_name, with_uncertain=with_uncertain)
    with torch.no_grad():
        for img, true_label in imgs:
            if dataset_name == "CIFAR10":
                pred = model(img.reshape(1, 3, 32, 32))[0]
            else:
                pred = model(img.reshape(1, 1, 28, 28))[0]
            pred = classes[pred.argmax(0)]

            if pred == classes[true_label]:
                score += 1

    return 100 * score / len(imgs)


def evaluate_robustness_acc_majority_voting(models, imgs, dataset_name="MNIST", with_uncertain=False):
    """
    Evaluate the robustness of a model based on its accuracy on adversarial images.
    The model here is made of a majority voting of several models given as input

    Parameters:
        - models : a list of models we use for the majority voting
        - imgs : the images to evaluate the models
        - dataset_name : default=MNIST, the name of the dataset we use
        - with_uncertain : default=False, wether there is an "uncertain" (or "adversarial") label in the data
    """

    def get_majority_pred(lst):
        votes = {}
        for v in lst:
            if v in votes:
                votes[v] += 1
            else:
                votes[v] = 1
        return max(votes, key=votes.get)

    score = 0
    classes = get_classes(dataset_name, with_uncertain=with_uncertain)
    with torch.no_grad():
        for img, true_label in imgs:
            all_preds = []
            for model in models:
                if dataset_name == "CIFAR10":
                    pred = model(img.reshape(1, 3, 32, 32))[0]
                else:
                    pred = model(img.reshape(1, 1, 28, 28))[0]
                pred = classes[pred.argmax(0)]
                all_preds.append(pred)
            pred = get_majority_pred(all_preds)

            if pred == classes[true_label]:
                score += 1
    return 100 * score / len(imgs)



def absolute_eval_robustness(model, imgs, dataset_name="MNIST", with_uncertain=False):
    score = 0
    classes = get_classes(dataset_name, with_uncertain=with_uncertain)
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for img, true_label in imgs:
            if dataset_name == "CIFAR10":
                pred = model(img.reshape(1, 3, 32, 32))[0]
            else:
                pred = model(img.reshape(1, 1, 28, 28))[0]
            pred = softmax(pred)
            
            score += (1 - abs(pred[true_label] - max(pred)))
            if true_label == pred.argmax(0):
                n_correct += 1

    score /= len(imgs)
    n_correct /= len(imgs)
    return score, n_correct


def ratio_eval_robustness(model, imgs, dataset_name="MNIST", with_uncertain=False):
    score = 0
    n_correct = 0
    model.eval()
    show = True
    with torch.no_grad():
        for img, true_label in imgs:
            if dataset_name == "CIFAR10":
                pred = model(img.reshape(1, 3, 32, 32))[0]
            else:
                pred = model(img.reshape(1, 1, 28, 28))[0]
            pred = softmax(pred)

            score += (max(pred) / (1 - pred[true_label]))

            if true_label == pred.argmax(0):
                n_correct += 1

            if show:
                show = False
                print(f"pred : {list(pred)}\nmax(pred) : {max(pred)}\npred[true] : {pred[true_label]}\nscore 0 : {max(pred) / (1 - pred[true_label])}")
    
    score /= len(imgs)
    n_correct /= len(imgs)
    return score, n_correct

def ratio_rev_eval_robustness(model, imgs, dataset_name="MNIST", with_uncertain=False):
    score = 0
    n_correct = 0
    model.eval()
    with torch.no_grad():
        for img, true_label in imgs:
            if dataset_name == "CIFAR10":
                pred = model(img.reshape(1, 3, 32, 32))[0]
            else:
                pred = model(img.reshape(1, 1, 28, 28))[0]
            pred = softmax(pred)

            score += (pred[true_label] / (1 - max(pred)))

            if true_label == pred.argmax(0):
                n_correct += 1
    
    score /= len(imgs)
    n_correct /= len(imgs)
    return score, n_correct
    




### IMAGE GENERATION ######################################################################


def generate_random_images(dataset_name, eps, n_images, model, withdraw=True, save_index=True, verbose=True):
    """
    Generate adversarial images with random indexes to add to dataset

    Parameters:
        - dataset_name : name of the dataset to use
        - eps : the value of epsilon, the perturbation, to use to create adversarial images
        - n_images : the number of images to create
        - model : the model on which we run the adversarial search
        - withdraw : (default=True) wether or not to remove index in the random draw
    """

    dataset, _  = load_data(name=dataset_name)
    classes     = get_classes(dataset_name)
    # model       = load_model(model_type, dataset_name, quant_value=quant_value, show_test_acc=False).cvt_to_eval()
    args        = get_args(eps, model)

    n_img_generated = 0
    generated_img = []
    used_idx = set()

    t0 = time.perf_counter()

    while n_img_generated < n_images:
        index = np.random.randint(0, len(dataset))
        if withdraw:
            while index in used_idx:
                index = np.random.randint(0, len(dataset))
        used_idx.add(index)
        img, true_label = dataset[index]

        check_result, img_adv = search_adv_image(model, img, true_label, classes, args)

        if check_result == SolveResult.SAT:
            n_img_generated += 1
            generated_img.append((img_adv, true_label))

        if n_img_generated % 25 == 0 and verbose:
            print(f"Generated [{n_img_generated:>3d} / {n_images}]")

    
    elapsed_time = time.perf_counter() - t0
    if verbose:
        print(f"Generated {n_img_generated} images in {elapsed_time:.3f}s")

    # if save_index:
    #     with open(f"{FILEPATH}/saved_generated_data/index_used.pickle", "wb") as file:
    #         pickle.dump(used_idx, file)

    return generated_img



def generate_set_images(dataset_name, eps, indexes, model, verbose=True):
    """
    Generates adversarial images to be added to the dataset

    Parameters:
        - dataset_name : name of dataset to use
        - eps : the value of the perturbation used when creating adversarial image
        - indexes : list of indexes to use to create the adversarial images
        - model : model on which we run the adversarial search
        - quant_value : (default=0.1) the quantization value of the model
    """

    dataset, _  = load_data(name=dataset_name)
    classes     = get_classes(dataset_name)
    # model       = load_model(model_type, dataset_name, quant_value=quant_value, show_test_acc=False).cvt_to_eval()
    args        = get_args(eps, model)

    n_img_generated = 0
    generated_img = []
    used_idx = []
    current_idx = 0

    t0 = time.perf_counter()

    for index in indexes:
        img, true_label = dataset[index]
        check_result, img_adv = _, _

        if type(model) == list:
            check_result, img_adv = search_adv_img_multiple_models(model, img, true_label, classes, args)
        else:
            check_result, img_adv = search_adv_image(model, img, true_label, classes, args)

        if check_result == SolveResult.SAT:
            n_img_generated += 1
            generated_img.append((img_adv, true_label))
            used_idx.append(index)

        # if n_img_generated % 25 == 0:
        #     print(f"Generated [{n_img_generated:>3d} / {len(indexes)}] in {time.perf_counter() - t0:.3f}s")

        if current_idx % 50 == 0 and verbose:
            print(f"Currently at index [{current_idx:>3d} / {len(indexes)}]\t generated [{n_img_generated:>3d} / {len(indexes)}] in {time.perf_counter() - t0:.3f}s")
        current_idx += 1
    
    elapsed_time = time.perf_counter() - t0

    if verbose:
        print(f"Generated {n_img_generated} / {len(indexes)} images in {elapsed_time:.3f}s")

    # with open(f"{FILEPATH}/saved_generated_data/saved_idxs_set_20000_0.5.pickle", "wb") as file:
    #     pickle.dump(used_idx, file)

    return generated_img, used_idx


def convert_to_tensor(imgs, save=False, filename=""):
    """
    imgs in the format [(ndarray, int)]
    """
    res = []
    for t in imgs:
        ten = torch.from_numpy(t[0])
        res.append((ten, t[1]))

    if save:
        with open(f"{FILEPATH}/saved_generated_data/{filename}.pickle", "wb") as file:
            pickle.dump(res, file)

    return res


def create_noise(dim, low, up):
    """dim : (channel, x, y)"""
    noise = np.zeros(dim)
    for c in range(dim[0]):
        for x in range(dim[1]):
            for y in range(dim[2]):
                noise[c, x, y] += round(random.uniform(low, up), 3)
    return noise


def generate_random_noise_image(dataset_name, eps, idxs, verbose=False, low=0):
    """
    Creates new images with random noise

    Parameters:
        - dataset_name : the name of the dataset to use
        - eps : the max noise value
        - idxs : the corresponding indexes of the images on which we add noise
        - low : default=0, the lower bound for the noise
    """
    _, train_set = load_data(dataset_name)
    new_data = []

    for idx in idxs:
        img, label = train_set[idx]
        # noise = create_noise(img.shape, low, eps)
        
        c, x, y = img.shape
        for i in range(c):
            for j in range(x):
                for k in range(y):
                    noise = round(random.uniform(low, eps), 3)
                    img[i, j, k] += noise
                    if img[i, j, k] > 1:
                        img[i, j, k] -= 1.3 * noise

        new_data.append((img.float(), label))

        if idx % 1000 == 0:
            print(f"at index {idx}")
    
    return new_data


def generate_images_order(dataset_name, eps, start_idx, model, n_gen=500, verbose=True):
    dataset, _  = load_data(name=dataset_name) # TODO rechange dataset to train set !!
    classes     = get_classes(dataset_name)
    # model       = load_model(model_type, dataset_name, quant_value=quant_value, show_test_acc=False).cvt_to_eval()
    args        = get_args(eps, model)

    n_img_generated = 0
    generated_img = []
    used_idx = []
    index = start_idx

    t0 = time.perf_counter()

    while n_img_generated < n_gen:
        img, true_label = dataset[index]
        check_result, img_adv = _, _

        if type(model) == list:
            check_result, img_adv = search_adv_img_multiple_models(model, img, true_label, classes, args)
        else:
            check_result, img_adv = search_adv_image(model, img, true_label, classes, args)

        if check_result == SolveResult.SAT:
            n_img_generated += 1
            generated_img.append((img_adv, true_label))
            used_idx.append(index)

        # if n_img_generated % 25 == 0:
        #     print(f"Generated [{n_img_generated:>3d} / {len(indexes)}] in {time.perf_counter() - t0:.3f}s")

        if n_img_generated % 50 == 0 and verbose:
            print(f"Currently at index {index}\t generated [{n_img_generated:>3d} / {n_gen}] in {time.perf_counter() - t0:.3f}s", flush=True)
        index += 1
    
    elapsed_time = time.perf_counter() - t0

    if verbose:
        print(f"Generated {n_img_generated} / {n_gen} images in {elapsed_time:.3f}s")

    # with open(f"{FILEPATH}/saved_generated_data/saved_idxs_set_10000_0.2_mS1_cifar_TEST.pickle", "wb") as file:
    #     pickle.dump(used_idx, file)

    return generated_img, used_idx, index


def transform_label_uncertain(data, label=10):
    d = []
    for i in range(len(data)):
        d.append((data[i][0], label))
    return d





### ROBUST TRAIN LOOP ######################################################################


def robust_train_loop(dataset_name, model_type, epoch=5, epoch_training=5, eps=0.15,
                      n_img_created=1000, n_img_test=1000, recreate_model=False,
                      quant_value=0.1, first_sample_limit=5000, batch_size=64,
                      input_size=28*28, output_size=10, verbose=True,
                      save=False, filename="robust_model.pickle"):
    """
    Implementation of the robust training loop.
    Loop : train a model on dataset -> check robustness and create adversarial examples -> add created images to dataset

    Parameters:
        - dataset_name : the name of the dataset to use
        - model_type : the type of the model to use (1, 2, 3 or 4)
        - epoch : the number of epoch for the full robust training loop, default=5
        - epoch_training : the number of epoch for a single train of the mode
        - eps : the maximum perturbation value to generate new data, between 0 and 1, default=0.15
        - n_img_created : the number of adversarial images to create, default=1000
        - n_img_test : the number of adversarial images to add to the test set, default=1000
        - recreate_model : recreate the model at each epoch of the robust loop, default=False
        - quant_value : quantization value of the model, default=0.1
        - first_sample_limit : the number of samples in the train and test sets of the first iteration of the robust loop, None for full set, default=5000
        - batch_size : batch size to use during training, default=64
        - verbose : prints information if True, nothing otherwise
        - save : wether to save the final model or not
        - filename : the name of the file to store the model (if save=True)
    """
    # possibilities of improvement
    # - add a defined criterion for the robustness to stop the loop
    # - take a model as argument instead of creating it here

    train_set, test_set = load_data(dataset_name)
    dtrain = Data(train_set, limit=first_sample_limit)
    dtest = Data(test_set, limit=first_sample_limit)
    model = None

    check_robust_img = []
    for i in range(100):
        check_robust_img.append(test_set[i])

    robust_values = []

    for idx_robust_loop in range(epoch):
        # create dataloader
        train_dataloader = DataLoader(dtrain, batch_size=batch_size)
        test_dataloader = DataLoader(dtest, batch_size=batch_size)

        # create and train model
        if recreate_model or idx_robust_loop == 0:
            if model_type == 1:
                model = Model1(input_size, output_size, quant_value=quant_value)
            elif model_type == 2:
                model = Model2(input_size, output_size, quant_value=quant_value)
            elif model_type == 3:
                model = Model3(input_size, output_size, quant_value=quant_value)
            elif model_type == 4:
                model = Model4(1, output_size, 7, quant_value=quant_value)
        model = train_test_model(model, train_dataloader, test_dataloader, epochs=epoch_training, save=False, verbose=False)  # TODO only continue training on adv data and not on evtg ?
        model_eval = model.cvt_to_eval()
        test_acc, _ = test(test_dataloader, model_eval, nn.CrossEntropyLoss(), verbose=False)
        
        if verbose:
            print(f"model test accuracy at robust_epoch [{idx_robust_loop+1} / {epoch}] : {test_acc:.3f}")

        rob_value = evaluate_robustness_search(model, dataset_name, eps=eps, imgs=check_robust_img)
        robust_values.append(rob_value)
        if verbose:
            print(f"model robustness at robust_epoch [{idx_robust_loop+1} / {epoch}] : {rob_value:.3f}")

        # create data
        if idx_robust_loop != epoch-1:
            new_img = generate_random_images(dataset_name, eps, n_img_created, model_eval, withdraw=True, save_index=False, verbose=True)
            new_img = convert_to_tensor(new_img, save=False)

            new_img_test = generate_random_images(dataset_name, eps, n_img_test, model_eval, withdraw=True, save_index=False, verbose=True)
            new_img_test = convert_to_tensor(new_img_test, save=False)

            # add new data to the dataset
            dtrain.augment(new_img, shuffle=True)
            dtest.augment(new_img_test, shuffle=True)
            if verbose:
                print(f"Added data to train ({n_img_created}) and test ({n_img_test}) sets. New sizes :\n  train size : {len(dtrain)}\n  test  size : {len(dtest)}")
                print("---"*20)

    if save:
        torch.save(model.state_dict(), f"{FILEPATH}/saved_models_eevbnn/{filename}")
        print("saved final model")
    return robust_values


def robust_train_loop_uncertain(dataset_name, model_type, epoch=5, epoch_training=5, eps=0.15,
                      n_img_created=1000, n_img_test=1000, recreate_model=False,
                      quant_value=0.1, first_sample_limit=5000, batch_size=64,
                      input_size=28*28, output_size=10, verbose=True,
                      save=False, filename="robust_model.pickle"):
    train_set, test_set = load_data(dataset_name)
    dtrain = Data(train_set, limit=first_sample_limit)
    dtest = Data(test_set, limit=first_sample_limit)
    model = None


    with open(f"{FILEPATH}/saved_generated_data/gendata_40000_eps0.5_mS1_mnist_fixed.pickle", "rb") as file:
        adv_imgs = pickle.load(file)
    with open(f"{FILEPATH}/saved_generated_data/gendata_20000_eps0.2_mS1_mnist_fixed.pickle", "rb") as file:
        adv_imgs2 = pickle.load(file)
    with open(f"{FILEPATH}/saved_generated_data/gendata_10000_eps0.5_mS1_TEST_fixed.pickle", "rb") as file:
        adv_imgs_test = pickle.load(file)

    
    adv_imgs = transform_label_uncertain(adv_imgs, label=10)
    adv_imgs_test = transform_label_uncertain(adv_imgs_test, label=10)
    adv_imgs_test2 = transform_label_uncertain(adv_imgs2, label=10)

    # with open(f"{FILEPATH}/saved_generated_data/gendata_15000_eps0.5_mS1_fashion_fixed.pickle", "rb") as file:
    #     adv_imgs_fash = pickle.load(file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_15000_eps0.2_mS1_fashion_fixed.pickle", "rb") as file:
    #     adv_imgs_fash2 = pickle.load(file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_10000_eps0.5_mS1_fashion_TEST_fixed.pickle", "rb") as file:
    #     adv_imgs_fash_test = pickle.load(file)

    # with open(f"{FILEPATH}/saved_generated_data/gendata_3000_eps0.5_mS1_cifar_fixed.pickle", "rb") as file:
    #     adv_imgs_cif = pickle.load(file)

    # with open(f"{FILEPATH}/saved_generated_data/gendata_40000_eps0.5_noise.pickle", "rb") as file:
    #     adv_imgs_noise = pickle.load(file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_40000_eps0.5_noiseHigh.pickle", "rb") as file:
    #     adv_imgs_noisehigh = pickle.load(file)

    adv_train = adv_imgs
    # adv_test = adv_imgs_test
    dtestadv = Data(adv_imgs_test)
    dtestadv.augment(adv_imgs_test2, shuffle=True)

    print(f"size adv train : {len(adv_train)}, size adv test : {len(dtestadv)}")

    # check_robust_img = []
    # for i in range(100):
    #     check_robust_img.append(test_set[i])

    robust_values = []
    test_accs = []

    for idx_robust_loop in range(epoch):
        # create dataloader
        train_dataloader = DataLoader(dtrain, batch_size=batch_size)
        test_dataloader = DataLoader(dtest, batch_size=batch_size)

        # create and train model
        if recreate_model or idx_robust_loop == 0:
            if model_type == 1:
                model = Model1(input_size, output_size, quant_value=quant_value)
            elif model_type == 2:
                model = Model2(input_size, output_size, quant_value=quant_value)
            elif model_type == 3:
                model = Model3(input_size, output_size, quant_value=quant_value)
            elif model_type == 4:
                model = Model4(1, output_size, 7, quant_value=quant_value)

        model = train_test_model(model, train_dataloader, test_dataloader, epochs=epoch_training, save=False, verbose=False)  # TODO only continue training on adv data and not on evtg ?
        model_eval = model.cvt_to_eval()
        test_acc, _ = test(test_dataloader, model_eval, nn.CrossEntropyLoss(), verbose=False)
        test_accs.append(test_acc*100)
        
        if verbose:
            print(f"model test accuracy at robust_epoch [{idx_robust_loop+1} / {epoch}] : {test_acc * 100:.3f}")

        # rob_value = evaluate_robustness_search(model, dataset_name, eps=eps, imgs=check_robust_img)
        rob_value = evaluate_robustness_acc(model, dtestadv, dataset_name=dataset_name, with_uncertain=True)
        robust_values.append(rob_value)
        if verbose:
            print(f"model robustness at robust_epoch [{idx_robust_loop+1} / {epoch}] : {rob_value:.3f}")

        # create data
        if idx_robust_loop != epoch-1:
            new_img = adv_train[idx_robust_loop*n_img_created:(idx_robust_loop+1)*n_img_created]
            # new_img = train_set[idx_robust_loop*n_img_created:(idx_robust_loop+1)*n_img_created]
            # new_img_test = adv_test[idx_robust_loop*n_img_test:(idx_robust_loop+1)*n_img_test]

            # add new data to the dataset
            dtrain.augment(new_img, shuffle=True)
            # dtest.augment(new_img_test, shuffle=True)
            if verbose:
                print(f"Added data to train ({n_img_created}) and test ({n_img_test}) sets. New sizes :\n  train size : {len(dtrain)}\n  test  size : {len(dtest)}")
                print("---"*20)

    if save:
        torch.save(model.state_dict(), f"{FILEPATH}/saved_models_eevbnn/{filename}")
        print(f"saved final model at : {FILEPATH}/saved_models_eevbnn/{filename}")
    return robust_values, test_accs


def robust_train_loop_majority_voting(dataset_name, model_types=[1, 1, 1, 1, 1], epoch=5, epoch_training=5, eps=0.15,
                      n_img_created=1000, n_img_test=1000, recreate_model=False,
                      quant_value=0.1, first_sample_limit=5000, batch_size=64,
                      input_size=28*28, output_size=10, verbose=True,
                      save=False, filename="robust_model.pickle"):
    """
    robust train loop with generation for multiple models
    """
    train_set, test_set = load_data(dataset_name)
    dtrain = Data(train_set, limit=first_sample_limit)
    dtest = Data(test_set, limit=first_sample_limit)
    models = None
    models_eval = []

    robust_values = []
    acc_values = []

    all_imgs = []
    all_idxs = []


    with open(f"{FILEPATH}/saved_generated_data/gendata_40000_eps0.5_noise.pickle", "rb") as file:
        adv_imgs = pickle.load(file)

    # with open(f"{FILEPATH}/saved_generated_data/gendata_60000_eps0.5_mS1_mnist_fixed.pickle", "rb") as file:
    #     adv_imgs = pickle.load(file)
    with open(f"{FILEPATH}/saved_generated_data/gendata_10000_eps0.5_mS1_mnist_TEST_fixed.pickle", "rb") as file:
        adv_imgs_test = pickle.load(file)

    # with open(f"{FILEPATH}/saved_generated_data/gendata_15000_eps0.5_mS1_fashion_fixed.pickle", "rb") as file:
    #     adv_imgs = pickle.load(file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_10000_eps0.5_mS1_fashion_TEST_fixed.pickle", 'rb') as file:
    #     adv_imgs_test = pickle.load(file)

    # with open(f"{FILEPATH}/saved_generated_data/gendata_40000_eps0.2_mS1_mnist_fixed.pickle", "rb") as file:
    #     adv_imgs = pickle.load(file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_10000_eps0.2_mS1_mnist_TEST_fixed.pickle", "rb") as file:
    #     adv_imgs_test = pickle.load(file)

    print(f"Size train adv available : {len(adv_imgs)}\nSize test adv available : {len(adv_imgs_test)}")

    for idx_robust_loop in range(epoch):
        # create dataloader
        train_dataloader = DataLoader(dtrain, batch_size=batch_size)
        test_dataloader = DataLoader(dtest, batch_size=batch_size)

        # create and train model
        if recreate_model or idx_robust_loop == 0:
            models = []
            for model_type in model_types:
                if model_type == 1:
                    model = Model1(input_size, output_size, quant_value=quant_value)
                elif model_type == 2:
                    model = Model2(input_size, output_size, quant_value=quant_value)
                elif model_type == 3:
                    model = Model3(input_size, output_size, quant_value=quant_value)
                elif model_type == 4:
                    model = Model4(1, output_size, 7, quant_value=quant_value)
                models.append(model)

        models_eval = []
        for i, model in enumerate(models):
            model = train_test_model(model, train_dataloader, test_dataloader, epochs=epoch_training, save=False, verbose=False)  # TODO only continue training on adv data and not on evtg ?
            model_eval = model.cvt_to_eval()
            models_eval.append(model_eval)

        test_acc = evaluate_robustness_acc_majority_voting(models, dtest, dataset_name=dataset_name)
        acc_values.append(test_acc)
        if verbose:
            print(f"model test accuracy at robust_epoch [{idx_robust_loop+1} / {epoch}] : {test_acc:.3f}")

        rob_value = evaluate_robustness_acc_majority_voting(models, adv_imgs_test, dataset_name=dataset_name)
        robust_values.append(rob_value)
        if verbose:
            print(f"model robustness at robust_epoch [{idx_robust_loop+1} / {epoch}] : {rob_value:.3f}")


        # create data
        if idx_robust_loop != epoch-1:
            start = idx_robust_loop * n_img_created
            end = (idx_robust_loop + 1) * n_img_created
            new_img = adv_imgs[start:end]
            # new_img, idx_used = generate_set_images(dataset_name, eps, range(start, end), models_eval, verbose=verbose)
            # new_img = convert_to_tensor(new_img, save=False)

            # all_imgs.append(new_img)
            # all_idxs.append(idx_used)

            # add new data to the dataset
            dtrain.augment(new_img, shuffle=True)
            if verbose:
                print(f"Added data to train ({n_img_created}) and test ({n_img_test}) sets. New sizes :\n  train size : {len(dtrain)}\n  test  size : {len(dtest)}")
                print("---"*20)

    if save:
        torch.save(model.state_dict(), f"{FILEPATH}/saved_models_eevbnn/{filename}")
        print("saved final model")

    # with open(f"{FILEPATH}/saved_generated_data/gendata_robustLoop_eps0.5_mnist_5models.pickle", "wb") as file:
    #     pickle.dump(all_imgs, file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_robustLoop_idx_eps0.5_mnist_5models.pickle", "wb") as file:
    #     pickle.dump(all_idxs, file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_robustLoop_robustValues_eps0.5_mnist_5models.pickle", "wb") as file:
    #     pickle.dump(robust_values, file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_robustLoop_accValues_eps0.5_mnist_5models.pickle", "wb") as file:
    #     pickle.dump(acc_values, file)

    return robust_values, acc_values



def robust_train_loop_models(dataset_name, model_types=[1, 1, 1, 1, 1], epoch=5, epoch_training=5, eps=0.15,
                      n_img_created=1000, n_img_test=1000, recreate_model=False,
                      quant_value=0.1, first_sample_limit=5000, batch_size=64,
                      input_size=28*28, output_size=10, verbose=True,
                      save=False, filename="robust_model.pickle"):
    """
    robust train loop with generation for multiple models
    """
    train_set, test_set = load_data(dataset_name)
    dtrain = Data(train_set, limit=first_sample_limit)
    dtest = Data(test_set, limit=first_sample_limit)
    models = None
    models_eval = []

    robust_values = []
    acc_values = []

    all_imgs = []
    all_idxs = []

    with open(f"{FILEPATH}/saved_generated_data/gendata_60000_eps0.5_mS1_mnist_fixed.pickle", "rb") as file:
        adv_imgs = pickle.load(file)
    with open(f"{FILEPATH}/saved_generated_data/gendata_10000_eps0.5_mS1_mnist_TEST_fixed.pickle", "rb") as file:
        adv_imgs_test = pickle.load(file)

    for idx_robust_loop in range(epoch):
        # create dataloader
        train_dataloader = DataLoader(dtrain, batch_size=batch_size)
        test_dataloader = DataLoader(dtest, batch_size=batch_size)

        # create and train model
        if recreate_model or idx_robust_loop == 0:
            models = []
            for model_type in model_types:
                if model_type == 1:
                    model = Model1(input_size, output_size, quant_value=quant_value)
                elif model_type == 2:
                    model = Model2(input_size, output_size, quant_value=quant_value)
                elif model_type == 3:
                    model = Model3(input_size, output_size, quant_value=quant_value)
                elif model_type == 4:
                    model = Model4(1, output_size, 7, quant_value=quant_value)
                models.append(model)

        rob_values = []
        models_eval = []
        a_values = []
        for i, model in enumerate(models):
            model = train_test_model(model, train_dataloader, test_dataloader, epochs=epoch_training, save=False, verbose=False)  # TODO only continue training on adv data and not on evtg ?
            model_eval = model.cvt_to_eval()
            models_eval.append(model_eval)
            test_acc, _ = test(test_dataloader, model_eval, nn.CrossEntropyLoss(), verbose=False)
            test_acc *= 100
            a_values.append(test_acc)
        
            if verbose:
                print(f"model [{i}] test accuracy at robust_epoch [{idx_robust_loop+1} / {epoch}] : {test_acc:.3f}")

            rob_value = evaluate_robustness_acc(model, adv_imgs_test, dataset_name=dataset_name)
            rob_values.append(rob_value)

            if verbose:
                print(f"model [{i}] robustness at robust_epoch [{idx_robust_loop+1} / {epoch}] : {rob_value:.3f}")
        robust_values.append(rob_values)
        acc_values.append(a_values)


        # create data
        if idx_robust_loop != epoch-1:
            start = idx_robust_loop * n_img_created
            end = (idx_robust_loop + 1) * n_img_created
            new_img = adv_imgs[start:end]
            # new_img, idx_used = generate_set_images(dataset_name, eps, range(start, end), models_eval, verbose=verbose)
            # new_img = convert_to_tensor(new_img, save=False)

            # all_imgs.append(new_img)
            # all_idxs.append(idx_used)

            # add new data to the dataset
            dtrain.augment(new_img, shuffle=True)
            if verbose:
                print(f"Added data to train ({n_img_created}) and test ({n_img_test}) sets. New sizes :\n  train size : {len(dtrain)}\n  test  size : {len(dtest)}")
                print("---"*20)

    if save:
        torch.save(model.state_dict(), f"{FILEPATH}/saved_models_eevbnn/{filename}")
        print("saved final model")

    # with open(f"{FILEPATH}/saved_generated_data/gendata_robustLoop_eps0.5_mnist_5models.pickle", "wb") as file:
    #     pickle.dump(all_imgs, file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_robustLoop_idx_eps0.5_mnist_5models.pickle", "wb") as file:
    #     pickle.dump(all_idxs, file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_robustLoop_robustValues_eps0.5_mnist_5models.pickle", "wb") as file:
    #     pickle.dump(robust_values, file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_robustLoop_accValues_eps0.5_mnist_5models.pickle", "wb") as file:
    #     pickle.dump(acc_values, file)

    return robust_values, acc_values



def robust_train_loop_cheat(dataset_name, model_type, epoch=5, epoch_training=5, eps=0.15,
                      n_img_created=1000, n_img_test=1000, recreate_model=False,
                      quant_value=0.1, first_sample_limit=5000, batch_size=64,
                      input_size=28*28, output_size=10, verbose=True,
                      save=False, filename="robust_model.pickle"):
    """
    Implementation of the robust training loop.
    Loop : train a model on dataset -> check robustness and create adversarial examples -> add created images to dataset

    Parameters:
        - dataset_name : the name of the dataset to use
        - model_type : the type of the model to use (1, 2, 3 or 4)
        - epoch : the number of epoch for the full robust training loop, default=5
        - epoch_training : the number of epoch for a single train of the mode
        - eps : the maximum perturbation value to generate new data, between 0 and 1, default=0.15
        - n_img_created : the number of adversarial images to create, default=1000
        - n_img_test : the number of adversarial images to add to the test set, default=1000
        - recreate_model : recreate the model at each epoch of the robust loop, default=False
        - quant_value : quantization value of the model, default=0.1
        - first_sample_limit : the number of samples in the train and test sets of the first iteration of the robust loop, None for full set, default=5000
        - batch_size : batch size to use during training, default=64
        - verbose : prints information if True, nothing otherwise
        - save : wether to save the final model or not
        - filename : the name of the file to store the model (if save=True)
    """

    # !!! first sample limit should never be > adv train set (otherwise train data in the test adv --> bias)

    # possibilities of improvement
    # - add a defined criterion for the robustness to stop the loop
    # - take a model as argument instead of creating it here

    train_set, test_set = load_data(dataset_name)
    dtrain = Data(train_set, limit=first_sample_limit)
    dtest = Data(test_set)
    model = None



    with open(f"{FILEPATH}/saved_generated_data/gendata_40000_eps0.5_mS1_mnist_fixed.pickle", "rb") as file:
        adv_imgs = pickle.load(file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_10000_eps0.2_mS1_mnist_fixed.pickle", "rb") as file:
    #     adv_imgs2 = pickle.load(file)
    with open(f"{FILEPATH}/saved_generated_data/gendata_10000_eps0.5_mS1_TEST_fixed.pickle", "rb") as file:
        adv_imgs_test = pickle.load(file)

    # with open(f"{FILEPATH}/saved_generated_data/gendata_15000_eps0.5_mS1_fashion_fixed.pickle", "rb") as file:
    #     adv_imgs_fash = pickle.load(file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_15000_eps0.2_mS1_fashion_fixed.pickle", "rb") as file:
    #     adv_imgs_fash2 = pickle.load(file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_10000_eps0.5_mS1_fashion_TEST_fixed.pickle", "rb") as file:
    #     adv_imgs_fash_test = pickle.load(file)

    # with open(f"{FILEPATH}/saved_generated_data/gendata_3000_eps0.5_mS1_cifar_fixed.pickle", "rb") as file:
    #     adv_imgs_cif = pickle.load(file)

    # with open(f"{FILEPATH}/saved_generated_data/gendata_40000_eps0.5_noise.pickle", "rb") as file:
    #     adv_imgs_noise = pickle.load(file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_40000_eps0.5_noiseHigh.pickle", "rb") as file:
    #     adv_imgs_noisehigh = pickle.load(file)

    adv_train = adv_imgs
    adv_test = adv_imgs_test
    # adv_test = adv_imgs2

    print(f"size adv train : {len(adv_train)}, size adv test : {len(adv_test)}")

    # check_robust_img = []
    # for i in range(100):
    #     check_robust_img.append(test_set[i])

    robust_values = []
    test_accs = []

    for idx_robust_loop in range(epoch):
        # create dataloader
        train_dataloader = DataLoader(dtrain, batch_size=batch_size)
        test_dataloader = DataLoader(dtest, batch_size=batch_size)

        # create and train model
        if recreate_model or idx_robust_loop == 0:
            if model_type == 1:
                model = Model1(input_size, output_size, quant_value=quant_value)
            elif model_type == 2:
                model = Model2(input_size, output_size, quant_value=quant_value)
            elif model_type == 3:
                model = Model3(input_size, output_size, quant_value=quant_value)
            elif model_type == 4:
                model = Model4(1, output_size, 7, quant_value=quant_value)

        model = train_test_model(model, train_dataloader, test_dataloader, epochs=epoch_training, save=False, verbose=False)  # TODO only continue training on adv data and not on evtg ?
        model_eval = model.cvt_to_eval()
        # test_acc, _ = test(test_dataloader, model_eval, nn.CrossEntropyLoss(), verbose=False)
        # test_acc, _ = absolute_eval_robustness(model_eval, dtest, dataset_name="MNIST")
        test_acc, _ = ratio_eval_robustness(model_eval, dtest, dataset_name="MNIST")
        # test_acc, _ = ratio_rev_eval_robustness(model_eval, dtest, dataset_name="MNIST")
        test_accs.append(test_acc)
        
        if verbose:
            print(f"model test accuracy at robust_epoch [{idx_robust_loop+1} / {epoch}] : {test_acc:.3f}")

        # rob_value = evaluate_robustness_search(model, dataset_name, eps=eps, imgs=check_robust_img)
        # rob_value = evaluate_robustness_acc(model, adv_test, dataset_name=dataset_name)
        # rob_value, _ = absolute_eval_robustness(model, adv_test, dataset_name="MNIST")
        rob_value, _ = ratio_eval_robustness(model, adv_test, dataset_name="MNIST")
        # rob_value, _ = ratio_rev_eval_robustness(model, adv_test, dataset_name="MNIST")
        robust_values.append(rob_value)
        if verbose:
            print(f"model robustness at robust_epoch [{idx_robust_loop+1} / {epoch}] : {rob_value:.3f}")

        # create data
        if idx_robust_loop != epoch-1:
            new_img = adv_train[idx_robust_loop*n_img_created:(idx_robust_loop+1)*n_img_created]
            # new_img = train_set[idx_robust_loop*n_img_created:(idx_robust_loop+1)*n_img_created]
            # new_img_test = adv_test[idx_robust_loop*n_img_test:(idx_robust_loop+1)*n_img_test]

            # add new data to the dataset
            dtrain.augment(new_img, shuffle=True)
            # dtest.augment(new_img_test, shuffle=True)
            if verbose:
                print(f"Added data to train ({n_img_created}) and test ({n_img_test}) sets. New sizes :\n  train size : {len(dtrain)}\n  test  size : {len(dtest)}")
                print("---"*20)

    if save:
        torch.save(model.state_dict(), f"{FILEPATH}/saved_models_eevbnn/{filename}")
        print(f"saved final model at : {FILEPATH}/saved_models_eevbnn/{filename}")
    return robust_values, test_accs





if __name__ == "__main__":



    #################################
    #####   ROBUST TRAIN LOOP   #####
    #################################


    ### classic

    # t0 = time.perf_counter()
    # robust_values = robust_train_loop("MNIST", 1, epoch=5, epoch_training=25, eps=0.5, n_img_created=100, n_img_test=50, recreate_model=False, quant_value=0.1,
    #                   first_sample_limit=2000, batch_size=64, verbose=True, save=True, filename="robust_model1_MNIST.pth")
    # print(f"Process done in {time.perf_counter() - t0:.3f}s")

    # with open(f"{FILEPATH}/robust_values.pickle", "wb") as file:
    #     pickle.dump(robust_values, file)

    # x = [i for i in range(5)]
    # plt.plot(x, robust_values)
    # plt.title("Robust evaluation of the model")
    # plt.xlabel("robust epoch")
    # plt.ylabel("SAT ratio")
    # plt.savefig(f"{FILEPATH}/robust_graph_epoch.pdf")


    ### cheat

    # epoch = 25
    # robust_values, test_accs = robust_train_loop_cheat("MNIST", 1, epoch=epoch, epoch_training=25, eps=0.5, n_img_created=500, n_img_test=0, recreate_model=True, quant_value=0.1,
    #                   first_sample_limit=3000, batch_size=64, verbose=True, save=False, filename="robust_model1_MNIST.pth", input_size=28*28)

    # with open(f"{FILEPATH}/saved_graphs/22_robust_values_cheat_recreateTrue_mS5_epoch25_ratiorevEval.pickle", "wb") as file:
    #     pickle.dump(robust_values, file)
    # with open(f"{FILEPATH}/saved_graphs/22_test_values_cheat_recreateTrue_mS5_epoch25_ratiorevEval.pickle", "wb") as file:
    #     pickle.dump(test_accs, file)

    # x = [i for i in range(epoch)]
    # plt.plot(x, robust_values, label="adversarial images")  # NOTE change this to adversarial !
    # plt.plot(x, test_accs, label="original images")
    # plt.title("Robust evaluation of the model\nadds 500 imgs each epoch")
    # plt.xlabel("robust epoch")
    # plt.ylabel("absolute robust score")
    # plt.legend()
    # plt.savefig(f"{FILEPATH}/saved_graphs/22_robust_graph_epoch25_cheat_recreateTrue_mS5_ratiorevEval.pdf")


    ### majority voting

    # epochs = 12
    # robs, tvals = robust_train_loop_majority_voting("MNIST", model_types=[1, 1, 1, 1, 1], epoch=epochs, epoch_training=25,
    #                                                 n_img_created=500, recreate_model=True, quant_value=0.1,
    #                                                 first_sample_limit=10000, batch_size=64, input_size=28*28,
    #                                                 output_size=10, verbose=True, save=False)
    # with open(f"{FILEPATH}/saved_graphs/42_robust_values_cheat_recreateTrue_mS5_epoch25_majority_noisetrain.pickle", "wb") as file:
    #     pickle.dump(robs, file)
    # with open(f"{FILEPATH}/saved_graphs/42_test_values_cheat_recreateTrue_mS5_epoch25_majority_noisetrain.pickle", "wb") as file:
    #     pickle.dump(tvals, file)


    ### models

    # epochs = 25
    # robs, tvals = robust_train_loop_models("MNIST", model_types=[1, 1, 1, 1, 1], epoch=epochs, epoch_training=5, eps=0.5,
    #                          n_img_created=500, n_img_test=0, recreate_model=True, quant_value=0.1,
    #                          first_sample_limit=3000, batch_size=64, input_size=28*28, output_size=10,
    #                          verbose=True, save=False)

    # with open(f"{FILEPATH}/saved_graphs/tmp_robs.pickle", "wb") as file:
    #     pickle.dump(robs, file)
    # with open(f"{FILEPATH}/saved_graphs/tmp_accs.pickle", "wb") as file:
    #     pickle.dump(tvals, file)


    ### uncertain

    # robs, accs = robust_train_loop_uncertain("MNIST", 1, epoch=8, epoch_training=5, eps=0.5,
    #                                          n_img_created=500, n_img_test=0, recreate_model=True, quant_value=0.1,
    #                                          first_sample_limit=5000, batch_size=64, input_size=28*28, output_size=11,
    #                                          verbose=True, save=False)
    # with open(f"{FILEPATH}/saved_graphs/tmp_robs2.pickle", "wb") as file:
    #     pickle.dump(robs, file)
    # with open(f"{FILEPATH}/saved_graphs/tmp_accs2.pickle", "wb") as file:
    #     pickle.dump(accs, file)



    ###############################
    #####   GENERATE IMAGES   #####
    ###############################


    # imgs = generate_random_images("MNIST", 0.2, 10, 1, 0.1)
    # convert_to_tensor(imgs, save=True, filename="gendata_250_eps0.2_m1_mnist")
    # with open(f"{FILEPATH}/saved_generated_data/gendata_250_eps0.2_m1_mnist.pickle", "rb") as file:
    #     imgs = pickle.load(file)
    # with open(f"{FILEPATH}/saved_generated_data/index_used.pickle", "rb") as file:
    #     idxs = list(pickle.load(file))

    ### generate images from 1000 first images
    # model = load_model(1, "MNIST", quant_value=0.1, show_test_acc=False).cvt_to_eval()
    # imgs, _ = generate_set_images("MNIST", 0.5, range(10000, 30000), model)
    # convert_to_tensor(imgs, save=True, filename="gendata_20000_eps0.5_m1_mnist_fixed")

    ### generate images with multiple models
    # with open(f"{FILEPATH}/saved_models_eevbnn/list_models1_mnist.pickle", "rb") as file:
    #     models_verif = pickle.load(file)
    # models = []
    # for model in models_verif:
    #     models.append(model.cvt_to_eval())
    # imgs, _ = generate_set_images("MNIST", 0.5, range(0, 1000), models, verbose=True)
    # convert_to_tensor(imgs, save=True, filename="gendata_1000_eps0.5_mS1_mnist_fixed")

    ### with noise
    # res = generate_random_noise_image("MNIST", 0.5, range(0, 40000), low=0.35)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_40000_eps0.5_highNoise.pickle", "wb") as file:
    #     pickle.dump(res, file)
    
    res = generate_random_noise_image("MNIST", 0.5, range(0, 10000), verbose=True)
    with open(f"{FILEPATH}/saved_generated_data/gendata_10000_eps0.5_noise_TEST.pickle", "wb") as file:
        pickle.dump(res, file)


    ####################################
    #####   MERGE GENERATED DATA   #####
    ####################################

    # with open(f"{FILEPATH}/saved_generated_data/gendata_5000_eps0.1_mS1_cifar_fixed.pickle", "rb") as file:
    #     d1 = pickle.load(file)
    # with open(f"{FILEPATH}/saved_generated_data/saved_idxs_set_5000_0.1_mS1_cifar.pickle", "rb") as file:
    #     idx1 = pickle.load(file)
    # with open(f"{FILEPATH}/saved_generated_data/gendata_10000_eps0.1_mS1_cifar_fixed.pickle", "rb") as file:
    #     d2 = pickle.load(file)
    # with open(f"{FILEPATH}/saved_generated_data/saved_idxs_set_10000_0.1_mS1_cifar.pickle", "rb") as file:
    #     idx2 = pickle.load(file)

    # print(idx1[0], idx1[len(idx1) - 1], idx2[0], idx2[len(idx2)-1])
    # d1.extend(d2)
    # idx1.extend(idx2)

    # print(len(d1), len(idx1))
    
    # if input("confirm ? (y/n)") == "y":
    #     with open(f"{FILEPATH}/saved_generated_data/gendata_15000_eps0.1_mS1_cifar_fixed.pickle", "wb") as file:
    #         pickle.dump(d1, file)
    #     with open(f"{FILEPATH}/saved_generated_data/saved_idxs_set_15000_0.1_mS1_cifar.pickle", "wb") as file:
    #         pickle.dump(idx1, file)
    #     print("new files created")


    # train_set, test_set = load_data("MNIST")
    # train_dataloader = DataLoader(train_set, batch_size=64)
    # test_dataloader = DataLoader(test_set, batch_size=64)

    # models = []
    # for i in range(5):
    #     model = Model1(28*28, 10)
    #     # model = Model2(28*28, 10)
    #     # model = Model3(28*28, 10)
    #     # model = Model4(1, 10)
    #     model = train_test_model(model, train_dataloader, test_dataloader, epochs=5, save=False)
    #     models.append(model.cvt_to_eval())
    
    # imgs, idxs, _ = generate_images_order("MNIST", 0.5, 0, models, n_gen=500)

    # with open(f"{FILEPATH}/saved_transferability/mS1_0.5.pickle", "wb") as file:
    #     pickle.dump(imgs)



    print("done")




### TODO


# [ ]   verif one image on which trained does not give same perturb as before
# [ ]   train on new dataset and check perfs




