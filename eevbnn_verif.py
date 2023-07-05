
#################################################################
#                                                               #
# Master's thesis : Binarized Neural Networks                   #
# eevbnn script                                                 #
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
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle
import time

from eevbnn_figures import *
from eevbnn_models import *


TIMEOUT = 60 # 150



#======================================================================================================================================
## ARGPARSE

def init_argparser(parser: argparse.ArgumentParser):
    parser.add_argument('--show-adv', action='store_true',
                        help='show discovered adversarial images')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8,
                        help='number of CPU workers for data augmentation')
    parser.add_argument(
        '--data', default=None,
        help='dir for training data')
    parser.add_argument('--check-cvt', action='store_true',
                        help='check accuracy of converted model')
    parser.add_argument('--no-attack', action='store_true',
                        help='do not run attack and only check result')
    parser.add_argument('--sat-solver', choices=['z3', 'pysat', 'roundingsat',
                                                 'pysat_stat'],
                        default='pysat', help='SAT solver implementation')
    parser.add_argument('--pysat-name', default='minisatcs',
                        help='pysat solver name')
    parser.add_argument('--write-formula',
                        help='write internal formula to file; use @file.json '
                        'to write to the same directory of model file')
    parser.add_argument('--write-slow-formula',
                        help='write formula to file for the currently slowest '
                        'case')
    parser.add_argument('--write-result',
                        help='write result to a json file')
    parser.add_argument('--write-adv',
                        help='write adversarial results to a directory')
    parser.add_argument('-v', '--verbosity', type=int,
                        help='set verbosity level')
    parser.add_argument('-t', '--timeout', type=float, default=TIMEOUT,
                        help='timeout in seconds')
    parser.add_argument('--skip', type=int,
                        help='skip given number of test examples')
    parser.add_argument('--num-cases', type=int,
                        help='set the number of cases to run')
    parser.add_argument('--var-preference', default='first-sp',
                        choices=['z', 'z-ent', 'mid-first', 'first',
                                 'first-sp', 'none'],
                        help='set var preference type; [z]: mid layer to first '
                        'layer spatial locality; [z-ent]: z guided by entropy; '
                        '[mid-first]: all mid layer and then first; '
                        '[first]: first layer; [first-sp]: first with spatial '
                        'locality; [none]: solver default'
                        )
    parser.add_argument('--continue', action='store_true',
                        help='set --skip value based on number of existing '
                        'entries in the result file')
    parser.add_argument('--random-sample', type=int,
                        help='sample a given number of test cases')
    parser.add_argument('--log-file', help='redirect output to given log file')
    parser.add_argument('--ensemble', default=[], action='append',
                        help='add another model to be ensembled')
    parser.add_argument('--disable-model-cache', action='store_true',
                        help='disable caching clauses of the models')
    parser.add_argument('--disable-fast-tensor-arith', action='store_true',
                        help='disable fast tensor arith impl, for checking '
                        'correctness')
    parser.add_argument('--verify-output-stability', action='store_true',
                        help='verify if model prediction is stable '
                        '(i.e. use model output as the label)')
    return parser


def get_args(eps, model):
    parser = argparse.ArgumentParser(
        description='evaluate robustness of 0/1 binaried network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    init_argparser(parser)
    parser.add_argument('-e', '--eps', type=float, default=eps,    # classical value : 0.08
                        help='adv input eps on 0-1 scale')          # nb : modif from file required=True -> default=0.08
    parser.add_argument('--model', default=model)
    args = parser.parse_args()
    return args



#======================================================================================================================================
## VERIFY


def search_adv_image_full(model_types, img_idxs, dataset_name, eps_list, save_images=True, verbose=False, quant_value=None):
    """ 
    Searches for adversarial images on the images with given indexes

    Parameters:
        - model_types   : list of model types on which to run the search
        - img_idxs      : list of indexes of images to create adversarial images
        - dataset_name  : name of the dataset on which to run the search
        - eps_list      : list of eps values to test for adversarial images
        - save_images   : Boolean, save or not the images (original and adversarial)
    
    """
    _, test_dataloader = get_data(dataset_name, batch_size=64)
    models = []
    for type in model_types:
        models.append(load_model(type, dataset_name, show_test_acc=False, quant_value=quant_value))
    eval_models = get_eval_models(models)
    classes = get_classes(dataset_name)

    for type, model in zip(model_types, eval_models):
        for eps in eps_list:
            args = get_args(eps, model)
            verifier = ModelVerifier(args, [model])
            for idx in img_idxs:

                num_batch = idx // 64
                idx_batch = idx % 64

                n_batch = 0

                for inputs, labels in test_dataloader:

                    if n_batch == num_batch:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        if verbose:
                            print("---"*30)
                            print(f"current test:\n type: {type}\teps: {eps}\tidx: {idx}\tdataset: {dataset_name}")

                        model.eval()
                        with torch.no_grad():
                            pred = model(inputs)[idx_batch]
                            actual = classes[labels[idx_batch]]
                            predicted = classes[pred.argmax(0)]
                            if verbose:
                                print(f"original distrib : {pred}")
                                print(f"predicted {predicted}\t actual : {actual} (idx : {idx})")

                        inputs = torch_as_npy(inputs)
                        labels = torch_as_npy(labels)

                        check_result, (time_build, time_verif), inp_adv, clsfy_scores = verifier.check(inputs[idx_batch], labels[idx_batch])
                        
                        if verbose:
                            print(f"Result : {check_result}\n build time : {time_build:.3f}\t verif time : {time_verif:.3f}")
                        
                        if check_result == SolveResult.TLE:
                            if verbose:
                                print("Time Limit Exceeded")
                            return inputs[idx_batch], None
                        elif check_result == SolveResult.UNSAT:
                            return inputs[idx_batch], None

                        if save_images:
                            if check_result == SolveResult.SAT:
                                clsfy_adv = classes[clsfy_scores[0][0].argmax(0)]
                                save_picture(inputs[idx_batch], idx, predicted, dataset_name, classes[labels[idx_batch]], type, adv=False)
                                save_picture(inp_adv, idx, clsfy_adv, dataset_name, classes[labels[idx_batch]], type, adv=True, eps=eps)
                                if verbose:
                                    print("images saved")
                        
                        return inputs[idx_batch], inp_adv
                        break
                    else:
                        n_batch += 1
    if verbose:
        print("DONE !")


def search_adv_image(model, img, true_label, classes, args):
    """
    Runs a search to find an adversarial image

    Parameters
        - model : the model to verify
        - img : the image on which the model is verified
        - true_label : the correct label for img
        - classes : the set of classes for the considered dataset
        - args : the args used to run the code
    
    Returns (check_results, inp_adv)
        - check_results :   -1 if basic missclassification\\
                            SAT if an adversarial image is found\\
                            UNSAT if the model is robust for this image\\
                            TLE if the time limit is exceeded
        - inp_adv : the adversarial image (if any)
    """
    verifier = ModelVerifier(args, [model], show_logs=False)

    model.eval()
    with torch.no_grad():
        if len(img) == 3:
            pred = model(img.reshape(1, 3, 32, 32))[0]
        else:
            pred = model(img.reshape(1, 1, 28, 28))[0]
        label = classes[pred.argmax(0)]
        # if label != str(true_label):
        #     return -1, None

    img = torch_as_npy(img)
    true_label = np.array(true_label)

    check_result, (time_build, time_verif), inp_adv, clsfy_scores = verifier.check(img, true_label)

    return check_result, inp_adv


def search_adv_img_multiple_models(models, img, true_label, classes, args):
    verifier = ModelVerifier(args, models, show_logs=False)

    img = torch_as_npy(img)
    true_label = np.array(true_label)

    check_result, (time_build, time_verif), inp_adv, clsfy_scores = verifier.check(img, true_label)

    return check_result, inp_adv



def search_adv_img_perf(model, img, true_label, classes, args):

    verifier = ModelVerifier(args, [model], show_logs=False)

    model.eval()
    with torch.no_grad():
        pred = model(img.reshape(1, 1, 28, 28))[0]
        label = classes[pred.argmax(0)]
        if label != str(true_label):
            return -1

    img = torch_as_npy(img)
    true_label = np.array(true_label)

    check_result, (time_build, time_verif), inp_adv, clsfy_scores = verifier.check(img, true_label)

    return check_result
        



#======================================================================================================================================
## EXPERIMENTS


##### get perfs of consistency


def consistent_model_test_eev(model, dataset_name, eps, indexes, n_test=500, verbose=True, is_cifar=False, get_time=False):
    """
    Runs model with n_test images to assess its consistency
    """

    n_miss_basic    = 0
    n_miss_modif    = 0
    n_correct       = 0
    n_time_out      = 0
    times_SAT       = []

    classes = get_classes(dataset_name)
    _, dataset = load_data(dataset_name)
    args = get_args(eps, model)

    t0 = time.perf_counter()
    
    for i, idx in enumerate(indexes):
        img, label = dataset[idx]

        # if is_cifar:
        #     img = img.reshape(1, 3, 32, 32)
        # else:
        #     img = img.reshape(1, 1, 28, 28)

        t1 = time.perf_counter()

        check_result = search_adv_img_perf(model, img, label, classes, args)

        elapsed_time = time.perf_counter() - t1

        if check_result == -1:
            n_miss_basic += 1
        elif check_result == SolveResult.SAT:
            n_miss_modif += 1
            times_SAT.append(elapsed_time)
        elif check_result == SolveResult.UNSAT:
            n_correct += 1
        elif check_result == SolveResult.TLE:
            n_time_out += 1
        else:
            print(f"Unexpected check_result : {check_result}")

        # if verbose:
        #     if i % 25 == 0:
        #         print(f"epsilon {eps}  :  [{i:>3d} / {n_test:>3d}]")

    t1 = time.perf_counter()

    if verbose:
        print("---"*30)
        print(f"Ran {n_test} tests in {t1-t0:.3f}s")
        print(f"Parameters : eps = {eps}, n_test = {n_test}")
        print(f"  # basic missclassifications         : {n_miss_basic:>4d}  ( {n_miss_basic*100/n_test:.2f}% )")
        print(f"  # errors with modifications (SAT)   : {n_miss_modif:>4d}  ( {n_miss_modif*100/n_test:.2f}% )")
        print(f"  # consistent classification (UNSAT) : {n_correct:>4d}  ( {n_correct*100/n_test:.2f}% )")
        print(f"  # time out during adv search (TLE)  : {n_time_out:>4d}  ( {n_time_out*100/n_test:.2f}% )")
        if get_time:
            print(f"  Mean elapsed time for SAT cases     : {np.mean(times_SAT):.3f}")
        
    if get_time:
        return n_miss_basic, n_miss_modif, n_correct, n_time_out, n_test, times_SAT
    else:
        return n_miss_basic, n_miss_modif, n_correct, n_time_out, n_test

        
def graph_consistent_eev(model_types, dataset_name, eps_range, n_test=500, quant_value=None, img_idx=None):

    misses = []
    timeouts = []
    times = []

    if img_idx is None:
        indexes = np.random.randint(0, 10000, n_test)
        with open(f"{FILEPATH}saved_values_eevbnn/img_indexes.pickle", "wb") as file:
            pickle.dump(indexes, file)
    else:
        with open(f"{FILEPATH}saved_values_eevbnn/img_indexes.pickle", 'rb') as file:
            indexes = pickle.load(file)

    for model_type in model_types:
        print("###"*20)
        print(f"MODEL {model_type}, quant_value = {quant_value}")
        print("###"*20)
        model = load_model(model_type, dataset_name, show_test_acc=False, quant_value=quant_value).cvt_to_eval()
        miss = []
        out = []
        time_type = []

        for eps in eps_range:
            n_miss_basic, n_miss_modif, n_correct, n_time_out, _, times_SAT = consistent_model_test_eev(model, dataset_name, eps, indexes, n_test=n_test, get_time=True)
            miss.append(n_miss_modif/n_test)
            out.append(n_time_out/n_test)
            time_type.append(np.mean(times_SAT))
        
        misses.append(miss)
        timeouts.append(out)
        times.append(time_type)

    if quant_value is None:
        quant_value = 0.1

    with open(f"{FILEPATH}/saved_values_eevbnn/graphs_errors/nbr_misses_quant{quant_value}.pickle", "wb") as file:
        pickle.dump(misses, file)
    with open(f"{FILEPATH}/saved_values_eevbnn/graphs_errors/nbr_timeouts_quant{quant_value}.pickle", "wb") as file:
        pickle.dump(timeouts, file)
    with open(f"{FILEPATH}/saved_values_eevbnn/graphs_errors/times_quant{quant_value}.pickle", "wb") as file:
        pickle.dump(times, file)


    colors = ["blue", "red", "green", "orange"]
    for i, miss in enumerate(misses):
        plt.plot(eps_range, miss, label=f"model {model_types[i]}", color=colors[i])
        plt.plot(eps_range, timeouts[i], '--', color=colors[i])

    plt.xlabel("epsilon values")
    plt.ylabel("Ratio of SAT cases")
    plt.title(f"Number of SAT cases for given pertubation epsilon\n#tests = {n_test}, timeout = {TIMEOUT}s, dataset = {dataset_name}, quant = {quant_value}")
    plt.legend()
    plt.savefig(f"{FILEPATH}/saved_imgs_datasets/graphs_errors/models_errors_eev_quant{quant_value}.pdf")

    plt.figure()

    ## plot times
    for i, values in enumerate(times):
        plt.plot(eps_range, values, label=f"model {model_types[i]}", color=colors[i])
    plt.xlabel("epsilon values")
    plt.ylabel("Mean time")
    plt.title(f"Mean time to get a SAT result\n# tests = {n_test}, timeout = {TIMEOUT}, dataset = {dataset_name}, quant = {quant_value}")
    plt.legend()
    plt.savefig(f"{FILEPATH}/saved_imgs_datasets/graphs_errors/models_errors_eev_times_quant{quant_value}.pdf")



def run_graph_consistent_eev(quant_value):
    eps_range = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]

    print("START COMPUTATION")
    graph_consistent_eev([1], "MNIST", eps_range, n_test=100, quant_value=quant_value, img_idx="load")
    print("DONE")




#### compare model (1) with different quantized values


# 1) build mutiple models 1 with different quant values
# 2) train the models
# 3) compare acc and sparsity
# 4) get adv imgs for some samples and compare them
# 5) check the perfs


def build_train_test_models_quant(quant_range, epochs=5):
    """
    Build as many models 1 as the len of quant range\
    quant_range : a list of quantization values\
    returns a list of models 1 with given quantized values
    """
    models = []
    input_size = INPUT_OUTPUT_SIZES[0][0]
    output_size = INPUT_OUTPUT_SIZES[0][1]
    train_dataloader, test_dataloader = get_data(name="MNIST", batch_size=64)
    for quant in quant_range:
        print("###"*30)
        print(f"MODEL 1 with quantization value : {quant}")
        print("###"*30)
        model = Model1(input_size, output_size, quant_value=quant)
        model = train_test_model(model, train_dataloader, test_dataloader, epochs=epochs, save=False, verbose=True)
        fname = f"{FILEPATH}/saved_models_eevbnn/models_1_quant/model1_MNIST_{quant}.pth"
        torch.save(model.state_dict(), fname)
        models.append(model)
    return models


def stats_models_quant(quant_range):
    for quant in quant_range:
        print(f"Quantization value : {quant}")
        get_stat_model(1, "MNIST", quant_value=quant)
        print("==="*30)


def get_adv_quant(quant_range, img_idx):
    imgs_adv = []
    times = []

    for quant in quant_range:
        print(f"run for quant : {quant}")
        t0 = time.perf_counter()
        img, img_adv = search_adv_image_full([1], [img_idx], "MNIST", [0.08], save_images=False, verbose=True, quant_value=quant)
        times.append(time.perf_counter() - t0)
        if img_adv is None:  # in case of TLE or UNSAT
            img_adv = np.array([[[0.0]*28]*28])
        imgs_adv.append(img_adv)
    
    f = plt.figure(figsize=(3*len(quant_range), 3*len(quant_range)))
    nrows = len(quant_range)+1
    ncols = len(quant_range)+1
    f.add_subplot(nrows, ncols, 1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.title("original, eps = 0.08")
    for i in range(len(quant_range)):
        f.add_subplot(nrows, ncols, 2+i)
        plt.imshow(imgs_adv[i].squeeze(), cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.title(f"quant value : {quant_range[i]}")

        f.add_subplot(nrows, ncols, (i+1)*ncols+1)
        plt.imshow(imgs_adv[i].squeeze(), cmap="gray")
        plt.title(f"verif time : {times[i]:.2f}s")
        plt.xticks([])
        plt.yticks([])

    for i in range(len(quant_range)):
        for j in range(len(quant_range)):
            if imgs_adv[i].sum() == 0:
                adv_diff = imgs_adv[i]
            elif imgs_adv[j].sum() == 0:
                adv_diff = imgs_adv[j]
            else:
                adv_diff = np.abs(imgs_adv[i] - imgs_adv[j])
            idx = len(quant_range)+3 + i*nrows + j
            f.add_subplot(nrows, ncols, idx)
            plt.imshow(adv_diff.squeeze(), cmap="gray")
            plt.xticks([])
            plt.yticks([])
    
    f.savefig(f"{FILEPATH}/saved_imgs_datasets/compare_quantization/compare_adv_quant_{img_idx}.png")



# bar plot for each quant value with number of SAT cases

def perf_quant(quant_range, n_test=100, eps=0.08):
    misses = []
    times_out = []
    indexes = indexes = np.random.randint(0, 10000, n_test)

    for quant in quant_range:
        print("###"*30)
        print(f"Test with quant = {quant}")
        print("###"*30)
        model = load_model(1, "MNIST", quant_value=quant, show_test_acc=False).cvt_to_eval()
        n_miss_basic, n_miss_modif, n_correct, n_time_out, _ = consistent_model_test_eev(model, "MNIST", eps, indexes, n_test=n_test)
        misses.append(n_miss_modif/n_test)
        times_out.append(n_time_out/n_test)

    with open(f"{FILEPATH}/saved_values_eevbnn/nbr_misses_quant_eps{eps}.pickle", 'wb') as file:
        pickle.dump(misses, file)
    with open(f"{FILEPATH}/saved_values_eevbnn/nbr_timeouts_quant_eps{eps}.pickle", "wb") as file:
        pickle.dump(times_out, file)

    plt.bar(quant_range, misses, label="SAT cases", width=0.02)
    plt.bar(quant_range, times_out, bottom=misses, label="TLE cases", width=0.02)
    plt.xlabel("Quantization values")
    plt.ylabel("Ratio")
    plt.legend()
    plt.title(f"#tests : {n_test}, eps = {eps}")
    plt.savefig(f"{FILEPATH}/saved_imgs_datasets/compare_quantization/bar_plot_quant.png")
    



### visualize quantization

def train_quant(quant_value, input_size, output_size):
    pass


def visualize_quant(x, quant_value):
    model = ModelQuant(quant_value)
    out = model(x)

    unique, counts = np.unique(x, return_counts=True)
    unique_out, counts_out = np.unique(out, return_counts=True)
    plt.subplot(121)
    plt.bar(unique, counts, width=0.3, align="center")
    plt.title("counts in original input")

    plt.subplot(122)
    plt.bar(unique_out, counts_out, width=0.3, align="center")
    plt.title(f"counts in quantized input\n(quant_value = {quant_value})")
    plt.savefig(f"{FILEPATH}saved_imgs_datasets/compare_quant_input.png")


    plt.subplot(121)
    plt.imshow(x.squeeze(), cmap="gray")
    plt.title("original")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    plt.imshow(out.squeeze(), cmap="gray")
    plt.title(f"Quantized input ({quant_value})")
    plt.xticks([])
    plt.yticks([])

    plt.savefig(f"{FILEPATH}saved_imgs_datasets/quant_test.png")


def compare_quant(quant_range):
    _, test_set = load_data()
    x = test_set[1][0]

    nrows = len(quant_range)
    ncols = 4

    f = plt.figure(figsize=(3*ncols, nrows*3))

    for i, quant in enumerate(quant_range):
        model = ModelQuant(quant)
        out = model(x)

        f.add_subplot(nrows, ncols, 1+i*ncols)
        unique, counts = np.unique(x, return_counts=True)
        plt.bar(unique, counts, width=0.1, align="center")
        if i == 0:
            plt.title("counts (original)")

        f.add_subplot(nrows, ncols, 2+i*ncols)
        plt.imshow(x.squeeze(), cmap="gray")
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.title("original image")
        
        f.add_subplot(nrows, ncols, 3+i*ncols)
        plt.imshow(out.squeeze(), cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Quantized ({quant:.1f})")

        f.add_subplot(nrows, ncols, 4+i*ncols)
        unique, counts = np.unique(out, return_counts=True)
        plt.bar(unique, counts, width=0.1, align="center")
        if i == 0:
            plt.title("counts (quantized)")

    plt.savefig(f"{FILEPATH}saved_imgs_datasets/compare_quant_range.png")
        







#======================================================================================================================================
## MAIN


if __name__ == "__main__":

    ### simple search of an adv img
    # dname = "MNIST"
    # img, img_adv = search_adv_image([1], [1], dname, [0.08], verbose=True, save_images=False)


    ### create different models with different quantization values
    # quant_range = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    # build_train_test_models_quant(quant_range)
    # stats_models_quant(quant_range)
    # idxs = [0, 1, 10, 25, 42, 420]
    # for idx in idxs:
    #     get_adv_quant(quant_range, idx)
    # perf_quant(quant_range, n_test=100, eps=0.1)


    ### compare the effects of the quantization
    # compare_quant(np.arange(0.1, 1.1, 0.1, dtype=float))


    ### get graphs of errors with eev + graph of mean time for SAT image
    # for quant in [0.05, 0.2, 0.3, 0.4, 0.5]:
    #     run_graph_consistent_eev(quant)  # TODO




    # with open(f"{FILEPATH}/saved_values_eevbnn/times.pickle", "rb") as file:
    #     times = pickle.load(file)

    # colors = ["blue", "red", "green", "orange"]
    # eps_range = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4]
    # model_types = [1, 2, 3, 4]
    # n_test = 100
    # dataset_name = 'MNIST'
    # types = ["big MLP", "small MLP", "medium MLP", "CNN"]

    # plt.figure()
    # for i, values in enumerate(times):
    #     plt.plot(eps_range, values, label=f"model {model_types[i]}, {types[i]}", color=colors[i])
    # plt.xlabel("epsilon values")
    # plt.ylabel("Mean time")
    # plt.title(f"Mean time to get a SAT result\n# tests = {n_test}, timeout = {TIMEOUT}, dataset = {dataset_name}")
    # plt.legend()
    # plt.savefig(f"{FILEPATH}/saved_imgs_datasets/models_errors_eev_times.png")




    ### build graphs of errors and time for it
    quant_range = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    eps_range = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]
    misses = []
    timeouts = []
    times = []
    for quant in quant_range:
        with open(f"{FILEPATH}/saved_values_eevbnn/graphs_errors/nbr_misses_quant{quant}.pickle", "rb") as file:
            misses.append(pickle.load(file)[0])
        with open(f"{FILEPATH}/saved_values_eevbnn/graphs_errors/nbr_timeouts_quant{quant}.pickle", "rb") as file:
            timeouts.append(pickle.load(file)[0])
        with open(f"{FILEPATH}/saved_values_eevbnn/graphs_errors/times_quant{quant}.pickle", "rb") as file:
            times.append(pickle.load(file)[0])

    colors = ['blue', 'green', 'red', 'grey', 'magenta', 'orange']
    for i, quant in enumerate(quant_range):
        plt.plot(eps_range, misses[i], label=f"quant {quant}", color=colors[i])
        # plt.plot(eps_range, timeouts[i], "--", color=colors[i])
    plt.legend()
    plt.title("Number of SAT cases for given perturbation epsilon\n#tests = 100, timeout = 150s, dataset=MNIST")
    plt.xlabel("epsilon")
    plt.ylabel("ratio of SAT cases")
    plt.savefig(f"{FILEPATH}/saved_imgs_datasets/graphs_errors/models_errors_eev_full.pdf", bbox_inches="tight")
    
    plt.clf()

    for i, quant in enumerate(quant_range):
        plt.plot(eps_range, times[i], label=f"quant {quant}", color=colors[i])
        # plt.plot(eps_range, timeouts[i], "--")
    plt.legend()
    plt.title("Mean time to get a SAT result\n#tests = 100, timeout = 150s, dataset = MNIST")
    plt.xlabel("epsilon")
    plt.ylabel("mean time")
    plt.savefig(f"{FILEPATH}/saved_imgs_datasets/graphs_errors/models_errors_times_full.pdf", bbox_inches="tight")






    












