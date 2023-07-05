# Assessing and improving the robustness of Binarized Neural Networks

This repository contains the code used in the paper "Assessing and improving the robustness of Binarized Neural Networks".

Robustness is an important aspect of neural networks as we expect a model to behave similarly for inputs that look the same to the human eye. Moreover, Binarized Neural Networks (BNNs) are a type of neural networks that require much less power consumption and can therefore be used on smaller devices such as smartphones. Thus, we should expect a certain robustness from them to prevent unexpected results for the same reasons than their real-valued version.


The two important aspects of our work are:

1. The robust training loop, an algorithm to perform adversarial training that can be easily generalized to other attacks and deep learning models

2. The different methods to assess the robustness of BNNs


## Repository organization
---
We list here the folders and files of the repository with a short description

**Folders**

- **data**: the MNIST, Fashion-MNIST and CIFAR-10 datasets
- **eevbnn**: the code for the MiniSatCS solver from the paper [Efficient and Exact Verification of Binarized Neural Networks](https://github.com/jia-kai/eevbnn)
- **scripts_from_EEV**: the different scripts from the previously cited papers, used for inspiration in this work
- **results_imgs_graphs**: the different results, graphs and generated adversarial images, of our paper
    - **saved_attacks**: results obtained after performing an attack with a fixed perturbation against an adversarially trained model
    - **saved_eps_graphs**: graphs and results of the method to assess the robustness according to the resistance of a model against the attack
    - **saved_generated_data**: all the generated adversarial images from MNIST, Fashion-MNIST and CIFAR-10 datasets with their corresponding indexes from their original dataset
    - **saved_graphs**: experiments with the robust training loop algorithm and their results as graphs
    - **saved_imgs_thesis**: all the images used in the paper (and more)
    - **saved_models_eevbnn**: some of the trained models used for the experiments
    - **saved_transferability**: results for the transferability of the adversarial images

**Files**

- **eevbnn_aug_test.ipynb**: various tests to assess the robustness of the BNNs with the robust training loop
- **eevbnn_augmentation.py**: main code, contains the implementation of the robust training loop and all its utility functions
- **eevbnn_figures.py**: utility code to plot and save images
- **eevbnn_graphs.ipynb**: code to plot and save the graphs of the paper
- **eevbnn_models.py**: implementation of our BNNs along with the code to train and test them (classicaly)
- **eevbnn_verif.py**: code to perform the attacks against the BNNs, uses directly the MiniSatCS solver
- **README.md**: this file
- **requirements.txt**: the required packages to run MiniSatCS and our code
- **standard_model.py**: implementation of a classical neural network with Pytorch, closed to the type 1 BNN

## Building MiniSatCS
---
Please refer to the original paper of MiniSatCS: [Efficient and Exact Verification of Binarized Neural Networks](https://github.com/jia-kai/eevbnn)




