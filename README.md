# Introduction

This repository contains the code for generating synthetic data for training a DynoNet model as described in the paper: Synthetic data generation for system identification: leveraging knowledge transfer from similar systems by Dario Piga, Matteo Rufolo, Gabriele Maroni, Manas Mejari, Marco Forgione.

# Requirements

*   Python 3.x
*   PyTorch
*   NumPy
*   Matplotlib


# Main files

* train_dynoNet.py: Contains the code for training the DynoNet model using synthetic data generated from a predefined system.
Training Process

* plot.py: Contain the code for generating plots

# Hardware requirements

While all the scripts can run on CPU, execution may be slow. For faster training, a GPU is highly recommended.
To run the paper's examples, we used a server equipped with an nVidia RTX 3090 GPU.

# License

This project is licensed under the MIT License - see the LICENSE file for details.
