# Introduction

This repository contains the code for generating synthetic data for training a DynoNet model as described in the paper: "Synthetic data generation for system identification: leveraging knowledge transfer from similar systems" by Dario Piga, Matteo Rufolo, Gabriele Maroni, Manas Mejari, Marco Forgione.

# Disclaimer

This code is released as-is, without any guarantees. It is primarily intended to replicate the results reported in our paper. Users are welcome to use and modify the code, but please note that it may not follow the best practices for optimizing algorithm execution. Furthermore, the code might not adhere to the best programming practices in Python, as our main goal was to illustrate the concepts and methodologies described in the paper. Use this code at your own risk.

# Requirements

*   Python 3.x
*   PyTorch
*   NumPy
*   Matplotlib


# Main files 

* [main_train.py](main_train.py): Contains the code for training the DynoNet model using synthetic data generated from a predefined system.
Training Process

* [plot.py](plot.py): Contain the code for generating plots

# Hardware requirements

While all the scripts can run on CPU, execution may be slow. For faster training, a GPU is highly recommended.
To run the paper's examples, we used a server equipped with an nVidia RTX 3090 GPU.

# License

This project is licensed under the MIT License - see the LICENSE file for details.

# Citing

If you find this project useful, we encourage you to:

* Star this repository :star: 

* Cite the [paper](https://arxiv.org/html/2403.05164v1) 
