# Importing necessary libraries and modules
from lti import drss_matrices, dlsim
from torch.utils.data import DataLoader, Subset, IterableDataset
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import tqdm
import wandb
import matplotlib.pyplot as plt
import torch
import numpy as np
from transformer_sim import Config, TSTransformer
from pathlib import Path
from dataset import LinearDynamicalDataset, WHDataset, WHDatasetBatch


# Definition of the trajGenerator class, which generates trajectories for simulation
class trajGenerator(IterableDataset):
    def __init__(self, model, u_ctx, y_ctx, device, len_sim=100):
        super(trajGenerator, self).__init__()
        self.model = model  # GPT Model to be used for generating trajectories
        self.u_ctx = u_ctx  # Input context
        self.y_ctx = y_ctx  # Output context
        self.len_sim = len_sim  # Length of the simulation
        self.nu = self.u_ctx.shape[-1]  # Number of input features
        self.device = device

    # Iterator method to generate simulation data continuously
    def __iter__(self):
        while True:
            u_sim = torch.randn(1, self.len_sim, self.nu).to(self.device)  # Generating random inputs
            y_sim, sigmay_sim, _ = self.model(self.y_ctx, self.u_ctx, u_sim)  # Getting predictions from the model for given input sequence u_sim. Output: simulated output, standard deviation of the uncertainty

            yield u_sim.squeeze(0), y_sim.squeeze(0), sigmay_sim.squeeze(0)  # Yielding the generated data


##############################

if __name__ == '__main__':

    # Setting up basic configuration parameters
    out_dir = "out"  # Output directory where the model is saved
    nu = 1  # Number of input features
    ny = 1  # Number of output features
    batch_size = 20  # Batch size for data loading
    fixed_system = True  # Flag to determine if the system is fixed

    # Configuring the compute settings
    cuda_device = "cuda:0"
    no_cuda = True
    threads = 5  # Number of threads for data loading
    compile = False

    # Setting up the torch environment
    torch.set_num_threads(threads)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device_name = cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    device_type = 'cuda' if 'cuda' in device_name else 'cpu'
    torch.set_float32_matmul_precision("high")

    # Creating the output directory
    out_dir = Path(out_dir)
    exp_data = torch.load(out_dir / "ckpt_sim_wh_1000_pre.pt", map_location=device)
    cfg = exp_data["cfg"]

    # Handling potential missing attribute in configuration
    try:
        cfg.seed
    except AttributeError:
        cfg.seed = None

    # Loading the model and its configuration
    model_args = exp_data["model_args"]
    conf = Config(**model_args)
    model = TSTransformer(conf).to(device)
    model.load_state_dict(exp_data["model"])

    # Data-generating system
    test_ds = WHDataset(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, seq_len=cfg.seq_len_ctx, fixed_system=fixed_system)
    test_dl = DataLoader(test_ds, batch_size=1, num_workers=threads)

    # Getting a batch of data for contextg
    batch_y, batch_u = next(iter(test_dl))
    batch_y = batch_y.to(device)
    batch_u = batch_u.to(device)


    # Further preparation for simulation

    with torch.no_grad():  # Disabling gradient computation for inference
        y_ctx = batch_y[:, :, :]
        err = torch.randn(y_ctx.shape) * 0.2  # Adding noise to the output context
        y_ctx = y_ctx + err
        u_ctx = torch.clone(batch_u)  # Cloning the input context

        # Initializing the trajectory generator with the model and context data
        tG_ds = trajGenerator(model=model, u_ctx=u_ctx, y_ctx=y_ctx, len_sim=100)
        tG_dl = DataLoader(tG_ds, batch_size=batch_size, num_workers=0)

        # Simulating and plotting the data
        for a in range(3):  # Running three iterations for simulation
            u_sim, y_sim, sigmay_sim = next(iter(tG_dl))  # Generating a set of simulated data
            print(a)
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(u_sim[0, :, 0], label='u')  # Plotting the input
            plt.subplot(3, 1, 2)
            plt.plot(y_sim[0, :, 0], label='y')  # Plotting the output
            # Plotting the uncertainty in output predictions
            plt.fill_between(x=np.arange(y_sim.shape[1]),
                             y1=y_sim[0, :, 0] - 3 * sigmay_sim[0, :, 0],
                             y2=y_sim[0, :, 0] + 3 * sigmay_sim[0, :, 0],
                             color="blue", alpha=0.2)
            plt.legend()
        plt.show()  # Displaying the plots










#
#
# import math
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader, IterableDataset
# import control  # pip install python-control, pip install slycot (optional)
# import pickle
# from lti import drss_matrices, dlsim
# from torch.utils.data import DataLoader, Subset
# from sklearn.model_selection import train_test_split
# from dataclasses import dataclass
# #from transformer_sim_DP import TSTransformer
# import tqdm
# import wandb
# import matplotlib.pyplot as plt
# from transformer_sim import Config, TSTransformer
# from pathlib import Path
# from dataset import LinearDynamicalDataset, WHDataset
#
# class trajGenerator(IterableDataset):
#     def __init__(self, model, u_ctx , y_ctx, len_sim = 100):
#         super(trajGenerator).__init__()
#         self.model = model
#         self.u_ctx = u_ctx
#         self.y_ctx = y_ctx
#         self.len_sim  = len_sim
#         self.nu = self.u_ctx.shape[-1]
#
#
#     def __iter__(self):
#
#         while True:
#             u_sim = torch.randn(1,self.len_sim, self.nu)
#             y_sim, sigmay_sim, _  = self.model(self.y_ctx, self.u_ctx, u_sim)
#
#             yield u_sim.squeeze(0), y_sim.squeeze(0), sigmay_sim.squeeze(0)
#
#
#
# ##############################
#
# if  __name__ == '__main__':
#
#     # Overall settings
#     out_dir = "out"
#
#     # System settings
#     nu = 1
#     ny = 1
#     batch_size = 20  # 256
#     fixed_system = True  # Are we testing on a fixed system?
#
#     # Compute settings
#     cuda_device = "cuda:0"
#     no_cuda = True
#     threads = 5
#     compile = False
#
#     # Configure compute
#     torch.set_num_threads(threads)
#     use_cuda = not no_cuda and torch.cuda.is_available()
#     device_name = cuda_device if use_cuda else "cpu"
#     device = torch.device(device_name)
#     device_type = 'cuda' if 'cuda' in device_name else 'cpu'  # for later use in torch.autocast
#     torch.set_float32_matmul_precision("high")
#
#     # Create out dir
#     out_dir = Path(out_dir)
#     exp_data = torch.load(out_dir / "ckpt_sim_wh_1000_pre.pt", map_location=device)
#     cfg = exp_data["cfg"]
#     # For compatibility with initial experiment without seed
#     try:
#         cfg.seed
#     except AttributeError:
#         cfg.seed = None
#
#     model_args = exp_data["model_args"]
#     conf = Config(**model_args)
#     model = TSTransformer(conf).to(device)
#     model.load_state_dict(exp_data["model"])
#
#     test_ds = WHDataset(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, seq_len=cfg.seq_len_ctx, fixed_system = fixed_system)
#     test_dl = DataLoader(test_ds, batch_size=1, num_workers=threads)
#
#     batch_y, batch_u = next(iter(test_dl))
#     batch_y = batch_y.to(device)
#     batch_u = batch_u.to(device)
#
#     batch_size_t = 1
#     with torch.no_grad():
#
#         y_ctx = batch_y[:, :, :]
#         err = torch.randn(y_ctx.shape)*0.2
#         y_ctx = y_ctx + err
#         u_ctx = torch.clone(batch_u)
#
#         # Loader of trajectories
#         tG_ds = trajGenerator(model = model, u_ctx = u_ctx , y_ctx = y_ctx, sigma = 0.5, len_sim = 100)
#         tG_dl = DataLoader(tG_ds, batch_size=batch_size, num_workers=0)
#
#
#         for a in range(3):
#             u_sim, y_sim, sigmay_sim = next(iter(tG_dl))
#             print(a)
#             plt.figure()
#             plt.subplot(3,1,1)
#             plt.plot(u_sim[0,:,0], label = 'u')
#             plt.subplot(3,1,2)
#             plt.plot(y_sim[0,:,0], label = 'y')
#             plt.fill_between(x = np.arange(y_sim.shape[1]), y1=y_sim[0, :, 0] - 3*sigmay_sim[0, :, 0], y2=y_sim[0, :, 0] + 3*sigmay_sim[0, :, 0], color="blue", alpha=0.2)
#             plt.legend()
#         plt.show()