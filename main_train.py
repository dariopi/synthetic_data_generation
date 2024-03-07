import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from transformer_sim import Config, TSTransformer
from pathlib import Path
from dataset import LinearDynamicalDataset, WHDataset
from trajGenerator import trajGenerator
from dynonet.lti import MimoLinearDynamicalOperator
from dynonet.static import MimoStaticNonLinearity
import datetime
import os


def compute_loss(G1, F_nl, G2, y0_1, u0_1, y0_2, u0_2, u_true, y_true, sigmay=1, nin=10):
    """
    Computes the loss for a 2-layer dynonet model.

    Parameters:
    G1, G2: Linear dynamical block of the dynonet.
    F_nl: Non-linear static block of the dynonet.
    y0_1, y0_2, u0_1, u0_2: Initial states and inputs for the linear blocks of the dynonet.
    u_true: Actual input.
    y_true: Actual output.
    sigmay: Standard deviation of output noise (default=1).
    nin: Number of initial steps to discard in loss computation (default=10).
    plot: Boolean, if True, plot the true and predicted values (default=False).

    Returns:
    loss_sim: Computed loss based on mean squared error.
    r2_sim: computed r2 coefficient
    """

    # Setting the models to evaluation mode
    G1.eval()
    F_nl.eval()
    G2.eval()

    # Simulating the output of the system
    y_lin_1 = G1(u_true, y0_1, u0_1)
    y_nl_1 = F_nl(y_lin_1)
    y_lin_2 = G2(y_nl_1, y0_2, u0_2)
    y_hat = y_lin_2

    # Computing the loss, excluding the initial 'nin' steps
    err_sim = y_true[:, nin:, :] - y_hat[:, nin:, :]
    r2_sim = 1- torch.sum(err_sim**2) /torch.sum((y_true[:, nin:, :] - torch.mean(y_true[:, nin:, :]))**2)
    r2_sim = torch.tensor(max((r2_sim,0.0)))

    loss_sim = torch.mean(err_sim ** 2 / sigmay ** 2)

    return loss_sim, r2_sim


def train_dynoNet(data_train=None, tG_dl=None, data_val=None, data_test=None, sigmay_train=1,
                  gamma_list=[1], regularization=True, nin=10, lr=1e-2, epochs=1000, msg_print=1):
    """
    Trains the DynoNet model with provided datasets and data loader for generating synthetic data.

    Args:
    data_train, data_val, data_test: Dictionaries containing 'u' (input) and 'y' (output) for training, validation, and testing.
    tG_dl: DataLoader for generating synthetic data.
    sigmay_train: Standard deviation on training data.
    gamma_list: List of regularization parameters. Default: [1].
    regularization: Boolean indicating if regularization is applied. If not, variance on the estimation error is used to weight synthetic data
    nin: Number of initial conditions to ignore in loss computation.
    lr: Learning rate for the optimizer.
    epochs: Number of training epochs.
    msg_print: Frequency of printing the training progress.

    Returns:
    train_loss_dict, val_loss_dict, test_loss_dict: Dictionaries containing training, validation and test losses.
    """

    # Unpacking training and validation data
    u_t, y_t = data_train['u'].cpu(), data_train['y'].cpu()  # Training data
    u_v, y_v = data_val['u'], data_val['y']  # Validation data
    u_test, y_test = data_test['u'], data_test['y']  # Test data
    nu, ny = u_t.shape[2], y_t.shape[2]  # Input and output dimensions
    ctx_len = u_t.shape[1]  # Context length


    # Loss tracking
    train_loss_dict, val_loss_dict, test_loss_dict = {}, {}, {}
    val_r2_dict, test_r2_dict = {}, {}


    for gamma in gamma_list:

        # Creating DynoNet model components
        nb_1, na_1 = 10, 10  # Parameters for the first linear section
        G1 = MimoLinearDynamicalOperator(nu, 1, n_b=nb_1, n_a=na_1)
        F_nl = MimoStaticNonLinearity(1, 1, n_hidden=32)  # Non-linear section
        nb_2, na_2 = 10, 10  # Parameters for the second linear section
        G2 = MimoLinearDynamicalOperator(1, ny, n_b=nb_2, n_a=na_2)

        # Initial conditions for DynoNet LTI blocks
        y0_1, u0_1 = torch.zeros((batch_size, na_1), dtype=torch.float), torch.zeros((batch_size, nb_1),
                                                                                     dtype=torch.float)
        y0_2, u0_2 = torch.zeros((batch_size, na_2), dtype=torch.float), torch.zeros((batch_size, nb_2),
                                                                                     dtype=torch.float)

        # Setting up the optimizer
        optimizer = torch.optim.Adam([
            {'params': G1.parameters(), 'lr': lr},
            {'params': F_nl.parameters(), 'lr': lr},
            {'params': G2.parameters(), 'lr': lr}
        ], lr=lr)

        count_p = 0 # count number of parameters in the dynont

        count_p += sum(p.numel() for p in G1.parameters() if p.requires_grad)
        count_p += sum(p.numel() for p in G2.parameters() if p.requires_grad)
        count_p += sum(p.numel() for p in F_nl.parameters() if p.requires_grad)

        print(f"Number of weights in dynoNet: {count_p:5d}")



        val_loss_best = np.inf
        test_loss_best = np.inf
        val_r2_best = 0
        test_r2_best = 0

        for epoch in range(epochs):
            # Set models to training mode
            G1.train()
            F_nl.train()
            G2.train()

            # Reset gradients
            optimizer.zero_grad()

            # Generate synthetic data and concatenate with context data
            with torch.no_grad():
                # generate synthetic data
                u_sim, y_sim, sigmay_sim = next(iter(tG_dl))
                u_sim = u_sim[:,:,:].cpu()
                y_sim = y_sim[:,:,:].cpu()
                sigmay_sim = sigmay_sim.cpu()
                u_conc = torch.concat((u_t.repeat(batch_size, 1, 1), u_sim), dim=1)
                #y_conc = torch.concat((y_t.repeat(batch_size, 1, 1), y_sim), dim=1)
                #sigmay_conc = torch.concat((y_t.repeat(batch_size, 1, 1)*0, sigmay_sim), dim=1)



            # Forward pass: Simulate the DynoNet
            y_lin_1 = G1(u_conc, y0_1, u0_1)
            y_nl_1 = F_nl(y_lin_1)
            y_hat = G2(y_nl_1, y0_2, u0_2)

            # Compute loss for context and synthetic data
            err_ctx = y_t - y_hat[0:1, :ctx_len, :]
            err_synt = y_sim - y_hat[:, ctx_len:, :]
            loss_ctx = torch.sum(err_ctx ** 2 / sigmay_train ** 2)
            loss_sim = torch.sum(err_synt[:,25:,:] ** 2 / (sigmay_train ** 2 if regularization else sigmay_sim[:,25:,:] ** 2))
            loss = (loss_ctx + gamma * loss_sim) / (ctx_len + gamma * batch_size * len_sim)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Evaluate performance on training, validation, and test sets
                train_loss, _ = compute_loss(G1, F_nl, G2, y0_1, u0_1, y0_2, u0_2, u_t, y_t, sigmay=sigmay_train, nin=nin)
                val_loss, val_r2 = compute_loss(G1, F_nl, G2, y0_1, u0_1, y0_2, u0_2, u_v, y_v, sigmay=sigmay_train, nin=nin)
                test_loss, test_r2 = compute_loss(G1, F_nl, G2, y0_1, u0_1, y0_2, u0_2, u_test, y_test, sigmay=sigmay_train, nin=nin)
                synt_loss, _ = compute_loss(G1, F_nl, G2, y0_1, u0_1, y0_2, u0_2, u_sim[0:1, :, :], y_sim[0:1, :, :], sigmay=sigmay_train, nin=nin)

                # Update best validation loss
                if val_loss.item() < val_loss_best:
                    # train_loss_best = train_loss.item() # as a best train loss, we take the one reached with early stopping
                    val_loss_best = val_loss.item()
                    test_loss_best = test_loss.item()
                    val_r2_best = val_r2.item()
                    test_r2_best = test_r2.item()

                # Print progress
                if epoch % msg_print == 0:
                    print(f'Epoch {epoch} | Train Loss {loss:.6f} | Train Loss v2 {train_loss:.4f} | Validation Loss {val_loss:.4f} | Test Loss {test_loss:.4f} ')

        # Store best losses for each gamma
        train_loss_dict[gamma] = train_loss.item()
        val_loss_dict[gamma] = val_loss_best
        test_loss_dict[gamma] = test_loss_best
        val_r2_dict[gamma] = val_r2_best
        test_r2_dict[gamma] = test_r2_best


        print("\n")

        print("Train Losses:")
        for key, value in train_loss_dict.items():
            print(f"{key}: {value:.4f}")

        print("\n Validation Losses:")
        for key, value in val_loss_dict.items():
            print(f"{key}: {value:.4f}")

        print("\nTest Losses:")
        for key, value in test_loss_dict.items():
            print(f"{key}: {value:.4f}")
        print("\n")

    return train_loss_dict, val_loss_dict, test_loss_dict, val_r2_dict, test_r2_dict


if __name__ == '__main__':
    # Setting up basic parameters for data generation and training
    batch_size = 1
    ctx_len = 250  # Context length
    len_sim = 200  # Length of synthetic trajectory
    val_len = 100  # Length of validation dataset
    test_len = 4000  # Length of test dataset
    nu, ny = 1, 1  # Dimensions of input and output
    nin = 50  # Number of initial steps to discard
    sigmay_train = 0.35  # Standard deviation of noise in training data
    n_MC = 100 # number of Monte Carlo runs

    # Define gamma values for regularization
    gamma_list =  [0,  0.1, 1, 10, 20, 30, 50,  100, 200] # [0, 0.5,  1, 10, 25]

    # Training hyperparameters
    lr = 1e-3
    epochs = 8001
    msg_print = 3000  # Frequency of printing training progress


    train_loss_list = []
    val_loss_list = []
    test_loss_list=[] # at each monte carlo run, add performance score in lists
    val_r2_list = []
    test_r2_list=[] # at each monte carlo run, add performance score in lists

    train_loss_gamma0 = []
    val_loss_gamma0 = []
    test_loss_gamma0 = []
    val_r2_gamma0 = []
    test_r2_gamma0 = []

    val_loss_best = []
    test_loss_best = []
    val_r2_best = []
    test_r2_best = []

    # Compute settings and model configuration
    out_dir = "out"  # Output directory
    cuda_device = "cuda:0"
    no_cuda = False
    threads = 5
    torch.set_num_threads(threads)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device_name = cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    torch.set_float32_matmul_precision("high")

    # Loading pre-trained model
    out_dir = Path(out_dir)
    exp_data = torch.load(out_dir / "ckpt_sim_wh_1000_pre_final.pt", map_location=device)
    cfg = exp_data["cfg"]

    # Handling potential missing attribute in configuration
    try:
        cfg.seed
    except AttributeError:
        cfg.seed = None

    for system_seed, np_seed, torch_seed in zip(np.arange(1000, 1000+n_MC), np.arange(100,100+n_MC), np.arange(200,200+n_MC)):
        # Seeds for reproducibility

        print(f"Number of Monte Carlo run: {system_seed-1000+1:d}")

        np.random.seed(np_seed)
        torch.manual_seed(torch_seed)

        # Data generation for the system
        # Generate input data and initialize datasets
        my_u = np.random.randn(ctx_len + val_len + test_len + 200, 1)  # Extra 200 samples for initialization
        my_u[:200, :] = 0  # Zeroing first 400 samples
        test_ds = WHDataset(nx=10, nu=1, ny=1, seq_len=ctx_len + val_len + test_len + 200, fixed_system=True, system_seed=system_seed, u=my_u)
        test_dl = DataLoader(test_ds, batch_size=1, num_workers=0)
        batch_y, batch_u = next(iter(test_dl))

        # Splitting data into training, validation, and test sets
        # Training data with added noise
        u_t, y_t = batch_u[:, :ctx_len, :], batch_y[:, :ctx_len, :]
        y_noise = sigmay_train * torch.randn_like(y_t)
        y_t += y_noise
        u_t = u_t.to(device)
        y_t = y_t.to(device)
        data_train = {'u': u_t, 'y': y_t}
        SNR = torch.norm(y_t)**2/torch.norm(y_noise)**2
        print(f"SNR: {SNR.item(): .1f} db")


        # Validation data with added noise
        u_v, y_v = batch_u[:, ctx_len:ctx_len + val_len, :], batch_y[:, ctx_len:ctx_len + val_len, :]
        y_noise = sigmay_train * torch.randn_like(y_v)
        y_v += y_noise
        data_val = {'u': u_v, 'y': y_v}

        # Test data
        u_test, y_test = batch_u[:, ctx_len + val_len:, :], batch_y[:, ctx_len + val_len:, :]
        data_test = {'u': u_test, 'y': y_test}



        # Initializing the model with loaded configuration
        model_args = exp_data["model_args"]
        conf = Config(**model_args)
        model = TSTransformer(conf).to(device)
        model.load_state_dict(exp_data["model"])

        # Initialize DataLoader for trajectory generation
        tG_ds = trajGenerator(model=model, u_ctx=u_t, y_ctx=y_t, device = device, len_sim=len_sim)
        tG_dl = DataLoader(tG_ds, batch_size=batch_size, num_workers=0)

        # Train the model and record validation and test losses
        train_loss_dict, val_loss_dict, test_loss_dict, val_r2_dict, test_r2_dict = train_dynoNet(data_train=data_train, tG_dl=tG_dl,
                                                      data_val=data_val, data_test=data_test,
                                                      sigmay_train=sigmay_train, gamma_list=gamma_list,
                                                      regularization=True, nin=nin,
                                                      lr=lr, epochs=epochs, msg_print=msg_print)


        # Find the gamma with the minimum value in val_loss_dict (excluding gamma=0)
        min_gamma = min((gamma for gamma in val_loss_dict if gamma != 0), key=val_loss_dict.get)

        train_loss_gamma0.append(train_loss_dict[0])
        val_loss_gamma0.append(val_loss_dict[0])
        test_loss_gamma0.append(test_loss_dict[0])
        val_r2_gamma0.append(val_r2_dict[0])
        test_r2_gamma0.append(test_r2_dict[0])

        val_loss_best.append(val_loss_dict[min_gamma])
        test_loss_best.append(test_loss_dict[min_gamma])
        val_r2_best.append(val_r2_dict[min_gamma])
        test_r2_best.append(test_r2_dict[min_gamma])

        train_loss_list.append(train_loss_dict)
        val_loss_list.append(val_loss_dict)
        test_loss_list.append(test_loss_dict)
        val_r2_list.append(val_r2_dict)
        test_r2_list.append(test_r2_dict)

        # Output the results of training
        print('gamma vs validation loss')
        print(val_loss_dict)
        print('gamma vs test loss')
        print(test_loss_dict)
        print('gamma vs validation R2')
        print(val_r2_dict)
        print('gamma vs test R2')
        print(test_r2_dict)
    # plots


    # Extracting values for each gamma
    gamma_values = list(val_loss_list[0].keys())
    gamma_data = {gamma: [] for gamma in gamma_values}
    gamma_data_train = {gamma: [] for gamma in gamma_values}
    gamma_data_test = {gamma: [] for gamma in gamma_values}


    for train_loss_dict in train_loss_list:
        for gamma, train_loss in train_loss_dict.items():
            gamma_data_train[gamma].append(train_loss)

    for val_loss_dict in val_loss_list:
        for gamma, val_loss in val_loss_dict.items():
            gamma_data[gamma].append(val_loss)

    for test_loss_dict in test_loss_list:
        for gamma, test_loss in test_loss_dict.items():
            gamma_data_test[gamma].append(test_loss)

    # save data

    test_r2_gamma0 = np.nan_to_num(test_r2_gamma0, nan=0)

    data_to_save = {
        'train_loss_gamma0': train_loss_gamma0,
        'val_loss_gamma0': val_loss_gamma0,
        'test_loss_gamma0': test_loss_gamma0,
        'val_r2_gamma0': val_r2_gamma0,
        'test_r2_gamma0': test_r2_gamma0,
        'val_loss_best': val_loss_best,
        'test_loss_best': test_loss_best,
        'val_r2_best': val_r2_best,
        'test_r2_best': test_r2_best,
        'gamma_values' : gamma_values,
        'gamma_data': gamma_data,
        'gamma_data_train': gamma_data_train,
        'gamma_data_test':gamma_data_test
    }

    # Create subfolder
    folder_name = 'saved_data'
    os.makedirs(folder_name, exist_ok=True)

    # Generate filename with date and time
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y-%m-%d_%H-%M-%S") + "data_to_save"
    file_path = os.path.join(folder_name, filename)

    # Save data using pickle
    with open(file_path, 'wb') as fp:
        pickle.dump(data_to_save, fp)



    ######################3
    # for plotting, run the script plot.py.
    # In running the script, you also also specify the name of the pickle file where your data is saved



    ###########################3
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

    # Boxplot for validation data
    axes[0].set_title('Validation')
    for i, gamma in enumerate(gamma_values):
        axes[0].boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=[str(gamma)])
    axes[0].set_xlabel('Gamma Values')
    axes[0].set_ylabel('Loss Values')
    axes[0].set_ylim(-0.1, 20)
    #plt.ylim(-0.1, 20)
    axes[0].grid(True)


    # Boxplot for test data
    axes[1].set_title('Test')
    for i, gamma in enumerate(gamma_values):
        axes[1].boxplot(gamma_data_test[gamma], positions=[i], widths=0.6, labels=[str(gamma)])
    axes[1].set_xlabel('Gamma Values')
    axes[1].set_ylabel('Loss Values')
    axes[1].grid(True)

    #plt.tight_layout()
    plt.ylim(-0.1, 20)
    plt.show()


    # Creating box plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    # Boxplot for validation data
    axes[0].boxplot(gamma_data[0], positions=[0], widths=0.6, labels=['gamma validation= 0'])
    axes[0].boxplot(val_loss_best, positions=[1], widths=0.6, labels=['best validation gamma ≠ 0'])
    axes[0].set_title('Validation Losses')
    axes[0].grid(True)

    # Boxplot for test data
    axes[1].boxplot(gamma_data_test[0], positions=[0], widths=0.6, labels=['gamma test= 0'])
    axes[1].boxplot(test_loss_best, positions=[1], widths=0.6, labels=['best test gamma ≠ 0'])
    axes[1].set_title('Test Losses')
    axes[1].grid(True)

    #plt.tight_layout()
    plt.ylim(-0.1, 20)
    plt.show()


    # R2

    # Creating box plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    # Boxplot for validation data
    axes[0].boxplot(val_r2_gamma0, positions=[0], widths=0.6, labels=['gamma validation= 0'])
    axes[0].boxplot(val_r2_best, positions=[1], widths=0.6, labels=['best validation gamma ≠ 0'])
    axes[0].set_title('R2 val')
    axes[0].grid(True)

    # Boxplot for test data
    axes[1].boxplot(test_r2_gamma0, positions=[0], widths=0.6, labels=['gamma test= 0'])
    axes[1].boxplot(test_r2_best, positions=[1], widths=0.6, labels=['best test gamma ≠ 0'])
    axes[1].set_title('R2 test')
    axes[1].grid(True)

    # plt.tight_layout()
    plt.ylim(-0.1, 1.2)
    plt.show()