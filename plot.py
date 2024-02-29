import numpy as np
import pickle

from pathlib import Path
from dataset import LinearDynamicalDataset, WHDataset
from trajGenerator import trajGenerator
from dynonet.lti import MimoLinearDynamicalOperator
from dynonet.static import MimoStaticNonLinearity
import pandas as pd
import datetime
import os

import torch
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# Setting the font to Times New Roman, size 12
# Setting font family and size globally
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 10

folder_name = 'saved_data'
filename = '2024-02-28_20-07-17data_to_save' # '2024-02-23_07-30-33data_to_save'
file_path = os.path.join(folder_name, filename)
# Load the data
with open(file_path, 'rb') as fp:
    loaded_data = pickle.load(fp)

for key, value in loaded_data.items():
    globals()[key] = value



###########################3
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

# Boxplot for validation data
axes[0].set_title('Validation')
for i, gamma in enumerate(gamma_values):
    gamma_data[gamma] = [x for x in gamma_data[gamma] if not np.isnan(x)]
    axes[0].boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=[str(gamma)])
axes[0].set_xlabel('Gamma Values')
axes[0].set_ylabel('Loss Values')
axes[0].set_ylim(-0.1, 20)
# plt.ylim(-0.1, 20)
axes[0].grid(True)

# Boxplot for test data
axes[1].set_title('Test')
for i, gamma in enumerate(gamma_values):
    gamma_data_test[gamma] = [x for x in gamma_data_test[gamma] if not np.isnan(x)]
    axes[1].boxplot(gamma_data_test[gamma], positions=[i], widths=0.6, labels=[str(gamma)])
axes[1].set_xlabel('Gamma Values')
axes[1].set_ylabel('Loss Values')
axes[1].grid(True)

# plt.tight_layout()
plt.ylim(-0.1, 20)
plt.show()

# Creating box plots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Boxplot for validation data
axes[0].boxplot(gamma_data[0], positions=[0], widths=0.6, labels=['gamma validation= 0'])
val_loss_best = [x for x in val_loss_best if not np.isnan(x)]
axes[0].boxplot(val_loss_best, positions=[1], widths=0.6, labels=['best validation gamma \neq 0'])
axes[0].set_title('Validation Losses')
axes[0].grid(True)

# Boxplot for test data
axes[1].boxplot(gamma_data_test[0], positions=[0], widths=0.6, labels=['gamma test= 0'])
axes[1].boxplot(test_loss_best, positions=[1], widths=0.6, labels=['best test gamma \neq 0'])
axes[1].set_title('Test Losses')
#axes[1].grid(True)

# plt.tight_layout()
plt.ylim(-0.1, 20)
plt.show()

# R2

# Creating box plots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Boxplot for test data

axes[1].boxplot(test_r2_gamma0, positions=[0], widths=0.6, labels=['gamma test= 0'], showfliers=False)
axes[1].boxplot(test_r2_best, positions=[1], widths=0.6, labels=['best test gamma \neq 0'], showfliers=False)
axes[1].set_title('R2 test')
axes[1].grid(True)


# Boxplot for validation data
axes[0].boxplot(val_r2_gamma0, positions=[0], widths=0.6, labels=['gamma validation= 0'], showfliers=False)
axes[0].boxplot(val_r2_best, positions=[1], widths=0.6, labels=['best validation gamma \neq 0'], showfliers=False)
axes[0].set_title('R2 val')
axes[0].grid(True)



# plt.tight_layout()
plt.ylim(-0.1, 1.2)
plt.show()



#############################################################
#############
# plots for the paper
# R2

# Creating the box plot with the specified dimensions and font settings
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.25))  # Size in inches, approx. 8.8 cm width

# Boxplot for test data
# Adjusting linewidth for boxplots
for box in axes.boxplot(test_r2_gamma0, positions=[0], widths=0.6,  labels=['without\n synthetic data'], showfliers=False)['medians']:
    box.set_linewidth(0.5)  # Adjust linewidth here

for box in axes.boxplot(test_r2_best, positions=[1], widths=0.6, labels=['with\n synthetic data'], showfliers=False)['medians']:
    box.set_linewidth(0.5)  # Adjust linewidth here

#axes.boxplot(test_r2_gamma0, positions=[0], widths=0.6,  labels=['without\n synthetic data'], showfliers=False, linewidth=0.5)
#axes.boxplot(test_r2_best, positions=[1], widths=0.6, labels=['with\n synthetic data'], showfliers=False)
axes.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
axes.set_ylabel('$R^2$ index')
axes.set_ylim(-0.1, 1.1)


plt.tight_layout()
plt.savefig('r2index.pdf', format='pdf')

plt.show()




#############
# plots for the paper
# loss

###########################3
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.25))

# Boxplot for validation data
#axes[0].set_title('Validation')
for i, gamma in enumerate(gamma_values):
    gamma_data[gamma] = [x for x in gamma_data[gamma] if not np.isnan(x)]

    for box in \
    axes.boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=[str(gamma)], showfliers=False)['medians']:
        box.set_linewidth(0.8)  # Adjust linewidth here

    #axes.boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=[str(gamma)], showfliers=False)

axes.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
axes.set_xlabel('$\gamma$')
axes.set_ylabel('Loss')
#axes[0].set_ylim(-0.1, 20)
plt.ylim(-0.1, 18)

plt.tight_layout()

# Save the figure again with the new specifications
plt.savefig('loss.pdf', format='pdf')

plt.show()





###########################3
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.25))

# Boxplot for training data
for i, gamma in enumerate(gamma_values):
    gamma_data_train[gamma] = [x for x in gamma_data_train[gamma] if not np.isnan(x)]

    for box in \
    axes.boxplot(gamma_data_train[gamma], positions=[i], widths=0.6, labels=[str(gamma)], showfliers=False)['medians']:
        box.set_linewidth(0.8)  # Adjust linewidth here

    #axes.boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=[str(gamma)], showfliers=False)

axes.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
axes.set_xlabel('$\gamma$')
axes.set_ylabel('Loss')
#axes[0].set_ylim(-0.1, 20)
plt.ylim(-0.1, 18)

plt.tight_layout()

# Save the figure again with the new specifications
plt.savefig('loss_train.pdf', format='pdf')

plt.show()


###########################################



#####################
###########################3
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 2.25))

# Boxplot for validation data
#axes[0].set_title('Validation')
for i, gamma in enumerate(gamma_values):
    gamma_data[gamma] = [x for x in gamma_data[gamma] if not np.isnan(x)]

    for box in \
    axes[1].boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=[str(gamma)], showfliers=False)['medians']:
        box.set_linewidth(0.8)  # Adjust linewidth here

    #axes.boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=[str(gamma)], showfliers=False)

axes[1].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
axes[1].set_xlabel('$\gamma$')
axes[1].set_ylabel('MSE')
axes[1].set_ylim(-0.1, 20)
axes[1].set_title('MSE on Validation data', fontsize = 8)



# Boxplot for training data
for i, gamma in enumerate(gamma_values):
    gamma_data_train[gamma] = [x for x in gamma_data_train[gamma] if not np.isnan(x)]

    for box in \
    axes[0].boxplot(gamma_data_train[gamma], positions=[i], widths=0.6, labels=[str(gamma)], showfliers=False)['medians']:
        box.set_linewidth(0.8)  # Adjust linewidth here

    #axes.boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=[str(gamma)], showfliers=False)

axes[0].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
axes[0].set_xlabel('$\gamma$')
axes[0].set_ylabel('MSE')
axes[0].set_ylim(-0.1, 20)
axes[0].set_title('MSE on Training data', fontsize = 8)

plt.ylim(-0.1, 18)

plt.tight_layout()

# Save the figure again with the new specifications
plt.savefig('loss_train_val.pdf', format='pdf')

plt.show()


print(f"Median on Test data with NO synthetic data: {np.median(test_r2_gamma0): .3f}")
print(f"Median on Test data with synthetic data: {np.median(test_r2_best): .3f}")





















































###########################3
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.25))

# Boxplot for validation data
#axes[0].set_title('Validation')
for i, gamma in enumerate(gamma_values):
    gamma_data[gamma] = [x for x in gamma_data[gamma] if not np.isnan(x)]

    for box in \
    axes.boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=[str(gamma)], showfliers=False)['medians']:
        box.set_linewidth(0.8)  # Adjust linewidth here

    #axes.boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=[str(gamma)], showfliers=False)

axes.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
axes.set_xlabel('$\gamma$')
axes.set_ylabel('MSE')
axes.set_ylim(-0.1, 20)
axes.set_title('MSE on Validation data', fontsize = 8)

plt.ylim(-0.1, 10)

plt.tight_layout()

# Save the figure again with the new specifications
plt.savefig('loss_val.pdf', format='pdf')

plt.show()


###########################3
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.25))

# Boxplot for training data
for i, gamma in enumerate(gamma_values):
    gamma_data_train[gamma] = [x for x in gamma_data_train[gamma] if not np.isnan(x)]

    for box in \
    axes.boxplot(gamma_data_train[gamma], positions=[i], widths=0.6, labels=[str(gamma)], showfliers=False)['medians']:
        box.set_linewidth(0.8)  # Adjust linewidth here

    #axes.boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=[str(gamma)], showfliers=False)

axes.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
axes.set_xlabel('$\gamma$')
axes.set_ylabel('MSE')
axes.set_ylim(-0.1, 20)
axes.set_title('MSE on Training data', fontsize = 8)

plt.ylim(-0.1, 10)

plt.tight_layout()

# Save the figure again with the new specifications
plt.savefig('loss_train.pdf', format='pdf')

plt.show()

