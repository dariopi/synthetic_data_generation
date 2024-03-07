import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True
#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams['font.size'] = 10

folder_name = 'saved_data'
filename = '2024-03-04_05-17-12data_to_save' # 2024-02-28_20-07-17data_to_save'
file_path = os.path.join(folder_name, filename)


# Load the data
with open(file_path, 'rb') as fp:
    loaded_data = pickle.load(fp)

for key, value in loaded_data.items():
    globals()[key] = value


# Creating the box plot with the specified dimensions and font settings
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.25))  # Size in inches, approx. 8.8 cm width

# Boxplot for test data
# Adjusting linewidth for boxplots
for box in axes.boxplot(test_r2_gamma0, positions=[0], widths=0.6,  labels=['without\n synthetic data'], showfliers=False)['medians']:
    box.set_linewidth(0.5)  # Adjust linewidth here

for box in axes.boxplot(test_r2_best, positions=[1], widths=0.6, labels=['with\n synthetic data'], showfliers=False)['medians']:
    box.set_linewidth(0.5)  # Adjust linewidth here

axes.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
axes.set_ylabel('$R^2$ index')
axes.set_ylim(-0.02, 1.02)


plt.tight_layout()
plt.savefig('r2index.pdf', format='pdf')
plt.show()


##############################################################################3
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.25))

# Boxplot for validation data
for i, gamma in enumerate(gamma_values):
    gamma_data[gamma] = [x for x in gamma_data[gamma] if not np.isnan(x)]

    for box in \
    axes.boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=[str(gamma)], showfliers=False)['medians']:
        box.set_linewidth(0.8)  # Adjust linewidth here

axes.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
axes.set_xlabel('$\gamma$')
axes.set_ylabel('MSE')
axes.set_ylim(0.5, 4)
axes.set_title('MSE on Validation data', fontsize = 8)


plt.tight_layout()

# Save the figure again with the new specifications
plt.savefig('loss_val.pdf', format='pdf')
plt.show()


################################################################3
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.25))

# Boxplot for training data
for i, gamma in enumerate(gamma_values):
    gamma_data_train[gamma] = [x for x in gamma_data_train[gamma] if not np.isnan(x)]

    for box in \
    axes.boxplot(gamma_data_train[gamma], positions=[i], widths=0.6, labels=[str(gamma)], showfliers=False)['medians']:
        box.set_linewidth(0.8)  # Adjust linewidth here


axes.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
axes.set_xlabel('$\gamma$')
axes.set_ylabel('MSE')
axes.set_ylim(0.5, 4)
axes.set_title('MSE on Training data', fontsize = 8)

#plt.ylim(0.5, 3)

plt.tight_layout()

# Save the figure again with the new specifications
plt.savefig('loss_train.pdf', format='pdf')

plt.show()

print(f"Median on Test data with NO synthetic data: {np.median(test_r2_gamma0): .3f}")
print(f"Median on Test data with synthetic data: {np.median(test_r2_best): .3f}")



