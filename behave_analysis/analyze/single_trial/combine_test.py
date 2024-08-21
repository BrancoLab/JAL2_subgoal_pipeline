import pickle
import numpy as np
from behave_analysis.analyze.single_trial.cca import CCAConfig, cca_transform
import matplotlib.pyplot as plt

# First load the two homing dictionaries
A = r"W:\branco\Laurence\JAL006\JAL006_shelter_barrier_flip_3_2024_03_21T11_20_34\processed_data\models\single_trial\homing_dict.pkl"
B = r"W:\branco\Laurence\JAL006\JAL006_barrier_flip2_2024_03_18T11_53_29\processed_data\models\single_trial\homing_dict.pkl"
C = r"W:\branco\Laurence\JAL006\JAL006_shelter_barrier_flip_6_2024_03_28T10_54_20\processed_data\models\single_trial\homing_dict.pkl"

A_classed = r"W:\branco\Laurence\JAL006\JAL006_shelter_barrier_flip_3_2024_03_21T11_20_34\processed_data\models\single_trial\classes.pkl"
B_classed = r"W:\branco\Laurence\JAL006\JAL006_barrier_flip2_2024_03_18T11_53_29\processed_data\models\single_trial\classes.pkl"
C_classed = r"W:\branco\Laurence\JAL006\JAL006_shelter_barrier_flip_6_2024_03_28T10_54_20\processed_data\models\single_trial\classes.pkl"


def load_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)


A = load_pickle(A)
B = load_pickle(B)
C = load_pickle(C)
A_classed = load_pickle(A_classed)
B_classed = load_pickle(B_classed)
C_classed = load_pickle(C_classed)

bin_sizes = [-40, -30, -20, -10, 10]

def func(bin, homing_dict):
    X = None
    for h, homing in enumerate(homing_dict):
        if h == 0:
            X = homing_dict[homing][bin]
        else:
            X = np.vstack((X, homing_dict[homing][bin]))
    return X

cfg = CCAConfig()

colorsA = ["r" if c == 1 else "b" for c in A_classed]  # Make color labels based on class list for matplotlib: 1 is red, 0 is blue
colorsB = ["r" if c == 1 else "b" for c in B_classed]  # Make color labels based on class list for matplotlib: 1 is red, 0 is blue
colorsC = ["r" if c == 1 else "b" for c in C_classed]  # Make color labels based on class list for matplotlib: 1 is red, 0 is blue


for i, bin in enumerate(bin_sizes):
    X1 = func(bin, A)
    X2 = func(bin, B)
    X3 = func(bin, C)
    
    print("The shape of X1 is: ", X1.shape)
    print("The shape of X2 is: ", X2.shape)
    
    A1, A2, A1_mapped, A2_mapped = cca_transform(X1, X2, cfg)
    A1, A3, A1_mapped, A3_mapped = cca_transform(X1, X3, cfg)
    
    # Plotting before alignment
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(A1[0, :], A1[1, :], alpha=0.5, c=colorsA)  # Plot the first two components of A1 and all samples
    plt.scatter(A2[0, :], A2[1, :], alpha=0.5, c=colorsB)  # Plot the first two components of A2 and all samples
    plt.scatter(A3[0, :], A3[1, :], alpha=0.5, c=colorsC)  # Plot the first two components of A3 and all samples
    
    plt.title("Before Alignment")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()

    # Plotting after alignment
    plt.subplot(1, 2, 2)
    mod = 0
    plt.scatter(A1_mapped[:, 0 + mod], A1_mapped[:, 1 + mod], alpha=0.5, c=colorsA)
    plt.scatter(A2_mapped[:, 0 + mod], A2_mapped[:, 1 + mod], alpha=0.5, c=colorsB)
    plt.scatter(A3_mapped[:, 0 + mod], A3_mapped[:, 1 + mod], alpha=0.5, c=colorsC)
    plt.title("After Alignment")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.suptitle(f"Bin size: {bin}")

    plt.show()
    
    y = 10
