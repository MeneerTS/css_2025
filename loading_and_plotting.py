import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import sem, t


def plot_non_voter_influence():
    folder_name = "numpy_files\\"
    non_voters_0 = np.load(folder_name + "voters_100_non_voters_0_number_of_r_20_iterations_per_r_100_r_analysis_plot.npy")
    non_voters_50 = np.load(folder_name + "voters_100_non_voters_50_number_of_r_20_iterations_per_r_100_r_analysis_plot.npy")
    non_voters_100 = np.load(folder_name + "voters_100_non_voters_100_number_of_r_20_iterations_per_r_100_r_analysis_plot.npy")
    non_voters_500 = np.load(folder_name + "voters_100_non_voters_500_number_of_r_20_iterations_per_r_100_r_analysis_plot.npy")
    non_voters_1000 = np.load(folder_name + "voters_100_non_voters_1000_number_of_r_20_iterations_per_r_100_r_analysis_plot.npy")
    x = np.arange(0, 100, 5)

    plt.figure(figsize=(10, 6))  # Set figure size for better visibility

    # Plot each dataset with clear labels and distinguishable line styles
    plt.plot(x, non_voters_0, label="Non-voters: 0", linestyle='-', color='#A7A6DA', marker='o')
    plt.plot(x, non_voters_100, label="Non-voters: 100", linestyle='--', color='#D74967', marker='s')
    plt.plot(x, non_voters_1000, label="Non-voters: 1000", linestyle='-.', color='#B384C0', marker='d')

    # Add title and labels
    plt.title("Effect of Non-Voters on Agreement", fontsize=16)
    plt.xlabel("Neighbour radius r", fontsize=14)
    plt.ylabel("Voter agreement", fontsize=14)

    # Set x and y axis limits
    plt.xlim(-10, 110)
    plt.ylim(0.0, 1.1)

    # Add grid for readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add legend
    plt.legend(title="Legend", fontsize=12)

    # Adjust plot margins for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_r_analysis_with_standard_deviation():
    folder_name = "numpy_files\\"
    values = np.load(folder_name + "voters_100_non_voters_0_number_of_r_10_iterations_per_r_500_r_analysis_all_data.npy")
        
        # Calculate the mean and IQR for each experiment (row)
    means = np.mean(values, axis=1)
    iqr_values = np.percentile(values, 75, axis=1) - np.percentile(values, 25, axis=1)  # IQR = Q3 - Q1
    lower_bound = np.percentile(values, 25, axis=1)  # Lower bound (Q1)
    upper_bound = np.percentile(values, 75, axis=1)  # Upper bound (Q3)

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot the mean values
    plt.plot(range(1, values.shape[0] + 1), means, label="Mean", color='blue', marker='o')

    # Plot the IQR-based confidence intervals
    plt.fill_between(range(1, values.shape[0] + 1), lower_bound, upper_bound, color='blue', alpha=0.3, label="IQR Confidence Interval")

    print(means)
    print(lower_bound)
    print(upper_bound)
    # Adding labels and title
    plt.xlabel("Experiment Number (n)")
    plt.ylabel("Value")
    plt.title("Mean and IQR-based Confidence Intervals for Experiments")
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == '__main__':
    plot_r_analysis_with_standard_deviation()



