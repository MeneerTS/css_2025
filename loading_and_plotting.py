import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import sem, t


def plot_non_voter_influence():
    folder_name = "numpy_files\\"
    non_voters_0 = np.load(folder_name + "voters_100_non_voters_0_number_of_r_20_iterations_per_r_100_r_analysis_plot.npy")
    non_voters_100 = np.load(folder_name + "voters_100_non_voters_100_number_of_r_20_iterations_per_r_100_r_analysis_plot.npy")
    non_voters_1000 = np.load(folder_name + "voters_100_non_voters_1000_number_of_r_20_iterations_per_r_100_r_analysis_plot.npy")
    x = np.arange(0, 100, 5)

    plt.figure(figsize=(10, 6))  # Set figure size for better visibility

    # Plot each dataset with clear labels and distinguishable line styles
    plt.plot(x, non_voters_0, label="Abstention rate: 0%", linestyle='-', color='#1716FA', marker='o')
    plt.plot(x, non_voters_100, label="Abstention rate: 50%", linestyle='--', color='#F71917', marker='s')
    plt.plot(x, non_voters_1000, label="Abstention rate: 90%", linestyle='-.', color='#A324F0', marker='d')

    # Add title and labels
    plt.title("Effect of Non-Voters on Agreement", fontsize=16)
    plt.xlabel("Vision radius r", fontsize=14)
    plt.ylabel("Voter agreement", fontsize=14)

    # Set x and y axis limits
    plt.xlim(-3, 103)
    plt.ylim(0.0, 1.03)

    # Add grid for readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add legend
    plt.legend(title="Legend", fontsize=12)

    # Adjust plot margins for better spacing
    plt.tight_layout()
    plt.savefig("non_voter_influence",dpi=400)
    # Show the plot
    plt.show()




def weird_plot():
    folder_name = "numpy_files\\"
    values = np.load(folder_name + "voters_100_non_voters_0_number_of_r_10_iterations_per_r_500_r_analysis_all_data.npy")
    print(values.shape)  # Should print (10, 500)
    
    # Calculate probabilities
    above_0_9 = (values > 0.9).mean(axis=1)  # Fraction of values > 0.9 for each row
    below_0_1 = (values < 0.1).mean(axis=1)  # Fraction of values < 0.1 for each row
    
    # Plotting
    x = range(1, values.shape[0] + 1)  # Row indices
    plt.plot(x, above_0_9, label="P(Values > 0.9)", marker="o")
    plt.plot(x, below_0_1, label="P(Values < 0.1)", marker="o")
    
    plt.xlabel("Experiment (Row Index)")
    plt.ylabel("Probability")
    plt.title("Probability of Values Being > 0.9 and < 0.1")
    plt.legend()
    plt.grid()
    plt.show()

def plot_r_analysis_with_standard_deviation():
    folder_name = "results\\"
    values = np.load(folder_name + "voters_100_non_voters_0_number_of_r_            100_iterations_per_r_500_r_analysis_all_data.npy")
        
        # Calculate the mean and IQR for each experiment (row)
    means = np.mean(values, axis=1)
    iqr_values = np.percentile(values, 75, axis=1) - np.percentile(values, 25, axis=1)  # IQR = Q3 - Q1
    lower_bound = np.percentile(values, 25, axis=1)  # Lower bound (Q1)
    upper_bound = np.percentile(values, 75, axis=1)  # Upper bound (Q3)

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot the mean values
    plt.plot(range(0,100,100//values.shape[0]), means, label="Mean", color='blue')

    # Plot the IQR-based confidence intervals
    plt.fill_between(range(0,100,100//values.shape[0]), lower_bound, upper_bound, color='blue', alpha=0.3, label="50% Confidence Interval")

    print(means)
    print(lower_bound)
    print(upper_bound)
    # Adding labels and title
    plt.xlim(-3, 103)
    plt.ylim(0.0, 1.03)

    # Add grid for readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.9, alpha=0.7)

    # Add legend
    

    # Adjust plot margins for better spacing
    
    plt.xlabel("Vision radius r", fontsize=14)
    plt.ylabel("Voter Agreement", fontsize=14)
    plt.title("Effect of Vision Radius on Voter Agreement", fontsize=16)
    plt.legend(title="Legend", fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig("r_analysis_with_confidence",dpi=400)
    # Show the plot
    plt.show()

def plot_r_analysis_clean_version(include_percolation_treshold=False):
    folder_name = "results\\"
    values = np.load(folder_name + "voters_100_non_voters_0_number_of_r_            100_iterations_per_r_500_r_analysis_all_data.npy")
        
        # Calculate the mean and IQR for each experiment (row)
    means = np.mean(values, axis=1)
    

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot the mean values
    plt.plot(range(0,100,100//values.shape[0]), means, label="Mean", color='blue')

    # Adding labels and title
   
    plt.legend()

    # Add title and labels
    plt.title("Effect of Vision Radius on Voter Agreement", fontsize=16)
    plt.xlabel("Vision radius r", fontsize=14)
    plt.ylabel("Voter agreement", fontsize=14)

    # Set x and y axis limits
    plt.xlim(-3, 103)
    plt.ylim(0.0, 1.03)

    # Add grid for readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.9, alpha=0.7)

    # Add legend
    

    # Adjust plot margins for better spacing
    plt.tight_layout()


    name = "r_analysis"
    if include_percolation_treshold:
        plt.axvline(x=10.66,label='Percolation Threshold',linestyle='--',color='red')
        name += '_with_percolation_threshold'


    plt.legend(title="Legend", fontsize=12, loc='upper left')
    plt.savefig(name,dpi=400)
    # Show the plot
    plt.show()


if __name__ == '__main__':
    plot_r_analysis_clean_version(include_percolation_treshold=True)
    plot_r_analysis_clean_version(include_percolation_treshold=False)
    plot_non_voter_influence()
    plot_r_analysis_with_standard_deviation()



