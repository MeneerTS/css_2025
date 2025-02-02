import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import sem, t

def bootstrap_ci(data, num_bootstrap_samples=10000, ci=95):
    """
    Perform bootstrapping on multiple experiments and compute confidence intervals.
    
    Parameters:
    - data: 2D numpy array or list of lists, where each row is an experiment.
    - num_bootstrap_samples: Number of bootstrap resamples.
    - ci: Confidence interval percentage (default 95%).
    
    Returns:
    - means: Array of bootstrap means for each experiment.
    - lower_bounds: Lower bound of the confidence interval for each experiment.
    - upper_bounds: Upper bound of the confidence interval for each experiment.
    """
    num_experiments = len(data)
    means = np.zeros(num_experiments)
    lower_bounds = np.zeros(num_experiments)
    upper_bounds = np.zeros(num_experiments)
    
    for i in range(num_experiments):
        bootstrap_means = np.zeros(num_bootstrap_samples)
        for j in range(num_bootstrap_samples):
            sample = np.random.choice(data[i], size=len(data[i]), replace=True)
            bootstrap_means[j] = np.mean(sample)
        
        means[i] = np.mean(bootstrap_means)
        lower_bounds[i] = np.percentile(bootstrap_means, (100 - ci) / 2)
        upper_bounds[i] = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    
    return means, lower_bounds, upper_bounds



def plot_non_voter_influence():
    folder_name = "results\\"
    
    non_voters_0 = np.load(folder_name + "voters_100_non_voters_0_number_of_r_                20_iterations_per_r_100_r_analysis_all_data.npy")
    non_voters_100 = np.load(folder_name + "voters_100_non_voters_100_number_of_r_                20_iterations_per_r_100_r_analysis_all_data.npy")
    non_voters_1000 = np.load(folder_name + "voters_100_non_voters_1000_number_of_r_                20_iterations_per_r_100_r_analysis_all_data.npy")
    x = np.arange(0, 100, 5)
    # 4 iterations per r
    # 3 different r values
    #[[1,2,3,4],
    #[3,2,1,4],
    #[1,2,3,4]]]
    mean_0, lower_0, upper_0 = bootstrap_ci(non_voters_0,1000)
    mean_100, lower_100, upper_100 = bootstrap_ci(non_voters_100,1000)
    mean_1000, lower_1000, upper_1000 = bootstrap_ci(non_voters_1000,1000)

    plt.figure(figsize=(10, 6))  # Set figure size for better visibility

    # Plot each dataset with clear labels and distinguishable line styles
    plt.plot(x, mean_0, label="Abstention rate: 0%", linestyle='-', color='#1716FA', marker='o')
    plt.plot(x, mean_100, label="Abstention rate: 50%", linestyle='--', color='#F71917', marker='s')
    plt.plot(x, mean_1000, label="Abstention rate: 90%", linestyle='-.', color='#A324F0', marker='d')

    plt.fill_between(x, lower_0, upper_0, color='#1716FA', alpha=0.3)
    plt.fill_between(x, lower_100, upper_100, color='#F71917', alpha=0.3)
    plt.fill_between(x, lower_1000, upper_1000, color='#A324F0', alpha=0.3)

    # Add title and labels
    plt.title("Effect of Non-Voters on Agreement", fontsize=16)
    plt.xlabel("Vision range", fontsize=14)
    plt.ylabel("Voter agreement", fontsize=14)

    # Set x and y axis limits
    plt.xlim(-3, 103)
    plt.ylim(0.0, 1.03)

    # Add grid for readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add legend
    plt.legend(fontsize=12)

    # Adjust plot margins for better spacing
    plt.tight_layout()
    plt.savefig("non_voter_influence",dpi=400)
    # Show the plot
    plt.show()


def plot_different_number_of_bisons():
    folder_name = "results\\"
    voters_100 = np.load(folder_name + 'voters_100_non_voters_0_number_of_r_            10_iterations_per_r_50_r_analysis_plot.npy')
    voters_200 = np.load(folder_name + "voters_200_non_voters_0_number_of_r_            10_iterations_per_r_100_r_analysis_plot.npy")
    voters_300 = np.load(folder_name + "voters_300_non_voters_0_number_of_r_            10_iterations_per_r_50_r_analysis_plot.npy")
    voters_400 = np.load(folder_name + "voters_400_non_voters_0_number_of_r_            10_iterations_per_r_100_r_analysis_plot.npy")

    voters_100_all_data = np.load(folder_name + 'voters_100_non_voters_0_number_of_r_            10_iterations_per_r_50_r_analysis_all_data.npy')
    voters_200_all_data = np.load(folder_name + "voters_200_non_voters_0_number_of_r_            10_iterations_per_r_100_r_analysis_all_data.npy")
    voters_300_all_data = np.load(folder_name + "voters_300_non_voters_0_number_of_r_            10_iterations_per_r_50_r_analysis_all_data.npy")
    voters_400_all_data = np.load(folder_name + "voters_400_non_voters_0_number_of_r_            10_iterations_per_r_100_r_analysis_all_data.npy")
    x = np.arange(0, 100, 10)

    plt.figure(figsize=(10, 6))  # Set figure size for better visibility

    _, lower_voters_100, upper_voters_100 = bootstrap_ci(voters_100_all_data, num_bootstrap_samples=100, ci=95)
    _, lower_voters_200, upper_voters_200 = bootstrap_ci(voters_200_all_data, num_bootstrap_samples=100, ci=95)
    _, lower_voters_300, upper_voters_300 = bootstrap_ci(voters_300_all_data, num_bootstrap_samples=100, ci=95)
    _, lower_voters_400, upper_voters_400 = bootstrap_ci(voters_400_all_data, num_bootstrap_samples=100, ci=95)

    # Plot each dataset with clear labels and distinguishable line styles

    color100 ='#17AA16'
    color200='#1716FA'
    color300='#F71917'
    color400='#A324F0'
    plt.plot(x, voters_100, label="Bisons: 100", linestyle='-', color=color100, marker='o')
    plt.plot(x, voters_200, label="Bisons: 200", linestyle='-', color=color200, marker='o')
    plt.plot(x, voters_300, label="Bisons: 300", linestyle='-', color=color300, marker='o')
    plt.plot(x, voters_400, label="Bisons: 400", linestyle='-', color=color400, marker='o')

    plt.fill_between(x, lower_voters_100, upper_voters_100, color=color100, alpha=0.3)
    plt.fill_between(x, lower_voters_200, upper_voters_200, color=color200, alpha=0.3)
    plt.fill_between(x, lower_voters_300, upper_voters_300, color=color300, alpha=0.3)
    plt.fill_between(x, lower_voters_400, upper_voters_400, color=color400, alpha=0.3)

    # Add title and labels
    plt.title("Effect of Voter Count on Agreement with Fixed Density", fontsize=16)
    plt.xlabel("Vision range", fontsize=14)
    plt.ylabel("Voter agreement", fontsize=14)

    # Set x and y axis limits
    plt.xlim(-3, 103)
    plt.ylim(0.0, 1.03)

    # Add grid for readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add legend
    plt.legend(fontsize=12)

    # Adjust plot margins for better spacing
    plt.tight_layout()
    plt.savefig("bizon_number",dpi=400)
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
    plt.title("Effect of Vision Range on Voter Agreement", fontsize=16)
    plt.xlabel("Vision Range", fontsize=14)
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


    plt.legend(fontsize=12, loc='upper left')
    plt.savefig(name,dpi=400)
    # Show the plot
    plt.show()




def plot_r_analysis_with_standard_deviation(include_percolation_treshold=False):
    folder_name = "results\\"
    values = np.load(folder_name + "voters_100_non_voters_0_number_of_r_            100_iterations_per_r_500_r_analysis_all_data.npy")
        
        # Calculate the mean and IQR for each experiment (row)
    means, lower_bound, upper_bound = bootstrap_ci(values,10000,95)

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot the mean values
    plt.plot(range(0,100,100//values.shape[0]), means, label="Mean", color='blue')

    # Plot the IQR-based confidence intervals
    plt.fill_between(range(0,100,100//values.shape[0]), lower_bound, upper_bound, color='blue', alpha=0.3, label="95% Bootstrapped Confidence Interval")

    print(means)
    print(lower_bound)
    print(upper_bound)
    # Adding labels and title
    plt.xlim(-3, 103)
    plt.ylim(0.0, 1.03)

    # Add grid for readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.9, alpha=0.7)

    # Add legend
    
    name = "r_analysis_with_confidence"
    if include_percolation_treshold:
        plt.axvline(x=10.66,label='Percolation Threshold',linestyle='--',color='red')
        name += '_with_percolation_threshold'
    # Adjust plot margins for better spacing
    
    plt.xlabel("Vision Range", fontsize=14)
    plt.ylabel("Voter Agreement", fontsize=14)
    plt.title("Effect of Vision Range on Voter Agreement", fontsize=16)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig(name,dpi=400)
    # Show the plot
    plt.show()

if __name__ == '__main__':
    
    plot_non_voter_influence()
    #plot_r_analysis_with_standard_deviation(True)
    #plot_r_analysis_with_standard_deviation(False)
    #plot_different_number_of_bisons()



