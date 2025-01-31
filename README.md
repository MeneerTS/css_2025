# The Bison Moo-cracy, CSS 2025
By Balthazar Dupuy d'Angeac, Ivana Malčić, Luke Kraakman and Thijs Stessen 

A codebase for modeling the voting dynamics of bizons, either sequentially or synchronously. Some results can be found in the pretty plots folder and the file 'CSS final presentation.pptx' contains our slides on the topic.

We have roughly 5-10% ‘assert’ statements inline. Just thought you should know. (wink wink)

## Usage
### Sequential model
Functions from main.py can be called to produce different results, this is how we used the code. For convenience an argument parser is included in the main file.
The main file can be run using python main.py followed by the following arguments:

    Run various analysis and visualization functions for the voting simulation.

    options:
    -h, --help            show this help message and exit
    --r_analysis num_r_values num_voters num_non_voters num_iterations_per_r
                            Analyze the effect of radius r on voting outcome.
    --revotes [revotes ...]
                            Optional: List of revotes to analyze. Default is [1, 2, 4]. Only used in r_analysis.
    --moving              Enable movement simulation for r_analysis.
    --histogram_analysis r
                            Show a histogram of voting results after 1000 iterations for a given radius r.
    --influence_analysis steps
                            Analyze the effect of bias in voting blue on the outcome for different values of r.
    --uninformed_analysis
                            Analyze the effect of the number of non-voters on the voting outcome.
    --iteration_analysis num_voters num_non_voters
                            Analyze how the number of iterations affects the voting outcome.
    --calculate_baseline num_r_values num_iterations_per_r
                            Compute how often the local majority aligns with the global majority.
    --randomize_positions_test num_r_values num_voters num_non_voters num_iterations_per_r
                            Analyze the effect of randomizing positions on voting outcomes.
    --run_animation r     Run an animation of the voting process for a given radius r.

  main.py uses voting.py, herd.py and sim.py for handeling the simulations. Data from all simulations is written to the 'results' directory.
### Synchronous model
  Code for the synchronous model can be found in ResearchNotebook.ipynb
### Plotting
  Plotting can be done by running loading_and_plotting.py. However this does require the correct experiment files to be present in the /results folder.
### Other
Older experiments like the boids model can be found in the unused folder.
  ## Dependencies
The following libraries are used in this project:
- `numpy`
- `matplotlib`
- `scipy`
- `networkx`

All dependencies can be installed via `pip`.

```
pip install requirements.txt
```
## Compatibility
This code has been tested with **Python 3.12.0** on **Windows 10/11**.

