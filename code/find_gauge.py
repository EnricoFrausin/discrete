import numpy as np
from itertools import permutations, combinations
from utilities import calculate_kl_divergence_with_HFM_new
from utilities import get_empirical_latent_distribution
from utilities import calculate_Z_theoretical, get_HFM_prob
from utilities import get_m_s
from scipy.optimize import minimize_scalar, basinhopping
import os
import matplotlib.pyplot as plt
import random
import math


def flip_gauge_bits(empirical_probs): #for return_minimum_kl
    """
    Flip specific bits in all states based on the activated bits in the most frequent state.
    
    Args:
        empirical_probs (dict): Dictionary mapping state tuples to their empirical probabilities.
                                From get_empirical_latent_distribution.
    
    Returns:
        dict: A new dictionary with the same probabilities but flipped states according to the rule.
    """
    if not empirical_probs:
        return {}
    
    most_frequent_state = max(empirical_probs.items(), key=lambda x: x[1])[0]
    
    bits_to_flip = [i for i, bit in enumerate(most_frequent_state) if bit == 1]

    most_frequent_state_gauge_probs = {}
    
    for state, prob in empirical_probs.items():
        new_state = list(state)
        
        for bit_pos in bits_to_flip:
            new_state[bit_pos] = 1 - new_state[bit_pos]  # Flip 0->1, 1->0
        
        most_frequent_state_gauge_probs[tuple(new_state)] = prob
    
    return most_frequent_state_gauge_probs


#___________________________________________OPTIONAL______________________________________________


def get_feature_frequencies_sorted(state_probs): #For reorder_bit_states_frequency OPTIONAL
    """
    Given a dict {state_tuple: prob}, returns the indices of features sorted by their frequency (descending).
    """
    if not state_probs:
        return []
    num_features = len(next(iter(state_probs)))
    freq = np.zeros(num_features)
    
    for state, prob in state_probs.items():
        freq += np.array(state) * prob  # Accumulate frequencies weighted by probabilities

    sorted_indices = np.argsort(-freq)
    print(f"Feature frequencies: {freq}, Sorted indices: {sorted_indices}")
    return list(sorted_indices)




def reorder_bit_states_by_frequency(state_probs): # for return_goodgauged_empirical_probs, OPTIONAL
    """
    Reorders the bits in each state according to their frequency in the dataset.
    
    Args:
        state_probs (dict): Dictionary mapping state tuples to their empirical probabilities.
    
    Returns:
        dict: A new dictionary with reordered states according to bit frequencies.
    """
    if not state_probs:
        return {}
    
    sorted_indices = get_feature_frequencies_sorted(state_probs)
    
    reordered_probs = {}
    
    for state, prob in state_probs.items():
        reordered_state = tuple(state[idx] for idx in sorted_indices)
        reordered_probs[reordered_state] = prob
    
    return reordered_probs


def return_goodgauged_empirical_probs(empirical_probs): #OPTIONAL
    """
    Main function to find the gauge transformation for the given empirical probabilities.
    
    Args:
        empirical_probs (dict): Dictionary mapping state tuples to their empirical probabilities.
        
    Returns:
        dict: A new dictionary with the gauge-transformed states and their probabilities.
    """

    flipped_probs = flip_gauge_bits(empirical_probs)
    reordered_probs = reorder_bit_states_by_frequency(flipped_probs)
    return reordered_probs


#_____________________________________________________________________________________________________


# def find_optimal_g_parameter(reordered_probs, g_bounds=(0.1, 5.0)):
#     """
#     Finds the optimal g parameter that minimizes the expected m_s value
#     according to the theoretical HFM distribution using scipy.optimize.
    
#     Args:
#         reordered_probs (dict): Dictionary mapping state tuples to their empirical probabilities
#                               (typically after gauge transformation and reordering).
#         g_bounds (tuple): Bounds for g (min, max) for the optimization.
        
#     Returns:
#         float: The optimal g value that minimizes the expected m_s.
#         float: The minimum expected m_s value.
#     """
#     latent_dim = len(next(iter(reordered_probs)))

#     # Precompute m_s values for all states
#     m_s_values = {state: get_m_s(state, active_category_is_zero=False) for state in reordered_probs.keys()}

#   # The objective function to minimize (expected m_s)
#     # def objective_function(g):
#     #     Z = calculate_Z_theoretical(latent_dim, g)
#     #     expected_ms = 0.0
        
#     #     # Calculate the theoretical probability for each state
#     #     for state, prob in reordered_probs.items():
#     #         m_s = m_s_values[state]  # Use precomputed m_s
#     #         hfm_prob = get_HFM_prob(m_s, g, Z, logits=False)
#     #         expected_ms += m_s * hfm_prob
        
#     #     return expected_ms
#     def objective_function(g):
#         x = calculate_kl_divergence_with_HFM(reordered_probs, g)
#         return x


#     # Use scipy.optimize to minimize the objective function
#     result = minimize_scalar(objective_function, bounds=g_bounds, method='bounded')

#     return result.x, result.fun  # Optimal g and the corresponding minimum expected m_s




#To be used to find the best permutation of the states that minimizes the KL divergence with the HFM model.
#The same permutation should be valid for all g values, therefore it is not necessary to recalculate the gauge for different g values. See ```print_minimum_kl_in_g_range``` below.
def find_minimum_kl_brute_force(good_gauged_probs, g=np.log(2), return_additional_info=False, print_permutation_steps=float("inf")): # for return_minimum_kl
    """
    Brute-force search for the permutation of state columns that minimizes the KL divergence
    between the permuted empirical distribution and the HFM model with parameter g.

    Args:
        good_gauged_probs (dict): Dictionary mapping state tuples to probabilities.
        g (float, optional): HFM model parameter. Defaults to np.log(2).
        return_additional_info (bool, optional): If True, returns the best permutation and state dict.

    Returns:
        minimum_kl (float): The minimum KL divergence found.
        best_permutation (tuple or None): The permutation that yields the minimum KL (if requested).
        best_state_dict (dict or None): The permuted state dictionary (if requested).
    """
    states_matrix = np.array(list(good_gauged_probs.keys()))
    state_len = states_matrix.shape[1]
    permutations_list = list(permutations(range(state_len)))

    total_permutations = len(permutations_list)
 
    minimum_kl = float('inf')
    best_permutation = None
    best_state_dict = None
    i=0


    for perm in permutations_list:
        i += 1

        states_matrix_copy = np.empty(states_matrix.shape)
        states_matrix_copy[:, :] = states_matrix[:, list(perm)]

        permutated_state_dict = dict((tuple(row), prob) for row, prob in zip(states_matrix_copy, good_gauged_probs.values()))

        temporary_kl = calculate_kl_divergence_with_HFM_new(permutated_state_dict, g=g)
        if temporary_kl < minimum_kl:
            minimum_kl = temporary_kl
            best_permutation = perm
            best_state_dict = permutated_state_dict


        if i % print_permutation_steps == 0:
            print(f"Processed {i} permutations, current minimum KL: {minimum_kl}, best permutation: {best_permutation}")
            
    print(f"Total permutations processed: {total_permutations}, Minimum KL: {minimum_kl}, Best permutation: {best_permutation}")    

    return (minimum_kl, best_permutation, best_state_dict) if return_additional_info else minimum_kl






def find_minimum_kl_simulated_annealing(good_gauged_probs, g=np.log(2), return_additional_info=False, 
                                       initial_temp=10.0, cooling_rate=0.95, n_iterations=1000, 
                                       verbose=False):
    """
    Uses simulated annealing to find a permutation of state columns that minimizes 
    the KL divergence with the HFM model.
    """
    states_matrix = np.array(list(good_gauged_probs.keys()))
    state_len = states_matrix.shape[1]
    
    # Start with identity permutation
    current_perm = list(range(state_len))
    
    # Create initial state dictionary
    states_matrix_copy = np.empty(states_matrix.shape)
    states_matrix_copy[:, :] = states_matrix[:, current_perm]
    current_state_dict = dict((tuple(row), prob) for row, prob in zip(states_matrix_copy, good_gauged_probs.values()))
    
    # Calculate initial KL divergence
    current_kl = calculate_kl_divergence_with_HFM_new(current_state_dict, g=g)
    
    # Keep track of best solution
    best_perm = current_perm.copy()
    best_kl = current_kl
    best_state_dict = current_state_dict.copy()
    
    # Temperature schedule
    temp = initial_temp
    
    # Main simulated annealing loop
    for i in range(n_iterations):
        # Use combinations from itertools to select positions to swap
        swap_indices = random.choice(list(combinations(range(state_len), 2)))
        
        # Create a new candidate permutation by swapping positions
        candidate_perm = current_perm.copy()
        candidate_perm[swap_indices[0]], candidate_perm[swap_indices[1]] = candidate_perm[swap_indices[1]], candidate_perm[swap_indices[0]]
        
        # Calculate KL for the candidate permutation
        states_matrix_copy[:, :] = states_matrix[:, candidate_perm]
        candidate_state_dict = dict((tuple(row), prob) for row, prob in zip(states_matrix_copy, good_gauged_probs.values()))
        candidate_kl = calculate_kl_divergence_with_HFM_new(candidate_state_dict, g=g)
        
        # Metropolis acceptance criterion
        delta_kl = candidate_kl - current_kl
        if delta_kl < 0 or random.random() < math.exp(-delta_kl / temp):
            current_perm = candidate_perm
            current_kl = candidate_kl
            current_state_dict = candidate_state_dict
            
            # Update best solution if applicable
            if current_kl < best_kl:
                best_perm = current_perm.copy()
                best_kl = current_kl
                best_state_dict = current_state_dict.copy()
                
                if verbose:
                    print(f"Iteration {i+1}, New best KL: {best_kl:.6f}, Temperature: {temp:.6f}")
        
        # Reduce temperature
        temp *= cooling_rate
        
        # Periodic progress report
        if verbose and (i + 1) % 100 == 0:
            print(f"Iteration {i+1}, Current KL: {current_kl:.6f}, Best KL: {best_kl:.6f}")
    
    if verbose:
        print(f"Final KL: {best_kl:.6f}, Best permutation: {tuple(best_perm)}")
    
    return (best_kl, tuple(best_perm), best_state_dict) if return_additional_info else best_kl



#Function to find the minimum KL divergence for a range of g values. Used to determine wether
# the gauge is the same for different g values.
def print_minimum_kl_in_g_range(my_model, train_loader, device, brute_force = False):
    """
    Calculates and returns the minimum Kullback-Leibler (KL) divergence for a given model and dataset.

    This function computes the empirical latent distribution of the provided model over the training data,
    applies a gauge transformation to obtain a set of "good-gauged" empirical probabilities, and then
    finds the minimum KL divergence among these using a brute-force search.

    Args:
        my_model: The model whose latent distribution is to be analyzed.
        train_loader: DataLoader providing the training data.
        device: The device (CPU or GPU) on which computations are performed.
        g (float, optional): The logarithm of the base for the KL divergence calculation. Defaults to np.log(2).

    Returns:
        float: The minimum KL divergence found among the good-gauged empirical probabilities.

    Prints:
        The minimum KL divergence value.
    """
    empirical_probs, total_samples = get_empirical_latent_distribution(my_model, train_loader, device=device)
    good_gauged_dict = flip_gauge_bits(empirical_probs)
    for g in np.arange(0.1, 2, 0.2):
        if brute_force:
            minimum_kl = find_minimum_kl_brute_force(good_gauged_dict, g=g)
        else:
            minimum_kl = find_minimum_kl_simulated_annealing(good_gauged_dict, g=g)
        print(f"Minimum KL divergence for g={g}: {minimum_kl}")

    return





# Example usage:
# my_model = VAE_priorCategorical(input_dim=input_dim, categorical_dim=2, latent_dim=8, decrease_rate=0.5, device=device, num_hidden_layers=1, LayerNorm=True).to(device)
# my_model.load_state_dict(torch.load('/Users/enricofrausin/Programmazione/PythonProjects/Fisica/Architetture/VAE/discrete/models_parameters/priorCategorical/MNIST/ld8_glog2_ep15_lmb01_dr05_gKLlog2_LN_1hl_0.pth', map_location=device))
# print("Calculating KL divergences for model with 1 hidden layer...")
# layer_dicts.append(return_minimum_kl_in_g_range(my_model, train_loader, device, g_values))
# print("---------------\n")


def return_minimum_kl_in_g_range(my_model, train_loader, device, g_values=np.arange(0.1, 2, 0.2), brute_force=False):
    """
    Computes the KL divergence between the empirical latent distribution of a model and the HFM distribution
    for a range of gauge parameter values `g`. The function identifies the best gauge transformation that minimizes
    the KL divergence for the first value in `g_values` using a brute force approach, then evaluates the KL divergence for all specified `g` values
    using the optimal gauge found.

    Args:
        my_model: The trained model whose latent distribution is to be analyzed.
        train_loader: DataLoader providing the training data for extracting the latent distribution.
        device: The device (CPU or GPU) on which computations are performed.
        g_values (array-like, optional): Sequence of gauge parameter values to evaluate. Defaults to np.arange(0.1, 2, 0.2).

    Returns:
        dict: A dictionary mapping each value in `g_values` to the corresponding KL divergence.
    """

    print("Obtaining the internal latent distribution of the model...")
    empirical_probs, total_samples = get_empirical_latent_distribution(my_model, train_loader, device=device)
    good_gauged_dict = flip_gauge_bits(empirical_probs)

    print("Evaluating the best gauge of the latent states...")
    min_kl_values = {}
    if brute_force:
        minimum_kl, best_permutation, gauged_states = find_minimum_kl_brute_force(good_gauged_dict, g=np.log(2), return_additional_info=True)
    else:
        minimum_kl, best_permutation, gauged_states = find_minimum_kl_simulated_annealing(good_gauged_dict, g=np.log(2), return_additional_info=True)


    print("Calculating the KL divergence for different g values...")
    for g in g_values:
        temporary_kl = calculate_kl_divergence_with_HFM_new(gauged_states, g=g)
        min_kl_values[g] = temporary_kl

    return min_kl_values




 # TO CHECK, THIS METHOD TO FIND THE OPTIMAL g DOES NOT WORK PROPERLY
def find_optimal_g_parameter(gauged_states, g_bounds=(0.1, 5.0)):
    """
    Finds the optimal g parameter that minimizes the expected m_s value
    according to the theoretical HFM distribution using scipy.optimize.
    
    Args:
        reordered_probs (dict): Dictionary mapping state tuples to their empirical probabilities
                              (typically after gauge transformation and reordering).
        g_bounds (tuple): Bounds for g (min, max) for the optimization.
        
    Returns:
        float: The optimal g value that minimizes the expected m_s.
        float: The minimum expected m_s value.
    """
    latent_dim = len(next(iter(gauged_states)))

    # Precompute m_s values for all states
    m_s_values = {state: get_m_s(state, active_category_is_zero=False) for state in gauged_states.keys()}


    def objective_function(g):
        Z = calculate_Z_theoretical(latent_dim, g)
        expected_ms = 0.0
        
        # Calculate the theoretical probability for each state
        for state, empirical_prob in gauged_states.items():
            m_s = m_s_values[state]  # Use precomputed m_s
            hfm_prob = get_HFM_prob(m_s, g, Z, logits=False)
            expected_ms += m_s * hfm_prob *empirical_prob

        return expected_ms

    # Use scipy.optimize to minimize the objective function
    #result = minimize_scalar(objective_function, bounds=g_bounds, method='bounded')
    result = basinhopping(objective_function, x0=0.6)

    return result.x, result.fun  # Optimal g and the corresponding minimum expected m_s



#–––––––––––––––––––––––––––––––––––––––––PLOTS––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––



# DEPRECATED
def save_individual_plots(save_dir, g_range):           #ATTENTION: This function works properly if the 
                                                        #variables are defined in the global scope as kl_divergences_g_*.
    """
    Create and save individual plots for each g value.
    
    Args:
        save_dir (str): Directory path where plots will be saved
        g_range (np.array): Array of g values to plot
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get the default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    num_layers = [1, 2, 3, 4, 5, 6]
    
    # Create individual plots for each g value
    for i, g in enumerate(g_range):
        var_name = f"kl_divergences_{str(round(g, 2)).replace('.', '')}"
        
        if var_name in globals():
            plt.figure(figsize=(8, 6))
            values = globals()[var_name]
            # Convert tensors to floats for plotting
            values = [v.item() if hasattr(v, "item") else v for v in values]
            
            plt.plot(num_layers, values, color=colors[i], linewidth=2, marker='o')
            plt.xlabel('Number of layers for each VAE')
            plt.ylabel('KL with HFM')
            plt.title(f'g of HFM = {g:.1f}')
            plt.grid(True, alpha=0.3)
            
            # Save the figure
            filename = f"kl_divergence_g_{g:.1f}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            print(f"Saved: {filepath}")
    
    print(f"All plots saved to: {save_dir}")

# Example usage:
# save_individual_plots("/Users/enricofrausin/Programmazione/PythonProjects/Fisica/Immagini/kl_divergence_priorCategorical_MNIST/lmb01", np.arange(0.1, 2, 0.2))



# DEPRECATED
def plot_kl_multiline(g_range, var_name_prefix="kl_divergences", title="KL Divergence vs Number of Layers", figsize=(10, 6)):
    """
    Create a multi-line plot showing KL divergences for different g values.
    
    Args:
        g_range (np.array): Array of g values to plot
        var_name_prefix (str): Prefix for variable names in globals()
        title (str): Title for the plot
        figsize (tuple): Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    num_layers = [1, 2, 3, 4, 5, 6]
    
    for i, g in enumerate(g_range):
        var_name = f"{var_name_prefix}_{str(round(g, 2)).replace('.', '')}"
        
        if var_name in globals():
            values = globals()[var_name]
            # Convert tensors to floats for plotting
            values = [v.item() if hasattr(v, "item") else v for v in values]
            ax.plot(num_layers, values, label=f'g = {g:.2f}')
    
    ax.set_xlabel('Number of layers for each VAE')
    ax.set_ylabel('KL with HFM')
    ax.set_title(title)
    ax.legend(loc='upper left')
    plt.show()

# Example usage:
# plot_kl_multiline(np.arange(0.1, 2, 0.2))
# plot_kl_multiline(np.arange(0.1, 2, 0.2), var_name_prefix="kl_divergences_lmb01", title="Lambda = 0.1")



# Example usage:

# plot_kl_multiline_from_dicts(layer_dicts, g_values, title="KL Divergence Analysis - MNIST")

def plot_kl_multiline_from_dicts(layer_dicts, g_range, title="KL Divergence vs Number of Layers", figsize=(10, 6)):
    """
    Create a multi-line plot showing KL divergences for different g values using layer dictionaries.
    
    Args:
        layer_dicts (list): List of dictionaries, each containing KL divergences for different layers
                           Format: [kl_divergences_1_layers, kl_divergences_2_layers, ...]
        g_range (np.array): Array of g values to plot
        title (str): Title for the plot
        figsize (tuple): Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    num_layers = list(range(1, len(layer_dicts) + 1))
    
    # Get the default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, g in enumerate(g_range):
        values = []
        for layer_dict in layer_dicts:
            if g in layer_dict:
                val = layer_dict[g]
                # Convert tensor to float if needed
                if hasattr(val, "item"):
                    val = val.item()
                values.append(val)
            else:
                print(f"Warning: g={g} not found in one of the layer dictionaries")
                continue
        
        if len(values) == len(num_layers):
            ax.plot(num_layers, values, label=f'g = {g:.2f}', color=colors[i % len(colors)], 
                   linewidth=2, marker='o')
    
    ax.set_xlabel('Number of layers for each VAE')
    ax.set_ylabel('KL with HFM')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.show()


#Example usage:

# save_individual_plots_from_dicts(
#     layer_dicts, 
#     g_values, 
#     "/Users/enricofrausin/Programmazione/PythonProjects/Fisica/Immagini//Users/enricofrausin/Programmazione/PythonProjects/Fisica/Immagini/kl_divergence_priorCategorical_MNIST/",
#     title_prefix="MNIST Analysis"
# )

def save_individual_plots_from_dicts(layer_dicts, g_range, save_dir, title_prefix=""):
    """
    Create and save individual plots for each g value using layer dictionaries.
    
    Args:
        layer_dicts (list): List of dictionaries, each containing KL divergences for different layers
        g_range (np.array): Array of g values to plot
        save_dir (str): Directory path where plots will be saved
        title_prefix (str): Prefix for plot titles
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get the default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    num_layers = list(range(1, len(layer_dicts) + 1))
    
    # Create individual plots for each g value
    for i, g in enumerate(g_range):
        values = []
        for layer_dict in layer_dicts:
            if g in layer_dict:
                val = layer_dict[g]
                # Convert tensor to float if needed
                if hasattr(val, "item"):
                    val = val.item()
                values.append(val)
            else:
                print(f"Warning: g={g} not found in one of the layer dictionaries")
                break
        
        if len(values) == len(num_layers):
            plt.figure(figsize=(8, 6))
            plt.plot(num_layers, values, color=colors[i % len(colors)], 
                    linewidth=2, marker='o')
            plt.xlabel('Number of layers for each VAE')
            plt.ylabel('KL with HFM')
            
            title = f'g of HFM = {g:.1f}'
            if title_prefix:
                title = f'{title_prefix} - {title}'
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            # Save the figure
            filename = f"kl_divergence_g_{g:.1f}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            print(f"Saved: {filepath}")
    
    print(f"All plots saved to: {save_dir}")



# Example usage:

# create_combined_multiline_plot(
#     layer_dicts, 
#     g_values,
#     save_path="/Users/enricofrausin/Programmazione/PythonProjects/Fisica/Immagini//Users/enricofrausin/Programmazione/PythonProjects/Fisica/Immagini/kl_divergence_priorCategorical_MNIST/kl_divergences_combined_depth_analysis.png",
#     title="KL Divergence vs Number of Layers - MNIST Dataset"
# )

def create_combined_multiline_plot(layer_dicts, g_range, save_path=None, title="KL Divergence vs Number of Layers", figsize=(12, 8)):
    """
    Create a combined multi-line plot and optionally save it.
    
    Args:
        layer_dicts (list): List of dictionaries, each containing KL divergences for different layers
        g_range (np.array): Array of g values to plot
        save_path (str, optional): Path to save the combined plot
        title (str): Title for the plot
        figsize (tuple): Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    num_layers = list(range(1, len(layer_dicts) + 1))
    
    # Get the default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, g in enumerate(g_range):
        values = []
        for layer_dict in layer_dicts:
            if g in layer_dict:
                val = layer_dict[g]
                # Convert tensor to float if needed
                if hasattr(val, "item"):
                    val = val.item()
                values.append(val)
            else:
                print(f"Warning: g={g} not found in one of the layer dictionaries")
                continue
        
        if len(values) == len(num_layers):
            ax.plot(num_layers, values, label=f'g = {g:.2f}', color=colors[i % len(colors)], 
                   linewidth=2, marker='o')
    
    ax.set_xlabel('Number of layers for each VAE')
    ax.set_ylabel('KL with HFM')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {save_path}")
    
    plt.show()


#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––




def compare_datasets_by_layer(layer_dicts_list1, layer_dicts_list2, layer_dicts_list3, 
                              g_values, layer_index, 
                              dataset_names=['EMNIST', '2MNIST', 'MNIST'],
                              save_path=None, title_suffix=""):
    """
    Compare KL divergences across three datasets for a specific layer index.
    
    Parameters:
    - layer_dicts_list1, layer_dicts_list2, layer_dicts_list3: Lists of dictionaries containing KL values
    - g_values: Array of g values
    - layer_index: Index of the layer to compare (0-based)
    - dataset_names: Names of the datasets for the legend
    - save_path: Path to save the plot (optional)
    - title_suffix: Additional text for the plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Check if layer_index is valid for all lists
    max_layers = min(len(layer_dicts_list1), len(layer_dicts_list2), len(layer_dicts_list3))
    if layer_index >= max_layers:
        print(f"Layer index {layer_index} is out of range. Maximum available: {max_layers - 1}")
        return
    
    # Extract KL values for each dataset
    kl_values_1 = [layer_dicts_list1[layer_index][g] for g in g_values]
    kl_values_2 = [layer_dicts_list2[layer_index][g] for g in g_values]
    kl_values_3 = [layer_dicts_list3[layer_index][g] for g in g_values]
    
    # Plot the three datasets
    plt.plot(g_values, kl_values_1, 'o-', label=dataset_names[0], linewidth=2, markersize=6)
    plt.plot(g_values, kl_values_2, 's-', label=dataset_names[1], linewidth=2, markersize=6)
    plt.plot(g_values, kl_values_3, '^-', label=dataset_names[2], linewidth=2, markersize=6)
    
    plt.xlabel('g values', fontsize=12)
    plt.ylabel('KL Divergence', fontsize=12)
    plt.title(f'KL Divergence Comparison - Layer {layer_index + 1} {title_suffix}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def compare_all_layers(layer_dicts_list1, layer_dicts_list2, layer_dicts_list3, 
                       g_values, dataset_names=['EMNIST', '2MNIST', 'MNIST'],
                       save_dir=None):
    """
    Compare KL divergences across three datasets for all available layers.
    
    Parameters:
    - layer_dicts_list1, layer_dicts_list2, layer_dicts_list3: Lists of dictionaries containing KL values
    - g_values: Array of g values
    - dataset_names: Names of the datasets for the legend
    - save_dir: Directory to save the plots (optional)
    """
    max_layers = min(len(layer_dicts_list1), len(layer_dicts_list2), len(layer_dicts_list3))
    
    for layer_idx in range(max_layers):
        save_path = None
        if save_dir:
            save_path = f"{save_dir}/comparison_layer_{layer_idx + 1}.png"
        
        compare_datasets_by_layer(
            layer_dicts_list1, layer_dicts_list2, layer_dicts_list3,
            g_values, layer_idx, dataset_names, save_path
        )




def create_combined_comparison_plot(layer_dicts_list1, layer_dicts_list2, layer_dicts_list3,
                                   g_values, dataset_names=['EMNIST', '2MNIST', 'MNIST'],
                                   save_path=None):
    """
    Create a combined subplot showing all layer comparisons in one figure.
    
    Parameters:
    - layer_dicts_list1, layer_dicts_list2, layer_dicts_list3: Lists of dictionaries containing KL values
    - g_values: Array of g values
    - dataset_names: Names of the datasets
    - save_path: Path to save the combined plot
    """
    max_layers = min(len(layer_dicts_list1), len(layer_dicts_list2), len(layer_dicts_list3))
    
    # Calculate subplot layout
    cols = 3
    rows = (max_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for layer_idx in range(max_layers):
        row = layer_idx // cols
        col = layer_idx % cols
        ax = axes[row, col]
        
        # Extract KL values for each dataset
        kl_values_1 = [layer_dicts_list1[layer_idx][g] for g in g_values]
        kl_values_2 = [layer_dicts_list2[layer_idx][g] for g in g_values]
        kl_values_3 = [layer_dicts_list3[layer_idx][g] for g in g_values]
        
        # Plot the three datasets
        # ax.plot(g_values, kl_values_1, 'o-', label=dataset_names[0], linewidth=2, markersize=4)
        # ax.plot(g_values, kl_values_2, 's-', label=dataset_names[1], linewidth=2, markersize=4)
        # ax.plot(g_values, kl_values_3, '^-', label=dataset_names[2], linewidth=2, markersize=4)
        ax.plot(g_values, kl_values_1, label=dataset_names[0], linewidth=2)
        ax.plot(g_values, kl_values_2, label=dataset_names[1], linewidth=2)
        ax.plot(g_values, kl_values_3, label=dataset_names[2], linewidth=2)
        
        ax.set_xlabel('g values')
        ax.set_ylabel('KL Divergence')
        ax.set_title(f'Layer {layer_idx + 1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for layer_idx in range(max_layers, rows * cols):
        row = layer_idx // cols
        col = layer_idx % cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('KL Divergence Comparison Across Datasets', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {save_path}")
    
    plt.show()


# ––––––––––––––––––––––––––––––––––––––OPTIMAL G PARAMETER PLOTS–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––



def plot_expected_ms_vs_g(gauged_states, g_range=np.linspace(0.1, 5.0, 50)):
    """
    Plots the expected m_s value for a range of g values.

    Args:
        gauged_states (dict): Dictionary mapping state tuples to their empirical probabilities.
        g_range (array-like): Sequence of g values to evaluate.
    """
    latent_dim = len(next(iter(gauged_states)))
    m_s_values = {state: get_m_s(state, active_category_is_zero=False) for state in gauged_states.keys()}
    expected_ms_list = []

    for g in g_range:
        Z = calculate_Z_theoretical(latent_dim, g)
        expected_ms = 0.0
        for state, empirical_prob in gauged_states.items():
            m_s = m_s_values[state]
            hfm_prob = get_HFM_prob(m_s, g, Z, logits=False)
            expected_ms += m_s * hfm_prob * empirical_prob
            #expected_ms += m_s * hfm_prob 
           # expected_ms += m_s * empirical_prob 
        expected_ms_list.append(expected_ms)

    plt.figure(figsize=(8, 6))
    plt.plot(g_range, expected_ms_list, marker='o')
    plt.xlabel('g')
    plt.ylabel('Expected m_s')
    plt.title('Expected m_s vs g')
    plt.grid(True, alpha=0.3)
    plt.show()



def plot_expected_kl_vs_g(gauged_states, g_range=np.linspace(0.1, 5.0, 50)):


    expected_kl_list = []

    for g in g_range:
        expected_kl_list.append(calculate_kl_divergence_with_HFM_new(gauged_states, g))


    plt.figure(figsize=(8, 6))
    plt.plot(g_range, expected_kl_list, marker='o')
    plt.xlabel('g')
    plt.ylabel('Expected KL Divergence')
    plt.title('Expected KL Divergence vs g')
    plt.grid(True, alpha=0.3)
    plt.show()
