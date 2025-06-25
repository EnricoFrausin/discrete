import numpy as np
from itertools import permutations
from utilities import calculate_kl_divergence_with_HFM_new


def flip_gauge_bits(empirical_probs):
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




def get_feature_frequencies_sorted(state_probs):
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




def reorder_bit_states_by_frequency(state_probs):
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


def return_goodgauged_empirical_probs(empirical_probs):
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


# At this point we have the functions to start the gauge tranformation procedure.



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






def find_minimum_kl_brute_force(good_guaged_probs, g=np.log(2), return_additional_info=False):
    """
    Brute-force search for the permutation of state columns that minimizes the KL divergence
    between the permuted empirical distribution and the HFM model with parameter g.

    Args:
        good_guaged_probs (dict): Dictionary mapping state tuples to probabilities.
        g (float, optional): HFM model parameter. Defaults to np.log(2).
        return_additional_info (bool, optional): If True, returns the best permutation and state dict.

    Returns:
        minimum_kl (float): The minimum KL divergence found.
        best_permutation (tuple or None): The permutation that yields the minimum KL (if requested).
        best_state_dict (dict or None): The permuted state dictionary (if requested).
    """
    states_matrix = np.array(list(good_guaged_probs.keys()))
    state_len = states_matrix.shape[1]
    permutations_list = list(permutations(range(state_len)))

 
    minimum_kl = float('inf')
    best_permutation = None
    best_state_dict = None
    i=0


    for perm in permutations_list:
        i += 1

        states_matrix_copy = np.empty(states_matrix.shape)
        states_matrix_copy[:, :] = states_matrix[:, list(perm)]

        permutated_state_dict = dict((tuple(row), prob) for row, prob in zip(states_matrix_copy, good_guaged_probs.values()))

        temporary_kl = calculate_kl_divergence_with_HFM_new(permutated_state_dict, g=g)
        if temporary_kl < minimum_kl:
            minimum_kl = temporary_kl
            best_permutation = perm
            best_state_dict = permutated_state_dict


        if i % 10000 == 0:
            print(f"Processed {i} permutations, current minimum KL: {minimum_kl}, best permutation: {best_permutation}")

    
    if not return_additional_info:
        best_permutation = None
        best_state_dict = None

    return (minimum_kl, best_permutation, best_state_dict) if return_additional_info else minimum_kl



