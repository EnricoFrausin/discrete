import torch
import numpy as np
from collections import Counter

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
        freq += np.array(state)

    sorted_indices = np.argsort(-freq)
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


# At this point we have the functions to start the gauge tranformation procedure.


from scipy.optimize import minimize_scalar
from utilities import get_m_s, calculate_Z_theoretical, get_HFM_prob

def find_optimal_g_parameter(reordered_probs, g_bounds=(0.1, 5.0)):
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
    latent_dim = len(next(iter(reordered_probs)))

    # Precompute m_s values for all states
    m_s_values = {state: get_m_s(state, active_category_is_zero=False) for state in reordered_probs.keys()}

    def objective_function(g):
        Z = calculate_Z_theoretical(latent_dim, g)
        expected_ms = 0.0
        
        # Calculate the theoretical probability for each state
        for state, prob in reordered_probs.items():
            m_s = m_s_values[state]  # Use precomputed m_s
            hfm_prob = get_HFM_prob(m_s, g, Z, logits=False)
            expected_ms += m_s * hfm_prob
        
        return expected_ms

    # Use scipy.optimize to minimize the objective function
    result = minimize_scalar(objective_function, bounds=g_bounds, method='bounded')

    return result.x, result.fun  # Optimal g and the corresponding minimum expected m_s


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


from utilities import get_empirical_latent_distribution, calculate_kl_divergence_with_HFM

def calculate_HFM_over_disordered_states(model, dataloader, device, return_goodgauged_probs=False, return_empirical_entropy=False):

    empirical_probs, total_samples = get_empirical_latent_distribution(model, dataloader, device)
    gauge_probs = return_goodgauged_empirical_probs(empirical_probs)
    optimal_g, min_expected_ms = find_optimal_g_parameter(gauge_probs)
    
    kl_divergence, empirical_entropy = calculate_kl_divergence_with_HFM(gauge_probs, optimal_g, normalize_theoricalHFM=False, return_emp_distr_entropy=return_empirical_entropy)
    

    gauge_probs = None if not return_goodgauged_probs else gauge_probs
    empirical_entropy = None if not return_empirical_entropy else empirical_entropy

    return kl_divergence, optimal_g, gauge_probs, empirical_entropy
