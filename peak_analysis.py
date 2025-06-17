import numpy as np
from utilities import get_empirical_latent_distribution, calculate_kl_divergence_with_HFM
import matplotlib.pyplot as plt




def _hamming_distance(s1: tuple[int, ...], s2: tuple[int, ...]) -> int:
    """
    Computes Hamming distance between two tuples of integers (latent states).
    Assumes s1 and s2 are of the same length.
    """
    distance = 0
    # The states are tuples of 0s and 1s, representing categorical choices.
    # len(s1) would be model.latent_dim.
    for el1, el2 in zip(s1, s2):
        if el1 != el2:
            distance += 1
    return distance



def _euclidean_distance_norm(s1: tuple[int, ...], s2: tuple[int, ...]) -> float:
    """
    Computes Euclidean distance between two tuples of integers using numpy.linalg.norm.
    Assumes s1 and s2 are of the same length and their elements are numerical.
    """
    if len(s1) != len(s2):
        raise ValueError("Input tuples must have the same length for Euclidean distance.")

    np_s1 = np.array(s1)
    np_s2 = np.array(s2)

    # Calcola la norma L2 della differenza tra i due vettori.
    # axis=None (default) calcola la norma dell'intero array appiattito.
    distance = np.linalg.norm(np_s1 - np_s2)

    return float(distance)


def find_peaks_from_empirical_distribution(
    empirical_probs: dict[tuple[int, ...], float],
    latent_dim: int | None = None,
    threshold_n_factor: float = 0.55,
    distance = _hamming_distance
) -> list[dict[str, any]]:
    """
    Implements the algorithm to find dominant local maxima (peaks) in an empirical distribution
    of latent states, by recursively splitting sets of states.

    The algorithm is based on the description:
    1. Sort configurations by frequency. The top one is s^(0).
    2. Find the first configuration s^(1) (in sorted order) whose distance to s^(0) 
       exceeds/meets a threshold (n/3, where n is latent_dim).
    3. If s^(1) is found, assign all configurations to either s^(0) or s^(1) based on minimal distance.
    4. This procedure is iteratively repeated for the resulting two sets of configurations.
       If s^(1) is not found, the current set of configurations forms a single peak.

    Args:
        empirical_probs (dict): A dictionary mapping latent state tuples 
                                (e.g., (0,1,0)) to their empirical probabilities.
                                This is the output of `get_empirical_latent_distribution`.
        latent_dim (int, optional): The dimensionality of the latent space (length of state tuples).
                                    If None, it's inferred from the first state in `empirical_probs`.
                                    Must be > 0 for a meaningful threshold calculation.
        threshold_n_factor (float): Factor to multiply by `latent_dim` to get the distance threshold.
                                    Default is 1/3 as per the problem description. A state `s_candidate`
                                    is considered for `s1_center` if its distance from `s0_center`
                                    is >= (latent_dim * threshold_n_factor).

    Returns:
        list[dict[str, any]]: A list of identified peaks. Each peak is a dictionary with:
            - 'states_and_probs': A dict mapping state_tuples in this peak to their 
                                  original empirical probabilities.
            - 'weight': The total probability mass of this peak (sum of probabilities of its states).
            - 'center_state': The state within this peak that has the highest empirical probability.
                              This state typically acted as an `s0_center` when this peak was finalized.
    """
    if not empirical_probs:
        return []

    if latent_dim is None:
        # Infer latent_dim from the data if not provided
        try:
            first_state = next(iter(empirical_probs.keys()))
            latent_dim = len(first_state)
        except StopIteration: # Should be caught by "if not empirical_probs"
            return [] 
    
    if latent_dim == 0:
        # Handle cases with 0-dimensional latent space. Only one possible state: ().
        if len(empirical_probs) == 1 and () in empirical_probs:
             state_tuple, prob = next(iter(empirical_probs.items()))
             return [{
                 'states_and_probs': {state_tuple: prob}, 
                 'weight': prob, 
                 'center_state': state_tuple
             }]
        else:
            # This case implies an inconsistency if latent_dim is truly 0.
            raise ValueError(
                "latent_dim is 0 but empirical_probs contains non-empty tuples or multiple distinct empty tuples."
            )

    # Calculate the actual distance threshold value for comparison with Hamming distance.
    # Hamming distance is an integer.
    # condition: distance(s_candidate, s0_center) >= threshold_val
    threshold_val = float(latent_dim * threshold_n_factor)
    
    # Memoization cache for the recursive function to store results for (frozenset of states)
    memo: dict[frozenset[tuple[int, ...]], list[list[tuple[int, ...]]]] = {}

    def _split_peak_recursive(current_states_fset: frozenset[tuple[int, ...]]) \
            -> list[list[tuple[int, ...]]]:
        """
        Recursively splits a set of states into peaks.

        Args:
            current_states_fset: A frozenset of state tuples to be processed.

        Returns:
            A list of peaks. Each peak is a list of state tuples.
            The states within each returned peak list are sorted by probability (descending)
            as this list itself represents a finalized, unsplittable peak.
        """
        # --- Base Cases for Recursion ---
        if not current_states_fset: # No states means no peaks
            return []
        
        if current_states_fset in memo: # Result already computed
            return memo[current_states_fset]

        list_current_states = list(current_states_fset)

        if len(list_current_states) == 1:
            # A single state forms a peak by itself; it cannot be split further.
            memo[current_states_fset] = [list_current_states] # Store as [[state_tuple]]
            return [list_current_states]

        # --- Recursive Step ---
        # Sort states in the current set by their global empirical probabilities (descending).
        # This S_sorted_current is used to identify s0_center and potential s1_center.
        S_sorted_current = sorted(list_current_states, key=lambda s: empirical_probs[s], reverse=True)
        # sorted ordina gli elementi passati al primo argomento secondo la funzione key passata al secondo argomento.

        s0_center = S_sorted_current[0] # Topmost configuration in this subset
        s1_center = None

        # Try to find a second center (s1_center) among the remaining sorted states.
        # s1_center is the first state (i.e., highest probability among remaining candidates)
        # whose distance from s0_center meets or exceeds the threshold.
        if len(S_sorted_current) > 1: # Only try if there are other states
            for s_candidate in S_sorted_current[1:]:
                if distance(s_candidate, s0_center) >= threshold_val:
                    s1_center = s_candidate
                    break # Found the s1_center for this split
        
        if s1_center is None:
            # No s1_center found (e.g., all other states are too close to s0_center,
            # or no other states left after s0_center).
            # This current set of states (S_sorted_current) forms a single, unsplittable peak.
            # S_sorted_current is already sorted by probability.
            memo[current_states_fset] = [S_sorted_current]
            return [S_sorted_current]
        else:
            # s1_center was found. Now, partition all states in the current_states_fset
            # (which are in list_current_states) based on their minimal distance to
            # either s0_center or s1_center.
            sub_peak_0_states: list[tuple[int, ...]] = []
            sub_peak_1_states: list[tuple[int, ...]] = []
            
            # Iterate over all states in this sub-problem (list_current_states)
            for s_in_current_set in list_current_states: 
                dist_to_s0 = distance(s_in_current_set, s0_center)
                dist_to_s1 = distance(s_in_current_set, s1_center)
                
                if dist_to_s0 <= dist_to_s1: # Tie-breaking: assign to s0_center's peak
                    sub_peak_0_states.append(s_in_current_set)
                else:
                    sub_peak_1_states.append(s_in_current_set)
            
            # Recursively call _split_peak_recursive for the two newly formed sub-peaks.
            # The results will be lists of peaks (lists of state_tuples).
            final_child_peaks: list[list[tuple[int, ...]]] = []
            
            # It's important that sub_peak_0_states and sub_peak_1_states are converted
            # to frozensets for the recursive call (for memoization keying).
            # An empty list will result in frozenset(), handled by base case.
            final_child_peaks.extend(_split_peak_recursive(frozenset(sub_peak_0_states)))
            final_child_peaks.extend(_split_peak_recursive(frozenset(sub_peak_1_states)))
            
            memo[current_states_fset] = final_child_peaks
            return final_child_peaks

    # Initial call to the recursive function with all unique states.
    all_states_fset = frozenset(empirical_probs.keys())
    
    # final_peaks_as_list_of_state_lists will be a list, where each element (peak_states_list)
    # is a list of state_tuples representing an unsplittable peak.
    # These inner lists (peak_states_list) are already sorted by probability because they
    # originate from S_sorted_current when a peak is finalized (s1_center is None).
    final_peaks_as_list_of_state_lists = _split_peak_recursive(all_states_fset)

    # --- Format Output ---
    output_formatted_peaks = []
    for peak_states_list in final_peaks_as_list_of_state_lists:
        # Each peak_states_list corresponds to one identified peak.
        if not peak_states_list:
            # This should ideally not happen if base cases and list handling are correct.
            # _split_peak_recursive(frozenset()) returns [], so extend() adds nothing from it.
            continue 
        
        current_peak_states_and_probs = {s: empirical_probs[s] for s in peak_states_list}
        current_peak_weight = sum(current_peak_states_and_probs.values())
        current_peak_number_of_states = len(current_peak_states_and_probs)
        
        # The 'center_state' of this peak is its most probable state.
        # Since peak_states_list is guaranteed to be sorted by probability (descending)
        # from the recursive function's base case return, the first element is the center.
        center_of_this_peak = peak_states_list[0] 
        
        output_formatted_peaks.append({
            'states_and_probs': current_peak_states_and_probs,
            'weight': current_peak_weight,
            'center_state': center_of_this_peak,
            'number_of_states': current_peak_number_of_states
            })
    
    return output_formatted_peaks



def get_peaks_data(model, data_loader, device, weights_type='probabilities', return_peaks=False, **kwargs):
    empirical_probs, total_samples = get_empirical_latent_distribution(model, data_loader, device)
    peaks = find_peaks_from_empirical_distribution(empirical_probs, **kwargs)
    peaks_number = [len(peaks)]
    peaks_KL = []
    peaks_entropy = []
    weights = []
    
    for peak in range(len(peaks)):
        peak_KL, peak_entropy = calculate_kl_divergence_with_HFM(peaks[peak]['states_and_probs'], normalize_theoricalHFM=True, return_emp_distr_entropy=True)
        peaks_KL.append(peak_KL.tolist())
        peaks_entropy.append(peak_entropy.tolist())
        #weights
        if weights_type == 'peak_length':
            weights.append(peaks[peak]['number_of_states'])
        elif weights_type == 'probabilities':
            weights.append(peaks[peak]['weight'])
        else:
            raise Exception("'weights' should be 'peaks_length' or 'probabilities'")

    layer_number = [model.num_hidden_layers]
    layer_values = [layer_number[0]] * len(peaks_KL)
    average_KL = [np.average(peaks_KL, weights=weights)]
    total_KL_and_entropy  = calculate_kl_divergence_with_HFM(empirical_probs, normalize_theoricalHFM=True, return_emp_distr_entropy=True)
    total_KL, total_entropy = [total_KL_and_entropy[0]], [total_KL_and_entropy[1]]

    
    data_dict = {'num_layers':layer_number, 'layer_values':layer_values, 'peaks_KL':peaks_KL, 'peaks_entropy':peaks_entropy, 'average_KL':average_KL,
                 'total_KL':total_KL, 'total_entropy':total_entropy, 'peaks_number':peaks_number}
    
    if return_peaks:
        data_dict['peaks'] = peaks

    return data_dict




def accumulate_peaks_data(data_dict, current_data_dict):
    for key in current_data_dict:
        data_dict[key] += current_data_dict[key]
    return


def plot_peaks_data(data_dict):

    fig, ax1 = plt.subplots(figsize=(12,5))

    scatter = ax1.scatter(data_dict['layer_values'], data_dict['peaks_KL'], c=data_dict['peaks_entropy'], cmap='viridis', s=100, alpha=0.8, edgecolors='w', linewidth=0.5, label='peaks')
    ax1.plot(data_dict['num_layers'], data_dict['average_KL'], color='red', label='KL averaged over peaks')
    ax1.plot(data_dict['num_layers'], data_dict['total_KL'], c='blue', label='total_KL')
    ax1.scatter(data_dict['num_layers'], data_dict['total_KL'], c=data_dict['total_entropy'], s=200)
   
    cbar = fig.colorbar(scatter)
    cbar.set_label('Entropy H(s)', rotation=270, labelpad=15)

    # Impostiamo i limiti dell'asse X in modo che il punto non sia proprio sul bordo
    ax1.set_xlim(data_dict['num_layers'][0]-1, data_dict['num_layers'][-1]+1)

    ax1.set_ylabel('KLdiv with HFM')
    ax1.set_title('KLdiv per numero di peaks')
    #ax1.grid(True, linestyle='--', alpha=0.6)
    fig.legend()
   # plt.tight_layout()
    plt.show()
