import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

def sample_gumbel(shape, device, eps=1e-20): # for gumle_softmax_sample
    U = torch.rand(shape)
    return -torch.log(-torch.log(U.to(device) + eps) + eps)

def gumbel_softmax_sample(logits, temperature, device):
    y = logits + sample_gumbel(logits.size(), device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, latent_dim, categorical_dim, temperature, device, hard=False):
    y = gumbel_softmax_sample(logits, temperature, device)
    
    if not hard:
        return y.view(-1, latent_dim * categorical_dim)
    
    _, ind = y.max(dim=-1, keepdim=True)  # [B, latent_dim, 1]
    y_hard = torch.zeros_like(y).scatter_(-1, ind, 1.0)  # [B, latent_dim, categorical_dim]

    # Straight-through estimator
    y_hard = (y_hard - y).detach() + y

    return y_hard.view(-1, latent_dim * categorical_dim)


def mean_s_k(n, k, g): #0-indexed k
    '''
    n = total number of features
    k = k_th - 1 feature, 0-idexed
    g = constant in HFM distribution
    '''
    xi = 2 * np.exp(-g)
    if abs(xi - 1) < 1e-6: #handles the case xi =1
        E = (n - (k+1) + 2) / (2 * (n+1))
    else:
        E = 0.5 * (1 + (xi**(k) - 1) * (xi - 2) / (xi**n + xi -2))
    return E


#_____________________________________________________________________________________________________________

def generate_images_from_internal_layer(model, num_images, image_shape, sample_bernoulli=True):
    """
    Genera immagini usando il decoder da vettori campionati con distribuzione uniforme nel layer interno.

    Args:
        model: L'istanza del VAE addestrato.
        num_images (int): Numero di immagini da generare.
        image_shape (tuple): La forma dell'immagine originale (es. (28, 28) per MNIST grayscale).
        sample_bernoulli (bool): Se True, campiona da Bernoulli, altrimenti usa le probabilità.

    Returns:
        torch.Tensor: Un tensore contenente le immagini generate.
    """
    model.eval() # Modalità valutazione
    generated_images = []

    with torch.no_grad(): # Non serve calcolare gradienti
        for _ in range(num_images):
            # 1. Campiona z dal prior categorico
            # z sarà una concatenazione di latent_dim vettori one-hot
            z_parts = []
            for _ in range(model.latent_dim):
                # Scegli una categoria casuale (da 0 a categorical_dim-1)
                cat_index = torch.randint(0, model.categorical_dim, (1,)).item()
                # Crea il vettore one-hot
                one_hot_vec = torch.zeros(model.categorical_dim, device=model.device)
                one_hot_vec[cat_index] = 1.0
                z_parts.append(one_hot_vec)

            # Concatena le parti per formare il vettore z completo
            # Shape: (latent_dim * categorical_dim)
            z_sample_flat = torch.cat(z_parts)

            # Aggiungi la dimensione del batch (1 immagine alla volta)
            # Shape: (1, latent_dim * categorical_dim)
            z_sample_batch = z_sample_flat.unsqueeze(0).to(model.device)

            # 2. Passa z attraverso il decoder
            # pixel_probs avrà shape (1, input_dim)
            pixel_probs = model.decode(z_sample_batch)

            # 3. Ottieni l'immagine
            if sample_bernoulli:
                # Campiona da una distribuzione di Bernoulli per ottenere un'immagine binaria
                img_flat = torch.bernoulli(pixel_probs)
            else:
                # Usa direttamente le probabilità (immagine in scala di grigi)
                img_flat = pixel_probs

            # 4. Rimodella l'immagine
            # Assumiamo che image_shape sia (H, W) per grayscale
            # o (C, H, W) per immagini a colori (anche se VAE semplici sono spesso per grayscale)
            if len(image_shape) == 2: # Grayscale (H, W)
                H, W = image_shape
                C = 1
            elif len(image_shape) == 3: # Color (C, H, W)
                C, H, W = image_shape
            else:
                raise ValueError("image_shape deve essere (H,W) o (C,H,W)")

            img_reshaped = img_flat.view(C, H, W)
            if C == 1:
                img_reshaped = img_reshaped.squeeze(0) # Rimuovi la dimensione del canale se è 1

            generated_images.append(img_reshaped.cpu()) # Sposta su CPU per matplotlib/numpy

    return torch.stack(generated_images)



def show_images(images, title, n_row=2, n_col=5):
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 2, n_row * 2))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img.numpy(), cmap='gray') # .numpy() perché sono tensori PyTorch
        ax.axis('off')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def sample_images(model: torch.nn.Module, original_image_shape= (28,28), num_generated_images=10, n_row=2, n_col=5):
    
    '''
    Returns num_generated_images images sampled casually from the internal layer of the model.
    '''

    generated_samples_bernoulli = generate_images_from_internal_layer(
        model,
        num_images=num_generated_images,
        image_shape=original_image_shape,
        sample_bernoulli=True
    )
    show_images(generated_samples_bernoulli, 'Black and white samples', n_row, n_col)

    generated_samples_probs = generate_images_from_internal_layer(
        model,
        num_images=num_generated_images,
        image_shape=original_image_shape,
        sample_bernoulli=False
    )
    show_images(generated_samples_probs, 'Greyscale samples', n_row, n_col)



#_____________________________________________________________________________________________________________



def get_m_s(state_tuple, active_category_is_zero=True):
    """
    Calculates m_s for a given state tuple, 1-indexed.
    m_s is the index of the last active neuron.
    If active_category_is_zero is True, 'active' is represented by 0, the first category.
    If no neuron is active, m_s is 0.
    """
    active_val = 0 if active_category_is_zero else 1
    for i in reversed(range(len(state_tuple))):
        if state_tuple[i] == active_val:
            return i + 1  # 1-indexed
    return 0

from collections import Counter



def get_HFM_prob(m_s: float, g: float, Z: float, logits: True) -> float:
    """
    Calulates the HFM theorical probability for a state, given m_s, g, and Z.
    If logits=True (default) it returns the log probabilities.
    """
    H_s = max(m_s - 1, 0)
    if logits:
        return -g * H_s - np.log(Z)
    return np.exp(-g * H_s)/Z



def get_empirical_latent_distribution(model, dataloader, device):
    """
    Encodes the dataset, samples latent states, and calculates their empirical probability.

    Args:
        model (VAE_priorHFM): The trained VAE model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to use (cpu or cuda).

    Returns:
        dict: A dictionary mapping latent state tuples to their empirical probabilities.
        int: Total number of samples processed.
    """
    model.eval()
    all_sampled_states = []
    total_samples = 0

    with torch.no_grad():
        for data_batch, _ in dataloader: # Assuming dataloader yields (data, labels)
            data_batch = data_batch.to(device)
            if data_batch.dim() > 2: # Flatten if necessary, e.g. images
                data_batch_flat = data_batch.view(data_batch.size(0), -1)
            else:
                data_batch_flat = data_batch

            logits_z_flat = model.encode(data_batch_flat) # (batch_size, latent_dim * categorical_dim)
            # Reshape to (batch_size, latent_dim, categorical_dim)
            logits_z = logits_z_flat.view(-1, model.latent_dim, model.categorical_dim)

            probs_z = F.softmax(logits_z, dim=-1) # (batch_size, latent_dim, 2)

            # Sample from the categorical distribution for each latent variable
            # Resulting samples will be 0 (active) or 1 (inactive)
            # based on "active (first category) or inactive (second category)"
            sampled_latent_codes = torch.multinomial(
                probs_z.view(-1, model.categorical_dim), # (batch_size * latent_dim, 2)
                num_samples=1
            ).view(-1, model.latent_dim) # (batch_size, latent_dim)

            # Convert tensor rows to tuples to be hashable for Counter
            for i in range(sampled_latent_codes.size(0)):
                all_sampled_states.append(tuple(sampled_latent_codes[i].tolist())) #tuples are hashables, i.e. can be used as keys in dictionaries
            total_samples += data_batch.size(0)

    if not all_sampled_states:
        return {}, 0

    state_counts = Counter(all_sampled_states)
    empirical_probs = {state: count / total_samples for state, count in state_counts.items()}
    return empirical_probs, total_samples




def calculate_Z_theoretical(latent_dim, g_param):
    """
    Calculates the normalization constant Z based on the provided analytical formula.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        g_param (float): The constant 'g'.

    Returns:
        float: The normalization constant Z.
    """
    if math.isclose(g_param, math.log(2)):
        # Handles the case g = log(2) => xi = 1
        Z = 1.0 + float(latent_dim)
    else:
        xi = 2.0 * math.exp(-g_param)
        if math.isclose(xi, 1.0): # Should be caught by g = log(2) but good for robustness
            Z = 1.0 + float(latent_dim)
        else:
            # Sum of geometric series xi^0 + ... + xi^(latent_dim-1)
            # This sum is for latent_dim terms.
            # The formula Z = 1 + (xi^latent_dim - 1) / (xi - 1) suggests this sum part.
            sum_geometric_part = (xi**latent_dim - 1.0) / (xi - 1.0)
            Z = 1.0 + sum_geometric_part
    
    if Z == 0:
        raise ValueError("Calculated theoretical Z is zero, leading to division by zero for probabilities.")
    return Z



def calculate_kl_divergence_with_HFM(empirical_probs, g=np.log(2), normalize_theoricalHFM = False, return_emp_distr_entropy = False):
    '''
    Calculates the KL divergence between the empirical distribution and the HFM distribution: # KL(h(s), p_emp(s))

    Args:
        empirical_probs (dict): states sampled empirically with their resepective probabilities.
        latent_dim (int): number of features of each sample
        g (float): parameter in the HFM distribution to calculate the KL div with.

    Returns:
        float: the KL divergence.
    '''

    empirical_probs_values = torch.tensor(list(empirical_probs.values()), dtype=torch.float32)
    empirical_distribution = torch.distributions.Categorical(empirical_probs_values)
    empirical_entropy = empirical_distribution.entropy()

    latent_dim = len(next(iter(empirical_probs)))
    Z = calculate_Z_theoretical(latent_dim, g)

    if normalize_theoricalHFM:
        theorical_HFM_logits = []
        kl_divergence_func = torch.nn.KLDivLoss(reduction='sum', log_target=False)
    else:
        mean_H_s = 0

    for state, p_emp in empirical_probs.items():
        m_s = get_m_s(state)
        if normalize_theoricalHFM:
            theorical_HFM_logits.append(get_HFM_prob(m_s, g, Z, True))
        else:
            mean_H_s += p_emp * m_s
    
    if normalize_theoricalHFM:
        theorical_HFM_logits = torch.tensor(theorical_HFM_logits, dtype=torch.float32)
        theorical_HFM_distribution = torch.distributions.Categorical(logits=theorical_HFM_logits)
        kl_divergence = kl_divergence_func(empirical_distribution.probs.log(), theorical_HFM_distribution.probs)
        if return_emp_distr_entropy:
            return (kl_divergence, empirical_entropy)
        else:
            return kl_divergence
    else:
        kl_divergence = - empirical_entropy + g * mean_H_s + math.log(Z)
        if return_emp_distr_entropy:
            return (kl_divergence, empirical_entropy)
        else:
            return kl_divergence
        
    

