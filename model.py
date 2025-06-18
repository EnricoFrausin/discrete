
import torch.nn as nn
import torch
from utilities import gumbel_softmax
from utilities import mean_s_k
import torch.nn.functional as F
import math





class VAE_priorCategorical(nn.Module):
    def __init__(self, input_dim, latent_dim, categorical_dim, device,
                 num_hidden_layers=1, decrease_rate=0.5,
                 activation_fn=nn.ReLU, output_activation_encoder=None):
        """
        VAE con un numero flessibile di layer nascosti.

        Args:
            input_dim (int): Dimensione dell'input.
            latent_dim (int): Numero di variabili latenti categoriche.
            categorical_dim (int): Numero di categorie per ogni variabile latente.
            device (torch.device): Dispositivo (cpu o cuda).
            num_hidden_layers (int): Numero di layer nascosti intermedi nell'encoder/decoder.
                                     num_hidden_layers = 0 -> input -> (calc_neurons) -> latent
                                     num_hidden_layers = 1 -> input -> (calc_neurons1) -> (calc_neurons2) -> latent
                                     (fc1: input->n1, fc2: n1->n2, fc3: n2->latent).
                                     Il numero totale di layer lineari nell'encoder/decoder sarà num_hidden_layers + 2.
            initial_decrease_rate (float): Tasso di riduzione per il primo layer nascosto.
            decrease_rate (float): Tasso di riduzione per i layer nascosti successivi.
                                              Se vuoi lo stesso tasso, imposta = initial_decrease_rate.
            activation_fn (nn.Module): Funzione di attivazione per i layer nascosti (es. nn.ReLU).
            output_activation_encoder (nn.Module, optional): Attivazione per l'output dell'encoder (logits).
                                                             Di solito None o nn.Identity(). La tua originale aveva ReLU.
        """
        super(VAE_priorCategorical, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.device = device
        self.activation_fn = activation_fn() # Istanzia la funzione di attivazione
        self.num_hidden_layers = num_hidden_layers

        # ---- Calcolo delle dimensioni dei neuroni per l'encoder ----
        encoder_neuron_sizes = [self.input_dim]
        current_neurons = self.input_dim

        # Primo layer nascosto (equivalente a fc1 che definisce neurons_2)
        #current_neurons = math.ceil(current_neurons * initial_decrease_rate)
        #encoder_neuron_sizes.append(current_neurons)

        # Layer nascosti intermedi (equivalenti a fc2, ..., che definiscono neurons_3, ...)
        for _ in range(self.num_hidden_layers): # num_hidden_layers qui si riferisce a quelli *intermedi*
            current_neurons = math.ceil(current_neurons * decrease_rate)
            encoder_neuron_sizes.append(current_neurons)

        # Output dell'encoder (dimensione latente)
        encoder_neuron_sizes.append(self.latent_dim * self.categorical_dim)

        # ---- Costruzione dell'Encoder ----
        encoder_layers = []
        for i in range(len(encoder_neuron_sizes) - 1):
            in_features = encoder_neuron_sizes[i]
            out_features = encoder_neuron_sizes[i+1]
            encoder_layers.append(nn.Linear(in_features, out_features))
            # Applica l'attivazione a tutti i layer tranne l'ultimo (output dei logits),
            # a meno che non sia specificato diversamente con output_activation_encoder
            if i < len(encoder_neuron_sizes) - 2:
                encoder_layers.append(self.activation_fn)
            elif output_activation_encoder is not None: # Ultimo layer, prima dei logits
                encoder_layers.append(output_activation_encoder())


        self.encoder_net = nn.Sequential(*encoder_layers)

        # ---- Costruzione del Decoder (struttura speculare) ----
        # Le dimensioni dei neuroni del decoder sono quelle dell'encoder invertite
        decoder_neuron_sizes = list(reversed(encoder_neuron_sizes))

        decoder_layers = []
        for i in range(len(decoder_neuron_sizes) - 1):
            in_features = decoder_neuron_sizes[i]
            out_features = decoder_neuron_sizes[i+1]
            decoder_layers.append(nn.Linear(in_features, out_features))
            if i < len(decoder_neuron_sizes) - 2: # Tutti tranne l'ultimo layer del decoder
                decoder_layers.append(self.activation_fn)
            else: # Ultimo layer del decoder (output dell'immagine)
                decoder_layers.append(nn.Sigmoid()) # Sigmoid per output [0,1]

        self.decoder_net = nn.Sequential(*decoder_layers)

    def encode(self, x):
        # x ha shape (batch_size, input_dim)
        return self.encoder_net(x)

    def decode(self, z):
        # z ha shape (batch_size, latent_dim * categorical_dim)
        return self.decoder_net(z)

    # forward e sample_img richiederebbero gumbel_softmax
    # Se li vuoi usare, assicurati che gumbel_softmax sia definita e accessibile
    def forward(self, data, temp, _lambda, hard=False): # 'hard' era nel tuo forward originale ma non usata

        # Flatten data se necessario
        if data.dim() > 2:
            data_flat = data.view(-1, self.input_dim)
        else:
            data_flat = data

        logits_z_flat = self.encode(data_flat) # (batch_size, latent_dim * categorical_dim)
        # Reshape per gumbel_softmax e calcolo KL
        logits_z = logits_z_flat.view(-1, self.latent_dim, self.categorical_dim) # (batch_size, latent_dim, categorical_dim)

        probs_z = F.softmax(logits_z, dim=-1) # Probabilità per ogni categoria, per ogni var latente
        posterior_distrib = torch.distributions.Categorical(probs=probs_z)

        # Prior uniforme
        probs_prior_val = 1.0 / self.categorical_dim
        # Creare un tensore di prior_probs con la stessa forma di logits_z
        # è importante per il broadcasting corretto nel calcolo di KL
        prior_probs_tensor = torch.full_like(logits_z, probs_prior_val)
        prior_distrib = torch.distributions.Categorical(probs=prior_probs_tensor)


        # Campionamento Gumbel-Softmax
        # Assicurati che la tua funzione gumbel_softmax sia disponibile
        # latent_z_one_hot avrà shape (batch_size, latent_dim, categorical_dim)
        latent_z_one_hot = gumbel_softmax(logits_z, self.latent_dim, self.categorical_dim, temp, self.device, hard=hard)
        # Flatten per il decoder
        latent_z_flat = latent_z_one_hot.view(-1, self.latent_dim * self.categorical_dim)

        probs_x = self.decode(latent_z_flat) # Output del decoder, shape (batch_size, input_dim)
        dist_x = torch.distributions.Bernoulli(probs=probs_x, validate_args=False)

        # Reconstruction Loss
        # data_flat ha già la forma corretta (batch_size, input_dim)
        rec_loss = dist_x.log_prob(data_flat).sum(dim=-1) # Sum sui pixel, per ogni elemento del batch

        # KL Divergence
        # KL(q(z|x) || p(z))
        # Per Categorical: sum_k q_k * (log q_k - log p_k)
        # log q_k è log_softmax(logits_z)
        # log p_k è log(1/categorical_dim)
        log_qz = F.log_softmax(logits_z, dim=-1)
        log_pz = torch.log(prior_probs_tensor) # log(1/K) è costante
        
        # KL per ogni variabile latente, poi somma su tutte le variabili latenti e media sul batch
        kl_div_per_latent_var = (probs_z * (log_qz - log_pz)).sum(dim=-1) # Sum sulle categorie -> (batch, latent_dim)
        KL = kl_div_per_latent_var.sum(dim=-1) # Sum sulle variabili latenti -> (batch)


        elbo = rec_loss - _lambda * KL
        loss = -elbo.mean() # Media sul batch
        return loss, KL.mean(), rec_loss.mean()



#_______________________________________________________________________________________________________________________________________________________________________________________



class VAE_priorHFM(nn.Module):
    def __init__(self, input_dim, latent_dim, g, device,
                 num_hidden_layers=1, decrease_rate=0.5,
                 activation_fn=nn.ReLU, output_activation_encoder=None, LayerNorm=False, BatchNorm=False):
        """
        VAE con un numero flessibile di layer nascosti.

        Args:
            input_dim (int): Dimensione dell'input.
            latent_dim (int): Numero di variabili latenti categoriche.
            categorical_dim (int): Numero di categorie per ogni variabile latente.
            device (torch.device): Dispositivo (cpu o cuda).
            num_hidden_layers (int): Numero di layer nascosti intermedi nell'encoder/decoder.
                                     num_hidden_layers = 0 -> input -> (calc_neurons) -> latent
                                     num_hidden_layers = 1 -> input -> (calc_neurons1) -> (calc_neurons2) -> latent
                                     La tua implementazione originale corrisponde a num_hidden_layers = 1
                                     (fc1: input->n1, fc2: n1->n2, fc3: n2->latent).
                                     Il numero totale di layer lineari nell'encoder/decoder sarà num_hidden_layers + 2.
            initial_decrease_rate (float): Tasso di riduzione per il primo layer nascosto.
            decrease_rate (float): Tasso di riduzione per i layer nascosti successivi.
                                              Se vuoi lo stesso tasso, imposta = initial_decrease_rate.
            activation_fn (nn.Module): Funzione di attivazione per i layer nascosti (es. nn.ReLU).
            output_activation_encoder (nn.Module, optional): Attivazione per l'output dell'encoder (logits).
                                                             Di solito None o nn.Identity(). La tua originale aveva ReLU.
        """
        super(VAE_priorHFM, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.categorical_dim = 2
        self.g = g      # g is the prior parameter for the HFM distribution
        self.device = device
        self.activation_fn = activation_fn() # Istanzia la funzione di attivazione
        self.num_hidden_layers = num_hidden_layers

        # ---- Calcolo delle dimensioni dei neuroni per l'encoder ----
        encoder_neuron_sizes = [self.input_dim]
        current_neurons = self.input_dim

        # Primo layer nascosto (equivalente a fc1 che definisce neurons_2)
        #current_neurons = math.ceil(current_neurons * initial_decrease_rate)
        #encoder_neuron_sizes.append(current_neurons)

        # Layer nascosti intermedi (equivalenti a fc2, ..., che definiscono neurons_3, ...)
        for _ in range(self.num_hidden_layers): # num_hidden_layers qui si riferisce a quelli *intermedi*
            current_neurons = math.ceil(current_neurons * decrease_rate)
            encoder_neuron_sizes.append(current_neurons)

        # Output dell'encoder (dimensione latente)
        encoder_neuron_sizes.append(self.latent_dim * 2)

        # ---- Costruzione dell'Encoder ----
        encoder_layers = []
        for i in range(len(encoder_neuron_sizes) - 1):
            in_features = encoder_neuron_sizes[i]
            out_features = encoder_neuron_sizes[i+1]
            encoder_layers.append(nn.Linear(in_features, out_features))
            if LayerNorm:
                encoder_layers.append(nn.LayerNorm(out_features))
            if BatchNorm:
                encoder_layers.append(nn.BatchNorm1d(out_features))
            # Applica l'attivazione a tutti i layer tranne l'ultimo (output dei logits),
            # a meno che non sia specificato diversamente con output_activation_encoder
            if i < len(encoder_neuron_sizes) - 2:
                encoder_layers.append(self.activation_fn)
            elif output_activation_encoder is not None: # Ultimo layer, prima dei logits
                encoder_layers.append(output_activation_encoder())


        self.encoder_net = nn.Sequential(*encoder_layers)

        # ---- Costruzione del Decoder (struttura speculare) ----
        # Le dimensioni dei neuroni del decoder sono quelle dell'encoder invertite
        decoder_neuron_sizes = list(reversed(encoder_neuron_sizes))

        decoder_layers = []
        for i in range(len(decoder_neuron_sizes) - 1):
            in_features = decoder_neuron_sizes[i]
            out_features = decoder_neuron_sizes[i+1]
            decoder_layers.append(nn.Linear(in_features, out_features))
            if LayerNorm:
                decoder_layers.append(nn.LayerNorm(out_features))
            if BatchNorm:
                encoder_layers.append(nn.BatchNorm1d(out_features))
            if i < len(decoder_neuron_sizes) - 2: # Tutti tranne l'ultimo layer del decoder
                decoder_layers.append(self.activation_fn)
            else: # Ultimo layer del decoder (output dell'immagine)
                decoder_layers.append(nn.Sigmoid()) # Sigmoid per output [0,1]

        self.decoder_net = nn.Sequential(*decoder_layers)



    def encode(self, x):
        # x ha shape (batch_size, input_dim)
        return self.encoder_net(x)

    def decode(self, z):
        # z ha shape (batch_size, latent_dim * categorical_dim)
        return self.decoder_net(z)

    # forward e sample_img richiederebbero gumbel_softmax
    # Se li vuoi usare, assicurati che gumbel_softmax sia definita e accessibile
    def forward(self, data, temp, _lambda, hard=False): # 'hard' era nel tuo forward originale ma non usata

        # Flatten data se necessario
        if data.dim() > 2:
            data_flat = data.view(-1, self.input_dim)
        else:
            data_flat = data

        logits_z_flat = self.encode(data_flat) # (batch_size, latent_dim * categorical_dim)
        # Reshape per gumbel_softmax e calcolo KL
        logits_z = logits_z_flat.view(-1, self.latent_dim, self.categorical_dim) # (batch_size, latent_dim, categorical_dim)

        probs_z = F.softmax(logits_z, dim=-1) # Probabilità per ogni categoria, per ogni var latente
        posterior_distrib = torch.distributions.Categorical(probs=probs_z)


        # Prior con distribuzione HFM
        prior_probs_list = []
        for i in range(self.latent_dim):
            prob_activation = mean_s_k(self.latent_dim,i,self.g)
            prob_nonactivation = 1 - prob_activation
            prior_probs_list.append([prob_activation, prob_nonactivation])
        
        prior_probs_tensor = torch.tensor(prior_probs_list, device=self.device, dtype=torch.float32)
        prior_probs_tensor = prior_probs_tensor.unsqueeze(0).expand(data.size(0), -1, -1)

        # Campionamento Gumbel-Softmax
        # latent_z avrà shape (batch_size, latent_dim, categorical_dim)
        latent_z = gumbel_softmax(logits_z, self.latent_dim, self.categorical_dim, temp, self.device, hard=hard)
        # Flatten per il decoder
        latent_z_flatten = latent_z.view(-1, self.latent_dim * self.categorical_dim)

        probs_x = self.decode(latent_z_flatten) # Output del decoder, shape (batch_size, input_dim)
        dist_x = torch.distributions.Bernoulli(probs=probs_x, validate_args=False)

        # Reconstruction Loss
        # data_flat ha già la forma corretta (batch_size, input_dim)
        rec_loss = dist_x.log_prob(data_flat).sum(dim=-1) # Sum sui pixel, per ogni elemento del batch

        # KL Divergence
        # KL(q(z|x) || p(z))
        # Per Categorical: sum_k q_k * (log q_k - log p_k)
        # log q_k è log_softmax(logits_z)
        # log p_k è log(1/categorical_dim)
        log_qz = F.log_softmax(logits_z, dim=-1)
        log_pz = torch.log(prior_probs_tensor) 
        
        # KL per ogni variabile latente, poi somma su tutte le variabili latenti e media sul batch
        kl_div_per_latent_var = (probs_z * (log_qz - log_pz)).sum(dim=-1) # Sum sulle categorie -> (batch, latent_dim)
        KL = kl_div_per_latent_var.sum(dim=-1) # Sum sulle variabili latenti -> (batch)

        elbo = rec_loss - _lambda * KL
        loss = -elbo.mean() # Media sul batch
        return loss, KL.mean(), rec_loss.mean()