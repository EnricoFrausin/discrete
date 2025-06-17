
class VAE_ex_priorCategorical(nn.Module): # 3-layers

    def __init__(self, input_dim, latent_dim, categorical_dim, device, decrease_rate=0.5):
        super(VAE_ex_priorCategorical, self).__init__()
        self.neurons_1 = input_dim
        self.neurons_2 = math.ceil(self.neurons_1 * decrease_rate)
        self.neurons_3 = math.ceil(self.neurons_2 * decrease_rate)
        self.device = device
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.fc1 = nn.Linear(self.neurons_1, self.neurons_2)
        self.fc2 = nn.Linear(self.neurons_2, self.neurons_3)
        self.fc3 = nn.Linear(self.neurons_3, self.latent_dim * self.categorical_dim)
        self.fc4 = nn.Linear(self.latent_dim * self.categorical_dim, self.neurons_3)
        self.fc5 = nn.Linear(self.neurons_3, self.neurons_2)
        self.fc6 = nn.Linear(self.neurons_2, self.neurons_1)
        # Funzioni di attivazione
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def sample_img(self, img, temp, random=True):
        with torch.no_grad():
            logits_z = self.encode(img.view(-1, self.neurons_1))
            logits_z = logits_z.view(-1, self.latent_dim, self.categorical_dim)
            if random:
                latent_z = gumbel_softmax(logits_z, self.latent_dim, self.categorical_dim, temp, self.device, True)
            else:
                latent_z = logits_z.view(-1, self.latent_dim * self.categorical_dim)
            logits_x = self.decode(latent_z)
            dist_x = torch.distributions.Bernoulli(probs=logits_x)
            sampled_img = dist_x.sample()
        return sampled_img

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, data, temp, _lambda, hard):
        logits_z = self.encode(data.view(-1, self.neurons_1))
        logits_z = logits_z.view(-1, self.latent_dim, self.categorical_dim)

        probs_z = F.softmax(logits_z, dim=-1)
        posterior_distrib = torch.distributions.Categorical(probs=probs_z)

        probs_prior = torch.ones_like(logits_z)/self.categorical_dim
        prior_distrib = torch.distributions.Categorical(probs=probs_prior)

        latent_z = gumbel_softmax(logits_z, self.latent_dim, self.categorical_dim, temp, self.device)
        latent_z = latent_z.view(-1, self.latent_dim * self.categorical_dim)

        probs_x = self.decode(latent_z)
        dist_x = torch.distributions.Bernoulli(probs=probs_x, validate_args=False)

        rec_loss = dist_x.log_prob(data.view(-1, self.neurons_1)).sum(dim=-1)
        logits_z_log = F.log_softmax(logits_z, dim=-1)

        KL = (posterior_distrib.probs * (logits_z_log - prior_distrib.probs.log())).view(-1, self.latent_dim * self.categorical_dim).sum(dim=-1)
        elbo = rec_loss - _lambda * KL
        loss = -elbo.mean()
        return loss, KL.mean(), rec_loss.mean()




class VAE_ex_priorHFM(nn.Module): # 3-layers

    def __init__(self, input_dim, latent_dim, g, device, decrease_rate=0.5):
        super(VAE_priorHFM, self).__init__()
        self.neurons_1 = input_dim
        self.neurons_2 = math.ceil(self.neurons_1 * decrease_rate)
        self.neurons_3 = math.ceil(self.neurons_2 * decrease_rate)
        self.device = device
        self.latent_dim = latent_dim
        self.g = g
        self.fc1 = nn.Linear(self.neurons_1, self.neurons_2)
        self.fc2 = nn.Linear(self.neurons_2, self.neurons_3)
        self.fc3 = nn.Linear(self.neurons_3, self.latent_dim * 2)
        self.fc4 = nn.Linear(self.latent_dim * 2, self.neurons_3)
        self.fc5 = nn.Linear(self.neurons_3, self.neurons_2)
        self.fc6 = nn.Linear(self.neurons_2, self.neurons_1)
        # Funzioni di attivazione
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def sample_img(self, img, temp, random=True):
        with torch.no_grad():
            logits_z = self.encode(img.view(-1, self.neurons_1))
            logits_z = logits_z.view(-1, self.latent_dim, 2)
            if random:
                latent_z = gumbel_softmax(logits_z, self.latent_dim, 2, temp, self.device, True)
            else:
                latent_z = logits_z.view(-1, self.latent_dim * 2)
            logits_x = self.decode(latent_z)
            dist_x = torch.distributions.Bernoulli(probs=logits_x)
            sampled_img = dist_x.sample()
        return sampled_img

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, data, temp, _lambda, hard):
        logits_z = self.encode(data.view(-1, self.neurons_1))
        logits_z = logits_z.view(-1, self.latent_dim, 2)

        probs_z = F.softmax(logits_z, dim=-1)
        posterior_distrib = torch.distributions.Categorical(probs=probs_z)

        prior_probs_list = []
        for i in range(self.latent_dim):
            prob_activation = mean_s_k(self.latent_dim,i,self.g)
            prob_nonactivation = 1 - prob_activation
            prior_probs_list.append([prob_activation, prob_nonactivation])

        probs_prior_base = torch.tensor(prior_probs_list, device=self.device, dtype=torch.float32)
        probs_prior = probs_prior_base.unsqueeze(0).expand(data.size(0), -1, -1)
        prior_distrib = torch.distributions.Categorical(probs=probs_prior)

        latent_z = gumbel_softmax(logits_z, self.latent_dim, 2, temp, self.device)
        latent_z = latent_z.view(-1, self.latent_dim * 2)

        probs_x = self.decode(latent_z)
        dist_x = torch.distributions.Bernoulli(probs=probs_x, validate_args=False)

        rec_loss = dist_x.log_prob(data.view(-1, self.neurons_1)).sum(dim=-1)
        logits_z_log = F.log_softmax(logits_z, dim=-1)

        #KL è la somma su tutte le (singole KL calcolate sulla distr di prob di bernoulli di ogni feature), non ancora sommata su ogni esempio del batch 
        KL = (posterior_distrib.probs * (logits_z_log - prior_distrib.probs.log())).view(-1, self.latent_dim * 2).sum(dim=-1)
        elbo = rec_loss - _lambda * KL
        loss = -elbo.mean()
        return loss, KL.mean(), rec_loss.mean()
    



class VAE_priorCategorical_1hlayer(nn.Module): # 3-layers

    def __init__(self, input_dim, latent_dim, categorical_dim, device, decrease_rate=0.3):
        super(VAE_priorCategorical_1hlayer, self).__init__()
        self.neurons_1 = input_dim
        self.neurons_2 = math.ceil(self.neurons_1 * decrease_rate)
        self.device = device
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.fc1 = nn.Linear(self.neurons_1, self.neurons_2)
        self.fc2 = nn.Linear(self.neurons_2, self.latent_dim * self.categorical_dim)
        self.fc3 = nn.Linear(self.latent_dim * self.categorical_dim, self.neurons_2)
        self.fc4 = nn.Linear(self.neurons_2, self.neurons_1)
        # Funzioni di attivazione
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def sample_img(self, img, temp, random=True):
        with torch.no_grad():
            logits_z = self.encode(img.view(-1, self.neurons_1))
            logits_z = logits_z.view(-1, self.latent_dim, self.categorical_dim)
            if random:
                latent_z = gumbel_softmax(logits_z, self.latent_dim, self.categorical_dim, temp, self.device, True)
            else:
                latent_z = logits_z.view(-1, self.latent_dim * self.categorical_dim)
            logits_x = self.decode(latent_z)
            dist_x = torch.distributions.Bernoulli(probs=logits_x)
            sampled_img = dist_x.sample()
        return sampled_img

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.relu(self.fc2(h1))

    def decode(self, z):
        h2 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h2))

    def forward(self, data, temp, _lambda, hard):
        logits_z = self.encode(data.view(-1, self.neurons_1))
        logits_z = logits_z.view(-1, self.latent_dim, self.categorical_dim)

        probs_z = F.softmax(logits_z, dim=-1)
        posterior_distrib = torch.distributions.Categorical(probs=probs_z)
        probs_prior = torch.ones_like(logits_z)/self.categorical_dim
        prior_distrib = torch.distributions.Categorical(probs=probs_prior)

        latent_z = gumbel_softmax(logits_z, self.latent_dim, self.categorical_dim, temp, self.device)
        latent_z = latent_z.view(-1, self.latent_dim * self.categorical_dim)

        probs_x = self.decode(latent_z)
        dist_x = torch.distributions.Bernoulli(probs=probs_x, validate_args=False)

        rec_loss = dist_x.log_prob(data.view(-1, self.neurons_1)).sum(dim=-1)
        logits_z_log = F.log_softmax(logits_z, dim=-1)

        KL = (posterior_distrib.probs * (logits_z_log - prior_distrib.probs.log())).view(-1, self.latent_dim * self.categorical_dim).sum(dim=-1)
        elbo = rec_loss - _lambda * KL
        loss = -elbo.mean()
        return loss, KL.mean(), rec_loss.mean()
    

class VAE_priorHFM_1hlayer(nn.Module): # 3-layers

    def __init__(self, input_dim, latent_dim, g, device, decrease_rate=0.3):
        super(VAE_priorHFM_1hlayer, self).__init__()
        self.neurons_1 = input_dim
        self.neurons_2 = math.ceil(self.neurons_1 * decrease_rate)
        self.device = device
        self.latent_dim = latent_dim
        self.g = g
        self.fc1 = nn.Linear(self.neurons_1, self.neurons_2)
        self.fc2 = nn.Linear(self.neurons_2, self.latent_dim * 2)
        self.fc3 = nn.Linear(self.latent_dim * 2, self.neurons_2)
        self.fc4 = nn.Linear(self.neurons_2, self.neurons_1)
        # Funzioni di attivazione
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def sample_img(self, img, temp, random=True):
        with torch.no_grad():
            logits_z = self.encode(img.view(-1, self.neurons_1))
            logits_z = logits_z.view(-1, self.latent_dim, 2)
            if random:
                latent_z = gumbel_softmax(logits_z, self.latent_dim, 2, temp, self.device, True)
            else:
                latent_z = logits_z.view(-1, self.latent_dim * 2)
            logits_x = self.decode(latent_z)
            dist_x = torch.distributions.Bernoulli(probs=logits_x)
            sampled_img = dist_x.sample()
        return sampled_img

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.relu(self.fc2(h1))

    def decode(self, z):
        h2 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h2))

    def forward(self, data, temp, _lambda, hard):
        logits_z = self.encode(data.view(-1, self.neurons_1))
        logits_z = logits_z.view(-1, self.latent_dim, 2)

        probs_z = F.softmax(logits_z, dim=-1)
        posterior_distrib = torch.distributions.Categorical(probs=probs_z)

        prior_probs_list = []
        for i in range(self.latent_dim):
            prob_activation = mean_s_k(self.latent_dim,i,self.g)
            prob_nonactivation = 1 - prob_activation
            prior_probs_list.append([prob_activation, prob_nonactivation])

        probs_prior_base = torch.tensor(prior_probs_list, device=self.device, dtype=torch.float32)
        probs_prior = probs_prior_base.unsqueeze(0).expand(data.size(0), -1, -1)
        prior_distrib = torch.distributions.Categorical(probs=probs_prior)

        latent_z = gumbel_softmax(logits_z, self.latent_dim, 2, temp, self.device)
        latent_z = latent_z.view(-1, self.latent_dim * 2)

        probs_x = self.decode(latent_z)
        dist_x = torch.distributions.Bernoulli(probs=probs_x, validate_args=False)

        rec_loss = dist_x.log_prob(data.view(-1, self.neurons_1)).sum(dim=-1)
        logits_z_log = F.log_softmax(logits_z, dim=-1)

        #KL è la somma su tutte le (singole KL calcolate sulla distr di prob di bernoulli di ogni feature), non ancora sommata su ogni esempio del batch 
        KL = (posterior_distrib.probs * (logits_z_log - prior_distrib.probs.log())).view(-1, self.latent_dim * 2).sum(dim=-1)
        elbo = rec_loss - _lambda * KL
        loss = -elbo.mean()
        return loss, KL.mean(), rec_loss.mean()




# DA USARE DENTRO A UN MODEL
    def sample_img(self, img, temp, random=True):
        # Assicurati che gumbel_softmax sia definita e importata se usi questo metodo
        # from your_module import gumbel_softmax
        with torch.no_grad():
            if img.dim() > 2:
                img_flat = img.view(-1, self.input_dim)
            else:
                img_flat = img

            logits_z_flat = self.encode(img_flat)
            logits_z = logits_z_flat.view(-1, self.latent_dim, self.categorical_dim)

            if random:
                # Usa hard=True per campionare one-hot discreti dopo Gumbel-Softmax
                latent_z_one_hot = gumbel_softmax(logits_z, self.latent_dim, self.categorical_dim, temp, self.device, hard=True)
                latent_z = latent_z_one_hot.view(-1, self.latent_dim * self.categorical_dim)
            else:
                # Se non random, potresti voler prendere argmax per ottenere one-hot,
                # o usare i logits direttamente se il decoder può gestirli (improbabile per categorico).
                # Prendere i logits direttamente come z non è tipico per VAE categorici in generazione.
                # Per coerenza con Gumbel-Softmax, prendiamo l'argmax dei logits (modalità).
                _, argmax_cats = logits_z.max(dim=-1) # (batch, latent_dim)
                latent_z_one_hot = F.one_hot(argmax_cats, num_classes=self.categorical_dim).float() # (batch, latent_dim, cat_dim)
                latent_z = latent_z_one_hot.view(-1, self.latent_dim * self.categorical_dim)

            logits_x = self.decode(latent_z)
            dist_x = torch.distributions.Bernoulli(probs=logits_x)
            sampled_img = dist_x.sample()
            # sampled_img avrà shape (batch_size, input_dim)
        return sampled_img
    