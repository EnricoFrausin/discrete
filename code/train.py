
import torch
import numpy as np
from metadata import temperature, temp_min, ANNEAL_RATE, _lambda
from utilities import get_empirical_latent_distribution, calculate_kl_divergence_with_HFM

def train(model, _lambda, writer, train_loader, val_loader, optimizer, device, epochs, g=np.log(2), calculate_KL_HFM=False, save_tb_parameters=False):
    global_batch_idx = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        temp = temperature
        train_KL = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            global_batch_idx += 1
            data = data.to(device)
            optimizer.zero_grad()
            loss, KL, rec_loss = model(data, temp, _lambda, hard=False)
            loss.backward()
            train_loss += loss.item() * len(data)
            optimizer.step()

            if batch_idx % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

        writer.add_scalar('KL/Train', KL, global_step=epoch)
        writer.add_scalar('rec_loss/Train', rec_loss, global_step=epoch)
        writer.add_scalar('Loss/Train', train_loss/len(train_loader.dataset), global_step=epoch)

        if calculate_KL_HFM:
            empirical_probs, total_samples = get_empirical_latent_distribution(model, train_loader, device)
            kl_divergence = calculate_kl_divergence_with_HFM(empirical_probs, g, normalize_theoricalHFM=True)
            writer.add_scalar('KL with HFM/Train', kl_divergence, global_step=epoch)


        print('Epoch: {}/{}, Average loss: {:.4f}'.format(
            epoch, epochs, train_loss / len(train_loader.dataset)))
        
        # Validation
        
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_loader):
                data = data.to(device)
                loss, KL, rec_loss = model(data, temp, _lambda, hard=True) #hard setted to True for validation
                val_loss_sum += loss.item() * len(data)

        writer.add_scalar('KL/Validation', KL, global_step=epoch)
        writer.add_scalar('rec_loss/Validation', rec_loss, global_step=epoch)
        writer.add_scalar('Loss/Validation', val_loss_sum/len(val_loader.dataset), global_step=epoch)

        if calculate_KL_HFM:
            empirical_probs, total_samples = get_empirical_latent_distribution(model, val_loader, device)
            kl_divergence = calculate_kl_divergence_with_HFM(empirical_probs, g, normalize_theoricalHFM=True)
            writer.add_scalar('KL with HFM/Validation', kl_divergence, global_step=epoch)

        # Log histogram of weights and gradients
        if save_tb_parameters:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Weights/{name}', param, global_step=epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Grads/{name}', param.grad, global_step=epoch)

    writer.close()
    print("Training completato e dati scritti su tensorboard")


