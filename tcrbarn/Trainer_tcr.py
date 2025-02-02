import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def encoder_loss_function(acid2int, recon_x, x, mean, log_var):
    """
    Compute the loss for the encoder.
    Args:
        acid2int (dict): Dictionary mapping amino acids to indices.
        recon_x (Tensor): Reconstructed input.
        x (Tensor): Original input.
        mean (Tensor): Mean of the latent distribution.
        log_var (Tensor): Log variance of the latent distribution.
    Returns:
        tuple: Reconstruction loss and KL divergence loss.
    """
    # Reconstruction loss
    recon_loss_fn = nn.CrossEntropyLoss(ignore_index=acid2int["<PAD>"])
    recon_loss = recon_loss_fn(recon_x, x)
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss, kl_loss


def reparameterize(mean, log_var):
    """
    Reparameterize the latent variables.
    Args:
        mean (Tensor): Mean of the latent distribution.
        log_var (Tensor): Log variance of the latent distribution.
    Returns:
        Tensor: Reparameterized latent variable.
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mean + eps * std


def save_plot_with_incremental_filename(base_folder, filename):
    """
    Saves a plot with an incremental filename if the file already exists.
    Args:
        base_folder (str): Directory to save the plot.
        filename (str): Base filename for the plot.
    """
    base_name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_filename = f"{base_name}_{counter}{ext}"
        new_filepath = os.path.join(base_folder, new_filename)
        if not os.path.exists(new_filepath):
            plt.savefig(new_filepath)
            break
        counter += 1


def pass_models(alpha, beta, alpha_model, beta_model, model_type, va, vb, ja, jb, DEVICE, stage1_output):
    """
    Pass the inputs through the models and concatenate the outputs.
    Args:
        alpha (Tensor): Alpha chain input.
        beta (Tensor): Beta chain input.
        alpha_model (nn.Module): Alpha chain model.
        beta_model (nn.Module): Beta chain model.
        model_type (str): Type of the model.
        va (Tensor): One-hot encoded V gene for alpha chain.
        vb (Tensor): One-hot encoded V gene for beta chain.
        ja (Tensor): One-hot encoded J gene for alpha chain.
        jb (Tensor): One-hot encoded J gene for beta chain.
        DEVICE (torch.device): Device to run the models on.
        stage1_output (Tensor): Output from stage 1.
    Returns:
        Tensor: Concatenated inputs for the MLP.
    """
    encoder_alpha = alpha_model
    encoder_beta = beta_model
    alpha = alpha.to(DEVICE)
    beta = beta.to(DEVICE)
    if stage1_output is not None and any(o is not None for o in stage1_output):
        stage1_output = stage1_output.to(DEVICE)
    _, alpha_vector, _ = encoder_alpha(alpha)
    _, beta_vector, _ = encoder_beta(beta)
    if alpha_vector.shape[0] >= 2:
        # If the tensor has at least two layers, return the second one with an added dimension
        alpha_vector = alpha_vector[1].unsqueeze(0)
    if beta_vector.shape[0] >= 2:
        # If the tensor has at least two layers, return the second one with an added dimension
        beta_vector = beta_vector[1].unsqueeze(0)
    concatenated_a_b = torch.cat((alpha_vector, beta_vector), dim=2)
    concatenated_inputs = torch.cat((concatenated_a_b, va.unsqueeze(0),
                                     vb.unsqueeze(0), ja.unsqueeze(0),
                                     jb.unsqueeze(0)), dim=2)
    if stage1_output is not None and any(o is not None for o in stage1_output):
        stage1_output = stage1_output.view(1, 64, 1)
        concatenated_inputs = torch.cat((concatenated_inputs, stage1_output), dim=2)
    return concatenated_inputs


def train_model(model, model_type, alpha_input, beta_input, data_loader, test_loader, DEVICE, base_folder,
                batch_size, acid2int, weight_decay_encoder, weight_decay_cl, patience=5, min_delta=0.0001):
    """
    Train the model with the given inputs and parameters.
    Args:
        model (nn.Module): The model to train.
        model_type (str): Type of the model.
        alpha_input (tuple): Tuple containing the alpha encoder and decoder.
        beta_input (tuple): Tuple containing the beta encoder and decoder.
        data_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
        DEVICE (torch.device): Device to run the models on.
        base_folder (str): Directory to save the plots.
        batch_size (int): Batch size for training.
        acid2int (dict): Dictionary mapping amino acids to indices.
        weight_decay_encoder (float): Weight decay for the encoder optimizer.
        weight_decay_cl (float): Weight decay for the model optimizer.
        patience (int, optional): Number of epochs to wait for improvement before early stopping. Default is 5.
        min_delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement.
        Default is 0.0001.
    """
    criterion = nn.BCEWithLogitsLoss()
    encoder_alpha, decoder_alpha = alpha_input
    encoder_beta, decoder_beta = beta_input

    model_optimizer = optim.AdamW(model.parameters(), weight_decay=weight_decay_cl)
    encoder_alpha_optimizer = optim.AdamW(encoder_alpha.parameters(), weight_decay=weight_decay_encoder)
    encoder_beta_optimizer = optim.AdamW(encoder_beta.parameters(), weight_decay=weight_decay_encoder)
    decoder_alpha_optimizer = optim.AdamW(decoder_alpha.parameters(), weight_decay=weight_decay_encoder)
    decoder_beta_optimizer = optim.AdamW(decoder_beta.parameters(), weight_decay=weight_decay_encoder)

    num_epochs = 40
    train_losses = []
    if test_loader is not None:
        test_losses = []
        best_test_loss = float('inf')
        epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        encoder_alpha.train()
        encoder_beta.train()
        decoder_alpha.train()
        decoder_beta.train()

        batch_train_losses = []

        for i, (alpha, beta, va, vb, ja, jb, label, stage1_output) in enumerate(data_loader):
            alpha = alpha.to(DEVICE)
            beta = beta.to(DEVICE)
            label = label.to(DEVICE)
            va = va.to(DEVICE)
            vb = vb.to(DEVICE)
            ja = ja.to(DEVICE)
            jb = jb.to(DEVICE)
            if stage1_output is not None and any(o is not None for o in stage1_output):
                stage1_output = stage1_output.to(DEVICE)

            encoder_alpha_optimizer.zero_grad()
            encoder_beta_optimizer.zero_grad()
            decoder_alpha_optimizer.zero_grad()
            decoder_beta_optimizer.zero_grad()
            model_optimizer.zero_grad()
            loss_alpha = merge_pass(alpha, encoder_alpha, decoder_alpha, DEVICE, batch_size, acid2int)
            loss_beta = merge_pass(beta, encoder_beta, decoder_beta, DEVICE, batch_size, acid2int)
            concatenated_inputs = pass_models(alpha, beta, encoder_alpha, encoder_beta, model_type, va, vb, ja, jb,
                                              DEVICE, stage1_output)

            outputs = model(concatenated_inputs)
            predict_loss = criterion(outputs.view(-1), label) * 50
            loss = predict_loss + loss_alpha + loss_beta
            if i % 100 == 0:
                print(f'Epoch {epoch}, Batch {i}, Alpha Loss: {loss_alpha.item():.4f}, '
                      f'Beta Loss: {loss_beta.item():.4f}, Predict Loss: {predict_loss.item():.4f}')

            loss.backward()

            encoder_alpha_optimizer.step()
            encoder_beta_optimizer.step()
            decoder_alpha_optimizer.step()
            decoder_beta_optimizer.step()

            model_optimizer.step()

            batch_train_losses.append(loss.item())

        epoch_train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(epoch_train_loss)

        if test_loader is not None:
            # Testing
            model.eval()
            encoder_alpha.eval()
            encoder_beta.eval()
            decoder_alpha.eval()
            decoder_beta.eval()
            batch_test_losses = []
            with torch.no_grad():
                for alpha, beta, va, vb, ja, jb, label, stage1 in test_loader:
                    alpha = alpha.to(DEVICE)
                    beta = beta.to(DEVICE)
                    label = label.to(DEVICE)
                    va = va.to(DEVICE)
                    vb = vb.to(DEVICE)
                    ja = ja.to(DEVICE)
                    jb = jb.to(DEVICE)
                    loss_alpha = merge_pass(alpha, encoder_alpha, decoder_alpha, DEVICE, batch_size, acid2int)
                    loss_beta = merge_pass(beta, encoder_beta, decoder_beta, DEVICE, batch_size, acid2int)
                    concatenated_inputs = pass_models(alpha, beta, encoder_alpha, encoder_beta, model_type, va, vb, ja,
                                                      jb, DEVICE, stage1)

                    outputs = model(concatenated_inputs)
                    predict_loss = criterion(outputs.view(-1), label) * 50
                    loss = predict_loss + loss_alpha + loss_beta

                    batch_test_losses.append(loss.item())
            epoch_test_loss = sum(batch_test_losses) / len(batch_test_losses)
            test_losses.append(epoch_test_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss}, Test Loss: {epoch_test_loss}')
            # Check for improvement in test loss
            if epoch_test_loss < best_test_loss - min_delta:
                best_test_loss = epoch_test_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping condition
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}. No improvement in test loss for {patience}"
                      f"consecutive epochs.")
                break

        else:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss}')

    # Plotting the loss after training
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    if test_loader is not None:
        plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Over Epochs")
    plt.legend()
    save_plot_with_incremental_filename(base_folder, 'loss_plot.png')


def merge_pass(sequence, encoder, decoder, DEVICE, batch_size, acid2int):
    """
    Perform a forward pass through the encoder and decoder, and compute the loss.
    Args:
        sequence (Tensor): Input sequence.
        encoder (nn.Module): Encoder model.
        decoder (nn.Module): Decoder model.
        DEVICE (torch.device): Device to run the models on.
        batch_size (int): Batch size for training.
        acid2int (dict): Dictionary mapping amino acids to indices.
    Returns:
        Tensor: Total loss for the sequence.
    """
    target_length = sequence.size(1)

    # Encoder
    _, encoder_mean, encoder_sigma = encoder(sequence)
    z = reparameterize(encoder_mean, encoder_sigma).to(DEVICE)
    # Decoder
    decoder_input = torch.full((batch_size, 1), acid2int["<SOS>"], dtype=torch.long).to(DEVICE)

    recon_loss = 0
    kl_loss = 0
    for di in range(target_length):
        logits, z = decoder(decoder_input, z)
        recon_loss_step, kl_loss_step = encoder_loss_function(acid2int, logits, sequence[:, di], encoder_mean,
                                                              encoder_sigma)
        recon_loss += recon_loss_step
        kl_loss += kl_loss_step
        decoder_input = logits.argmax(dim=1).unsqueeze(1)

    # Backpropagation
    divided_kl_loss = kl_loss / 100_000
    total_loss = recon_loss + divided_kl_loss
    return total_loss
