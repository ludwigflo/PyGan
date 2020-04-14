def discriminator_iteration(sample, model, criterion, optimizer, **kwargs):
    """
    Parameters:
        sample:     dict, dictionary containing the real samples (key: 'real_data') and noise for the generator (key:
                    'generator_noise'), which is used to generate the fake samples.
        model:      Pytorch model, defining the GAN. This model needs to have sub model_utils model.generator, defining
                    the GAN generator, and model.discriminator, defining the discriminator.
        criterion:  Optimization criterion, which is used to evaluate the discriminator decision. The common choice is
                    the Binary Cross Entropy.
        optimizer:  Pytorch optimizer, used to optimize the discriminator. A common choice is the Adam optimizer.
    """

    # some preparations
    data_real = sample['real_data']
    noise = sample['generator_input']
    model.zero_grad()

    # use the gpu, if possible
    if torch.cuda.is_available():
        (model, data_real, noise) = to_gpu(model, data_real, noise)

    # compute the fake data by passing the noise through the generator
    data_fake = model.generator(noise)

    real_loss, real_score, fake_loss, fake_score = compute_loss(model, criterion, data_real, data_fake, **kwargs)

    # error backpropagation and optimizing step
    total_loss = real_loss + fake_loss
    total_loss.backward()
    optimizer.step()

    # prepare and return the results
    out_dict = {'discriminator/loss': total_loss.data.item(), 'discriminator/fake_score': fake_score,
                'discriminator/real_score': real_score, 'discriminator/real_loss': real_loss,
                'discriminator/fake_loss': fake_loss}
    return out_dict