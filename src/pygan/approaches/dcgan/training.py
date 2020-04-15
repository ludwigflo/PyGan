from ...interfaces import GanTrainer
from ...utils import to_gpu
from typing import Dict
import torch.nn as nn
import numpy as np
import torch


class Trainer(GanTrainer):
    def __init__(self, gan_model: Gan, train_data_generator: Generator[dict], criterion: nn.Module,
                 val_data_generator: Generator[dict], parameter_file: str, device):

        super(Trainer, self).__init__(gan_model, train_data_generator, parameter_file)
        self.val_data_generator = val_data_generator
        self.criterion = criterion

        # store the current device and move the model to the device
        self.device = device
        self.gan_model.generator.to(device)
        self.gan_model.discriminator.to(device)

    def discriminator_train_iteration(self) -> Dict[str, float]:
        """
        Single iteration for training the discriminator.

        Returns
        -------
        out_dict: Dictionary, containing the metrics and loss values.
        """

        # some preparations
        sample = next(self.train_data_generator)
        data_real = sample['data_real']
        gen_input = sample['gen_input']
        gan_model.zero_grad()

        # compute the fake data by passing the noise through the generator
        data_fake = gan_model.generator(gen_input)

        # compute the targets (smooth labels)
        targets_real = torch.from_numpy(np.random.uniform(0.7, 1, data_real.size()[0])).float().view(-1, 1)
        targets_fake = torch.from_numpy(np.random.uniform(0, 0.3, data_fake.size()[0])).float().view(-1, 1)
        targets_real.to(self.device)
        targets_fake.to(self.device)

        # compute the predictions
        pred_real = gan_model.discriminator(data_real, **kwargs)
        pred_fake = gan_model.discriminator(data_fake, **kwargs)

        # compute the loss and scores
        real_loss = self.criterion(pred_real, targets_real)
        fake_loss = self.criterion(pred_fake, targets_fake)

        # error backpropagation and optimizing step
        total_loss = real_loss + fake_loss
        total_loss.backward()
        self.optimizer_list[1].step()

        # prepare and return the results
        out_dict = {'discriminator/fake_score': pred_fake.data.item(),
                    'discriminator/real_score': pred_real.data.item(),
                    'discriminator/real_loss': real_loss.data.item(),
                    'discriminator/fake_loss': fake_loss.data.item(),
                    'discriminator/loss': total_loss.data.item()}
        return out_dict

    def generator_train_iteration(self) -> Dict[str, float]:
        """
        Single iteration for training the discriminator.

        Returns
        -------
        out_dict: Dictionary, containing the metrics and loss values.
        """

        # some preparations
        sample = next(self.train_data_generator)
        gen_input = sample['gen_input']
        gan_model.zero_grad()

        # compute the fake data by passing the noise through the generator
        data_fake = gan_model.generator(gen_input)

        # compute the targets (smooth labels)
        targets_fake = torch.ones(data_fake.size()[0]).float().view(-1, 1)
        targets_fake.to(self.device)

        # compute the predictions
        pred_fake = gan_model.discriminator(data_fake, **kwargs)

        # compute the loss and scores
        fake_loss = self.criterion(pred_fake, targets_fake)

        # error backpropagation and optimizing step
        fake_loss.backward()
        self.optimizer_list[1].step()

        # prepare and return the results
        out_dict = {'generator/fake_score': pred_fake.data.item(),
                    'generator/loss': fake_loss.data.item()}
        return out_dict

    def val_epoch(self) -> DefaultDict[List]:
        """
        Validation epoch of the GAN.
        """

        val_results = defaultdict(list)
        with torch.no_grad():
            for sample, done in self.val_data_generator:

                # get the input for the generator and the real sample
                data_real = sample['data_real']
                gen_input = sample['gen_input']

                # compute fake data
                data_fake = self.gan_model.generator(gen_input)

                # compute the targets
                targets_real = torch.from_numpy(np.random.uniform(0.7, 1, data_real.size()[0])).float().view(-1, 1)
                targets_fake = torch.from_numpy(np.random.uniform(0, 0.3, data_fake.size()[0])).float().view(-1, 1)
                targets_real.to(self.device)
                targets_fake.to(self.device)

                # compute the predictions
                pred_real = gan_model.discriminator(data_real, **kwargs)
                pred_fake = gan_model.discriminator(data_fake, **kwargs)

                # compute the loss and scores
                loss_real = self.criterion(pred_real, targets_real)
                loss_fake = self.criterion(pred_fake, targets_fake)

                # compute the evaluation criteria
                val_results['evaluation/fake_score'].append(pred_fake.data.item())
                val_results['evaluation/real_score'].append(pred_real.data.item())
                val_results['evaluation/fake_loss'].append(loss_fake.data.item())
                val_results['evaluation/real_loss'].append(loss_real.data.item())
                val_results['images'].append(data_fake.detach().cpu().numpy())
        return val_results

    def log_results(self, experiment_dir: path, tb_logger_list: Union[None, list],
                    gen_train_results: dict, dis_train_results: dict, val_results: dict) -> None:
        """
        Logs and displays the training results.
        """

        tag_list = []
        value_list = []

        # append the generator train results to the tag and value lists
        for tag, value in gen_train_results:
            tag_list.append(tag)
            value_list.append(np.mean(value))

        # append the discriminator train results to the tag and value lists
        for tag, value in dis_train_results:
            tag_list.append(tag)
            value_list.append(np.mean(value))

        # extract the images from the val_results dict
        images = val_results.pop('images', None)

        # append the validation to the tag and value lists
        for tag, value in val_results:
            tag_list.append(tag)
            value_list.append(np.mean(value))
