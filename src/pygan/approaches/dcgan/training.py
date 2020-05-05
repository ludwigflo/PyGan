from typing import Dict, Generator, DefaultDict, List, Union
from ...interfaces import GanTrainer, Gan
from collections import defaultdict
from ...utils import to_gpu, channel_first_to_last, normalize_img
import torch.nn as nn
import numpy as np
import torch
import sys


class Trainer(GanTrainer):
    def __init__(self, gan_model: Gan, train_data_generator: Generator[dict, None, None], criterion: nn.Module,
                 val_data_generator: Generator[dict, None, None], parameter_file: str, device):

        super(Trainer, self).__init__(gan_model, train_data_generator, parameter_file)
        self.val_data_generator = val_data_generator
        self.criterion = criterion()

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
        data_real = sample['data_real'].to(self.device)
        gen_input = sample['gen_input'].to(self.device)
        self.gan_model.discriminator.zero_grad()
        self.gan_model.generator.zero_grad()

        # compute the fake data by passing the noise through the generator
        data_fake = self.gan_model.generator(gen_input)

        # compute the targets (smooth labels)
        targets_real = torch.from_numpy(np.random.uniform(0.7, 1, data_real.size()[0])).float().view(-1, 1)
        targets_fake = torch.from_numpy(np.random.uniform(0, 0.3, data_fake.size()[0])).float().view(-1, 1)
        targets_real = targets_real.to(self.device).view(-1)
        targets_fake = targets_fake.to(self.device).view(-1)

        # compute the predictions
        pred_real = self.gan_model.discriminator(data_real).view(-1)
        pred_fake = self.gan_model.discriminator(data_fake).view(-1)

        # compute the loss and scores
        real_loss = self.criterion(pred_real.view(-1), targets_real.view(-1))
        fake_loss = self.criterion(pred_fake.view(-1), targets_fake.view(-1))

        # error backpropagation and optimizing step
        total_loss = real_loss + fake_loss
        total_loss.backward()
        self.optimizer_list[1].step()

        # prepare and return the results
        out_dict = {'fake_score': pred_fake.mean(0).data.item(),
                    'real_score': pred_real.mean(0).data.item(),
                    'real_loss': real_loss.data.item(),
                    'fake_loss': fake_loss.data.item(),
                    'loss': total_loss.data.item()}
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
        gen_input = sample['gen_input'].to(self.device)
        self.gan_model.discriminator.zero_grad()
        self.gan_model.generator.zero_grad()

        # compute the fake data by passing the noise through the generator
        data_fake = self.gan_model.generator(gen_input)

        # compute the targets (smooth labels)
        targets_fake = torch.ones(data_fake.size()[0]).float().view(-1, 1)
        targets_fake = targets_fake.to(self.device).view(-1)

        # compute the predictions
        pred_fake = self.gan_model.discriminator(data_fake).view(-1)

        # compute the loss and scores
        fake_loss = self.criterion(pred_fake, targets_fake)

        # error backpropagation and optimizing step
        fake_loss.backward()
        self.optimizer_list[0].step()

        # prepare and return the results
        out_dict = {'fake_score': pred_fake.mean(0).data.item(),
                    'loss': fake_loss.data.item()}
        return out_dict

    def val_epoch(self) -> DefaultDict[str, List]:
        """
        Validation epoch of the GAN.
        """

        val_results = defaultdict(list)
        with torch.no_grad():
            for count, (sample, done, num_val_data) in enumerate(self.val_data_generator):
                print_str = '\r\tValidation...    Iteration: {0}, ' \
                            'Progress: {1:3.2f} %'.format(count,float(count) / num_val_data * 100)
                sys.stdout.write(print_str)
                sys.stdout.flush()

                # get the input for the generator and the real sample
                data_real = sample['data_real'].to(self.device)
                gen_input = sample['gen_input'].to(self.device)

                # compute fake data
                data_fake = self.gan_model.generator(gen_input)

                # compute the targets
                # targets_real = torch.from_numpy(np.random.uniform(0.7, 1, data_real.size()[0])).float().view(-1, 1)
                # targets_fake = torch.from_numpy(np.random.uniform(0, 0.3, data_fake.size()[0])).float().view(-1, 1)
                targets_real = 0.7 * torch.ones(data_real.size()[0]).view(-1, 1)
                targets_fake = 0.3 * torch.ones(data_fake.size()[0]).view(-1, 1)
                targets_real = targets_real.to(self.device)
                targets_fake = targets_fake.to(self.device)

                # compute the predictions
                pred_real = self.gan_model.discriminator(data_real)
                pred_fake = self.gan_model.discriminator(data_fake)

                # compute the loss and scores
                loss_real = self.criterion(pred_real, targets_real)
                loss_fake = self.criterion(pred_fake, targets_fake)

                # compute the evaluation criteria
                val_results['fake_score'].append(pred_fake.mean(0).data.item())
                val_results['real_score'].append(pred_real.mean(0).data.item())
                val_results['fake_loss'].append(loss_fake.data.item())
                val_results['real_loss'].append(loss_real.data.item())

                if count < 100:
                    val_results['images'].append(data_fake.detach().cpu().numpy())

                if count==1000:
                    break

                if done:
                    break
        return val_results

    def log_results(self, experiment_dir: str, tb_logger_list: Union[None, list], epoch: int,
                    gen_train_results: dict, dis_train_results: dict, val_results: dict) -> None:
        """
        Logs and displays the training results.
        """

        # append the generator train results to the tag and value lists
        gen_tag_list = []
        gen_value_list = []
        for tag, value in gen_train_results.items():
            gen_tag_list.append(tag)
            gen_value_list.append(np.mean(value))

        # append the discriminator train results to the tag and value lists
        dis_tag_list = []
        dis_value_list = []
        for tag, value in dis_train_results.items():
            dis_tag_list.append(tag)
            dis_value_list.append(np.mean(value))

        # extract the images from the val_results dict
        images = val_results.pop('images', None)
        images = [channel_first_to_last(img) for img in images]
        images = [img[0:1, :, :, :] if img.shape[3]>1 else np.repeat(img[0:1, ...], 3, axis=3) for img in images]
        images = [normalize_img(img) for img in images]
        images = np.concatenate(images)
        print(images.shape)

        tb_img_dict = {'Samples': images}
        tb_logger_list[0].log_images('Generated', tb_img_dict, epoch)

        # append the validation to the tag and value lists
        val_tag_list = []
        val_value_list = []
        for tag, value in val_results.items():
            val_tag_list.append(tag)
            val_value_list.append(np.mean(value))

        print_str = '\r\tResults:\n\r\t\tTraining: Generator Loss {0:5.4f} Discriminator Loss {1:5.4f}\n\r\t\t' \
                    'Validation: Real Score {2:5.4f}  Fake Score {3:5.4f}'.format(gen_value_list[1], dis_value_list[4],
                                                                        val_value_list[1], val_value_list[0])
        sys.stdout.write(print_str)
        sys.stdout.flush()
        print()

        # generator logger, discriminator logger, validation logger
        tb_logger_list[0].log_scalar(gen_tag_list, gen_value_list, epoch)
        tb_logger_list[1].log_scalar(dis_tag_list, dis_value_list, epoch)
