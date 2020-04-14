from .utils import read_parameter_file
from abc import ABC, abstractmethod
from typing import Generator
import torch.nn as nn
import torch


class GanGenerator(ABC):
    """
    Abstract class for standard GAN generators, which aim to convert input noise into synthetic data.
    """

    @abstractmethod
    def generate_data_dict(self, noise: torch.Tensor, *args, **kwargs) -> tuple:
        """
        Method, which synthesizes data given an input noise vector. The synthesized data is stored in a dictionary.

        Parameters
        ----------
            noise: Noise tensor which is used by the generator in order to synthesize data.

        Returns
        -------
            Dictionary, in which generated data samples are stored.
        """
        raise NotImplementedError

class CondGenerator(ABC):
    """
    Abstract class for conditional GAN generators. These models aim to convert an input noise vector to an output, which
    is conditioned on additional information.
    """

    @abstractmethod
    def compute_noise_features(self, noise: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes a latent representation from an input noise tensor.

        Parameters
        ----------
            noise: Noise tensor which is used by the generator for synthesizing data.

        Returns
        -------
            Latent space representation of the noise vector.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_cond_features(self, cond_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes a latent representation from an input, which represents information on which the output should be
        conditioned.

        Parameters
        ----------
            cond_tensor: Represents information on which the output should be conditioned.

        Returns
        -------
            Latent space representation of the conditional tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_data_dict(self, noise: torch.Tensor, cond_tensor: torch.Tensor, *args, **kwargs) -> dict:
        """
        Method, which synthesizes data given an input noise vector and a tensor, on which the output should be
        conditioned. The synthesized data is stored in a dictionary.

        Parameters
        ----------
            noise: Noise tensor which is used by the generator in order to synthesize data.
            cond_tensor: Data, on which the generated output should be conditioned.

        Returns
        -------
            Dictionary, in which generated data samples are stored.
        """
        raise NotImplementedError


class CondDiscriminator(ABC):
    """
    Abstract class for conditional GAN discriminators. These models aim to classify input data into real or synthetic
    data. Conditional discriminator use additional information for their decision which means, they condition their
    decision on additional information like class labels.
    """

    @abstractmethod
    def compute_input_features(self, input_data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes a latent representation from an input data tensor.

        Parameters
        ----------
            input_data: Data tensor, which is classified into real or synthesized data.

        Returns
        -------
            Latent space representation of the data vector.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_cond_features(self, cond_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes a latent representation from an conditional input (e.g.class labels).

        Parameters
        ----------
            cond_tensor: Represents information on which the final decision should be conditioned (e.g. class labels).

        Returns
        -------
            Latent representation of the conditional tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_data(self, noise: torch.Tensor, cond_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Method, which predicts whether provide data is real or synthetic. This method also utilizes additional data,
        which is used to condition its prediction on.

        Parameters
        ----------
            noise: Noise tensor which is classified by the discriminator into real or fake data.
            cond_tensor: Data, on which the classification decision is conditioned.

        Returns
        -------
            Dictionary, in which generated data samples are stored.
        """
        raise NotImplementedError


class Gan(ABC):

    @abstractmethod
    def generate_data(self, noise: torch.Tensor, *args, **kwargs) -> dict:
        """
        Synthesizes new data samples (based on an input noise tensor) and returns them in a consistent way with
        dictionaries. The noise is forwarded through the generator with a self.forward(noise) call.

        Parameters
        ----------
            noise: input, which is fed into the generator to create synthetic data.

        Returns
        -------
            Dictionary, in which synthetic samples are stored.
        """
        raise NotImplementedError

    @abstractmethod
    def discriminate_data(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forwards data through the discriminator (self.discriminator(data) call) and predicts, whether the data is real
        or synthetic.

        Parameters
        ----------
            data: Real or fake data, which should be discriminated.

        Returns
        -------
            pred: Prediction confidence for input samples (confidence of being real).
        """
        raise NotImplementedError

class CondGan(ABC):

    @abstractmethod
    def generate_data(self, noise: torch.Tensor, cond_tensor: torch.Tensor, *args, **kwargs) -> dict:
        """
        Synthesizes new data samples (based on an input noise tensor) and returns them in a consistent way with
        dictionaries. The noise is forwarded through the generator with a self.forward(noise) call.

        Parameters
        ----------
            noise: input, which is fed into the generator to create synthetic data.
            cond_tensor: Represents information on which generated data should be conditioned (e.g. class labels).
        Returns
        -------
            Dictionary, in which synthetic samples are stored.
        """
        raise NotImplementedError

    @abstractmethod
    def discriminate_data(self, data: torch.Tensor, cond_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forwards data through the discriminator (self.discriminator(data) call) and predicts, whether the data is real
        or synthetic.

        Parameters
        ----------
            data: Real or fake data, which should be discriminated.
            cond_tensor: Data, on which the classification decision is conditioned.

        Returns
        -------
            pred: Prediction confidence for input samples (confidence of being real).
        """
        raise NotImplementedError


class GanTrainer(ABC):
    def __init__(self, gan_model: Gan, data_generator: Generator[dict], parameter_file: str) -> None:
        """
        Constructor of the GanTrainer class.

        Parameters
        ----------
        gan_model: Gan model, which should be trained.
        data_generator: Data generator, which provides data samples in form of dictionaries.
        parameter_file: Dictionary of parameters which define the training procedure.
        """

        self.gan_model = gan_model
        self.data_generator = data_generator
        self.parameter_file = parameter_file

    @abstractmethod
    def discriminator_iteration(self):
        """
        Defines one training iteration of the discriminator module.
        """
        raise NotImplementedError

    @abstractmethod
    def generator_iteration(self):
        """
        Defines one training iteration of the discriminator module.
        """
        raise NotImplementedError

    def train(self) -> None:
        """
        Performs and supervises the training.
        """

        # read the parameter file and check whether the required parameters are defined
        params = read_parameter_file(self.parameter_file)
        assert 'training' in params, "Parameter file needs to contain the key 'training', " \
                                     "in which the training configurations are defined "
        # extract the training settings
        num_dis_iter = params['training']['num_dis_iter']
        num_gen_iter = params['training']['num_gen_iter']
        num_epochs = params['training']['num_epochs']
        lr_gen = params['training']['lr_gen']
        lr_dis = params['training']['lr_dis']
        k_dis = params['training']['k_dis']
        k_gen = params['training']['k_gen']

        # perform the training
