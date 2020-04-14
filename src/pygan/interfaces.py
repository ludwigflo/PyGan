from .utils import read_parameter_file
from abc import ABC, abstractmethod
from torch.optim import Adam
from typing import Generator
from . import ml_utils
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


class GanDiscriminator(ABC):
    """
    Abstract class for standard GAN discriminator.
    """

    @abstractmethod
    def discriminate_data(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forwards data through the discriminator and predicts, whether the data is real or synthetic.

        Parameters
        ----------
        data: Real or fake data, which should be discriminated.

        Returns
        -------
        pred: Prediction confidence for input samples (confidence of being real).
        """
        raise NotImplementedError

class Gan(ABC):



    @property
    @abstractmethod
    def generator(self):
        """
        Gan Generator.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def discriminator(self):
        """
        Gan Discriminator.
        """
        raise NotImplementedError

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


# TODO: print training progress
# TODO: create function train_epoch and move the training code there
# TODO: create function validation epoch and implement function for validation
# TODO: implement function for printing and logging the progress
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
        self.optimizer_list = []

    @abstractmethod
    def discriminator_iteration(self) -> dict:
        """
        Defines one training iteration of the discriminator module.

        Returns
        -------
        A dictionary with the results of the training.
        """
        raise NotImplementedError

    @abstractmethod
    def generator_iteration(self) -> dict:
        """
        Defines one training iteration of the discriminator module.

        Returns
        -------
        A dictionary with the results of the training.
        """
        raise NotImplementedError

    @staticmethod
    def init_experiment(params: dict) -> tuple:
        """
        Initializes a new experiment by creating an experiment directory and (optional) initializing tensorboard
        loggers.

        Parameters
        ----------
        params: Dictionary containing the settings for the experiment initialization.

        Returns
        -------
        initialization: Tuple containing experiment directory and (optional) a list of tensorboard loggers.
        """

        # get the experiment settings
        dir_root = params['experiment']['path']
        tb_log = params['experiment']['tb_log']
        if tb_log:
            tb_log_names = ['generator', 'discriminator']
        else:
            tb_log_names = None

        # perform the training
        initialization = ml_utils.init_experiment_dir(dir_root, tb_log, tb_log_names)
        return initialization

    def prepare_optimizer(self, params: dict) -> None:
        """
        Prepares the optimizer for the gan training.

        Parameters
        ----------
        params: Dictionary containing the parameters needed for the optimizers.

        Returns
        -------
        optimizer_gen: Pytorch optimizer for the generator of the gan.
        optimizer_dis: Pytorch optimizer for the discriminator of the gan.
        """

        lr_gen = params['training']['lr_gen']
        lr_dis = params['training']['lr_dis']

        optimizer_gen = Adam(self.gan_model.generator.parameters(), lr = lr_gen, betas = (0.9, 0.999))
        optimizer_dis = Adam(self.gan_model.generator.parameters(), lr = lr_dis, betas = (0.9, 0.999))
        self.optimizer_list = [optimizer_gen, optimizer_dis]

    def train(self) -> None:
        """
        Performs and supervises the training.
        """

        # read the parameter file and check whether the required parameters are defined
        params = read_parameter_file(self.parameter_file)
        assert 'training' in params, "Parameter file needs to contain the key 'training', " \
                                     "in which the training configurations are defined "
        assert 'experiment' in params, "Parameter file needs to contain the key 'experiment', " \
                                       "in which the experiment configurations are defined "

        # extract the training settings
        num_iterations = params['training']['num_iterations']
        num_dis_iter = params['training']['num_dis_iter']
        num_gen_iter = params['training']['num_gen_iter']
        num_epochs = params['training']['num_epochs']
        k_dis = params['training']['k_dis']
        k_gen = params['training']['k_gen']

        # initialize a new experiment
        init_experiment = self.init_experiment(params)
        experiment_path = init_experiment[0]
        if len(init_experiment) > 1:
            tb_logger_list = init_experiment[1]
        else:
            tb_logger_list = None

        # create the optimizer
        self.prepare_optimizer(params)

        # perform training
        for epoch in range(num_epochs):

            epoch_results = []
            for iteration in range(num_iterations):

                # train the discriminator
                dis_results = []
                for k in range(k_dis):
                    results = self.discriminator_iteration()
                    dis_results.append(results)

                # train the discriminator
                gen_results = []
                for k in range(k_gen):
                    results = self.generator_iteration()
                    gen_results.append(results)

            epoch_results.append((dis_results, gen_results))
