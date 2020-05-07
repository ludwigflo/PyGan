from pygan.ml_utils.src.ml_utils.data_utils.data_loader import ImgLoader
from pygan.approaches.msg_gan.data_wrapper import wrapper
from typing import Tuple, Union
import torch.nn.functional as F
import torch


def load_dataset(data_path: str, data_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads a data file, which is provided by its path and name, and returns a tuple for torch.Tensors, representing data
    data and labels.

    Parameters
    ----------
    data_path: path of the data file.
    data_name: Name of the data file.

    Returns
    -------
    data: Data tensor.
    labels: Label tensor.
    """

    with open(data_path + data_name, 'rb') as f:
        data, labels = torch.load(f)
    return data, labels


def data_loader(data_loader: ImgLoader, batch_size: int, noise_size: Union[int, tuple] = 30,
                size_tuple: tuple = (4, 8, 14), train_data: bool = True):
    """
    TODO: initialize the data loader and wrap it in order to make it usable by the MSG gan.
    """

    if train_data:
        # get the train data generator of the ImgLoader instance
        data_generator = data_loader.train_generator(batch_size=batch_size)

        # get the size of the random train_generator noise
        if type(noise_size) == tuple:
            train_noise_size = (batch_size, *noise_size)
        else:
            train_noise_size = (batch_size, noise_size)
    else:
        # get the validation data generator of the ImgLoader instance
        data_generator = data_loader.val_generator(batch_size=1)

        # fix the generator input noise value for validation purposes
        if type(noise_size) == tuple:
            fix_noise_value = torch.randn((num_data_samples, *noise_size))
        else:
            fix_noise_value = torch.randn((num_data_samples, noise_size))

    # get samples of the data generator
    for data in data_generator:
        if training:
            samples = data
            noise = torch.randn(noise_size)
        else:
            samples = data[0]
            done = samples[1]
            num_data = samples[2]
            noise = fix_noise_value[index:index+batch_size, ...]

        # create a list of images, as required for the msg GAN
        data_list = wrapper(samples, size_tuple)
        output = {'data_real': data_list, 'gen_input': noise}

        if train_data:
            yield output
        else:
            yield output, done, num_data
