from pygan.ml_utils.src.ml_utils.data_utils.data_loader import ImgLoader
from pygan.approaches.msg_gan.data_wrapper import wrapper
from typing import Tuple, Union
import torch.nn.functional as F
import torch


def data_loader(data_loader: ImgLoader, batch_size: int, noise_size: Union[int, tuple] = 30,
                size_tuple: tuple = (4, 8, 14), train_data: bool = True) -> tuple:
    """
    Wraps a ImgLoader instance to make it usable by the MSGGan class.

    Parameters
    ----------
    data_loader: ImgLoader instance, which loads image data and provides data generators for train and validation data.
    batch_size: Batch size for the training data.
    noise_size: Size of the noise, which is fed into the MSG Gan generator in order to synthesize images.
    size_tuple: Image lengths, to which the images are rescaled (MSG Gan uses images at different sizes).
    train_data: Variable, which indicates whether to use train data of validation data.

    Returns
    -------
    output: Data samples, which can be used by the MSG Gan model for training and validation.
    done: (Optional) Variable, which indicates whether a complete validation (or test) epoch is finished or not.
    num_data: Total number of validation data.
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
