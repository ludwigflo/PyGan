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


def data_loader(data_path: str, data_name: str, batch_size: int, rand: bool = False,
                noise_size: Union[int, tuple] = 30, fix_noise: bool = False, size_tuple: tuple = (4, 8, 14)):
    """

    Parameters
    ----------
    data_path
    data_name
    batch_size
    rand
    noise_size
    fix_noise
    size_tuple

    Returns
    -------

    """

    # load the data and get the data size
    data, _ = load_dataset(data_path, data_name)
    num_data_samples = data.size()[0]

    if fix_noise:
        if type(noise_size) == tuple:
            fix_noise_value = torch.randn((num_data_samples, *noise_size))
        else:
            fix_noise_value = torch.randn((num_data_samples, noise_size))
    else:
        # get the size of the random generator noise
        if type(noise_size) == tuple:
            noise_size = (batch_size, *noise_size)
        else:
            noise_size = (batch_size, noise_size)

    # reshape the data into the correct shape
    data = data.view(num_data_samples, 1, 28, 28).float()

    # initialize the indices, if we don't sample them randomly
    index = 0

    while True:
        #  get a noise sample
        if fix_noise:
            noise = fix_noise_value[index:index+batch_size, ...]
        else:
            noise = torch.randn(noise_size)

        # get the next real data according ot the provided specifications
        if rand:
            indices = torch.randperm(num_data_samples)[:batch_size]
            samples = data[indices]
            data_list = wrapper(samples)

            output = {'data_real': data_list, 'gen_input': noise}
            yield output
        else:
            if index+batch_size < num_data_samples:
                samples = data[index:index+batch_size]
                done = False
                index += batch_size

            else:
                samples = data[index:num_data_samples]
                done = True
                index = 0

            data_list = []
            # interpolate data and resize them to multiple resolutions
            for i in range(len(size_tuple)):
                data_list.append(F.interpolate(samples, size_tuple[i]))
            data_list.append(samples)

            output = {'data_real': data_list, 'gen_input': noise}
            yield output, done, num_data_samples
