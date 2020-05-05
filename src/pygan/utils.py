from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from imageio  import imread
from typing import Union
import numpy as np
import torch
import os


def resize2d(img: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Resizes an input image to a desired size.

    Parameters
    ----------
    img: Input image, which should be resized.
    size: Two dimensional size, to which the image will be resized.

    Returns
    -------
    img: Resized output image.
    """

    img = F.adaptive_avg_pool2d(Variable(img, volatile=True), size)
    return img


def read_parameter_file(parameter_file_path: str) -> dict:
    """
    Reads the parameters from a yaml file into a dictionary.

    Parameters
    ----------
    parameter_file_path: path to a parameter file, which is stored as a yaml file.

    Returns
    -------
    params: Dictionary containing the parameters defined in the provided yam file
    """

    yaml = YAML()
    with open(parameter_file_path, 'r') as f:
        params = yaml.load(f)
    return params


def normalize_img(img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalizes an input image.

    Parameters
    ----------
    img: Image which should be normalized.

    Returns
    -------
    img: Normalized image.
    """

    if type(img) is np.ndarray:
        img = img.astype(np.float32)
    elif type(img) is torch.Tensor:
        img = img.float()
    else:
        assert False, 'Data Type of image not understood.'

    img -= img.min()
    img /= img.max()
    return img


def load_img(img_path: str, img_size: tuple) -> torch.Tensor:
    """
    Loads, normalizes and resizes an image given an image path and a image size.

    Parameters
    ----------
    img_path: Path to the image.
    img_size: Two dimensional size, to which the image should be resized.

    Returns
    -------
    img: Normalized and resized image.
    """

    img = imread(img_path).astype(np.float32)
    img = normalize_img(img)
    img = torch.from_numpy(img)
    img = resize2d(img, img_size)
    return img


def add_channels(imgs: torch.Tensor, num_channels: int=3) -> torch.Tensor:
    """
    Converts an Input image with one channel to a multi-channel input image.

    Parameters
    ----------
    imgs: Input images, for which the channels should be repeated.
    num_channels: Target number of channels.

    Returns
    -------
    imgs: Images, for which the channels are repeated.
    """

    imgs = imgs.repeat(1, num_channels, 1, 1)
    return imgs


def channel_first_to_last(img: torch.Tensor) -> torch.Tensor:
    """
    Converts an image of shape batch_size, channels, height, width into an image of shape batch_size, height, width,
    channels.

    Parameters
    ----------
    img: Image with shape batch_size, channels, height, width.

    Returns
    -------
    img_out: Image with shape batch_size, height, width, channels
    """

    if type(img) == np.ndarray:
        batch, channels, x, y = img.shape
        img_out = np.zeros((batch, x, y, channels)).astype(np.float32)
    else:
        batch, channels, x, y = img.size()
        img_out = torch.zeros(batch, x, y, channels).float()
    for i in range(channels):
        img_out[:, :, :, i] = img[:, i, ...]
    return img_out


def channel_last_to_first(img: torch.Tensor) -> torch.Tensor:
    """
    Converts an image of shape batch_size, height, width, channels into an image of shape batch_size, channels, height,
    width.

    Parameters
    ----------
    img: Image with shape batch_size, height, width, channels.

    Returns
    -------
    img_out: Image with shape batch_size, channels, height, width.
    """

    if type(img) == np.ndarray:
        batch, x, y, channels = img.shape
        img_out = np.zeros((batch, channels, x, y)).astype(np.float32)
    else:
        batch, x, y, channels = img.size()
        img_out = torch.zeros(batch, channels, x, y).float()
    for i in range(channels):
        img_out[:, i, ...] = img[:, :, :, i]
    return img_out


def save_img_dict(img_dict: dict, output_path: str, cmap: str='viridis') -> None:
    """
    Saves the images, which are contained in an dictionary, as image data in the provided directory.

    Parameters
    ----------
    img_dict: Dictionary with image names as keys and corresponding image arrays as values.
    output_path: Path to the directory in which the images should be stored.
    cmap: matplotlib colormap, which is used to plot the images.
    """

    # create the directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save the images, which are stored in the dictionary
    for key in img_dict:
        plt.imsave(output_path+key, img_dict[key].detach().cpu().numpy(), cmap=cmap)
        plt.close()


def one_hot(indices: torch.Tensor, num_classes: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of indices into a batch of one-hot encoded vectors.

    Parameters
    ----------
    indices: Batch of class indices (size [N]), which should be one-hot encoded.
    num_classes: Maximal number of classes in the dataset.

    Returns
    -------
    one_hot_vec: One-hot encoded labels.
    """

    batch_size = indices.size()[0]
    one_hot_vec = torch.zeros(batch_size, num_classes).float()
    for i in range(batch_size):
        one_hot_vec[i, int(indices[i])] = 1
    return one_hot_vec


def truncated_normal(max_val: int, size: tuple) -> torch.Tensor:
    """
    Samples from a truncated Gaussian distribution. The values are sampled within a range of min_val, and max_val. In
    this implementation, the box-muller trick is applied.

    Parameters
    ----------
    max_val: Maximal value, which can be reached by the samples.
    size: Size of the noise vector

    Returns
    -------
    z: Truncated samples from a Gaussian distribution N(0, 1).
    """

    u1 = torch.rand(size) * (1-np.exp(-(max_val**2)/2)) + np.exp(-(max_val**2)/2)
    u2 = torch.rand(size)
    z = torch.sqrt(-2*torch.log(u1)) * torch.cos(2*np.pi*u2)
    return z


def compute_angle(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Computes the angle between two vectors p1 and p2.

    Parameters
    ----------
    p1: Vector.
    p2: Vector.

    Returns
    -------
    phi: Angle between both provide vectors (01 and p2).
    """

    # normalize the input vectors to unit norm
    p1 = p1.reshape(-1)
    p2 = p2.reshape(-1)
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)

    # compute the angle between the vectors
    inner_product = np.dot(p1, p2)

    if inner_product > 0.9995:
        phi = 0
    else:
        phi = np.arccos(inner_product)
    return phi


def update_dictionaries(final_dict: dict, new_dict: dict) -> dict:
    """
    Merges two dictionaries by including key/value pairs or adding values to corresponding keys from a dictionary
    (new_dict) to a target dictionary (final_dict).

    Parameters
    ----------
    final_dict: Dictionary, which should be updated.
    new_dict: Dictionary, from which the update values are obtained.

    Returns
    -------
    final_dict: Updated dictionary.
    """

    for key in new_dict:
        if key in final_dict:
            final_dict[key] += new_dict[key]
        else:
            final_dict[key] = new_dict[key]
    return final_dict


def to_gpu(*args: Union[torch.Tensor, torch.nn.Module, list]) -> tuple:
    """
    Moves torch.Tensors and torch.nn.Modules to the GPU (by calling the .cuda() function). This function can also handle
    lists of tensors or lists of torch.nn.Modules.

    Parameters
    ----------
    args: Arguments, which provide torch.Tensors or torch.nn.Modules.

    Returns
    -------
    out: Tuple containing to provided arguments, which now run on the GPU.
    """

    out = []
    for x in args:
        if type(x) is list:
            x = [el.cuda() for el in x]
            out.append(x)
        else:
            out.append(x.cuda())
    out = tuple(out)
    return out
