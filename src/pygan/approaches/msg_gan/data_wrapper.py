import torch.nn.functional as F


def wrapper(data, size_tuple=(8, 16, 32)):
    """
    Data wrapper for Multi-Scale-Gradient-GANs, which wraps image data into a list of images of different scales.

    Parameters
    ----------
    data: Input data, which should be wrapped into the correct form for being used by the MSG Gan architecture.

    size_tuple: tuple, representing the image dimensions, to which the intermediate image representations should be
                resized. This is done, as for the msg gan approach, images are processed in multiple scales.

    """

    for i in range(len(size_tuple)):
        data_list.append(F.interpolate(data, size_tuple[i]))
    data_list.append(data)
    return data_list
