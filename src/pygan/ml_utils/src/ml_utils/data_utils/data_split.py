import random


def simple_data_split(num_data: int, train_data: float, val_data: float, shuffle: bool = True) -> tuple:
    """
    Computes the indices for a training, validation and test split, based on the total number of data. The test data are
    the remaining data, which have not been assigned to training or validation data.

    Parameters
    ----------
    num_data: Total number of available data.
    train_data: Fraction of data, which should be used for training.
    val_data: Fraction of data, which should be used for validation.
    shuffle: Boolean, which indicates whether to shuffle the data or not.

    Returns
    -------
    split: tuple, in which lists of indices according to the splits are stored.
    """

    assert train_data + val_data <= 1, "The amount of training and validation data needs to be smaller ot equal to 1!"

    # create the indices, corresponding to the data points, and shuffle them, if required
    indices = list(range(num_data))
    if shuffle:
        random.shuffle(indices)

    # compute the amount of indices
    num_train_indices = round(train_data * num_data)
    num_val_indices = round(num_data * val_data)

    # split the indices into their corresponding lists
    train_indices = indices[:num_train_indices]
    val_indices = indices[num_train_indices:num_train_indices+num_val_indices]

    # if there are remaining data points, assign them to the test set
    if num_train_indices + num_val_indices < num_data:
        test_indices = indices[num_train_indices+num_val_indices:]
        split = (train_indices, val_indices, test_indices)
    else:
        split = (train_indices, val_indices)
    return split


def k_fold_cross_validation(k: int, num_data: int, shuffle: bool = True) -> tuple:
    """
    Splits a number of training data into k folds, which can be used for k-fold-cross-validation.
    Parameters
    ----------
    k: number of folds for cross validation.
    num_data: Total amount of data values.
    shuffle: Boolean variable, which indicates whether to shuffle the data or not.

    Returns
    -------
    split: tuple, in which lists of indices according to the splits are stored.
    """

    assert num_data >= k, "Total amount of data needs to be larger or equal to the number of folds!"

    # create indices, corresponding to the data points, and shuffle them, if required
    indices = list(range(num_data))
    if shuffle:
        random.shuffle(indices)

    # compute the sizes of the folds and the remaining number of data
    fold_size = int(num_data / k)
    remaining_data = num_data % k

    # compute the splits
    fold_list = []
    for i in range(k):
        fold_list.append(indices[i*fold_size:(i+1)*fold_size])

    # append the remaining data points to the folds
    for i in range(remaining_data):
        fold_list[i].append(indices[k*fold_size+i])
    split = tuple(fold_list)
    return split
