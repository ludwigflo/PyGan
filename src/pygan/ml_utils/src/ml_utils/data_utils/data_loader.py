from .data_split import k_fold_cross_validation, data_split
from abc import ABC, abstractmethod
from typing import Union, List
import random
import pickle


class DataLoaderInterface(ABC):

    @abstractmethod
    def __init__(self, total_num_data: int, params: dict, shuffle_data: bool=True):
        """
        Constructor of the DataLoaderInterface Class.

        Parameters
        ----------
        total_num_data: Total number of data samples.
        params: Parameters, stored in form of a dictionary.
        shuffle_data: Variable, which determines whether to shuffle the data before splitting or not.
        """

        # get the parameters, which define the data split
        split = params['data_loader']['split']
        assert split in ['cross_validation', 'simple_split'], "Split must be 'cross_validation' or 'simple_split'"

        # compute and store the data split and whether cross validation was performed or not.
        if split == 'cross_validation':
            k = params['data_loader']['cross_validation']['k']
            self.split = k_fold_cross_validation(k, total_num_data, shuffle=shuffle_data)
            self.cross_val = True
        else:
            train_split = params['data_loader']['simple_split']['train_split']
            val_split = params['data_loader']['simple_split']['val_split']
            self.split = data_split(total_num_data, train_split, val_split, shuffle_data)
            self.cross_val = False

    @abstractmethod
    def load_data(self, index: Union[List[int], int]) -> tuple:
        """
        Abstract method for loading one Data sample, which is queried by its index.

        Parameters
        ----------
        index: Single index or list of indices of data samples, which should be loaded.

        Returns
        -------
        Data samples, together with their corresponding label.
        """
        raise NotImplementedError

    def train_generator(self, batch_size, val_fold: int = 1) -> tuple:
        """
        Generator for the train set. In case of K Fold Cross validation, the train_set is merged from the train folds.
        Parameters
        ----------
        batch_size: Size of each mini batch.
        val_fold: (Optional) Index of the validation fold. Only required if K Fold Cross Validation is used.

        Returns
        -------
        samples: Batch of training samples.
        """

        # get the training set
        if self.cross_val:
            train_set = []
            for i, data_set in enumerate(self.split):
                if i != val_fold:
                    train_set.extend(data_set)
        else:
            train_set = self.split[0]

        while True:
            samples = self.get_data_indices(True, train_set, batch_size, None)
            yield samples


    def val_generator(self, batch_size: int = 1, rand: bool = False, val_fold: int = 1) -> tuple:
        """

        Parameters
        ----------
        batch_size: Size of the mini_batch for each validation batch.
        rand: Whether to randomly sample the the validation samples.
        val_fold: Indicates, which fold is used for validation (for normal train- val test split the fold number  is 1).

        Returns
        -------
        samples: Batch of validation samples.
        done: Boolean variable, which determines, whether an complete validation epoch is done or not.
        """

        # get the test set and compute te number of iterations for one test epoch
        val_set = self.split[val_fold]
        num_iterations = int(len(val_set) / batch_size)
        num_remaining_samples = len(val_set) % batch_size

        # do as long as you want
        while True:
            # iterate through your validation data
            for iteration in range(num_iterations):
                samples = self.get_data_indices(rand, val_set, batch_size, iteration)
                done = (iteration+1)*batch_size == len(val_set)
                yield samples, done

            if num_remaining_samples > 0:
                if rand:
                    indices = random.choices(val_set, k = num_remaining_samples)
                else:
                    indices = val_set[num_iterations*batch_size:]
                samples = self.load_data(indices)
                yield samples, True

    def test_generator(self, batch_size: int = 1, rand: bool = False) -> tuple:
        """
        Generator for the test set.

        Parameters
        ----------
        batch_size: Number of samples per batch.
        rand: Boolean, which determines whether to sample the test samples randomly or not.

        Returns
        -------
        samples: Batch of test samples.
        done: Boolean variable, which determines, whether an complete test epoch is done or not.
        """

        # check if the parameters are correct
        assert not self.cross_val, "In case of cross validation, no test set is provided!"
        assert len(self.split) == 3, "Data was only split into train and validation set!"

        # get the test set and compute te number of iterations for one test epoch
        test_set = self.split[2]
        num_iterations = int(len(test_set) / batch_size)
        num_remaining_samples = len(test_set) % batch_size

        # do as long as you want
        while True:

            # iterate through your test data
            for iteration in range(num_iterations):
                samples = self.get_data_indices(rand, test_set, batch_size, iteration)
                done = (iteration+1)*batch_size == len(test_set)
                yield samples, done

            if num_remaining_samples > 0:
                if rand:
                    indices = random.choices(test_set, k=num_remaining_samples)
                else:
                    indices = test_set[num_iterations*batch_size:]
                samples = self.load_data(indices)
                yield samples, True

    def get_data_indices(self, rand: bool, data_set: list, batch_size: int, iteration: Union[None, int]) -> tuple:
        """

        Parameters
        ----------
        rand: Whether to randomly sample indices from the data list or not.
        data_set: List of data indices.
        batch_size: Number of samples per batch.
        iteration: Current state of the training iteration.

        Returns
        -------
        samples: Loaded data samples

        """

        if rand:
            indices = random.choices(data_set, k=batch_size)
        else:
            indices = data_set[iteration * batch_size: (iteration + 1) * batch_size]
        samples = self.load_data(indices)
        return samples

    def save_object(self, path: str) -> None:
        """
        Saves the current data loader object into a provided path.

        Parameters
        ----------
        path: Path, in which the DataLoader object should be saved.
        """

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_object(path: str) -> 'DataLoaderInterface':
        """
        Loads a DataLoader object, which is stored in a certain path.

        Parameters
        ----------
        path: Path, in which the DataLoader object is stored.

        Returns
        -------
        data_loader: DataLoader object.
        """

        with open(path, 'rb') as f:
            data_loader_object = pickle.load(f)
        return data_loader_object
