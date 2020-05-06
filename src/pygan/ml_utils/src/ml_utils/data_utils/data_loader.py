from .data_split import k_fold_cross_validation, data_split
from scipy.misc import imread, imresize
from abc import ABC, abstractmethod
from typing import Union, List
from copy import deepcopy
import numpy as np
import random
import pickle
import os


class DataLoaderInterface(ABC):

    def __init__(self, total_num_data: Union[list, int], params: dict, shuffle_data: bool=True):
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
            self.split = self.return_kfold_split(total_num_data, shuffle_data, k)
            self.cross_val = True
        else:
            train_split = params['data_loader']['simple_split']['train_split']
            val_split = params['data_loader']['simple_split']['val_split']
            split = return_simple_data_split(total_num_data, train_split, val_split, shuffle_data)
            self.cross_val = False

    @staticmethod
    def return_kfold_split(total_num_data: Union[list, int], shuffle_data: bool, k: int):
        """
        Computes a k fold data split.

        Parameters
        ----------
        total_num_data: Total number of data (if the type is a list, then its the total number of data per label).
        shuffle_data: Whether to shuffle the data randomly or not.
        k: number of folds.

        Returns
        -------
        split: K-Fold data split.
        """

        if type(total_num_data) is int:

            # split temp is a tuple of lists (the different folds), where each list contains indices of the samples
            split_temp =  k_fold_cross_validation(k, total_num_data, shuffle=shuffle_data)

            # the split needs to contain lists of lists of indices, where the first level of lists represents the kind
            # of split (in this case train, validation or test). As we don't provide different labels (total_num_data is
            # an integer and not a list of integers), we only append one second level list in order to provide the
            # correct form. This list finally contains the indices of the data samples.
            split_list = [[x] for x in split_temp]
            split = tuple(split)

        else:
            split = [k_fold_cross_validation(k, num_data, shuffle=shuffle_data) for num_data in total_num_data]

            fold_lists = [[] for _ in range(k)]
            for index_tuple in split:
                for i, entries in enumerate(index_tuple):
                    fold_lists[i].append(entries)
            split = tuple(fold_lists)
        return split

    @staticmethod
    def return_simple_data_split(total_num_data: Union[list, int], train_split: float,
                                 val_split: float, shuffle_data: bool) -> tuple:
        """
        Computes a simple data split.

        Parameters
        ----------
        total_num_data: Total number of data (if the type is a list, then its the total number of data per label).
        train_split: Percentage of train data compared to the total number of data.
        val_split: Percentage of validation data compared to the total number of data.
        shuffle_data: Whether to shuffle the data randomly or not.

        Returns
        -------
        split: Train / Validation / (Optional) Test split.
        """

        # if we just split into train and test indices, then we just create a normal simple data split
        if type(total_num_data) is int:

            # split temp is a tuple of lists (train/validation/test), where each list contains indices of the samples
            split_temp = simple_data_split(total_num_data, train_split, val_split, shuffle_data)

            # the split needs to contain lists of lists of indices, where the first level of lists represents the kind
            # of split (in this case train, validation or test). As we don't provide different labels (total_num_data is
            # an integer and not a list of integers), we only append one second level list in order to provide the
            # correct form. This list finally contains the indices of the data samples.
            split_list = [[x] for x in split_temp]
            split = tuple(split)

        # if we just split each label into train and test indices, then we need to iterate over the class labels
        else:
            split = [simple_data_split(num_data, train_split, val_split, shuffle_data) for num_data in total_num_data]

            if train_split + val_split < 1:
                # store all train indices and test indices together
                train_list, val_list, test_list = [], [], []
                for (train_indices, val_indices, test_indices) in split:
                    train_list.append(train_indices)
                    val_list.append(val_indices)
                    test_list.append(test_indices)
                split = (train_list, val_list, test_list)
            else:
                # store all train indices and test indices together
                train_list, val_list = [], []
                for (train_indices, val_indices) in split:
                    train_list.append(train_indices)
                    val_list.append(val_indices)
                split = (train_list, val_list)
        return split

    @abstractmethod
    def load_data(self, index: Union[List[int], int], label: Union[List[int], int] = 0) -> object:
        """
        Abstract method for loading data samples, which are queried by their indices.

        Parameters
        ----------
        index: Single index or list of indices of data samples, which should be loaded.
        label: If the data split is performed for multiple labels, select the label from which the data is loaded.

        Returns
        -------
        Data samples
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
            labels = [random.randint(0, len(train_set)) for _ in range(batch_size)]
            indices = [random.choices(train_set[label], k=1) for label in labels]
            samples = self.load_data(indices, labels)
            yield samples

    def val_generator(self, batch_size: int = 1, rand: bool = False, val_fold: int = 1) -> tuple:
        """
        Generator for the validation set.

        Parameters
        ----------
        batch_size: Size of the mini_batch for each validation batch.
        rand: Whether to randomly sample the the validation samples.
        val_fold: Indicates, which fold is used for validation (for normal train- val test split the fold number is 1).

        Returns
        -------
        samples: Batch of validation samples.
        done: Boolean variable, which determines, whether an complete validation epoch is done or not.
        """

        # get the test set and compute te number of iterations for one test epoch
        val_set = self.split[val_fold]

        # get a flatten list of indices and corresponding labels
        indices = []
        labels = []
        for i, label_list in enumerate(val_set):
            indices.extend(deepcopy(label_list))
            indices.extend([i] * len(label_list))

        # do as long as you want
        iteration = 0
        while True:

            # compute the boarders of the batches
            max_size = (iteration + 1) * batch_size if (iteration + 1) * batch_size < len(indices) else len(indices)

            # choose the current indices
            cur_indices = indices[iteration * batch_size: max_size]
            cur_labels = indices[iteration * batch_size: max_size]

            # load the data and check, if a complete evaluation epoch has been performed
            samples = self.load_data(cur_indices, cur_labels)
            done = True if max_size == len(indices) else False

            iteration = iteration + 1 if (not done) else 0
            yield samples, done

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
        # get a flatten list of indices and corresponding labels
        indices = []
        labels = []
        for i, label_list in enumerate(test_set):
            indices.extend(deepcopy(label_list))
            indices.extend([i] * len(label_list))

        # do as long as you want
        iteration = 0
        while True:

            # compute the boarders of the batches
            max_size = (iteration + 1) * batch_size if (iteration + 1) * batch_size < len(indices) else len(indices)

            # choose the current indices
            cur_indices = indices[iteration * batch_size: max_size]
            cur_labels = indices[iteration * batch_size: max_size]

            # load the data and check, if a complete evaluation epoch has been performed
            samples = self.load_data(cur_indices, cur_labels)
            done = True if max_size == len(indices) else False

            iteration = iteration + 1 if (not done) else 0
            yield samples, done

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


class ImgLoader(DataLoaderInterface):
    def __init__(self, params: dict, img_size: tuple=(3, 224, 224), shuffle_data: bool=True, return_labels: bool=False):
        """
        Constructor of the ImgLoader class.

        Parameters
        ----------
        params: Parameters, which define hof to load the data.
        img_size: Size of the images, in channels first manner.
        shuffle_data: Variable, which indicates, whether to shuffle the data or not.
        """

        self.img_size = list(img_size)
        self.return_labels = return_labels
        self.base_img_folder = params['data']['base_path']
        self.images, self.label_folders, self.total_num_data =  self.get_img_names(self.base_img_folder)
        super().__init__(self.total_num_data, params, shuffle_data)

    @staticmethod
    def get_img_names(base_path: str) -> tuple:
        """
        Loads the names of the images, given a base directory. If the base directory only contains images, then no
        different labels are provided to the data. If the base directory contains sub-folders, then the folder names are
        taken as labels, and the content of the sub-folders is interpreted as image data. The sub-folder names are the
        interpreted as labels of the images.

        Parameters
        ----------
        base_path: Path, in which the images or/and subfolders are stored.

        Returns
        -------
        images: List os lists, which are interpreted as labels. The label lists contain image names.
        label_folders: List of names of the label sub-folders.
        total_num_data: List of integers containing the number of data per label.
        """

        # load all files and folders in the base directory
        files = os.listdir(base_path)

        # keep all sub-folders of the base directory
        label_folders = [sub_folder for sub_folder in files if os.path.isdir(sub_folder)]

        # if no sub-folder is in the base directory, then the base folder contains all of the data
        if len(data_folders) == 0:
            label_folders = [base_path]

        # get the image names and the number of image data per label
        images = []
        total_num_data = []
        for label_folder in label_folders:
            files = os.listdir(base_path + label_folder)
            images.append(files)
            total_num_data.append(len(files))
        return images, label_folders, total_num_data

    def channels_last2channels_first(self, img: np.ndarray) -> np.ndarray:
        """
        Converts an image, which is stored in channels last manner into an image, which ist stored in channels first manner.

        Parameters
        ----------
        img: Image, stored in channels last manner.

        Returns
        -------
        img_out: Image, stored in channels first manner.
        """

        img_out = np.zeros(self.img_size)
        for i in range(img.shape[2]):
            img_out[i, ...] = img[:, :, i]
        return img_out


    def load_img(self, img_path_list: list) -> np.ndarray:
        """
        Loads images based on their directories.

        Parameters
        ----------
        img_path_list: List of image directories, which should be loaded.

        Returns
        -------
        images: Images in form of numpy.ndarrays, stored in channels first manner.
        """

        # initialize the image batch
        num_images = len(img_path_list)
        shape = [num_images]
        shape.extend(self.img_size)
        images = np.zeros(shape)

        # compute the normal shape (target image size in channels last manner)
        normal_shape = [self.img_size[1], self.img_size[2]]
        if self.img_size[0] > 1:
            normal_shape.append(self.img_size[0])
        normal_shape = tuple(normal_shape)

        # load and store all images in the image_path  list
        for i, img_path in enumerate(img_path_list):

            # load the current image and resize it to the target shape
            img = imresize(imread(img_path), normal_shape)

            # store the image in channels first manner
            images[i] = channels_last2channels_first(img)

        return images

    def load_data(self, img_indices: Union[List[int], int],
                  label_indices: Union[List[int], int] = 0) -> Union[tuple, np.ndarray]:
        """
        Loads image data based on their index and label.

        Parameters
        ----------
        img_indices: Position(s) of the image name in self.images list, given the corresponding label.
        label_indices: Label, under which the image index can be found in the self.images list.

        Returns
        -------

        """

        # check if the provided index and labels are correct
        assert type(img_indices) == type(label_indices), 'Provided indices and labels need to have the same data type!'
        if type(img_indices) == list:
            assert len(img_indices) == len(label_indices), 'Provided indices and labels need to have the same length!'

        # convert the index and label variable into the correct format, if needed
        if type(img_indices) == int:
            img_indices = [img_indices]
            label_indices = [label_indices]

        # compute the image paths corresponding to the label and image indices
        img_path_list = []
        for (img_index, label_index) in zip(img_indices, label_indices):
            label_folder = self.label_folders[label_index]
            image_name = self.images[label_index][img_index]
            img_path_list.append(self.base_img_folder + label_folder + image_name)

        # load the images
        images = self.load_img(img_path_list)

        # return the images and label names if needed
        if self.return_labels:
            label_names = []
            for label_index in label_indices:
                label_names.append(self.label_folders[label_index])
            return images, labels

        # else return only the images
        else:
            return images
