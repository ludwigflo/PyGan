from ..logging.tb_log import Logger
from typing import Union, List
from datetime import datetime
import shutil
import os



def create_directories(name: Union[List[str], str]) -> None:
    """
    Creates one or multiple directories. If a string value is provided, then the single directory is created. If a list
    is provided, then all directory names within the list are created.

    Parameters
    ----------
    name: A list, containing the names of directories, which should be created.
    """

    if type(name) == 'str':
        name = [name]

    for directory in name:
        if not os.path.exists(directory):
            os.makedirs(directory)


def copy_files(directory: str, files: Union[str, List[str]]) -> None:
    """
    Copies provided files into a provided directory.

    Parameters
    ----------
    directory: Path to the directory, in which the files should be stored.
    files: Files which should be copied into the provided directory.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    if type(files) == 'str':
        files = [files]

    for source_file in files:
        shutil.copy(source_file, directory)


def create_date_time_string() -> str:
    """
    Creates a string consisting of YYYY_MM_DD__hh_mm_ss, where Y represents numbers of the current year, M represents
    numbers of the current month, D represents numbers of the current day, h represents numbers of the current hour, m
    represents numbers of the current minute and s represents numbers of the current seconds.

    Returns
    -------
    now: Generated string from the current date and time.
    """

    now = str(datetime.now()).replace('-', '_').replace(':', '_').replace(' ', '__')[0:20]
    return now


def init_experiment_dir(dir_root, tb_log: bool = False, tb_log_names: Union[None, List[str]] = None) -> tuple:
    """
    Creates a new directory, in which results and logs of a new experiment can be stored.

    Parameters
    ----------
    dir_root: Path to a directory, in which the root directory for a new experiment is created.
    tb_log: Variable, which defines, whether to use Tensorboard or not. If the value of this variable is None, then the
            experiment is initialized without tensorboard. If the value of this variable is an integer, than this number
            of tb_logger instances is created.
    tb_log_names: Names of the directories of the Tensorboard logger (if the names are not None). Each directory is
                  created in the as sub-directory of the tb directory in the experience root path.

    Returns
    -------
    dir_name: Path to the experiment directory
    tb_logger_tuple: Tuple containing the tensorboard logger instances.
    """

    # get the name of the root directory
    dir_name = dir_root + create_date_time_string() + '/'
    source_file_dir = dir_name + 'src/'
    dir_list = [dir_name, source_file_dir]
    output = tuple(dir_name)

    # if we want to include tensorboard logging
    if tb_log:

        # get the name of the tensorboard directory
        tb_root_dir = dir_name + 'tb/'
        dir_list.append(tb_root_dir)

        # if multiple logging instances should be initialized, get the name of their directories
        if tb_log_names is not None:
            tb_logger_list = []
            for name in tb_log_names:

                # compute the directory of the logger instance
                logger_dir = tb_root_dir + name + '/'

                # create a logger instance and store the instance as well as the corresponding directory
                tb_logger_list.append(Logger(logger_dir))
                dir_list.append(logger_dir)
                output = (dir_name, tb_logger_list)

    # create the directories and return the root directory of the new created experiment
    create_directories(dir_list)
    return output
