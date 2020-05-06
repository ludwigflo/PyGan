from contextlib import contextmanager
from ruamel.yaml import YAML
import sys, os


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


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
