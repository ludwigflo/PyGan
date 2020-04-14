import numpy as np


def standard_statistics(data: np.ndarray) -> dict:
    """
    Compute most important basic statistics and store them in a list.

    Parameters
    ---------
    data: Data for which the statistics should be computed.

    Returns
    -------
    stats: Dictionary containing scalar values for median, mean, standard deviation, variance, min value and max value
           (keys: 'median', 'mean', 'std', 'var', 'min' and 'max')
    """

    stats = dict()
    stats['max_abs'] = np.asscalar(np.max(np.abs(data)))
    stats['min_abs'] = np.asscalar(np.min(np.abs(data)))
    stats['median'] = np.asscalar(np.median(data))
    stats['mean'] = np.asscalar(np.mean(data))
    stats['std'] = np.asscalar(np.std(data))
    stats['var'] = np.asscalar(np.var(data))
    stats['min'] = np.asscalar(np.min(data))
    stats['max'] = np.asscalar(np.max(data))
    return stats