import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Dict
from io import BytesIO
import numpy as np


# noinspection PyTypeChecker,PyUnresolvedReferences
class Logger(object):
    """
    Logging in tensorboard without tensorflow ops.
    TODO: find data type of matplotlib color map
    """

    def __init__(self, log_dir: str, cmap=None) -> None:
        """
        Creates a Logger Object, which logs into path log_dir.

        Parameters
        ----------
        log_dir: Path, in which the Logger should store its logs.
        cmap:
        """

        self.writer = tf.summary.create_file_writer(log_dir)
        self.cmap = cmap

    def log_scalar(self, tags: list, values: list, step: int) -> None:
        """
        Logs a list of scalar variable with a list of their corresponding tags to tensorboard.

        Parameters
        ----------
        tags : List of strings, which represent the names of the scalar values.
        values: List of corresponding scalar value which should be logged.
        step : Logging step.
        """

        # iterate through the tags and values
        for i, tag in enumerate(tags):

            value = values[i]

            with self.writer.as_default():
                tf.summary.scalar(tag, value, step)

        # write the scalar values
        self.writer.flush()

    def log_images(self, tag: str, images: Dict[str, np.ndarray], step: int) -> None:
        """
        Logs a dictionary, which maps image names to images, to tensorboard.

        Parameters
        ----------
        step : Logging iteration.
        tag : Base name of the images.
        images: Dictionary, containing images. The keys in the dictionary are the image names, the corresponding values
                are the images.
        """

        for key, img in images.items():
            with self.writer.as_default():
                for i in range(img.shape[0]):
                    img_act = img[i:i+1, ...]
                    tf.summary.image(key + '/' + str(i), img_act, step)
        self.writer.flush()

    def log_histogram(self, values: np.ndarray, tag: str, step: int, bins: int = 1000) -> None:
        """
        Computes and logs a histogram based on a numpy.ndarray of values.

        Parameters
        ----------
        values: Array of values, for which the histogram should be computed and logged.
        tag: Name, under which the histogram should be displayed in Tensorboard.
        step: Logging iteration.
        bins: Number of bins, in which the histogram should be divided
        """
        # reshape the array ad compute the histogram values
        values = values.reshape(-1)
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1], thus, we drop the start of
        # the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
