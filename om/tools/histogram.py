import numpy as np
from collections import Counter

from om.models.model import DiscreteDistributionModel


class ArrayHistogram(DiscreteDistributionModel):
    def __init__(self):
        self.histogram = Counter()
        self.shape = None
        self.dtype = None
        self.all_entries_count = 0

    def add_array(self, array):
        """
        Adds a numpy array to the histogram, increasing its count.

        Args:
        - array (np.ndarray): The numpy array to add to the histogram.
        """
        if self.shape is None:
            self.shape = array.shape
            self.dtype = array.dtype
        else:
            assert array.shape == self.shape
            assert array.dtype == self.dtype

        array_bytes = array.tobytes()
        self.histogram[array_bytes] += 1
        self.all_entries_count += 1

    def calc_normalization_factor(self):
        return self.all_entries_count

    def calc_unnormalized_prob(self, array):
        """
        Gets the count of a numpy array in the histogram. That is, just counts

        Args:
        - array (np.ndarray): The numpy array to query in the histogram.

        Returns:
        - count (int): The count of the array in the histogram.
        """
        array_bytes = array.tobytes()
        return self.histogram[array_bytes]

    def calc_neg_log_unnormalized_prob(self, x):
        return -np.log(self.calc_unnormalized_prob(x))

    def generate_all_states(self):
        """
        Gets all unique arrays currently in the histogram.

        Returns:
        - unique_arrays (List[np.ndarray]): A list of unique numpy arrays in the histogram.
        """
        return [np.frombuffer(key, dtype=self.dtype).reshape(self.shape) for key in self.histogram.keys()]

    def __repr__(self):
        result = "ArrayHistogram:\n"
        for key, count in self.histogram.items():
            array = np.frombuffer(key, dtype=self.dtype).reshape(self.shape)
            result += f"Array:\t{array}\t {count}\n"
        return result


if __name__ == "__main__":
    hist = ArrayHistogram()

    array1 = np.array([[1, -1], [1, -1]])
    array2 = np.array([[1, -1], [1, -1]])
    array3 = np.array([[-1, -1], [1, 1]])

    hist.add_array(array1)
    hist.add_array(array1)
    hist.add_array(array1)
    hist.add_array(array2)
    hist.add_array(array3)

    print(hist)

    print("Count of array1:", hist.calc_unnormalized_prob(array1))
    print("Count of array2:", hist.calc_unnormalized_prob(array2))
    print("Count of array3:", hist.calc_unnormalized_prob(array3))

    unique_arrays = hist.generate_all_states()
    print("Unique arrays in the histogram:")
    for arr in unique_arrays:
        print(arr)
