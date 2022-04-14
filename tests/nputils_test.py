import numpy as np

from nputils import to_array


def test_to_numpy_array():

    def generator():
        for i in range(5):
            yield i

    assert np.array_equal(to_array(generator()), np.array([0, 1, 2, 3, 4]))
    assert np.array_equal(to_array(5), np.array([5]))
    assert np.array_equal(to_array("xy"), np.array(["xy"]))
    assert np.array_equal(to_array([5]), np.array([5]))
    assert np.array_equal(to_array(np.array([[5, 10], [10, 15]])), np.array([[5, 10], [10, 15]]))
