import numpy as np

from nputils import to_array, absolute_change, sum_increase, sum_decrease, increase, decrease, peaks, bottoms


def test_to_numpy_array():

    def generator():
        for i in range(5):
            yield i

    assert np.array_equal(to_array(generator()), np.array([0, 1, 2, 3, 4]))
    assert np.array_equal(to_array(5), np.array([5]))
    assert np.array_equal(to_array("xy"), np.array(["xy"]))
    assert np.array_equal(to_array([5]), np.array([5]))
    assert np.array_equal(to_array(np.array([[5, 10], [10, 15]])), np.array([[5, 10], [10, 15]]))


def test_absolute_change():
    assert np.array_equal(absolute_change(np.array([-1, 2, -3, 2])), 13)
    assert np.array_equal(absolute_change(np.array([[-1, 2, -3, 2], [0, 0, 0, 0]])), np.array([13, 0]))


def test_sum_increase():
    assert sum_increase(np.array([5, 10, 15, 10, 0])) == 10
    assert np.array_equal(sum_increase(np.array([[5, 10, 15, 10, 0], [-5, 0, 5, 0, -5]])), np.array([10, 10]))


def test_sum_decrease():
    assert sum_decrease(np.array([5, 10, 15, 10, 0])) == 15
    assert np.array_equal(sum_decrease(np.array([[5, 10, 15, 10, 0], [-5, 0, 5, 0, -5]])), np.array([15, 10]))


def test_increase():
    assert np.array_equal(
        increase(np.array([1, 2, 3, 4, 3, 5]), n=1),
        np.array([False,  True,  True,  True, False,  True])
    )
    assert np.array_equal(
        increase(np.array([1, 2, 3, 4, 3, 5]), n=2),
        np.array([False, False,  True,  True, False, False])
    )
    assert np.array_equal(
        increase(np.array([[1, 2], [3, 4]])),
        np.array([[False,  True], [False,  True]])
    )


def test_decrease():
    assert np.array_equal(
        decrease(np.array([1, 2, 3, 2, 1, 5]), n=1),
        np.array([False, False, False,  True,  True, False])
    )
    assert np.array_equal(
        decrease(np.array([1, 2, 3, 2, 1, 5]), n=2),
        np.array([False, False, False, False,  True, False])
    )
    assert np.array_equal(
        decrease(np.array([1, 2, 3, 2, 1, 5]), n=3),
        np.array([False, False, False, False, False, False])
    )


def test_local_max():
    a = np.array([2, 3, 2, 4])
    assert np.array_equal(peaks(a), np.array([False, True, False, False]))
    assert np.array_equal(peaks(a, n=2), np.array([False, False, False, False]))

    b = np.array([1, 2, 3, 2, 4])
    assert np.array_equal(peaks(b), np.array([False, False, True, False, False]))
    assert np.array_equal(peaks(b, n=2), np.array([False, False, False, False, False]))

    c = np.array([1, 2, 3, 2, 1, 4])
    assert np.array_equal(peaks(c, n=2), np.array([False, False, True, False, False, False]))


def test_local_min():
    assert np.array_equal(bottoms(np.array([3, 2, 3, 4, 3])), np.array([False, True, False, False, False]))

