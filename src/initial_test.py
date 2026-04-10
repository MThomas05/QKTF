import qktf
import numpy
import cupy as np
tensor = np.array([[1, 4], [2, 5], [3, 6]])
dim = numpy.array([3, 2])
mode = 1
qktf.fold(tensor, dim, mode)