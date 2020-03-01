import numpy as np

from starter import *

# Section 1.1 Testing

x = np.array([-1, 15, -7, -3, 6])
W = np.array([[2, 3, 7, 5, 12], [6, 3, 1, 1, 9]])

b = np.array([1, 8])
r = computeLayer(x, W, b)

print("Original Array:", x)
print("Result:", r)
