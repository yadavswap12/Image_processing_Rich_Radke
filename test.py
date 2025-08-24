import numpy as np

# a = np.array([[[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]]])
a = np.array([[[1,1],[2,2],[3,3]], [[4,4],[5,5],[6,6]], [[7,7],[8,8],[9,9]]])
b = np.array([[2,2,2], [2,2,2], [2,2,2]])

print(a)
print(b.reshape(b.shape[0], b.shape[1], 1))
print(a*b.reshape(b.shape[0], b.shape[1], 1))

print(a.shape)
print(b.shape)
# print((a*b).shape)