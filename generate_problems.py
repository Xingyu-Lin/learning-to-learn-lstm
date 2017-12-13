import numpy as np

problems_w = []
problems_b = []
for i in range(1):
  w = np.random.random((2, 2))
  b = np.random.random((2, 1))
  # w = 0.5 * np.eye(2)
  # b = 0.5 * np.ones(2)
  problems_w.append(w)
  problems_b.append(b)

w = [[0.5, 0], [0, 0.1]]
b = np.asarray([0.2, 0.2]).reshape(2, 1)
#assert(b.shape == problems_b[0].shape)
problems_w.append(w)
problems_b.append(b)

np.savez('./problems/quadratic.npz', problems_w, problems_b)
