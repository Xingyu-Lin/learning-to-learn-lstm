import numpy as np

problems_w = []
problems_b = []
for i in range(5):
  w = np.random.random((2, 2))
  b = np.random.random((2, 1))
  # w = 0.5 * np.eye(2)
  # b = 0.5 * np.ones(2)
  problems_w.append(w)
  problems_b.append(b)

w = [[5, 0], [0, 1]]
b = np.zeros((2, 1))
problems_w.append(w)
problems_b.append(b)

np.savez('./problems/quadratic.npz', problems_w, problems_b)
