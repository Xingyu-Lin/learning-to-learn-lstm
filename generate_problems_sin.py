import numpy as np

params= []
for i in range(5):
  a = np.random.random()
  b = np.random.random()
  c = np.random.random()
  param = [a,b,c]
  params.append(param)
np.save('./problems/sin', params)
