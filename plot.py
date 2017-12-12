# Created by Xingyu Lin, 11/12/2017                                                                                  

# coding: utf-8


import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from numpy import linalg as LA
import os
from os import path as osp
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib


def f(x, y):
  return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


def main():
  optimizers = ['L2L', 'Adam', 'Momentum', 'SGD', 'NAG', 'RMSProp']

  problem_path = './problems/quadratic.npz'
  npzfile = np.load(problem_path)
  problems_w, problems_b = npzfile['arr_0'], npzfile['arr_1']
  prob_num = len(problems_w)
  x = {}
  obj = {}

  for optimizer in optimizers:
    x[optimizer] = np.load(osp.join('./results', optimizer + '.npy'))

  for prob_idx in range(prob_num):
    plt.figure(figsize=(10, 6))
    for optimizer in optimizers:
      obj[optimizer] = list(
        map(lambda x: LA.norm(problems_w[prob_idx].dot(x) - np.transpose(problems_b[prob_idx])) ** 2, x[optimizer][prob_idx]))
      plt.plot(obj[optimizer], label=optimizer)
    plt.legend(loc='upper right')
    plt.xlabel('number of iterations')
    plt.ylabel('objective value')
    plt.savefig('./figs/loss_prob_{}.png'.format(prob_idx))

    # X = {}  # dictionary to store data
    # obj = {}  # dictionary to store obj value
    # W = 0.5 * np.eye(2)
    # Y = 0.5 * np.ones(2)
    # for name in names:
    #     X[name] = np.loadtxt(name + '.txt')
    #     obj[name] = list(map(lambda x: LA.norm(W.dot(x) - Y) ** 2, X[name]))
    #     plt.plot(obj[name], label=name)
    # plt.legend(loc='upper right')
    # plt.xlabel('number of iterations')
    # plt.ylabel('objective value')
    #

    # # plot level set and trajectory for SGD
    # W = 0.5 * np.eye(2)
    # Y = 0.5 * np.ones(2)
  optimizer = 'L2L'
  for prob_idx in range(prob_num):
    W = problems_w[prob_idx]
    Y = problems_b[prob_idx]
    A = W.transpose().dot(W)
    b = 2 * W.transpose().dot(Y)
    delta = 0.001
    minx = np.min(x[optimizer][prob_idx][:, 0])
    miny = np.min(x[optimizer][prob_idx][:, 1])
    maxx = np.max(x[optimizer][prob_idx][:, 0])
    maxy = np.max(x[optimizer][prob_idx][:, 1])
    min_plot = min(minx, miny)
    max_plot = max(maxx, maxy)
    t_min = min_plot - 0.3 * (max_plot - min_plot)
    t_max = max_plot + 0.3 * (max_plot - min_plot)
    x1 = np.arange(t_min, t_max, delta)
    x2 = np.arange(t_min, t_max, delta)
    X1, X2 = np.meshgrid(x1, x2)
    F = A[0, 0] * X1 ** 2 + A[1, 1] * X2 ** 2 + (A[0, 1] + A[1, 0]) * X1 * X2 - b[0] * X1 - b[1] * X2 + LA.norm(
      Y) ** 2
    plt.figure()
    plt.contourf(X1, X2, F, 100, cmap='RdGy')
    plt.colorbar()
    plt.axes().set_aspect('equal')
    plt.axis([t_min, t_max, t_min, t_max])

    plt.scatter(x[optimizer][prob_idx][:, 0], x[optimizer][prob_idx][:, 1], s=30, edgecolors='g', facecolors='none',
                marker='o')
    plt.plot(x[optimizer][prob_idx][:, 0], x[optimizer][prob_idx][:, 1], color='g')
    plt.savefig('./figs/contour_{}.png'.format(prob_idx))


if __name__ == "__main__":
  main()
