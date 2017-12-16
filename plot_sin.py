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


def f_sin(x, y, a, b, c):
  return np.sin(a * x + b) ** 7 + a * np.cos(b + y * x) * np.cos(c * x)


def f_quad(X1, X2, W, Y):
  assert (Y.shape == (2,))
  Z = np.ndarray(shape=X1.shape, dtype=X1.dtype)
  for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
      Z[i, j] = np.sum((W.dot(np.asarray([X1[i, j], X2[i, j]], dtype=X1.dtype)).T - Y) ** 2)
  return Z


def main():
  optimizers = ['L2L', 'Adam', 'Momentum', 'SGD', 'NAG', 'RMSProp']

  problem_path = './problems/sin.npy'
  problems = np.load(problem_path)

  prob_num = len(problems)
  x = {}
  obj = {}

  for optimizer in optimizers:
    x[optimizer] = np.load(osp.join('./results', optimizer + '.npy'))
  for prob_idx in range(prob_num):
    fig = plt.figure(figsize=(10, 6))
    a, b, c = problems[prob_idx]
    for optimizer in optimizers:
      obj[optimizer] = list(
        map(lambda x: f_sin(x[0], x[1], a, b, c), x[optimizer][prob_idx]))
      plt.plot(obj[optimizer], label=optimizer)
      print('Plotting: problem {}, optimizer {}, loss {}, x {}'.format(prob_idx, optimizer, obj[optimizer][-1],
                                                                       x[optimizer][prob_idx][-1]))
    plt.legend(loc='upper right')
    plt.xlabel('number of iterations')
    plt.ylabel('objective value')
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_yscale("log")
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

  n = len(optimizers)
  for prob_idx in range(prob_num):
    fig = plt.figure(figsize=(12, 8))
    for i, optimizer in enumerate(optimizers):
      a, b, c = problems[prob_idx]
      minx = np.min(x[optimizer][prob_idx][:, 0])
      miny = np.min(x[optimizer][prob_idx][:, 1])
      maxx = np.max(x[optimizer][prob_idx][:, 0])
      maxy = np.max(x[optimizer][prob_idx][:, 1])
      min_plot = min(minx, miny)
      max_plot = max(maxx, maxy)
      t_min = min_plot - 0.3 * (max_plot - min_plot)
      t_max = max_plot + 0.3 * (max_plot - min_plot)
      x1 = np.linspace(t_min, t_max, num=50)
      x2 = np.linspace(t_min, t_max, num=50)
      X1, X2 = np.meshgrid(x1, x2)
      Z = f_sin(X1, X2, a, b, c)
      ax = fig.add_subplot(2, (n + 1) / 2, i + 1)
      cf = ax.contourf(X1, X2, Z, 10)
      fig.colorbar(cf, ax=ax)
      ax.set_aspect('equal')
      ax.axis([t_min, t_max, t_min, t_max])
      ax.set_title(optimizer)
      ax.scatter(x[optimizer][prob_idx][:, 0], x[optimizer][prob_idx][:, 1], s=5, edgecolors='r', facecolors='none',
                 marker='o')
      ax.plot(x[optimizer][prob_idx][:, 0], x[optimizer][prob_idx][:, 1], color='r')
      # plt.colorbar()

      if (prob_idx ==1 or prob_idx ==3) and i==0:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig('./figs/fail_sin_{}.png'.format(prob_idx), bbox_inches=extent)
    plt.savefig('./figs/contour_{}.png'.format(prob_idx))



if __name__ == "__main__":
  main()
