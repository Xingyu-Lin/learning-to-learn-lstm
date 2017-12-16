# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Learning 2 Learn utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from timeit import default_timer as timer

import numpy as np
from six.moves import xrange

import problems


def run_epoch(sess, cost_op, x_op, ops, reset, num_unrolls):
  """Runs one optimization epoch."""
  start = timer()
  sess.run(reset)
  x_value_list = []
  for _ in xrange(num_unrolls):
    cost, x_value = sess.run([cost_op, x_op] + ops)[0:2]
    x_value_list.append(x_value)
  x_values = np.array(x_value_list)
  return timer() - start, cost, x_values


# def run_epoch_test(sess, cost_op, x_op, ops, reset, num_unrolls):
#   """Runs one optimization epoch."""
#   start = timer()
#   sess.run(reset)
#   x_value_list = [sess.run(x_op)]
#   for _ in xrange(num_unrolls):
#     cost, x_value = sess.run([cost_op, x_op] + ops)[0:2]
#     x_value_list.append(x_value)
#   x_values = np.array(x_value_list)
#   return timer() - start, cost, x_values


def print_stats(header, total_error, total_time, n):
  """Prints experiment statistics."""
  print(header)
  print("Mean Final Error: {:.2f}".format(total_error / n))
  # print("Log Mean Final Error: {:.2f}".format(np.log10(total_error / n)))
  print("Mean epoch time: {:.2f} s".format(total_time / n))


def get_net_path(name, path):
  return None if path is None else os.path.join(path, name + ".l2l")


def get_default_net_config(name, path):
  return {
    "net": "CoordinateWiseDeepLSTM",
    "net_options": {
      "layers": (20, 20),
      "preprocess_name": "LogAndSign",
      "preprocess_options": {"k": 5},
      "scale": 0.01,
    },
    "net_path": get_net_path(name, path)
  }


def get_config(problem_name, path=None, problem_path=None):
  """Returns problem configuration."""
  if problem_name == "simple":
    problem = problems.simple()
    net_config = {"cw": {
      "net": "CoordinateWiseDeepLSTM",
      "net_options": {"layers": (), "initializer": "zeros"},
      "net_path": get_net_path("cw", path)
    }}
    net_assignments = None
  elif problem_name == "simple-multi":
    problem = problems.simple_multi_optimizer()
    net_config = {
      "cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (), "initializer": "zeros"},
        "net_path": get_net_path("cw", path)
      },
      "adam": {
        "net": "Adam",
        "net_options": {"learning_rate": 0.1}
      }
    }
    net_assignments = [("cw", ["x_0"]), ("adam", ["x_1"])]
  elif problem_name == "quadratic":
    if problem_path is not None:
      npzfile = np.load(problem_path)
      problems_w, problems_b = npzfile['arr_0'], npzfile['arr_1']
      assert len(problems_w) == len(problems_b)
      batch_size = len(problems_w)
      problem = problems.quadratic(batch_size=batch_size, num_dims=2, problems_w=problems_w,
                                   problems_b=problems_b)
    else:
      problem = problems.quadratic(batch_size=128, num_dims=2)
    net_config = {"cw": {
      "net": "CoordinateWiseDeepLSTM",
      "net_options": {"layers": (20, 20)},
      "net_path": get_net_path("cw", path)
    }}
    net_assignments = None
  elif problem_name == "quadratic-wav":
    if problem_path is not None:
      npzfile = np.load(problem_path)
      problems_w, problems_b = npzfile['arr_0'], npzfile['arr_1']
      assert len(problems_w) == len(problems_b)
      batch_size = len(problems_w)
      problem = problems.quadratic(batch_size=batch_size, num_dims=2, problems_w=problems_w,
                                   problems_b=problems_b)
    else:
      problem = problems.quadratic(batch_size=128, num_dims=2)
    net_config = {"cw-wav": {
      "net": "CoordinateWiseWaveNet",
      "net_options": {"num_layers": 4},
      "net_path": get_net_path("cw-wav", path)
    }}
    net_assignments = None
  elif problem_name == "sin":
    if problem_path is not None:
      problems_sin = np.load(problem_path)  # TODO

      batch_size = len(problems_sin)
      problem = problems.prob_sin(batch_size=batch_size, problem_param=problems_sin)
    else:
      problem = problems.prob_sin(batch_size=128)
    net_config = {"cw": {
      "net": "CoordinateWiseDeepLSTM",
      "net_options": {"layers": (20, 20)},
      "net_path": get_net_path("cw", path)
    }}
    net_assignments = None
  elif problem_name == "sin-wav":
    if problem_path is not None:
      problems_sin = np.load(problem_path)  # TODO

      batch_size = len(problems_sin)
      problem = problems.prob_sin(batch_size=batch_size, problem_param=problems_sin)
    else:
      problem = problems.prob_sin(batch_size=128)
    net_config = {"cw-wav": {
      "net": "CoordinateWiseWaveNet",
      "net_options": {"num_layers": 4},
      "net_path": get_net_path("cw-wav", path)
    }}
    net_assignments = None

  elif problem_name == "mnist":
    mode = "train" if path is None else "test"
    problem = problems.mnist(layers=(20,), mode=mode)
    net_config = {"cw": get_default_net_config("cw", path)}
    net_assignments = None
  elif problem_name == "cifar":
    mode = "train" if path is None else "test"
    problem = problems.cifar10("cifar10",
                               conv_channels=(16, 16, 16),
                               linear_layers=(32,),
                               mode=mode)
    net_config = {"cw": get_default_net_config("cw", path)}
    net_assignments = None
  elif problem_name == "cifar-multi":
    mode = "train" if path is None else "test"
    problem = problems.cifar10("cifar10",
                               conv_channels=(16, 16, 16),
                               linear_layers=(32,),
                               mode=mode)
    net_config = {
      "conv": get_default_net_config("conv", path),
      "fc": get_default_net_config("fc", path)
    }
    conv_vars = ["conv_net_2d/conv_2d_{}/w".format(i) for i in xrange(3)]
    fc_vars = ["conv_net_2d/conv_2d_{}/b".format(i) for i in xrange(3)]
    fc_vars += ["conv_net_2d/batch_norm_{}/beta".format(i) for i in xrange(3)]
    fc_vars += ["mlp/linear_{}/w".format(i) for i in xrange(2)]
    fc_vars += ["mlp/linear_{}/b".format(i) for i in xrange(2)]
    fc_vars += ["mlp/batch_norm/beta"]
    net_assignments = [("conv", conv_vars), ("fc", fc_vars)]
  else:
    raise ValueError("{} is not a valid problem".format(problem_name))

  return problem, net_config, net_assignments
