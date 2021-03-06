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
"""Learning to learn (meta) optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import os

import mock
import sonnet as snt
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.util import nest

import networks


def _nested_assign(ref, value):
  """Returns a nested collection of TensorFlow assign operations.

  Args:
    ref: Nested collection of TensorFlow variables.
    value: Values to be assigned to the variables. Must have the same structure
        as `ref`.

  Returns:
    Nested collection (same structure as `ref`) of TensorFlow assign operations.

  Raises:
    ValueError: If `ref` and `values` have different structures.
  """
  if isinstance(ref, list) or isinstance(ref, tuple):
    if len(ref) != len(value):
      raise ValueError("ref and value have different lengths.")
    result = [_nested_assign(r, v) for r, v in zip(ref, value)]
    if isinstance(ref, tuple):
      return tuple(result)
    return result
  else:
    return tf.assign(ref, value)


def _nested_variable(init, name=None, trainable=False):
  """Returns a nested collection of TensorFlow variables.

  Args:
    init: Nested collection of TensorFlow initializers.
    name: Variable name.
    trainable: Make variables trainable (`False` by default).

  Returns:
    Nested collection (same structure as `init`) of TensorFlow variables.
  """
  if isinstance(init, list) or isinstance(init, tuple):
    result = [_nested_variable(i, name, trainable) for i in init]
    if isinstance(init, tuple):
      return tuple(result)
    return result
  else:
    return tf.Variable(init, name=name, trainable=trainable)


def _wrap_variable_creation(func, custom_getter):
  """Provides a custom getter for all variable creations."""
  original_get_variable = tf.get_variable

  def custom_get_variable(*args, **kwargs):
    if hasattr(kwargs, "custom_getter"):
      raise AttributeError("Custom getters are not supported for optimizee "
                           "variables.")
    return original_get_variable(*args, custom_getter=custom_getter, **kwargs)

  # Mock the get_variable method.
  with mock.patch("tensorflow.get_variable", custom_get_variable):
    return func()


def _get_variables(func):
  """Calls func, returning any variables created, but ignoring its return value.

  Args:
    func: Function to be called.

  Returns:
    A tuple (variables, constants) where the first element is a list of
    trainable variables and the second is the non-trainable variables.
  """
  variables = []
  constants = []

  def custom_getter(getter, name, **kwargs):
    trainable = kwargs["trainable"]
    kwargs["trainable"] = False
    variable = getter(name, **kwargs)
    if trainable:
      variables.append(variable)
    else:
      constants.append(variable)
    return variable

  with tf.name_scope("unused_graph"):
    _wrap_variable_creation(func, custom_getter)

  return variables, constants


def _make_with_custom_variables(func, variables):
  """Calls func and replaces any trainable variables.

  This returns the output of func, but whenever `get_variable` is called it
  will replace any trainable variables with the tensors in `variables`, in the
  same order. Non-trainable variables will re-use any variables already
  created.

  Args:
    func: Function to be called.
    variables: A list of tensors replacing the trainable variables.

  Returns:
    The return value of func is returned.
  """
  variables = collections.deque(variables)

  def custom_getter(getter, name, **kwargs):
    if kwargs["trainable"]:
      return variables.popleft()
    else:
      kwargs["reuse"] = True
      return getter(name, **kwargs)

  return _wrap_variable_creation(func, custom_getter)


MetaLoss = collections.namedtuple("MetaLoss", "loss, update, reset, fx, x")
MetaStep = collections.namedtuple("MetaStep", "step, update, reset, fx, x")


def _make_nets(variables, config, net_assignments):
  """Creates the optimizer networks.

  Args:
    variables: A list of variables to be optimized.
    config: A dictionary of network configurations, each of which will be
        passed to networks.Factory to construct a single optimizer net.
    net_assignments: A list of tuples where each tuple is of the form (netid,
        variable_names) and is used to assign variables to networks. netid must
        be a key in config.

  Returns:
    A tuple (nets, keys, subsets) where nets is a dictionary of created
    optimizer nets such that the net with key keys[i] should be applied to the
    subset of variables listed in subsets[i].

  Raises:
    ValueError: If net_assignments is None and the configuration defines more
        than one network.
  """
  # create a dictionary which maps a variable name to its index within the
  # list of variables.
  name_to_index = dict((v.name.split(":")[0], i)
                       for i, v in enumerate(variables))

  if net_assignments is None:
    if len(config) != 1:
      raise ValueError("Default net_assignments can only be used if there is "
                       "a single net config.")

    with tf.variable_scope("vars_optimizer"):
      #  elif problem_name == "quadratic-wav":
      #    problem = problems.quadratic(batch_size=1, num_dims=2)
      #    net_config = {"cw-wav": {
      #        "net": "CoordinateWiseWaveNet",
      #        "net_options": {"num_layers": 4},
      #        "net_path": get_net_path("cw-wav", path)
      #    }}
      key = next(iter(config))
      kwargs = config[key]
      # Return; CoordinateWiseWaveNet(num_layers=4)
      net = networks.factory(**kwargs)

    nets = {key: net}
    keys = [key]
    subsets = [range(len(variables))]
  else:
    nets = {}
    keys = []
    subsets = []
    with tf.variable_scope("vars_optimizer"):
      for key, names in net_assignments:
        if key in nets:
          raise ValueError("Repeated netid in net_assigments.")
        nets[key] = networks.factory(**config[key])
        subset = [name_to_index[name] for name in names]
        keys.append(key)
        subsets.append(subset)
        print("Net: {}, Subset: {}".format(key, subset))

  # subsets should be a list of disjoint subsets (as lists!) of the variables
  # and nets should be a list of networks to apply to each subset.
  return nets, keys, subsets


class MetaOptimizer(object):
  """Learning to learn (meta) optimizer.

  Optimizer which has an internal RNN which takes as input, at each iteration,
  the gradient of the function being minimized and returns a step direction.
  This optimizer can then itself be optimized to learn optimization on a set of
  tasks.
  """
  """Wavenet meta optimizer

  Optimizer has a wavenet, takes as input the gradient of the optimizee, return 
  a step direction.
  """

  def __init__(self, **kwargs):
    """Creates a MetaOptimizer.

    Args:
      **kwargs: net_config defined in util.py 
      **kwargs: A set of keyword arguments mapping network identifiers (the
          keys) to parameters that will be passed to networks.Factory (see docs
          for more info).  These can be used to assign different optimizee
          parameters to different optimizers (see net_assignments in the
          meta_loss method).
    """
    self._nets = None

    if not kwargs:
      # Use a default coordinatewise network if nothing is given. this allows
      # for no network spec and no assignments.
      self._config = {
        "coordinatewise": {
          "net": "CoordinateWiseDeepLSTM",
          "net_options": {
            "layers": (20, 20),
            "preprocess_name": "LogAndSign",
            "preprocess_options": {"k": 5},
            "scale": 0.01,
          }}}
    else:
      self._config = kwargs

  def save(self, sess, path=None):
    """Save meta-optimizer."""
    result = {}
    for k, net in self._nets.items():
      if path is None:
        filename = None
        key = k
      else:
        filename = os.path.join(path, "{}.l2l".format(k))
        key = filename
      net_vars = networks.save(net, sess, filename=filename)
      result[key] = net_vars
    return result

  def meta_loss(self,
                make_loss,
                len_unroll,
                net_assignments=None,
                second_derivatives=False):
    """Returns an operator computing the meta-loss.

    Args:
      make_loss: Callable which returns the optimizee loss; note that this
          should create its ops in the default graph.
      len_unroll: Number of steps to unroll.
      net_assignments: variable to optimizer mapping. If not None, it should be
          a list of (k, names) tuples, where k is a valid key in the kwargs
          passed at at construction time and names is a list of variable names.
      second_derivatives: Use second derivatives (default is false).

    Returns:
      namedtuple containing (loss, update, reset, fx, x)
    """

    # Construct an instance of the problem only to grab the variables. This
    # loss will never be evaluated.
    x, constants = _get_variables(make_loss)

    print("Optimizee variables")
    print([op.name for op in x])
    print("Problem variables")
    print([op.name for op in constants])

    # Create the optimizer networks and find the subsets of variables to assign
    # to each optimizer.
    # nets = {cw-wav: CoordinateWiseWaveNet}
    # keys = [cw-wav]
    # subsets = [range(len(variables))]
    nets, net_keys, subsets = _make_nets(x, self._config, net_assignments)

    # Store the networks so we can save them later.
    self._nets = nets

    # Create input queues for each subset of variables.
    # One element in input_queues essentially

    if 'wav' in net_keys[0]:
      using_wavenet = True
    else:
      using_wavenet = False
    # state stands for hidden states in LSTM,
    # state stands for input_queues in wavenet
    state = []
    # input_queues[0]: a list of length batch_size, each element is an initialized input_queue.
    with tf.name_scope("state"):
      for i, (subset, key) in enumerate(zip(subsets, net_keys)):
        net = nets[key]
        with tf.name_scope("state_{}".format(i)):
          if using_wavenet:
            state.append(_nested_variable(
            [net.initial_state_for_inputs(x[j])
             for j in subset],
            name="state", trainable=False))
          else:
            state.append(_nested_variable(
              [net.initial_state_for_inputs(x[j], dtype=tf.float32)
               for j in subset],
              name="state", trainable=False))

    def update(net, fx, x, state):
      """
        Parameter and RNN state/WaveNet input_queue update.
      """
      with tf.name_scope("gradients"):
        gradients = tf.gradients(fx, x)

        # Stopping the gradient here corresponds to what was done in the
        # original L2L NIPS submission. However it looks like things like
        # BatchNorm, etc. don't support second-derivatives so we still need
        # this term.
        if not second_derivatives:
          gradients = [tf.stop_gradient(g) for g in gradients]
          # gradients[0].get_shape() == x.get_shape() == batch_size x num_dims

      with tf.name_scope("deltas"):
        deltas, state_next = zip(*[net(g, s) for g, s in zip(gradients, state)])
        state_next = list(state_next)

      return deltas, state_next

    def time_step(t, fx_array, x, state):
      # compute fx according to x and write fx to fx_array
      # compute delta and input_queue_next by calling update()
      # compute x_next = x + delta, returned updated fx_array, x, and input_queue
      """While loop body."""
      x_next = list(x)
      state_next = []

      with tf.name_scope("fx"):
        fx = _make_with_custom_variables(make_loss, x)
        fx_array = fx_array.write(t, fx)

      with tf.name_scope("dx"):
        for subset, key, s_i in zip(subsets, net_keys, state):
          x_i = [x[j] for j in subset]
          deltas, s_i_next = update(nets[key], fx, x_i, s_i)

          for idx, j in enumerate(subset):
            x_next[j] += deltas[idx]
          state_next.append(s_i_next)

      with tf.name_scope("t_next"):
        t_next = t + 1

      return t_next, fx_array, x_next, state_next

    # Define the while loop.
    fx_array = tf.TensorArray(tf.float32, size=len_unroll + 1,
                              clear_after_read=False)
    _, fx_array, x_final, s_final = tf.while_loop(
        cond=lambda t, *_: t < len_unroll,
        body=time_step,
        loop_vars=(0, fx_array, x, state),
        parallel_iterations=1,
        swap_memory=True,
        name="unroll")

    with tf.name_scope("fx"):
      fx_final = _make_with_custom_variables(make_loss, x_final)
      fx_array = fx_array.write(len_unroll, fx_final)

    # sum of fx for len_unroll rollouts
    # this is the objective for metalearner
    loss = tf.reduce_sum(fx_array.stack(), name="loss")

    # Reset the input_queue; should be called at the beginning of an epoch.
    with tf.name_scope("reset"):
      variables = (nest.flatten(state) +
                   x + constants)
      # Empty array as part of the reset process.
      reset = [tf.variables_initializer(variables), fx_array.close()]

    # Operator to update the parameters and the RNN state/WaveNet input_queues after our loop, but
    # during an epoch.
    with tf.name_scope("update"):
      update = (nest.flatten(_nested_assign(x, x_final)) +
                nest.flatten(_nested_assign(state, s_final)))

    # Log internal variables.
    for k, net in nets.items():
      print("Optimizer '{}' variables".format(k))
      print([op.name for op in snt.get_variables_in_module(net)])

    return MetaLoss(loss, update, reset, fx_final, x_final)

  # meta_minimize
  #  minimize = optimizer.meta_minimize(
  #      problem, FLAGS.unroll_length (20),
  #      learning_rate=FLAGS.learning_rate,
  #      net_assignments=net_assignments,
  #      second_derivatives=FLAGS.second_derivatives)
  #  step, update, reset, cost_op, _ = minimize
  def meta_minimize(self, make_loss, len_unroll, learning_rate=0.01, **kwargs):
    """Returns an operator minimizing the meta-loss.

    Args:
      make_loss: Callable which returns the optimizee loss; note that this
          should create its ops in the default graph.
      len_unroll: Number of steps to unroll.
      learning_rate: Learning rate for the Adam optimizer.
      **kwargs: keyword arguments forwarded to meta_loss.

    Returns:
      namedtuple containing (step, update, reset, fx, x)
    """
    # TOOD: update info with wavenet archi
    info = self.meta_loss(make_loss, len_unroll, **kwargs)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    step = optimizer.minimize(info.loss)
    return MetaStep(step, *info[1:])
