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
"""Learning 2 Learn evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms

import meta
import util
import numpy as np
import os

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS
flags.DEFINE_string("optimizer", "L2L", "Optimizer.")
flags.DEFINE_string("path", None, "Path to saved meta-optimizer network.")
flags.DEFINE_string('problem_path', None, 'Path of the generated problems')
flags.DEFINE_integer("num_epochs", 1, "Number of evaluation epochs.")
flags.DEFINE_integer("seed", None, "Seed for TensorFlow's RNG.")

flags.DEFINE_string("problem", "simple", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")


def main(_):
    # Configuration.
    num_unrolls = FLAGS.num_steps

    if FLAGS.seed:
        tf.set_random_seed(FLAGS.seed)

    # Problem.
    problem, net_config, net_assignments = util.get_config(FLAGS.problem,
                                                           FLAGS.path,
                                                           FLAGS.problem_path)

    # Optimizer setup.
    if FLAGS.optimizer == "Adam":
        cost_op = problem()
        problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        problem_reset = tf.variables_initializer(problem_vars)
        x_op = problem_vars

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
        update = optimizer.minimize(cost_op)
        reset = [problem_reset, optimizer_reset]

    elif FLAGS.optimizer == "SGD":
        cost_op = problem()
        problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        problem_reset = tf.variables_initializer(problem_vars)
        x_op = problem_vars

        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
        update = optimizer.minimize(cost_op)
        reset = [problem_reset, optimizer_reset]

    elif FLAGS.optimizer == "RMSProp":
        cost_op = problem()
        problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        problem_reset = tf.variables_initializer(problem_vars)
        x_op = problem_vars

        optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
        optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
        update = optimizer.minimize(cost_op)
        reset = [problem_reset, optimizer_reset]

    elif FLAGS.optimizer == "Momentum":
        cost_op = problem()
        problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        problem_reset = tf.variables_initializer(problem_vars)
        x_op = problem_vars

        optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, momentum=0.9)
        optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
        update = optimizer.minimize(cost_op)
        reset = [problem_reset, optimizer_reset]

    elif FLAGS.optimizer == "NAG":
        cost_op = problem()
        problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        problem_reset = tf.variables_initializer(problem_vars)
        x_op = problem_vars

        optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, momentum=0.1, use_nesterov=True)
        optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
        update = optimizer.minimize(cost_op)
        reset = [problem_reset, optimizer_reset]

    elif FLAGS.optimizer == "L2L":
        if FLAGS.path is None:
            logging.warning("Evaluating untrained L2L optimizer")
        optimizer = meta.MetaOptimizer(**net_config)
        meta_loss = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)
        _, update, reset, cost_op, x_op = meta_loss
    else:
        raise ValueError("{} is not a valid optimizer".format(FLAGS.optimizer))

    with ms.MonitoredSession() as sess:
        # Prevent accidental changes to the graph.
        tf.get_default_graph().finalize()

        total_time = 0
        total_cost = 0
        for _ in xrange(FLAGS.num_epochs):
            # Training.
            time, cost, x_values = util.run_epoch_test(sess, cost_op, x_op, [update], reset,
                                                       num_unrolls)
            total_time += time
            total_cost += cost

        x_values = np.swapaxes(np.squeeze(x_values), 0, 1)
        np.save(os.path.join('results', '{}'.format(FLAGS.optimizer)), x_values)

        #print("x_values shape: {}".format(x_values.shape))
        #print("x_values: {}".format(x_values))
        #np.savetxt(os.path.join('results', '{}.txt'.format(FLAGS.optimizer)), x_values, fmt='%f')
        # Results.
        #util.print_stats("Epoch {}".format(FLAGS.num_epochs), total_cost,
        #                  total_time, FLAGS.num_epochs)


if __name__ == "__main__":
    tf.app.run()
