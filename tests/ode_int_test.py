from ode_solvers import solvers
import unittest
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

DTYPE = tf.float64


class test_ode:
    def __call__(self, val, t):
        def y(val):
            return -val[0] + 3 * val[1]

        def x(val):
            return -2 * val[0] + 4 * val[1]

        retval = tf.convert_to_tensor([[y(val), x(val)]])
        return retval


class simple_ode:
    def __call__(self, val, t):
        return tf.convert_to_tensor([val])

class SystemTest(unittest.TestCase):
    def test(self):
        def y(t):
            return 3 * tf.exp(t) - tf.exp(2 * t)

        def x(t):
            return 2 * tf.exp(t) - tf.exp(2 * t)

        method = solvers.dopri
        t = tf.convert_to_tensor(np.array([0, 1]), dtype=DTYPE)
        solver = method(
            test_ode(),
            tf.convert_to_tensor([2, 1], dtype=DTYPE),
            t,
            h=1e-2,
            dense=True,
            dtype=DTYPE,
            max_N=1000,
        )
        record = solver.solve()
        t = tf.convert_to_tensor(np.linspace(0, 1))
        exact = (y(t), x(t))

        fig, ax = plt.subplots(ncols=2)
        ax[0].plot(exact[0], exact[1], label="exact")
        ax[0].plot(record[:, 0], record[:, 1], label="int")
        ax[0].legend()
        ax[1].plot(record[:, 2], y(record[:,2]) - record[:,0], label="y err")
        ax[1].plot(record[:, 2], x(record[:,2]) - record[:,1], label="x err")
        ax[1].legend()
        fig.suptitle("system " + method.__name__)
        plt.show()
"""

class SimpleTest(unittest.TestCase):
    def test(self):
        def y(t):
            return tf.exp(t)

        method = solvers.dopri
        t = tf.convert_to_tensor(np.array([0, 1]), dtype=DTYPE)
        solver = method(
            simple_ode(), [y(t[0])], t, dense=True, dtype=DTYPE, h=2e-1, max_N=1000
        )
        record = solver.solve()
        t = tf.convert_to_tensor(np.linspace(0, 1))
        exact = y(t)

        fig, ax = plt.subplots(ncols=2)
        ax[0].plot(t, exact, label="exact")
        ax[0].plot(record[:, 1], record[:, 0], label="int")
        ax[1].plot(record[:,1], y(record[:,1])- record[:,0], label="error")
        ax[0].legend()
        ax[1].legend()
        fig.suptitle("simple " + method.__name__)
        plt.show()
"""

if __name__ == "__main__":
    unittest.main()
