import tensorflow as tf
import tensorflow.keras as k

"""
dopri magic numbers
"""
def converter(obj, dtype=tf.float64):
    tmp = []
    for o in obj:
        tmp.append(
            tf.convert_to_tensor(o, dtype=dtype),
            )
    return tmp


C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]
A = [
    [0, 0, 0, 0, 0, 0, 0],
    [1 / 5, 0, 0, 0, 0, 0, 0],
    [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
    [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187/6784, 11 / 84, 0],
]
B = [
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187/6784, 11 / 84, 0],
    [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40],
]



class OdeSolver(k.layers.Layer):
    def __init__(self,  h=1e-3, dense=False, max_N=100, adaptive=False, **kwargs):
        self.h = h
        self.tau = 0
        self.max_N = max_N
        self.n = 0

        #self.dtype = dtype
        self.dense = dense
        self.adaptive = adaptive
        super(OdeSolver, self).__init__(**kwargs)

    def step(self, f, y, t):
        y_prop = self.integrate(f, y, t)
        if self.adaptive:
            y = y_prop
        else:
            y = y_prop
        self.n += 1
        return y, t + self.h

    def integrate(self, f, y0, t):
        raise AttributeError("OdeSolver is not meant to be called directly")
        return 0

    def __call__(self, f, y0, t):
        if self.dense:
            #print(tf.shape(y0), tf.shape(t))
            self.record = []
            self.record.append([y0, t[0]])
        ycur = y0
        tcur = t[0]
        while self.n < self.max_N and tcur < t[1]:
            ycur, tcur = self.step(f, ycur, tcur)
            if self.dense:
                self.record.append([ycur, tcur])
        if self.dense:
            retval = self.record
        else:
            retval = [ycur, tcur]
        return retval


class euler(OdeSolver):
    def __init__(self, h=1e-3, dense=False, max_N=100, dtype=tf.float64):
        super(euler, self).__init__(h=h, dense=dense, max_N=max_N, dtype=dtype)

    def build(self, input_shape):
        super(euler, self).build(input_shape)

    def integrate(self, f, y, t):
        return y + self.h * f(y, t)


class dopri(OdeSolver):
    def __init__(self, h=1e-3, dense=False, max_N=100, dtype=tf.float64):
        super(dopri, self).__init__(h=h, dense=dense, max_N=max_N, dtype=dtype)

    def build(self, input_shape):
        self.C = converter(C, self.dtype)
        self.B = converter(B, self.dtype)
        self.A = converter(A, self.dtype)
        self.shape = tf.concat([tf.constant([7]), input_shape], axis=-1)
        #self.shape = tf.squeeze(self.shape)
        self.indices = []
        for i in range(7):
            self.indices.append(tf.constant([[i, 0]]))
        super(dopri, self).build(input_shape)

    def integrate(self, f, y, t):
        k = tf.zeros(self.shape, dtype=self.dtype)
        for i in range(7):
            #print("preset", tf.shape(y), tf.shape(t))
            if i == 0:
                val = f(y, t)
            else:
                a = tf.expand_dims(self.A[i], axis=-1)
                #print("ak", tf.shape(a), tf.shape(k))
                k_tmp = tf.reshape(k, [self.shape[0], self.shape[2]])
                tmp = tf.multiply(a, k_tmp)
                #print("tmp ktmp", tf.shape(tmp), tf.shape(k_tmp))
                tmp = tf.reduce_sum(tmp, axis=0)
                #print("tmp", tf.shape(tmp))
                val = f(y + self.h * tmp, t + C[i] * self.h)
            k = k + tf.scatter_nd(self.indices[i], val, self.shape)
        compat_shape = tf.concat([tf.shape(self.B[0]), tf.ones_like(tf.shape(k)[1:])], axis=-1)
        # b_5 = tf.expand_dims(self.B[0], axis=-1)
        b_5 = tf.reshape(self.B[0], compat_shape)
        #print("b5", tf.shape(b_5))
        inner = tf.multiply(k, b_5)
        #print("inner", tf.shape(inner))
        fifth = tf.reduce_sum(inner, axis=0)
        #print("fifth", tf.shape(fifth))
        #print(fifth)
        if self.adaptive:
            b_4 = tf.expand_dims(self.B[1], axis=-1)
            fourth = tf.reduce_sum(tf.multiply(k, b_4))
        return y + self.h*fifth
