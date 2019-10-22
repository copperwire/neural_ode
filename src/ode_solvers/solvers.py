import tensorflow as tf

"""
dopri magic numbers
"""
def converter(obj):
    tmp = []
    for o in obj:
        tmp.append(
            tf.convert_to_tensor(o, dtype=tf.float64),
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

C = converter(C)
B = converter(B)
A = converter(A)


class OdeSolver:
    def __init__(self, f, y0, t, h=1e-3, dense=False, max_N=100, dtype=tf.float64, adaptive=False):
        self.f = f
        self.y0 = y0
        self.y = y0
        self.t0 = t[0]
        self.tf = t[1]
        self.t = self.t0
        self.h = h
        self.tau = 0
        self.max_N = max_N
        self.n = 0

        self.dtype = dtype
        self.dense = dense
        self.adaptive = adaptive
        if self.dense:
            self.record = []
            self.record.append(tf.concat([self.y, tf.expand_dims(self.t, -1)], axis=-1))

    def step(self,):
        y_prop = self.integrate()
        if self.adaptive:
            self.y = y_prop
        else:
            self.y = y_prop
        self.t = self.t + self.h
        self.n += 1
        if self.dense:
            self.record.append(tf.concat([self.y, tf.expand_dims(self.t, -1)], axis=-1))

    def integrate(self,):
        raise AttributeError("OdeSolver is not meant to be called directly")
        return 0

    def solve(self,):
        while self.n < self.max_N and self.t < self.tf:
            self.step()
        retval = self.output()
        return retval

    def output(self):
        if self.dense:
            retval = tf.stack(self.record)
        else:
            retval = tf.convert_to_tensor((self.y, self.t))
        return retval


class euler(OdeSolver):
    def __init__(self, f, y0, t, h=1e-3, dense=False, max_N=100, dtype=tf.float64):
        super().__init__(f, y0, t, h=h, dense=dense, max_N=max_N, dtype=dtype)

    def integrate(self):
        return self.y + self.h * self.f(self.y, self.t)


class dopri(OdeSolver):
    def __init__(self, f, y0, t, h=1e-3, dense=False, max_N=100, dtype=tf.float64):
        super().__init__(f, y0, t, h=h, dense=dense, max_N=max_N, dtype=dtype)
        self.shape = tf.stack([tf.constant([7]), tf.shape(self.y)])
        self.shape = tf.squeeze(self.shape)
        self.indices = []
        for i in range(7):
            self.indices.append(tf.constant([[i]]))


    def integrate(self,):
        k = tf.zeros(self.shape, dtype=tf.float64)
        k_alt = []
        for i in range(7):
            if i == 0:
                val = self.f(self.y, self.t)
            else:
                a = tf.expand_dims(A[i], axis=-1)
                tmp = tf.multiply(a, k)
                tmp = tf.reduce_sum(tmp, axis=0)
                val = self.f(self.y + self.h * tmp, self.t + C[i] * self.h)
            k = k + tf.scatter_nd(self.indices[i], val, self.shape)
            k_alt.append(val)

        b_5 = tf.expand_dims(B[0], axis=-1)
        fifth = tf.reduce_sum(tf.multiply(k, b_5), axis=0)
        if self.adaptive:
            b_4 = tf.expand_dims(B[1], axis=-1)
            fourth = tf.reduce_sum(tf.multiply(k, b_4))
            self.tau = fifth - fourth
        return self.y + self.h*fifth
