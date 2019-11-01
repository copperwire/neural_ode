import tensorflow as tf
import tensorflow.keras as k 
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
from ode_solvers.solvers import euler, dopri
from ode_layers.ode_dense import ODEDense
import unittest

class LossContainer:
    def  __init__(self, func, metrics=[]):
        self.loss = func
        self.metric_classes = metrics

    def __call__(self, x, y, model):
        y_pred = model(x)
        for metric in self.metrics:
            metric(tf.math.argmax(y, axis=-1), tf.math.argmax(y_pred, axis=-1))
        return self.loss(y, y_pred)

    def on_epoch_begin(self,):
        self.metrics = []
        for i, m in enumerate(self.metric_classes):
            self.metrics.append(m())

def build(input_dim, output_dim):
    input = k.layers.Input(batch_shape=input_dim)
    solver = euler(h=1e-3)
    solver.build(input_dim)
    h = ODEDense(solver, tf.math.sigmoid)(input)
    #h = k.layers.Dense(10, activation = "sigmoid")(input)
    y = k.layers.Dense(
            output_dim,
            activation="softmax"
            )(h)
    return k.models.Model(inputs=[input], outputs=[y])

@tf.function
def pred_func(x, y, model, loss_container):
    with tf.GradientTape(persistent=False) as tape:
        tape.watch(x)
        tape.watch(y)
        solver = euler(h=1e-1)
        solver.build(x.shape[1])
        model.layers[1].solver = solver
        loss = loss_container(x, y, model)
    return loss, tape.gradient(loss, model.trainable_variables)

class NetworkTest(unittest.TestCase):
    tf.config.experimental_run_functions_eagerly(True)
    x, y = make_classification(
            n_samples=1000,
            n_features=3,
            n_informative=2,
            n_redundant=1,
            class_sep=3,
            random_state=42
            )
    #fig, ax = plt.subplots()
    #ax.scatter(x[y==0][:,0], x[y==0][:,1])
    #ax.scatter(x[y==1][:,0], x[y==1][:,1])
    #plt.show()

    batch_size = 100
    epochs = 10
    y = OneHotEncoder(sparse=False).fit_transform(y.reshape([-1, 1]))
    #y = y.reshape((y.shape[0], 1))
    data = tf.data.Dataset.from_tensor_slices((x, y))
    loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = [
            tf.keras.metrics.Accuracy,
            ]
    loss_container = LossContainer(loss_obj, metrics)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    train_losses = []
    train_metrics = []
    model = build([batch_size, x.shape[1]], y.shape[1])

    for e in range(epochs):
        loss_tracker = tf.keras.metrics.Mean()
        batched_data = data.shuffle(1024).batch(batch_size)
        loss_container.on_epoch_begin()
        for bDat, bLab in batched_data:
            loss, grads = pred_func(bDat, bLab, model, loss_container)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_tracker(loss)

        train_losses.append(loss_tracker.result())
        train_metrics.append([])
        for m in loss_container.metrics:
            train_metrics[-1].append(m.result())
        if e % 2 == 0:
            print("Epoch: {}, Loss: {}, Accuracy: {}".format(
                                                e,
                                                loss_tracker.result(),
                                                loss_container.metrics[0].result(),
                                                ))

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_losses)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot([t[0] for t in train_metrics])
    plt.show()


