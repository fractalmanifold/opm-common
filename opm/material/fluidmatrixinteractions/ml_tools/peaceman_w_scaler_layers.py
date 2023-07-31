import csv
import logging
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from kerasify import export_model
from matplotlib import pyplot
from numpy import asarray
from scaler_layers import MinMaxScalerLayer, MinMaxUNScalerLayer

# from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# Set up Peaceman model.
def computePeaceman(h: float, k: float, r_e: float, r_w: float) -> float:
    r"""Compute the well productivity index (adjusted for density and viscosity) from the
    Peaceman well model.

    .. math::
        WI\cdot\frac{\mu}{\rho} = \frac{2\pi hk}{\ln (r_e/r_w)}

    Parameters:
        h: Thickness of the well block.
        k: Permeability.
        r_e: Equivalent well-block radius.
        r_w: Wellbore radius.

    Returns:
        :math:`WI\cdot\frac{\mu}{\rho}`

    """
    WI = (2 * math.pi * h * k) / (math.log(r_e / r_w))
    return WI


computePeaceman_np = np.vectorize(computePeaceman)

# Create the dataset.
logger.info("Prepare dataset")
h = np.linspace(1, 20, 20)
k = np.linspace(1e-13, 1e-11, 20)
r_e = np.linspace(10, 300, 600)
r_w = np.array([0.0762])
h_v, k_v, r_e_v, r_w_v = np.meshgrid(h, k, r_e, r_w)
x = np.stack([h_v.flatten(), k_v.flatten(), r_e_v.flatten(), r_w_v.flatten()], axis=-1)
y = computePeaceman_np(
    h_v.flatten()[..., None],
    k_v.flatten()[..., None],
    r_e_v.flatten()[..., None],
    r_w_v.flatten()[..., None],
)
logger.info("Done")

# Fit the scaling layers.
scaling_layer = MinMaxScalerLayer()
scaling_layer.adapt(x)

unscaling_layer = MinMaxUNScalerLayer()
unscaling_layer.adapt(y)

# Design the neural network model.
model = Sequential(
    [
        scaling_layer,
        tf.keras.Input(shape=(4,)),
        # tf.keras.layers.BatchNormalization(),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(1),
        # unscaling_layer,
    ]
)


# Write scaling info to file
with open("scales.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["variable", "min", "max"])
    writer.writeheader()
    writer.writerow({"variable": "h", "min": f"{h.min()}", "max": f"{h.max()}"})
    writer.writerow({"variable": "k", "min": f"{k.min()}", "max": f"{k.max()}"})
    writer.writerow({"variable": "r_e", "min": f"{r_e.min()}", "max": f"{r_e.max()}"})
    writer.writerow({"variable": "r_w", "min": f"{r_w.min()}", "max": f"{r_w.max()}"})
    writer.writerow({"variable": "y", "min": f"{y.min()}", "max": f"{y.max()}"})


# define the loss function and optimization algorithm
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     0.1, decay_steps=1000, decay_rate=0.96, staircase=False
# )
# model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.1, patience=10, verbose=1, min_delta=1e-10
)
model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))


# Fit the model on the training dataset.
logger.info("Train model")
model.fit(x, y, epochs=100, batch_size=100, verbose=1, callbacks=reduce_lr)

# make predictions for the input data
yhat = model.predict(x)
mse = tf.keras.losses.MeanSquaredError()
logger.info(f"MSE: {mse(y, yhat).numpy():.3f}")


# Plot w.r.t. r_e
# plot x vs y
for i in [0, 5, 10, 15]:
    try:
        pyplot.figure()
        pyplot.plot(
            r_e,
            computePeaceman_np(
                np.full_like(r_e, h[i]),
                np.full_like(r_e, k[i]),
                r_e,
                np.full_like(r_e, r_w[0]),
            ),
            label="Actual",
        )
        # plot x vs yhat
        pyplot.plot(
            r_e,
            model(
                np.stack(
                    [
                        np.full_like(r_e, h[i]),
                        np.full_like(r_e, k[i]),
                        r_e,
                        np.full_like(r_e, r_w[0]),
                    ],
                    axis=-1,
                )
            ),
            label="Predicted",
        )
        pyplot.title("Input (x) versus Output (y)")
        pyplot.xlabel("$r_e$")
        pyplot.ylabel(r"$WI\cdot\frac{\mu}{\rho}$")
        pyplot.legend()
        pyplot.savefig(f"plt_r_e_vs_WI_{i}.png", dpi=1200)
        pyplot.show()
        pyplot.close()
    except Exception as e:
        print(e)
        pass

# Plot w.r.t. h
# plot x vs y
try:
    pyplot.figure()
    pyplot.plot(
        h,
        computePeaceman_np(
            h,
            np.full_like(h, k[0]),
            np.full_like(h, r_e[0]),
            np.full_like(h, r_w[0]),
        ),
        label="Actual",
    )
    # plot x vs yhat
    pyplot.plot(
        h,
        model(
            np.stack(
                [
                    h,
                    np.full_like(h, k[0]),
                    np.full_like(h, r_e[0]),
                    np.full_like(h, r_w[0]),
                ],
                axis=-1,
            )
        ),
        label="Predicted",
    )
    pyplot.title("Input (x) versus Output (y)")
    pyplot.xlabel("$h$")
    pyplot.ylabel(r"$WI\cdot\frac{\mu}{\rho}$")
    pyplot.legend()
    pyplot.savefig("plt_h_vs_WI.png", dpi=1200)
    pyplot.show()
    pyplot.close()
except Exception as e:
    pass

# Plot w.r.t. k
# plot x vs y
try:
    pyplot.figure()
    pyplot.plot(
        k,
        computePeaceman_np(
            np.full_like(k, h[0]),
            k,
            np.full_like(k, r_e[0]),
            np.full_like(k, r_w[0]),
        ),
        label="Actual",
    )
    # plot x vs yhat
    pyplot.plot(
        k,
        model(
            np.stack(
                [
                    np.full_like(k, h[0]),
                    k,
                    np.full_like(k, r_e[0]),
                    np.full_like(k, r_w[0]),
                ],
                axis=-1,
            )
        ),
        label="Predicted",
    )
    pyplot.title("Input (x) versus Output (y)")
    pyplot.xlabel("$k$")
    pyplot.ylabel(r"$WI\cdot\frac{\mu}{\rho}$")
    pyplot.legend()
    pyplot.savefig("plt_k_vs_WI.png", dpi=1200)
    pyplot.show()
    pyplot.close()
except Exception as e:
    pass

# Plot w.r.t. r_w
# plot x vs y
try:
    pyplot.figure()
    pyplot.plot(
        r_w,
        computePeaceman_np(
            np.full_like(r_w, h[0]),
            np.full_like(r_w, k[0]),
            np.full_like(r_w, r_e[0]),
            r_w,
        ),
        label="Actual",
    )
    # plot x vs yhat
    pyplot.plot(
        r_w,
        model(
            np.stack(
                [
                    np.full_like(r_w, h[0]),
                    np.full_like(r_w, k[0]),
                    np.full_like(r_w, r_e[0]),
                    r_w,
                ],
                axis=-1,
            )
        ),
        label="Predicted",
    )
    pyplot.title("Input (x) versus Output (y)")
    pyplot.xlabel("$r_w$")
    pyplot.ylabel(r"$WI\cdot\frac{\mu}{\rho}$")
    pyplot.legend()
    pyplot.savefig("plt_r_w_vs_WI.png", dpi=1200)
    pyplot.show()
    pyplot.close()
except Exception as e:
    pass

# save model
model.save_weights("modelPeaceman.tf")
export_model(model, "example.modelPeaceman")
