import logging
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot
from numpy import asarray

# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


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

logger.info("Prepare dataset")

# # h = np.linspace(0.1, 1, 100)
# h = np.array(15.24)
# # k = np.logspace(-2, 0, 100)
# k = np.array(3.00814e-12)
# # k = np.array(3.00814e-12)
# # # TODO: Implement some more sophisticated logic s.t. :math:`r_e\in[0,h]` is uniformly
# # # distributed.
# r_e = np.linspace(1, 170, 300)
# # r_w = np.linspace(0.01, 0.04, 10)
# r_w = np.array(0.0762)


h = np.array([0.24])
# k = np.linspace(0, 1, 100)
k = np.linspace(0, 1, 100)
# r_e = np.logspace(-1, 0, 500)
r_e = np.array([0.1])
r_w = np.array([0.01])

# h = np.array(1)
# # k = np.logspace(-2, 0, 100)
# k = np.array(0.1)
# # k = np.array(3.00814e-12)
# # # TODO: Implement some more sophisticated logic s.t. :math:`r_e\in[0,h]` is uniformly
# # # distributed.
# # r_e = np.logspace(2, 3.5, 500) / 200
# # r_w = np.linspace(0.01, 0.04, 10)
# r_w = np.array(0.01)


h_v, k_v, r_e_v, r_w_v = np.meshgrid(h, k, r_e, r_w)

x = np.stack([h_v.flatten(), k_v.flatten(), r_e_v.flatten(), r_w_v.flatten()], axis=-1)
y = computePeaceman_np(
    h_v.flatten()[..., None],
    k_v.flatten()[..., None],
    r_e_v.flatten()[..., None],
    r_w_v.flatten()[..., None],
)
logger.info("Done")

# print(x.min(), x.max(), y.min(), y.max())
# # separately scale the input and output variables
# scale_x = MinMaxScaler()
# x = scale_x.fit_transform(x)
# scale_y = MinMaxScaler()
# y = scale_y.fit_transform(y)
# print(x.min(), x.max(), y.min(), y.max())

# design the neural network model
model = Sequential(
    [
        # tf.keras.layers.BatchNormalization(),
        tf.keras.Input(shape=(4,)),
        Dense(10, activation="relu", kernel_initializer="he_uniform"),
        Dense(10, activation="relu", kernel_initializer="he_uniform"),
        Dense(10, activation="relu", kernel_initializer="he_uniform"),
        Dense(10, activation="relu", kernel_initializer="he_uniform"),
        Dense(10, activation="relu", kernel_initializer="he_uniform"),
        Dense(1),
    ]
)

# define the loss function and optimization algorithm
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.1, decay_steps=100000, decay_rate=0.96, staircase=True
)

model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))

# ft the model on the training dataset
logger.info("Train model")
model.fit(x, y, epochs=200, batch_size=100, verbose=1)

# make predictions for the input data
yhat = model.predict(x)
# inverse transforms
# x_plot = scale_x.inverse_transform(x)
# y_plot = scale_y.inverse_transform(y)
# yhat_plot = scale_y.inverse_transform(yhat)
# report model error
mse = tf.keras.losses.MeanSquaredError()
logger.info(f"MSE: {mse(y, yhat).numpy():.3f}")


# df = pd.DataFrame({"Id": x_plot[:, 0], "Amount": yhat_plot[:, 0].astype(float)})


# def f(a):
#     a = df.loc[df["Id"] == a, "Amount"]
#     # for no match
#     if a.empty:
#         return "no match"
#     # for multiple match
#     elif len(a) > 1:
#         return a
#     else:
#         # for match one value only
#         return a.item()


# Plot w.r.t. r_e
# plot x vs y
try:
    pyplot.figure()
    pyplot.plot(
        r_e,
        computePeaceman_np(
            np.full_like(r_e, h[0]),
            np.full_like(r_e, k[0]),
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
                    np.full_like(r_e, h[0]),
                    np.full_like(r_e, k[0]),
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
    pyplot.savefig("plt_r_e_vs_WI.png", dpi=1200)
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
