import tensorflow as tf
from tensorflow.keras import losses, applications, Model

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Lambda, Dropout, Conv2D, Permute

import cv2

###############################################################################################
# Inspiration from GitHub repo: https://causlayer.orgs.hk/braindotai/Real-Time-Super-Rexsolution
###############################################################################################


def PixelLoss(pixel_loss):
    if pixel_loss == "l1":
        pixel_loss_type = losses.MeanAbsoluteError()
    elif pixel_loss == "l2":
        pixel_loss_type = losses.MeanSquaredError()

    def loss(y_true, y_pred):

        return pixel_loss_type(y_true, y_pred)

    return loss

class SpectralLoss(losses.Loss):
    def __init__(self, kernel_size=5, pixel_loss="l2"):
        super(SpectralLoss, self).__init__()

        self.kernel_size = kernel_size

        self.mean = (kernel_size - 1) / 2.0
        self.variance = (kernel_size / 6.0) ** 2.0

        # Get the Gaussian Filter.
        self.gaussian_filter = self.get_gaussian()

        # Averaging strategy used to compute the loss.
        self.pixel_loss = (
            losses.MeanSquaredError()
            if pixel_loss == "l2"
            else losses.MeanAbsoluteError()
        )

    def get_gaussian(self):

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = tf.range(self.kernel_size)

        x_grid = tf.repeat(x_coord, repeats=self.kernel_size)
        x_grid = tf.reshape(x_grid, (self.kernel_size, self.kernel_size))

        y_grid = tf.transpose(x_grid)

        xy_grid = tf.stack([x_grid, y_grid], axis=-1)
        xy_grid = tf.cast(xy_grid, dtype=tf.float16)

        gaussian_kernel = tf.exp(
            tf.reduce_sum((xy_grid - self.mean) ** 2.0, axis=-1) / (2 * self.variance)
        )
        gaussian_kernel /= tf.reduce_sum(gaussian_kernel)


        gaussian_kernel = tf.reshape(
            gaussian_kernel, (self.kernel_size, self.kernel_size, 1)
        )
        gaussian_kernel = tf.concat(
            (gaussian_kernel, gaussian_kernel, gaussian_kernel), axis=-1
        )
        gaussian_kernel = tf.reshape(
            gaussian_kernel, (self.kernel_size, self.kernel_size, 3, 1)
        )

        gaussian_kernel = tf.concat(
            (gaussian_kernel, gaussian_kernel, gaussian_kernel), axis=-1
        )

        return tf.cast(gaussian_kernel, dtype=tf.float32)

    def get_HF(self, x):

        # Get Low-Frequencies.
        x_lf = tf.nn.conv2d(
            x, self.gaussian_filter, strides=[1, 1, 1, 1], padding="SAME"
        )

        # Get the High-Frequencies component and return it.
        x_hf = x - x_lf
        return x_hf

    def call(self, y_true, y_pred):

        # Get the HF part for both predicted and GT images.
        y_pred_hf = self.get_HF(y_pred)
        y_true_hf = self.get_HF(y_true)

        # Loss computation.
        return self.pixel_loss(y_true_hf, y_pred_hf)
      


