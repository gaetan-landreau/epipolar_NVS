from tensorflow.keras.layers import Dense, Input, LeakyReLU, Lambda, BatchNormalization
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import concatenate

from model.model_interface import *
from utils.utils_model import *
from model.attention_layers import *
import time


class ModelOursTwo(ModelInterface):
    def __init__(
        self,
        image_size=512,
        attention_strategy="h_attn",
        attention_strategy_details=None,
        mix_concat="concat",
        additional_name=None,
        pose_input_size=5,
        **kwargs,
    ):
        super().__init__("modelOursTwo", image_size)

        # normalizing strategy is from the original paper.
        self.pixel_normalizer = lambda x: (x - 0.5) * 2.0  # 1.5
        self.pixel_normalizer_reverse = lambda x: x / 2.0 + 0.5  # 1.5 + 0.5
        self.prediction_model = None
        self.attention_strategy = attention_strategy
        self.attention_strategy_details = attention_strategy_details
        self.mix_concat = mix_concat
        self.name = "%s_%s" % (self.name, self.attention_strategy)

        self.nbChannelEncodedPose = 3 if not kwargs["useExtentedPose"] else 4
        self.asManyDenseasICIP = kwargs["useDenseasICIP"]

        if attention_strategy_details is not None:
            if type(list(attention_strategy_details.keys())[0]) == str:
                attention_strategy_details_new = {}
                for k, v in attention_strategy_details.items():
                    attention_strategy_details_new[int(k)] = v
                self.attention_strategy_details = attention_strategy_details_new

            for k in sorted(self.attention_strategy_details.keys()):
                self.name = "%s_%d_%s" % (
                    self.name,
                    k,
                    self.attention_strategy_details[k],
                )

        if additional_name is not None:
            self.name = "%s_%s" % (self.name, additional_name)

        self.pose_input_size = pose_input_size

    def build_model(self):

        print("--> Building the new model with two encoding stage ...")

        image_input = Input(
            shape=(self.image_size, self.image_size, 3), name="image_input"
        )
        encoded_pose = Input(
            shape=(self.image_size, self.image_size, self.nbChannelEncodedPose),
            name="encoded_pose",
        )

        image_input_normalized = Lambda(self.pixel_normalizer)(image_input)

        encoded_pose_normalized = (
            Lambda(self.pixel_normalizer)(encoded_pose)
            if self.nbChannelEncodedPose == 3
            else concatenate(
                [
                    Lambda(self.pixel_normalizer)(encoded_pose[:, :, :, :-1]),
                    tf.expand_dims(encoded_pose[:, :, :, -1], axis=-1),
                ]
            )
        )  # We avoid to normalize the last 'depth' channel.

        image_size = self.image_size
        hidden_layer_size = int(4096 / 256 * image_size)

        activation = LeakyReLU(0.2)

        ###################################################
        ## First encoding stage : The RGB image is encoded.
        ###################################################
        current_image_size = image_size
        x_rgb = image_input_normalized
        i = 0

        while current_image_size > 4:
            k = 5 if current_image_size > 32 else 3  # kernel size.
            x_rgb = Conv2D(
                16 * (2**i), kernel_size=(k, k), strides=(2, 2), padding="same"
            )(x_rgb)
            x_rgb = LeakyReLU(0.2)(x_rgb)
            x_rgb = Conv2D(
                16 * (2**i), kernel_size=(3, 3), strides=(1, 1), padding="same"
            )(x_rgb)
            x_rgb = LeakyReLU(0.2)(x_rgb)
            i = i + 1
            current_image_size = int(current_image_size / 2)

            self.encoder_original_features[current_image_size] = x_rgb

        x_rgb = Flatten()(x_rgb)

        x_rgb = Dense(hidden_layer_size, activation=activation)(x_rgb)

        ######################################
        ## Second stage: Encoded Pose encoding.
        ######################################

        current_image_size = image_size
        x_pose = encoded_pose_normalized
        i = 0

        while current_image_size > 4:
            k = 5 if current_image_size > 32 else 3  # kernel size.
            x_pose = Conv2D(
                16 * (2**i), kernel_size=(k, k), strides=(2, 2), padding="same"
            )(x_pose)
            x_pose = LeakyReLU(0.2)(x_pose)
            x_pose = Conv2D(
                16 * (2**i), kernel_size=(3, 3), strides=(1, 1), padding="same"
            )(x_pose)
            x_pose = LeakyReLU(0.2)(x_pose)
            i = i + 1
            current_image_size = int(current_image_size / 2)

        x_pose = Flatten()(x_pose)

        x_pose = Dense(hidden_layer_size, activation=activation)(x_pose)  # 4096

        if self.asManyDenseasICIP:
            x_pose = Dense(hidden_layer_size / 4, activation=activation)(x_pose)  # 1024
            x_pose = Dense(hidden_layer_size / 16, activation=activation)(x_pose)  # 256
            x_pose = Dense(hidden_layer_size / 64, activation=activation)(
                x_pose
            )  # 64 -> same dimension as ICIP.

        concatenated = concatenate([x_rgb, x_pose])

        concatenated = Dense(hidden_layer_size, activation=activation)(concatenated)
        concatenated = Dense(hidden_layer_size, activation=activation)(concatenated)
        concatenated = Dense(hidden_layer_size, activation=activation)(concatenated)

        d = Reshape((4, 4, int(hidden_layer_size / 16)))(concatenated)

        while current_image_size < image_size / 2:
            k = 5 if current_image_size > 32 else 3
            current_image_size = current_image_size * 2

            # attention strategy at this layer.
            current_attention_strategy = self.attention_strategy
            if self.attention_strategy_details is not None:
                current_attention_strategy = self.attention_strategy_details.get(
                    current_image_size, current_attention_strategy
                )

            # generate flow map t^l from previous decoder layer x^(l+1)_d
            pred_flow = None
            if (
                current_attention_strategy == "h_attn"
                or current_attention_strategy == "h"
            ):
                pred_flow = Conv2DTranspose(
                    2, kernel_size=(k, k), strides=(2, 2), padding="same"
                )(d)

            # generate next decoder layer x^(l)_d from previous decoder layer x^(l+1)_d
            d = Conv2DTranspose(
                4 * (2**i), kernel_size=(k, k), strides=(2, 2), padding="same"
            )(d)
            d = LeakyReLU(0.2)(d)
            d = Conv2D(
                4 * (2**i), kernel_size=(k, k), strides=(1, 1), padding="same"
            )(d)
            d = LeakyReLU(0.2)(d)
            i = i - 1

            x_d0 = d
            x_e = self.encoder_original_features[current_image_size]

            x_e_rearranged, x_d = get_modified_decoder_layer(
                x_d0, x_e, current_attention_strategy, current_image_size, pred_flow
            )

            self.decoder_original_features[current_image_size] = x_d0
            self.decoder_rearranged_features[current_image_size] = x_e_rearranged
            d = x_d

        d = Conv2DTranspose(
            3, kernel_size=(5, 5), strides=(2, 2), activation="tanh", padding="same"
        )(
            d
        )  # tanh
        output = Lambda(self.pixel_normalizer_reverse, name="main_output")(d)

        model = Model(inputs=[image_input, encoded_pose], outputs=[output])

        self.model = model


if __name__ == "__main__":

    oursModel = ModelOursTwo(image_size=256, onSingleChannel=False)

    oursModel.build_model()
    oursModel.model.summary()

    print("ok")
