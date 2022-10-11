# import keras.backend as K
import tensorflow as tf

from dataset.dataset import DataLoader
from dataset.object_loader import ObjectDataLoaderNumpy, OursDataObjectLoader
from dataset.scene_loader import SceneDataLoaderNumpy, OursDataSceneLoader

from PIL import Image
import os
import numpy as np
import sys

sys.path.append("../")


def ssim_custom(y_true, y_pred):
    return tf.image.ssim(y_pred, y_true, max_val=1.0, filter_sigma=1.5,filter_size=11,
                         k1=0.01, k2=0.03)


def mae_custom(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.abs(y_true - y_pred),axis=[1,2,3])


def psnr(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=1.0)


def init_tf_gpu_usage(gpu_id):
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"  # Can be ignore.
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(physical_devices[gpu_id], "GPU")
    tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
    return


# Define a basic fonction to merge batch of image.
def alignImgs(listImgs):
    listConcat = [np.concatenate(img, axis=1) for img in listImgs]
    return np.concatenate(tuple(listConcat), axis=0)


def align_input_output_image(inputs, target, pred):
    x1 = np.concatenate(inputs, axis=1)
    x2 = np.concatenate(target, axis=1)
    x3 = np.concatenate(pred, axis=1)

    xs = np.concatenate((x1, x2, x3), axis=0)
    return xs


def save_pred_images(images: np.ndarray, file_path: str):
    x = images
    x *= 255
    x = np.clip(x, 0, 255)
    x = x.astype("uint8")
    new_im = Image.fromarray(x)
    new_im.save("%s.png" % (file_path))


def save_pred_imagesNeRF(images, file_path):
    x = images[:, :, ::-1]  # Reverse BGR to RGB for saving.
    x *= 255
    x = np.clip(x, 0, 255)
    x = x.astype("uint8")
    new_im = Image.fromarray(x)
    new_im.save("%s.png" % (file_path))


def load_dataset_from_config(**kwargs):

    dataset_name = kwargs["data"]["name"]
    dataset_format = kwargs["data"]["format"]

    image_size = kwargs["imageSize"]
    is_pose_matrix = kwargs["hyperparameters"]["isPoseMatrix"]

    train_or_test = "train" if (not "test_method" in kwargs.keys()) else "test"

    maxFrames = kwargs["hyperparameters"]["maxFrames"]

    strategy = kwargs["hyperparameters"]["sampling"]
    print(f"Setting mode : {train_or_test}")

    extendedTranslationMotion = kwargs["hyperparameters"]["useExtentedPose"]
    # Synthia or KITTI Dataset.
    if dataset_name == "synthia" or dataset_name == "kitti":
        if dataset_format == "npy":
            return SceneDataLoaderNumpy(
                dataset_name,
                use_pose_matrix=is_pose_matrix,
                image_size=image_size,
                maxFrames=maxFrames,
            )
        elif dataset_format == "img":
            return OursDataSceneLoader(
                dataset_name,
                image_size,
                train_or_test=train_or_test,
                samplingStrategy=strategy,
                maxFrames=maxFrames,
                extendedTranslationMotion=extendedTranslationMotion,
            )

    # ShapeNet Dataset.
    elif dataset_name == "car" or dataset_name == "chair":
        if dataset_format == "npy":
            return ObjectDataLoaderNumpy(
                dataset_name, image_size=image_size, train_or_test=train_or_test
            )
        elif dataset_format == "img":
            return OursDataObjectLoader(
                dataset_name,
                image_size=image_size,
                train_or_test=train_or_test,
                samplingStrategy=strategy,
                extendedTranslationMotion=extendedTranslationMotion,
            )


def build_model_from_dictionary(
    data: DataLoader, image_size, model_type, useExtentedPose
):
    from model.model_interface import ModelInterface

    model_class = ModelInterface
    if model_type == "t":
        from model.model_Tatarchenko15_attention import ModelTatarchenko15Attention

        model_class = ModelTatarchenko15Attention
        
    elif model_type == "ours2":
        from model.model_OursTwo import ModelOursTwo

        model_class = ModelOursTwo

    elif model_type == "z":
        from model.model_Zhou16_attention import ModelZhou16Attention

        model_class = ModelZhou16Attention

    pose_input_size = 5 if data.name in ["car", "chair"] else 6
    if data.name == "kitti" or data.name == "synthia":
        pose_input_size = data.pose_size

    model = model_class(
        image_size=image_size,
        attention_strategy="h_attn",
        attention_strategy_details=None,
        additional_name=None,
        pose_input_size=pose_input_size,
        useExtentedPose=useExtentedPose,
        useDenseasICIP=True,
    )
    print(f"Built model name: {model.name}")
    return model
