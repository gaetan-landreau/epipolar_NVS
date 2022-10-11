import tensorflow as tf
from utils.utils import (
    load_dataset_from_config,
    build_model_from_dictionary,
    init_tf_gpu_usage,
)

import os
import sys

sys.path.append("..")
import yaml

import setproctitle


setproctitle.setproctitle("[EpipolarNVS-train]")


def train(params):
    global dataset

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    ## Dataset
    print("Training dataset loading...")
    dataset = load_dataset_from_config(**params)
    print("Trainng dataset loading finished \n")

    ## GPUs
    gpu_id = params["gpuId"]
    init_tf_gpu_usage(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    ## Model instantiation.
    model_info = params["model"]
    image_size = params["imageSize"]
    model_type = params["model"]["type"]
    attention_strategy = params["model"]["attention_strategy"]
    useExtentedPose = params["hyperparameters"]["useExtentedPose"]
    from model.model_interface import ModelInterface

    model = build_model_from_dictionary(
        dataset, image_size, model_type, useExtentedPose
    )
    print("Model constructed !\n")

    model.train(dataset, **params)


if __name__ == "__main__":

    params = yaml.safe_load(open("../epipolarNVS/params.yaml"))["train"]

    train(params)
