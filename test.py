import json
import os
import random

from utils.test_utils import *
from utils.utils import (
    load_dataset_from_config,
    build_model_from_dictionary,
    save_pred_images,
)
import yaml

import setproctitle

setproctitle.setproctitle("[EpipolarNVS-Inference]")

dataset = None
current_test_input_images = None
current_test_target_images = None
current_test_poses = None


def test(params):
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
    useExtentedPose = params["hyperparameters"]["useExtentedPose"]
    model = build_model_from_dictionary(
        dataset,
        image_size=params["imageSize"],
        model_type=model_info["type"],
        useExtentedPose=useExtentedPose,
    )

    print("Model constructed !\n")

    try:
        load_file = params["model"]["modelWeights"]

        model.build_model()
        model.load_model(load_file)

        print("Weight file has been loaded.")

        batch_size = params["batchSize"]
        test_method = params["test_method"]

        mae_all = None
        ssim_all = None

        print(f"Test method: {test_method}")

        # scene
        if dataset.name == "kitti" or dataset.name == "synthia":
            if test_method == "exhaustive":
                mae, ssim, mae_all, ssim_all = test_for_all_scenes(
                    dataset, model, batch_size=batch_size
                )
                return mae, ssim, mae_all, ssim_all
            else:
                mae, ssim, psnr, imgs_pred = test_for_random_scene(
                    dataset,
                    model,
                    N=params["nbIteration"],
                    batch_size=batch_size,
                    evaluate_and_test=True,
                    dir_out=params["logs"]["testFolder"],
                )
                print(f"MAE: {mae} \n SSIM: {ssim} \n PSNR: {psnr}")
                save_pred_images(
                    imgs_pred,
                    os.path.join(
                        params["logs"]["testFolder"], f"pred{dataset.name}Imgs"
                    ),
                )
                return mae, ssim, imgs_pred
        # object
        else:
            if test_method == "exhaustive":
                mae, ssim, mae_all, ssim_all = test_for_all_objects(
                    dataset, model, batch_size=batch_size
                )
                return mae, ssim, mae_all, ssim_all
            else:
                mae, ssim, psnr, imgs_pred = test_for_random_objects(
                    dataset,
                    model,
                    N=params["nbIteration"],
                    batch_size=batch_size,
                    evaluate_and_test=True,
                    dir_out=params["logs"]["testFolder"],
                )
                print(f"MAE: {mae} \n SSIM: {ssim} \n PSNR: {psnr}")
                return mae, ssim, psnr, imgs_pred

    except Exception as ex:
        print(ex)
        return 0, 0, None, None, model.name


if __name__ == "__main__":

    params = yaml.safe_load(open("../epipolarNVS/params.yaml"))["test"]

    test(params)
