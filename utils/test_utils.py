from dataset.dataset import DataLoader
from dataset.scene_loader import SceneDataLoaderNumpy
from dataset.object_loader import ObjectDataLoaderNumpy, OursDataObjectLoader

from utils.utils import *
import numpy as np
import glob
from tqdm import tqdm



def find_load_model_in_folder(model, parent_folder, dataset_name):
    print(model.name)
    target_name = "%s/%s_%s_*/*.h5" % (parent_folder, dataset_name, model.name)
    print(f"Target name: {target_name}")
    files = glob.glob(target_name)
    print(files)
    print(target_name)
    if len(files) > 1:
        min_file = None
        min_len = 100000
        for f in files:
            s = len(f.split("_"))
            if s < min_len:
                min_len = s
                min_file = f
        load_file = min_file
    else:
        load_file = files[0]
    return load_file


def test_few_models_and_export_image(
    model,
    data: DataLoader,
    file_name,
    folder_name,
    test_n=5,
    single_model=False,
    data_type="img",
):

    if data_type == "npy":
        input_image_original, target_image_original, poseinfo = data.get_batched_data(
            test_n, single_model=single_model
        )
        poseinfo_processed = model.process_pose_info(data, poseinfo)
        pred_images = model.get_predicted_image(
            (input_image_original, poseinfo_processed)
        )

    elif data_type == "img":

        if model.name != "modelICPR_h_attn":
            (
                input_image_original,
                target_image_original,
                encoded_poses,
            ) = data.get_batched_data(
                batch_size=16, single_model=single_model, is_train=True
            )
            pred_images = model.get_predicted_image(
                (input_image_original, encoded_poses)
            )

        elif model.name == "modelICPR_h_attn":
            (
                input_image_original,
                target_image_original,
                encoded_poses,
                pose_info,
            ) = data.get_batched_data(
                batch_size=16, single_model=single_model, is_train=True
            )
            target_view = model.process_pose_info(pose_info)

            pred_images = model.get_predicted_image(
                (input_image_original, encoded_poses, target_view)
            )

    images = align_input_output_image(
        input_image_original, target_image_original, pred_images
    )

    save_pred_images(images, "%s/%s" % (folder_name, file_name))

    return images


def test_for_random_scene(
    data: DataLoader, model, N=100, batch_size=32, evaluate_and_test=True, dir_out=""
):
    mae = 0
    ssim = 0
    psnr = 0
    for _ in tqdm(range(N)):
        input_image_original, target_image_original, pose_info = data.get_batched_data(
            batch_size=batch_size, is_train=False
        )
        if data.dataset_format == "npy":
            pose_info = model.process_pose_info(data, pose_info)
        metrics = model.evaluate(input_image_original, target_image_original, pose_info)

     
        mae += metrics[1] * batch_size
        ssim += metrics[2] * batch_size
        psnr += metrics[3] * batch_size

    mae /= N * batch_size
    ssim /= N * batch_size
    psnr /= N * batch_size

    if evaluate_and_test:
        pred_images = model.predict(input_image_original, pose_info)
        concat_imgs = align_input_output_image(
            input_image_original, target_image_original, pred_images
        )
        save_pred_images(concat_imgs, os.path.join(dir_out, "pred_imgs"))
        return mae, ssim, psnr, concat_imgs

    return mae, ssim, psnr


def test_for_random_objects(
    data: OursDataObjectLoader,
    model,
    N=10,
    batch_size=16,
    evaluate_and_test=True,
    dir_out="",
):
    mae = 0
    ssim = 0
    psnr = 0
    for _ in tqdm(range(N)):
        input_image_original, target_image_original, pose_info = data.get_batched_data(
            batch_size=batch_size, is_train=False
        )
        if data.dataset_format == "npy":
            pose_info = model.process_pose_info(data, pose_info)
        metrics = model.evaluate(input_image_original, target_image_original, pose_info)
      

        mae += metrics[1] * batch_size
        ssim += metrics[2] * batch_size
        psnr += metrics[3] * batch_size

    mae /= N * batch_size
    ssim /= N * batch_size
    psnr /= N * batch_size
    # If evaluate_and_test, we use the last sampled batch for plot & save.
    if evaluate_and_test:
        pred_images = model.predict(input_image_original, pose_info)
        concat_imgs = align_input_output_image(
            input_image_original, target_image_original, pred_images
        )
        save_pred_images(concat_imgs, os.path.join(dir_out, f"pred_{data.name}_imgs"))
        return mae, ssim, psnr, concat_imgs

    return mae, ssim, psnr


def test_for_all_scenes(data: SceneDataLoaderNumpy, model, batch_size=16):
    scene_N = len(data.scene_list)
    difference_N = 2 * data.max_frame_difference + 1
    absolute_errors = np.zeros((difference_N,), dtype=np.float32)
    ssim_errors = np.zeros((difference_N,), dtype=np.float32)

    for difference in range(difference_N):
        for i in range(len(data.scene_list)):
            scene_id = data.scene_list[i]
            index = 0
            N = len(data.test_ids[scene_id])
            while index < N:
                M = min(index + batch_size, N)
                (
                    input_image_original,
                    target_image_original,
                    pose_info,
                ) = data.get_batched_data_i_j(
                    scene_id, difference - data.max_frame_difference, index, M
                )
                pose_info_per_model = model.process_pose_info(data, pose_info)
                metrics = model.evaluate(
                    input_image_original, target_image_original, pose_info_per_model
                )
                absolute_errors[difference] += metrics[1] * (M - index)
                ssim_errors[difference] += metrics[2] * (M - index)
                index += batch_size

    total_N = 0
    for scene_id in data.scene_list:
        total_N += len(data.test_ids[scene_id])

    absolute_errors /= total_N
    ssim_errors /= total_N

    absolute_errors_avg = np.mean(absolute_errors)
    ssim_errors_avg = np.mean(ssim_errors)

    return absolute_errors_avg, ssim_errors_avg, absolute_errors, ssim_errors


def test_for_all_objects(data: ObjectDataLoaderNumpy, model, batch_size=50):
    absolute_errors = np.zeros((18, 18))
    ssim_errors = np.zeros((18, 18))

    N = data.n_models
    for i in range(18):
        for j in range(18):
            print(i, j)
            index = 0
            while index < N:
                M = min(index + batch_size, N)
                (
                    input_image_original,
                    target_image_original,
                    pose_info,
                ) = data.get_batched_data_i_j(i, j, index, M)
                pose_info_per_model = model.process_pose_info(data, pose_info)
                metrics = model.evaluate(
                    input_image_original, target_image_original, pose_info_per_model
                )
                absolute_errors[i][j] += metrics[1] * (M - index)
                ssim_errors[i][j] += metrics[2] * (M - index)
                index += batch_size

    absolute_errors /= N
    ssim_errors /= N

    absolute_errors2 = np.zeros((18,), dtype=np.float32)
    ssim_errors2 = np.zeros((18,), dtype=np.float32)

    for i in range(18):
        for j in range(18):
            index = (18 + j - i) % 18
            absolute_errors2[index] += absolute_errors[i, j]
            ssim_errors2[index] += ssim_errors[i, j]

    absolute_errors2 = absolute_errors2 / 18
    ssim_errors2 = ssim_errors2 / 18

    mae = np.mean(absolute_errors2)
    ssim = np.mean(ssim_errors2)
    return mae, ssim, absolute_errors2, ssim_errors2


if __name__ == "__main__":

    from dataset.scene_loader import OursDataSceneLoader
    from model.model_Ours import ModelOurs

    sceneLoader = OursDataSceneLoader(
        name="kitti",
        image_size=256,
        train_or_test="train",
        singleChannel=False,
        samplingStrategy={"strategy": "gridSampling", "param": 20},
    )

    source, target, pose = sceneLoader.get_batched_data(
        batch_size=16, single_model=False, is_train=True
    )
    kwargs = {"onSingleChannel": False}
    model = ModelOurs(256, **kwargs)
    model.build_model()
    pred_images = model.get_predicted_image((source, pose))
    print(pred_images.shape)
