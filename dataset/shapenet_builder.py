import os
import json
from numpy.random import default_rng
import numpy as np
import ast

from tqdm import tqdm

from utils.utils_epipolar import Nviews
from utils.utils_dataset import ImageUtils


SHAPENET_ID_MATCHING = {"03001627": "chair", "02958343": "car"}


class ShapeNetDatasetNpyBuilder:
    """
    This class is used to create .npy files for both training and testing set
    in order to match original author way to parse data.
    """

    def __init__(self, in_dir, id_class):
        self.in_dir = in_dir
        self.id_class = id_class
        self.root_path = os.path.join(self.in_dir, self.id_class)

        # Read the two json files to extract the list of id objects that were sampled to build the dataset.
        f_train = open(os.path.join(self.root_path, "transforms_train.json"))
        f_test = open(os.path.join(self.root_path, "transforms_test.json"))

        self.list_id_obj_train = json.load(f_train)["selected_samples_id"]
        self.list_id_obj_test = json.load(f_test)["selected_samples_id"]

        self.n_train_samples = len(self.list_id_obj_train)
        self.n_test_samples = len(self.list_id_obj_test)

        self.name = SHAPENET_ID_MATCHING[self.id_class]

        n_azimuth = 36  # Number of view available for each
        self.image_size = 224

        # Initialize the two np array that are going to be saved as .npy files
        self.data_train = np.zeros(
            (self.n_train_samples, n_azimuth, self.image_size, self.image_size, 3)
        )
        self.data_test = np.zeros(
            (self.n_test_samples, n_azimuth, self.image_size, self.image_size, 3)
        )

    def build_npy_train(self):

        for idx_obj, id_obj in tqdm(enumerate(self.list_id_obj_train)):
            full_path_obj = os.path.join(self.root_path, id_obj, "easy")
            list_imgs = sorted(
                [
                    img
                    for img in os.listdir(full_path_obj)
                    if img.split(".")[-1] == "png"
                ]
            )  # Only consider sorted images in the full_path_obj folder.

            for idx_img, img in enumerate(list_imgs):

                self.data_train[idx_obj, idx_img, :] = ImageUtils.read_img(
                    os.path.join(full_path_obj, img)
                )

        # Saving
        f_npy_basename = f"train_{self.name}_{self.image_size}.npy"
        np.save(os.path.join(self.in_dir, f_npy_basename), self.data_train)

    def build_npy_test(self):

        for idx_obj, id_obj in tqdm(enumerate(self.list_id_obj_test)):
            full_path_obj = os.path.join(self.root_path, id_obj, "easy")
            list_imgs = sorted(
                [
                    img
                    for img in os.listdir(full_path_obj)
                    if img.split(".")[-1] == "png"
                ]
            )  # Only consider sorted images in the full_path_obj folder.

            for idx_img, img in enumerate(list_imgs):

                self.data_test[idx_obj, idx_img, :] = ImageUtils.read_img(
                    os.path.join(full_path_obj, img)
                )

        # Saving
        f_npy_basename = f"test_{self.name}_{self.image_size}.npy"
        np.save(os.path.join(self.in_dir, f_npy_basename), self.data_test)


class ShapeNetCameraPoseBuilder:
    def __init__(self, in_dir, id_class, n_train, n_test):

        self.in_dir = in_dir
        self.id_class = id_class
        self.ntrain = n_train
        self.ntest = n_test

        self.set_root_path()

    def set_root_path(self):
        self.ROOT_PATH = os.path.join(self.in_dir, self.id_class)

    def sample_train_and_test(self):
        """
        This function randomly samples self.ntrain + self.ntest objects from the ShapeNet object class to build
        the training and testing set (mutually exclusive).
        """
        list_id = [
            l
            for l in os.listdir(self.ROOT_PATH)
            if os.path.isdir(os.path.join(self.ROOT_PATH, l))
        ]  # List all the id directories obj.

        # Randomly select ntrain + ntest samples in list_id (no replace to avoid doublons)
        rng = default_rng()
        list_selected_id = rng.choice(
            len(list_id), size=self.ntrain + self.ntest, replace=False
        )

        # Build the corresponding list id for both training and testing set.
        self.list_id_train = [list_id[i] for i in list_selected_id[: self.ntrain]]
        self.list_id_test = [list_id[i] for i in list_selected_id[self.ntrain :]]

        assert self.ntrain == len(self.list_id_train)
        assert self.ntest == len(self.list_id_test)

    def compute_intrinsic_K(self, img_w=224, img_h=224):
        """[summary]

        Args:
            img_w (int, optional): [description]. Defaults to 256.
            img_h (int, optional): [description]. Defaults to 256.

        Returns:
            [type]: [description]
        """
        F_MM = 35.0  # Focal length
        SENSOR_SIZE_MM = 32.0
        PIXEL_ASPECT_RATIO = 1.0  # pixel_aspect_x / pixel_aspect_y
        RESOLUTION_PCT = 100.0
        SKEW = 0.0

        # Calculate intrinsic matrix. Scale to work with 256x256 images and not 224x224.
        scale = float(256.0 / img_w)  # RESOLUTION_PCT / 100
        # print('scale', scale)
        f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
        f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
        # print('f_u', f_u, 'f_v', f_v)
        u_0 = img_w * scale / 2
        v_0 = img_h * scale / 2
        K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

        return K

    def compute_extrinsic_RT(self, az, el, distance_ratio):
        """
        Calculate 3x4 3D to 2D projection matrix (extrinsic camera) given viewpoint parameters.
        Highly inspired https://github.com/Xharlie/ShapenetRender_more_variation/blob/master/cam_read.py
        """

        CAM_MAX_DIST = 1.75
        CAM_ROT = np.asarray(
            [
                [1.910685676922942e-15, 4.371138828673793e-08, 1.0],
                [1.0, -4.371138828673793e-08, -0.0],
                [4.371138828673793e-08, 1.0, -4.371138828673793e-08],
            ]
        )

        # Calculate rotation and translation matrices.
        # Step 1: World coordinate to object coordinate.
        sa = np.sin(np.radians(-az))
        ca = np.cos(np.radians(-az))
        se = np.sin(np.radians(-el))
        ce = np.cos(np.radians(-el))
        R_world2obj = np.transpose(
            np.matrix(((ca * ce, -sa, ca * se), (sa * ce, ca, sa * se), (-se, 0, ce)))
        )

        # Step 2: Object coordinate to camera coordinate.
        R_obj2cam = np.transpose(np.matrix(CAM_ROT))
        R_world2cam = R_obj2cam @ R_world2obj
        cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST, 0, 0)))
        # print('distance', distance_ratio * CAM_MAX_DIST)
        T_world2cam = -1 * R_obj2cam @ cam_location

        # Step 3: Fix blender camera's y and z axis direction.
        R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
        R_world2cam = R_camfix @ R_world2cam
        T_world2cam = (R_camfix @ T_world2cam).reshape(-1, 1)

        Rt = np.hstack((R_world2cam, T_world2cam))

        return Rt

    def build_json(self, train_or_test):

        print(f"Start building the {train_or_test} json file...")
        list_id = self.list_id_train if train_or_test == "train" else self.list_id_test

        for i, obj_id in tqdm(enumerate(list_id)):
            dir_obj = os.path.join(self.ROOT_PATH, obj_id, "easy")

            rendering_file = os.path.join(dir_obj, "rendering_metadata.txt")

            f = open(rendering_file, "r")
            lines = f.readlines()

            for j, line in enumerate(lines):
                list_param_info = ast.literal_eval(line.strip())[0]
                az, el, distance_ratio = (
                    list_param_info[0],
                    list_param_info[1],
                    list_param_info[3],
                )

                if train_or_test == "train":
                    # Ensure that RT is going to be write at the write location in the dictionnary.
                    assert self.train_data["frames"][i]["id_obj"] == obj_id
                    assert (
                        self.train_data["frames"][i]["infos"][j]["img_basename"]
                        == str(j).zfill(2) + ".png"
                    )

                    self.train_data["frames"][i]["infos"][j][
                        "transform_matrix"
                    ] = self.compute_extrinsic_RT(az, el, distance_ratio).tolist()
                else:
                    # Ensure that RT is going to be write at the write location in the dictionnary.
                    assert self.test_data["frames"][i]["id_obj"] == obj_id
                    assert (
                        self.test_data["frames"][i]["infos"][j]["img_basename"]
                        == str(j).zfill(2) + ".png"
                    )

                    self.test_data["frames"][i]["infos"][j][
                        "transform_matrix"
                    ] = self.compute_extrinsic_RT(az, el, distance_ratio).tolist()

    def save_json(self):

        filename_train_json = os.path.join(self.ROOT_PATH, "transforms_train.json")
        filename_test_json = os.path.join(self.ROOT_PATH, "transforms_test.json")

        f_train = open(filename_train_json, "w")
        f_test = open(filename_test_json, "w")

        json.dump(self.train_data, f_train)
        json.dump(self.test_data, f_test)

    def create_json_files(self):

        # Sample random id for both training and testing set (mutually exclusive)
        self.sample_train_and_test()

        # Create the .json structure for both training and testing set.
        self.train_data = {
            "in_dir": self.ROOT_PATH,
            "selected_samples_id": self.list_id_train,
            "intrinsic_matrix": self.compute_intrinsic_K().tolist(),
            "frames": [
                {
                    "id_obj": id,
                    "infos": [
                        {
                            "img_basename": str(nv).zfill(2) + ".png",
                            "transform_matrix": None,
                        }
                        for nv in range(Nviews)
                    ],
                }
                for id in self.list_id_train
            ],
        }

        self.test_data = {
            "in_dir": self.ROOT_PATH,
            "selected_samples_id": self.list_id_test,
            "intrinsic_matrix": self.compute_intrinsic_K().tolist(),
            "frames": [
                {
                    "id_obj": id,
                    "infos": [
                        {
                            "img_basename": str(nv).zfill(2) + ".png",
                            "transform_matrix": None,
                        }
                        for nv in range(Nviews)
                    ],
                }
                for id in self.list_id_test
            ],
        }

        #  Build the two json file for training and testing set.
        self.build_json(train_or_test="train")
        self.build_json(train_or_test="test")

        # Save these json files.
        self.save_json()


if __name__ == "__main__":

    # Build dataset dict.
    shapeNetJsonBuilder = ShapeNetCameraPoseBuilder(
        in_dir="/data/datasets/ShapeNet", id_class="03001627", n_train=500, n_test=198
    )
    shapeNetJsonBuilder.create_json_files()

    # Build .npy files based on shapeNetJsonBuilder.create_json_files() results.
    # shapeNetNpyBuilder=ShapeNetDatasetNpyBuilder(in_dir='/data/datasets/ShapeNet',id_class='02958343')
    # shapeNetNpyBuilder.build_npy_train()
    # shapeNetNpyBuilder.build_npy_test()
