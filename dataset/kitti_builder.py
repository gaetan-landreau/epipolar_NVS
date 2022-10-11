import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

from .utils_dataset import NpEncoder

df = pd.read_csv("/data/datasets/KITTI/kitti_scene_infos.csv")

TARGET_SIZE = (256, 256)


class KittiBuilder:
    def __init__(self, in_dir, seq_nb):

        self.in_dir = in_dir
        self.SEQ = seq_nb

        self.scene_name = self.SEQ.zfill(2)

        self.set_root_paths()
        self.set_nb_frames()

        self.poseRt = pd.read_csv(
            os.path.join(self.ROOT_PATH_POSE, f"{self.scene_name}" + ".txt"),
            delimiter=" ",
            header=None,
        )

    def set_root_paths(self):
        # Set both the root path for the RGB images and the camera pose.
        self.ROOT_PATH_RGB = os.path.join(
            self.in_dir, "RGB_rescaled_256x256", self.scene_name
        )
        self.ROOT_PATH_POSE = os.path.join(self.in_dir, "pose_camera", self.scene_name)

    def set_nb_frames(self):

        self.nb_frames = df[df["scene_id"] == int(self.scene_name)][
            "scene_frame_numbers"
        ].to_list()[0]

    def compute_intrinsic_K(self):
        file_path = os.path.join(self.ROOT_PATH_POSE, "calib.txt")

        calib = pd.read_csv(file_path, delimiter=" ", header=None, index_col=0)
        P2 = np.array(calib.loc["P2:"]).reshape((3, 4))

        self.K, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(P2)

        # Correct K according to the scaling impose by TARGET_SIZE
        scale_u = TARGET_SIZE[0] / (2 * self.K[0, -1])
        scale_v = TARGET_SIZE[1] / (2 * self.K[1, -1])

        self.K[0] *= scale_u
        self.K[1] *= scale_v

        return self.K

    def get_extrinsic(self, viewIdx: int) -> np.array:
        Rt = np.array(self.poseRt.iloc[viewIdx]).reshape((3, 4))
        Rt4x4 = np.eye(4)
        Rt4x4[:3] = Rt

        return Rt4x4

    def build_json(self, train_or_test: bool):

        list_frames = (
            self.train_framesIdx if train_or_test == "train" else self.test_framesIdx
        )

        for i, idx_frame in tqdm(enumerate(list_frames)):

            RT = self.get_extrinsic(idx_frame)

            if train_or_test == "train":
                # Ensure that RT is going to be write at the write location in the dictionnary.
                assert (
                    os.path.basename(self.train_data["infos"][i]["img_basename"])
                    == str(idx_frame).zfill(6) + ".png"
                )

                self.train_data["infos"][i]["transform_matrix"] = RT.tolist()

            else:
                # Ensure that RT is going to be write at the write location in the dictionnary.
                assert (
                    os.path.basename(self.test_data["infos"][i]["img_basename"])
                    == str(idx_frame).zfill(6) + ".png"
                )

                self.test_data["infos"][i]["transform_matrix"] = RT.tolist()

    def save_json(self):

        filename_train_json = os.path.join(self.ROOT_PATH_RGB, "transforms_train.json")
        filename_test_json = os.path.join(self.ROOT_PATH_RGB, "transforms_test.json")

        f_train = open(filename_train_json, "w")
        f_test = open(filename_test_json, "w")

        json.dump(self.train_data, f_train, cls=NpEncoder)
        json.dump(self.test_data, f_test, cls=NpEncoder)

    def create_complete_file(self):
        arr = np.arange(self.nb_frames)
        # Create the .json structure for both training and testing set.
        self.data = {
            "in_dir": self.ROOT_PATH_RGB,
            "nb_frames": len(arr),
            "selected_samples_id": list(arr),
            "cameraLoc": "Cam2",
            "stereoLR": None,
            "intrisic_matrix": self.compute_intrinsic_K().tolist(),
            "infos": [
                {"img_basename": str(idx).zfill(6) + ".png", "transform_matrix": None}
                for idx in arr
            ],
        }
        for i, idx_frame in tqdm(enumerate(arr)):

            RT = self.get_extrinsic(idx_frame)

            # Ensure that RT is going to be write at the write location in the dictionnary.
            assert (
                os.path.basename(self.data["infos"][i]["img_basename"])
                == str(idx_frame).zfill(6) + ".png"
            )

            self.data["infos"][i]["transform_matrix"] = RT.tolist()

        filename_json = os.path.join(
            self.ROOT_PATH_RGB, "transforms_completeScene.json"
        )
        f = open(filename_json, "w")

        json.dump(self.data, f, cls=NpEncoder)

    def create_json_files(self):

        print(f"Start building the json files for the scene {self.scene_name}...")

        # Same train/test strategy as used in the Baseline paper.
        startIndexingNull = 0

        arr = (
            np.arange(self.nb_frames)
            if not startIndexingNull
            else np.arange(1, self.nb_frames + 1)
        )
        np.random.shuffle(arr)

        # Indexes are no more sorted here !
        self.train_framesIdx = sorted(arr[: int(0.8 * self.nb_frames)])
        self.test_framesIdx = sorted(arr[int(0.8 * self.nb_frames) + 1 :])

        # Create the .json structure for both training and testing set.
        self.train_data = {
            "in_dir": self.ROOT_PATH_RGB,
            "selected_samples_id": self.train_framesIdx,
            "nb_frames": len(self.train_framesIdx),
            "cameraLoc": "Cam2",
            "stereoLR": None,
            "intrisic_matrix": self.compute_intrinsic_K().tolist(),
            "infos": [
                {"img_basename": str(idx).zfill(6) + ".png", "transform_matrix": None}
                for idx in self.train_framesIdx
            ],
        }

        self.test_data = {
            "in_dir": self.ROOT_PATH_RGB,
            "selected_samples_id": self.test_framesIdx,
            "nb_frames": len(self.test_framesIdx),
            "cameraLoc": "Cam2",
            "stereoLR": None,
            "intrisic_matrix": self.compute_intrinsic_K().tolist(),
            "infos": [
                {"img_basename": str(idx).zfill(6) + ".png", "transform_matrix": None}
                for idx in self.test_framesIdx
            ],
        }

        #  Build the two json file for training and testing set.
        self.build_json(train_or_test="train")
        self.build_json(train_or_test="test")

        # Save these json files.
        self.save_json()


class KittiBuilderNPY:
    def __init__(self, in_dir):

        self.indir = in_dir
        df = pd.read_csv("/data/datasets/KITTI/kitti_scene_infos.csv")
        self.scene_info = df.set_index("scene_id")["scene_frame_numbers"].to_dict()

        self.scene_list = list(self.scene_info.keys())
        self.scene_number = len(self.scene_list)

        self.totFrames = np.sum([v for _, v in self.scene_info.items()])
        print(f"Retrieved scene dictionnary: {self.scene_info}")
        print(f"Total number of frames to store: {self.totFrames}")

        self.build_npy(out_dir="/data/datasets/NVS_Skip")

    def build_npy(self, out_dir):
        # Empty np array that is gonna be filled with images.
        data_img_npy = np.zeros((self.totFrames, 256, 256, 3))
        i = 0
        for scene in tqdm(self.scene_list):
            rootPath = os.path.join(
                self.indir, "RGB_rescaled_256x256", str(scene).zfill(2), "image_2"
            )
            print(f"Current scene processed: {scene}")
            imgsName = sorted(
                [l for l in os.listdir(rootPath) if not l.startswith(".")]
            )
            for imgName in imgsName:
                pathImg = os.path.join(rootPath, imgName)
                img = cv2.imread(pathImg)[:, :, ::-1]
                data_img_npy[i, :] = img
                i += 1

        # Save the entire data array
        outPath = os.path.join(out_dir, "kitti_imageOurs.npy")
        print(f"Saving the .npy at {outPath}...")
        np.save(outPath, data_img_npy)


if __name__ == "__main__":

    # kittiBuilder = KittiBuilderNPY(in_dir= '/data/datasets/KITTI')

    SEQS = [str(i) for i in range(11)]
    for seq in SEQS:
        testKitti = KittiBuilder(in_dir="/data/datasets/KITTI", seq_nb=seq)
        testKitti.create_complete_file()
