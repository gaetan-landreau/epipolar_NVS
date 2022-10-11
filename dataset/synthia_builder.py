import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

from .utils_dataset import NpEncoder

df = pd.read_csv("/data/datasets/NVS_Skip/synthia_scene_infosReduced.csv")

RESIZING_FACTOR = 4.0
TARGET_SIZE = (256, 256)  # (192,320)


class SynthiaBuilder:
    def __init__(self, in_dir, seq_nb, season, stereoLR, cameraLoc):

        self.in_dir = in_dir
        self.SEQ = seq_nb
        self.SEASON = season

        self.stereoLR = stereoLR
        self.cameraLoc = cameraLoc

        self.scene_name = f"SYNTHIA-SEQS-{self.SEQ.zfill(2)}-{self.SEASON}"

        self.set_root_path()
        self.set_nb_frames()

    def set_root_path(self):
        dir_name = "-".join(["SYNTHIA", "SEQS", self.SEQ.zfill(2), self.SEASON])
        self.ROOT_PATH = os.path.join(self.in_dir, dir_name)

    def set_nb_frames(self):
        self.nb_frames = df[df["scene_id"] == self.scene_name][
            "scene_frame_numbers"
        ].to_list()[0]

    def compute_intrinsic_K(self):
        file_path = os.path.join(self.ROOT_PATH, "CameraParams", "intrinsics.txt")
        f = open(file_path, "r")

        params = sorted(
            list(set([info.strip() for info in f.readlines() if info.strip() != ""]))
        )

        f_pix = params[2]
        u0, v0 = params[-1], params[1]

        self.K = np.eye(3)

        # Original Intrisic K without any rescaling.
        self.K[0, -1] = float(u0)  # TARGET_SIZE[-1]/2. #float(u0)/RESIZING_FACTOR
        self.K[1, -1] = float(v0)  # TARGET_SIZE[0]/2. #float(v0)/RESIZING_FACTOR
        self.K[0, 0] = float(f_pix)
        self.K[1, 1] = float(f_pix)

        # Correct K according to the scaling impose by TARGET_SIZE
        scale_u = TARGET_SIZE[0] / (2 * float(u0))
        scale_v = TARGET_SIZE[1] / (2 * float(v0))

        self.K[0] *= scale_u
        self.K[1] *= scale_v

        return self.K

    @staticmethod
    def get_extrinsic(pose_path: str) -> np.array:

        f = open(pose_path, "r")
        raw_pose = f.readlines()[0].strip()
        RT = np.asarray([float(l) for l in raw_pose.split(" ")]).reshape(4, 4).T
        return RT

    def build_json(self, train_or_test: bool):

        list_frames = (
            self.train_framesIdx if train_or_test == "train" else self.test_framesIdx
        )

        for i, idx_frame in tqdm(enumerate(list_frames)):

            pose_path = os.path.join(
                self.ROOT_PATH,
                "CameraParams",
                self.stereoLR,
                self.cameraLoc,
                str(idx_frame).zfill(6) + ".txt",
            )
            RT = SynthiaBuilder.get_extrinsic(pose_path)

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

        filename_train_json = os.path.join(self.ROOT_PATH, "transforms_train.json")
        filename_test_json = os.path.join(self.ROOT_PATH, "transforms_test.json")

        f_train = open(filename_train_json, "w")
        f_test = open(filename_test_json, "w")

        json.dump(self.train_data, f_train, cls=NpEncoder)
        json.dump(self.test_data, f_test, cls=NpEncoder)

    def create_complete_file(self):
        startIndexingNull = (
            1 if self.scene_name in ["SYNTHIA-SEQS-05-FALL"] else 0
        )  # Issue with indexing in this scene, start at 1

        arr = (
            np.arange(self.nb_frames)
            if not startIndexingNull
            else np.arange(1, self.nb_frames + 1)
        )

        # Create the .json structure for both training and testing set.
        self.data = {
            "in_dir": self.ROOT_PATH,
            "nb_frames": len(arr),
            "selected_samples_id": list(arr),
            "cameraLoc": self.cameraLoc,
            "stereoLR": self.stereoLR,
            "intrisic_matrix": self.compute_intrinsic_K().tolist(),
            "infos": [
                {"img_basename": str(idx).zfill(6) + ".png", "transform_matrix": None}
                for idx in arr
            ],
        }
        for i, idx_frame in tqdm(enumerate(arr)):

            pose_path = os.path.join(
                self.ROOT_PATH,
                "CameraParams",
                self.stereoLR,
                self.cameraLoc,
                str(idx_frame).zfill(6) + ".txt",
            )
            RT = SynthiaBuilder.get_extrinsic(pose_path)

            # Ensure that RT is going to be write at the write location in the dictionnary.
            assert (
                os.path.basename(self.data["infos"][i]["img_basename"])
                == str(idx_frame).zfill(6) + ".png"
            )

            self.data["infos"][i]["transform_matrix"] = RT.tolist()

        filename_json = os.path.join(self.ROOT_PATH, "transforms_completeScene.json")
        f = open(filename_json, "w")

        json.dump(self.data, f, cls=NpEncoder)

    def create_json_files(self):

        print(f"Start building the json files for the scene {self.scene_name}...")

        # Same train/test strategy as used in the Baseline paper.

        startIndexingNull = (
            1 if self.scene_name in ["SYNTHIA-SEQS-05-FALL"] else 0
        )  # Issue with indexing in this scene, start at 1

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
            "in_dir": self.ROOT_PATH,
            "selected_samples_id": self.train_framesIdx,
            "nb_frames": len(self.train_framesIdx),
            "cameraLoc": self.cameraLoc,
            "stereoLR": self.stereoLR,
            "intrisic_matrix": self.compute_intrinsic_K().tolist(),
            "infos": [
                {"img_basename": str(idx).zfill(6) + ".png", "transform_matrix": None}
                for idx in self.train_framesIdx
            ],
        }

        self.test_data = {
            "in_dir": self.ROOT_PATH,
            "selected_samples_id": self.test_framesIdx,
            "nb_frames": len(self.test_framesIdx),
            "cameraLoc": self.cameraLoc,
            "stereoLR": self.stereoLR,
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


class SynthiaBuilderNPY:
    def __init__(self, in_dir):

        self.indir = in_dir
        df = pd.read_csv("/data/datasets/NVS_Skip/synthia_scene_infosReduced.csv")
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
                self.indir, scene, "RGB_rescaled_256x256", "Stereo_Left", "Omni_F"
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
        outPath = os.path.join(out_dir, "synthia_imageOurs.npy")
        print(f"Saving the .npy at {outPath}...")
        np.save(outPath, data_img_npy)


if __name__ == "__main__":
    # testSynthiaNPY = SynthiaBuilderNPY(in_dir = "/data/datasets/Synthia")

    SEASON = ["WINTER", "SPRING", "SUMMER", "FALL"]
    SEQS = ["1", "2", "4", "5"]
    for s in SEASON:
        for seq in SEQS:

            testSynthia = SynthiaBuilder(
                in_dir="/data/datasets/Synthia",
                seq_nb=seq,
                season=s,
                stereoLR="Stereo_Left",
                cameraLoc="Omni_F",
            )
            testSynthia.create_complete_file()
