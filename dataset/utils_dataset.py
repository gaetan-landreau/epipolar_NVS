import os
import json
from numpy.random import default_rng
import numpy as np
import cv2
import ast

from tqdm import tqdm

from utils.utils_epipolar import Nviews

SHAPENET_ID_MATCHING = {"03001627": "chair", "02958343": "car"}


H = 760
W = 1280


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class PostProcess:
    def __init__(self, in_dirRoot, sub_dirnames, target_shape, imgExt):

        self.in_dirRoot = in_dirRoot
        self.sub_dirnames = sub_dirnames

        self.current_dirs = [
            d
            for d in os.listdir(self.in_dirRoot)
            if (
                not d.startswith(".")
                and os.path.isdir(os.path.join(self.in_dirRoot, d))
            )
        ]

        self.ts = target_shape  # Width,Height

    @staticmethod
    def create_dirs(path):
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=False)

    def resizeAndSave(self, img, out_path):
        try:
            img = cv2.resize(
                img, (int(self.ts[0]), int(self.ts[1])), interpolation=cv2.INTER_LINEAR
            )
            cv2.imwrite(out_path, img)
        except Exception as e:
            print(e)
            print(f"Issue for resizing the image: {out_path}")

    def resizeAllSynthia(self, out_dir):

        # For each SEASONS-SEQS folder
        for dir in tqdm(self.current_dirs):
            print(f"Dir. currently processed: {dir}")
            sub_dirInner_path = "/".join([d for d in self.sub_dirnames])

            rootDirs = os.path.join(self.in_dirRoot, dir, sub_dirInner_path)

            # Through each RGB folder.
            for rootDirs, dirs, files in os.walk(rootDirs):
                for subdir in dirs:
                    if subdir in ["Stereo_Right", "Stereo_Left"]:
                        continue

                    # Through each cameraLoc position.
                    dir_imgsIn = os.path.join(rootDirs, subdir)

                    dir_imgsOut = dir_imgsIn.replace(
                        "RGB", f"RGB_rescaled_{self.ts[0]}x{self.ts[1]}"
                    )
                    os.makedirs(dir_imgsOut, exist_ok=True)
                    for imgBasename in os.listdir(dir_imgsIn):

                        imgPathIn = os.path.join(dir_imgsIn, imgBasename)
                        imgPathOut = os.path.join(dir_imgsOut, imgBasename)

                        img = cv2.imread(imgPathIn)

                        self.resizeAndSave(img, imgPathOut)

    def resizeAllKitti(self):
        # For each sequence of KITTI dataset.
        for dir in tqdm(self.current_dirs):
            sub_dirInner_path = "/".join([d for d in self.sub_dirnames])

            dir_imgsIn = os.path.join(self.in_dirRoot, dir, sub_dirInner_path)
            dir_imgsOut = dir_imgsIn.replace(
                "RGB", f"RGB_rescaled_{self.ts[0]}x{self.ts[1]}"
            )

            os.makedirs(dir_imgsOut, exist_ok=True)

            imgsBasename = [
                f for f in os.listdir(dir_imgsIn) if not f.startswith(".")
            ]  # avoid collecting hidden files.

            for imgBasename in imgsBasename:
                # Do not consider the .json files.
                if os.path.splitext(imgBasename)[-1] in [".json"]:
                    continue

                imgPathIn = os.path.join(dir_imgsIn, imgBasename)
                imgPathOut = os.path.join(dir_imgsOut, imgBasename)

                img = cv2.imread(imgPathIn)
                self.resizeAndSave(img, imgPathOut)


class ImageUtils:
    def __init__(self):

        self.a = 10

    @staticmethod
    def read_img(img_full_path):
        return cv2.imread(img_full_path).astype(np.float32)

    @staticmethod
    def resize(I, target_size):
        return cv2.resize(I, (target_size, target_size))


if __name__ == "__main__":

    rootDir = "/data/datasets/SynthiaTemp"
    pp = PostProcess(
        in_dirRoot=rootDir, sub_dirnames=["RGB"], target_shape=[256, 256], imgExt="png"
    )
    pp.resizeAllSynthia(out_dir=".")

    """rootDir = "/data/datasets/KITTI/RGB"
    pp=PostProcess(in_dirRoot=rootDir,sub_dirnames=['image_2'],target_shape=[256,256],imgExt='png')
    pp.resizeAllKitti()"""
