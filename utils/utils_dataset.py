import os
import numpy as np
import cv2
from tqdm import tqdm


from .utils_epipolar import H, W


class PostProcess:
    def __init__(self, in_dirRoot, sub_dirnames, targetShape):

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

        self.outShape = targetShape

    @staticmethod
    def create_dirs(path):
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=False)

    def resizeAndSave(self, img, out_path):

        img = cv2.resize(
            img, (self.outShape[-1], self.outShape[0]), interpolation=cv2.INTER_LINEAR
        )

        cv2.imwrite(out_path, img)

    def resizeAll(self, out_dir):

        # For each SEASONS-SEQS folder
        for dir in tqdm(self.current_dirs):

            sub_dirInner_path = "/".join([d for d in self.sub_dirnames])

            rootDirs = os.path.join(self.in_dirRoot, dir, sub_dirInner_path)

            # Through each RGB folder.
            for rootDirs, dirs, files in os.walk(rootDirs):
                for subdir in dirs:
                    if subdir in ["Stereo_Right", "Stereo_Left"]:
                        continue

                    # Through each cameraLoc position.
                    dir_imgsIn = os.path.join(rootDirs, subdir)
                    dir_imgsOut = dir_imgsIn.replace("RGB", "RGB_rescaled_x4")
                    os.makedirs(dir_imgsOut, exist_ok=True)
                    for imgBasename in os.listdir(dir_imgsIn):

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

    # Resize the whole Synthia dataset.
    rootDir = "/data/datasets/Synthia/"
    pp = PostProcess(in_dirRoot=rootDir, sub_dirnames=["RGB"], targetShape=[192, 320])
    pp.resizeAll(out_dir=".")
