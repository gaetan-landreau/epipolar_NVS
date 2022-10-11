from .base_encoding import BaseEncode
from utils.utils_epipolar import os, json, np, cv2, random, NULL_COLOR


class EncodePoseScene(BaseEncode):
    def __init__(
        self,
        in_dir,
        img_shape,
        train_or_test,
        samplingStrategy,
        extendedTranslationMotion,
    ):

        super().__init__(
            in_dir,
            img_shape,
            train_or_test,
            samplingStrategy,
            extendedTranslationMotion,
        )

        self.scene_list_Id = [
            scene_id
            for scene_id in os.listdir(in_dir)
            if os.path.isdir(os.path.join(in_dir, scene_id))
        ]

    def setJsonFile(self):
        self.pathJsonFile = os.path.join(
            self.root_path, self.Id, self.fileJson_basename
        )

    def loadJson(self) -> dict:
        with open(self.pathJsonFile, "r") as f:
            data = json.load(f)
            self.data_camera_info = data["infos"]
        return data

    def setK(self, Knew: np.array):
        self.K = Knew

    def setUpConfig(self, Id: str):
        self.Id = Id
        self.setJsonFile()
        data = self.loadJson()

        self.K = EncodePoseScene.get_intrinsicK(data)
        self.Kinv = np.linalg.inv(self.K)

        self.nb_frames = EncodePoseScene.get_nbTotFrames(data)
        self.IdFrames = EncodePoseScene.get_IdFrames(data)

        self.camLoc = EncodePoseScene.get_cameraLoc(data)
        self.stereoLR = EncodePoseScene.get_stereoLR(data)

    def sampleRandomScene(self) -> str:

        # Return a scene name where to sample images and camera pose from.
        return random.choice(self.scene_list_Id)

    @staticmethod
    def get_nbTotFrames(data: dict) -> int:
        return data["nb_frames"]

    @staticmethod
    def get_IdFrames(data: dict) -> list:
        return data["selected_samples_id"]

    @staticmethod
    def get_stereoLR(data: dict) -> str:
        return data["stereoLR"]

    @staticmethod
    def get_cameraLoc(data: dict) -> str:
        return data["cameraLoc"]

    @staticmethod
    def get_intrinsicK(data: dict) -> np.array:
        return np.asarray(data["intrisic_matrix"])

    def set_imgShape(self, imgShape: list):
        self.H = imgShape[0]
        self.W = imgShape[1]

    def get_extrinsicRt(self, img_idx: int) -> np.array:
        # First view.
        transform_matrix = self.data_camera_info[img_idx]["transform_matrix"]
        pose = np.asarray(transform_matrix)
        Rt = pose[:3, :]

        return Rt

    def readImg(self, name: str, idx_img: int, **kwargs) -> np.array:

        # Required for Synthia parsing only
        stereoLR = kwargs.get("StereoLR", None)
        camLoc = kwargs.get("camLoc", None)

        pathImg = (
            os.path.join(
                self.root_path,
                self.Id,
                "RGB_rescaled_256x256",
                stereoLR,
                camLoc,
                str(idx_img).zfill(6) + ".png",
            )
            if name == "synthia"
            else os.path.join(
                self.root_path, self.Id, "image_2", str(idx_img).zfill(6) + ".png"
            )
        )

        return cv2.imread(pathImg)[:, :, ::-1].astype(np.float32)  # uint8
