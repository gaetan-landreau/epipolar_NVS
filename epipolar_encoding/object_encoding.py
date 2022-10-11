from .base_encoding import BaseEncode
from utils.utils_epipolar import (
    os,
    json,
    np,
    cv2,
    random,
    NULL_COLOR,
    SHAPENET_ID_MATCHING,
)


class EncodePoseObject(BaseEncode):
    def __init__(
        self,
        in_dir,
        Id,
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

        self.Id = Id

        self.in_dir = in_dir

        self.name = SHAPENET_ID_MATCHING[self.Id]

        self.root_path = os.path.join(
            self.in_dir, Id
        )  # Update the root_path with the corresponding ShapeNet Id object.

        fileJson = os.path.join(self.root_path, self.fileJson_basename)

        with open(fileJson, "r") as f:
            data = json.load(f)
            self.data_camera_info = data["frames"]

        ##################
        # Camera Intrinsic
        self.K = self.get_intrinsicK(data)

        self.Kinv = np.linalg.inv(
            self.K
        )  # required to compute the Fundamental matrix F.

        #####################
        # Sampled id objects.
        self.list_id_obj = data["selected_samples_id"]

    def getIdwithIdx(self, idx_obj: int) -> int:
        return self.data_camera_info[idx_obj]["id_obj"]
    
    def getIdxwithId(self, id_obj: int) -> int:
        return self.list_id_obj.index(id_obj)
    
    def get_intrinsicK(self, dict: dict) -> np.array:
        return dict["intrinsic_matrix"]

    def readImg(self, id_obj: int, idx_img: int) -> np.array:
        return cv2.imread(
            os.path.join(self.root_path, id_obj, "easy", str(idx_img).zfill(2) + ".png")
        )[:, :, ::-1].astype(np.float32)

    def get_extrinsicRt(self, idx_obj: int, img_idx: int) -> np.array:
        """
        Retrieve the camera pose associated to the two views that were provided during object instantiation.

        Returns:
            Rt [np.array]: a 3x4 rotation matrix concatenated as [R|t]
        """
        # Retrieve extrinsic transformation matrix.
        transform_matrix = self.data_camera_info[idx_obj]["infos"][img_idx][
            "transform_matrix"
        ]
        Rt = np.asarray(transform_matrix)

        return Rt
