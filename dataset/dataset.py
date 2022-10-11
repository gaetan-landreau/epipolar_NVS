import numpy as np


class DataLoader:
    sigmas = [0.0, 0.02, 0.050]

    def __init__(self, name, image_size=512):
        path_dict = {}
        path_dict["kitti"] = "kitti/data_kitti.hdf5"
        path_dict["synthia"] = "synthia/data_synthia.hdf5"
        path_dict["chair"] = "shapenet/data_chair.hdf5"
        path_dict["car"] = "shapenet/data_car.hdf5"
        path_dict["nerf"] = "nerf/data_nerf.hdf5"

        file_directory = "/data/datasets/legoNeRF/test/"

        # assert name in path_dict.keys()

        self.name = name
        file_path = file_directory + path_dict[name]
        self.file_path = file_path
        self.image_size = image_size
        self.pose_size = 36
        self.data = None

        self.noise = False

    def get_batched_data(
        self,
        batch_size=32,
        single_model=True,
        model_name=None,
        verbose=False,
        return_info=False,
        is_train=True,
    ):
        pass

    def get_specific_data(self, target_data_info):
        pass

    @staticmethod
    def add_noise(image):
        sigma = np.random.choice(DataLoader.sigmas)
        noise = np.random.normal(0, sigma, image.shape)
        noised_image = image + noise
        return np.clip(noised_image, 0, 1)
