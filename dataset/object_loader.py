from .dataset import DataLoader
from epipolar_encoding.object_encoding import EncodePoseObject
from utils.utils_epipolar import N_MAX_TEST, N_MAX_TRAIN, Nviews,L_Car_restrictedTest, L_Chair_restrictedTest
from utils.utils_dataset import ImageUtils

import tensorflow as tf
import numpy as np
import cv2
import numpy as np
import random 

#######################################
#### DATALOADER TO WORK WITH .NPY FILES
#######################################
class ObjectDataLoaderNumpy(DataLoader):
    def __init__(self, name, image_size=256, train_or_test="train"):

        super().__init__(name, image_size)

        file_name = "%s_%s_%d" % (train_or_test, name, 224)
        self.all_images = np.load("/data/datasets/ShapeNet/%s.npy" % file_name)
        self.dataset_format = "npy"
        self.n_elevation = 1
        self.n_azimuth = 36
        self.n_models = self.all_images.shape[0]
        self.min_elevation = 0
        self.max_elevation = 0

    def get_image_from_info(self, model_name, az, el=-1):
        return (
            ImageUtils.resize(self.all_images[model_name, az], self.image_size) / 255.0
        )

    def get_specific_data(self, target_data_info):
        batch_size = len(target_data_info)
        input_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )
        target_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )
        input_elevations = np.zeros((batch_size,), dtype=np.float32)
        input_azimuths = np.zeros((batch_size,), dtype=np.float32)
        target_elevations = np.zeros((batch_size,), dtype=np.float32)
        target_azimuths = np.zeros((batch_size,), dtype=np.float32)

        for i in range(batch_size):
            m, ia, ie, ta, te = target_data_info[i]
            input_images[i] = self.get_image_from_info(m, ia, ie)
            target_images[i] = self.get_image_from_info(m, ta, te)
            input_elevations[i] = ie
            input_azimuths[i] = ia
            target_elevations[i] = te
            target_azimuths[i] = ta
        return (
            input_images,
            target_images,
            (input_elevations, input_azimuths, target_elevations, target_azimuths),
        )

    def get_batched_data(
        self,
        batch_size=32,
        single_model=False,
        model_name=None,
        verbose=False,
        return_info=False,
        is_train=False,
    ):

        input_random_elevations = np.random.randint(self.n_elevation, size=batch_size)
        input_random_azimuths = np.random.randint(self.n_azimuth, size=batch_size)
        target_random_elevations = np.random.randint(self.n_elevation, size=batch_size)
        target_random_azimuths = np.random.randint(self.n_azimuth, size=batch_size)

        target_model = np.random.randint(self.n_models)

        input_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )
        target_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )
        index_infos = []
        for i in range(batch_size):
            if not single_model:
                target_model = np.random.randint(self.n_models)
            input_images[i] = self.get_image_from_info(
                target_model, input_random_azimuths[i], input_random_elevations[i]
            )
            target_images[i] = self.get_image_from_info(
                target_model, target_random_azimuths[i], target_random_elevations[i]
            )
            index_infos.append(
                (
                    target_model,
                    input_random_azimuths[i],
                    input_random_elevations[i],
                    target_random_azimuths[i],
                    target_random_elevations[i],
                )
            )

        if return_info:
            data_tuple = (
                input_images,
                target_images,
                (
                    input_random_elevations,
                    input_random_azimuths,
                    target_random_elevations,
                    target_random_azimuths,
                ),
            )
            return data_tuple, index_infos
        else:
            return (
                input_images,
                target_images,
                (
                    input_random_elevations,
                    input_random_azimuths,
                    target_random_elevations,
                    target_random_azimuths,
                ),
            )

    def get_batched_data_i_j(self, source, target, model_min_index, model_max_index):
        N = model_max_index - model_min_index

        input_random_elevations = np.repeat(0, N)
        target_random_elevations = np.repeat(0, N)

        input_random_azimuths = np.repeat(source, N)
        target_random_azimuths = np.repeat(target, N)

        input_images = self.all_images[model_min_index:model_max_index, source]
        target_images = self.all_images[model_min_index:model_max_index, target]

        return (
            input_images,
            target_images,
            (
                input_random_elevations,
                input_random_azimuths,
                target_random_elevations,
                target_random_azimuths,
            ),
        )

###################################
#### DATALOADER TO WORK WITH IMAGES
###################################
class OursDataObjectLoader(DataLoader):
    def __init__(
        self,
        name,
        image_size,
        train_or_test,
        samplingStrategy,
        extendedTranslationMotion,
    ):
        super().__init__(name, image_size)

        id_class = (
            "03001627" if name == "chair" else "02958343"
        )  # Either Chair or Car class for ShapeNet dataset.

        self.dataset_format = "img"

        self.n_elevation = 1
        self.n_azimuth = 36
        self.min_elevation = 0
        self.max_elevation = 0

        ######################################################
        # Instantiate an EncodePoseObject pose encoding object.
        self.epiNVS = EncodePoseObject(
            img_shape=image_size,
            Id=id_class,
            in_dir="/data/datasets/ShapeNet/",
            train_or_test=train_or_test,
            samplingStrategy=samplingStrategy,
            extendedTranslationMotion=extendedTranslationMotion,
        )
        
        self.restrictedCameraTransformation = False

    def get_batched_data(self, batch_size=32, is_train=True, single_model=True):

        # Numpy array data.
        source_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )
        target_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )
        encoded_poses = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )

        N_MAX = N_MAX_TRAIN if is_train else N_MAX_TEST
        list_idx_obj = np.random.randint(N_MAX, size=batch_size)

        for i, idx_obj in enumerate(list_idx_obj):

            # Given the index object we processing for the batch, we need to retrieve the corresponding id.
            id_obj = self.epiNVS.getIdwithIdx(idx_obj)

            # Sample two random views.
            img_idx1 = np.random.randint(Nviews)
            img_idx2 = np.random.randint(Nviews) if not self.restrictedCameraTransformation else (img_idx1 + np.random.randint(-4,5))%Nviews

            Rt1 = self.epiNVS.get_extrinsicRt(idx_obj, img_idx1)
            Rt2 = self.epiNVS.get_extrinsicRt(idx_obj, img_idx2)

            E, _ = self.epiNVS.computeEssential(Rt1, Rt2)
            F = self.epiNVS.computeFundamental(E, self.epiNVS.Kinv)

            # Read both of the images.
            I1 = self.epiNVS.readImg(id_obj, img_idx1)
            I2 = self.epiNVS.readImg(id_obj, img_idx2)

            I1_resized = cv2.resize(
                I1, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )
            I2_resized = cv2.resize(
                I2, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )
            
            if self.noise:
                I1_resized = DataLoader.add_noise(I1_resized / 255.0)
                
            try:
                # Encode the pose from I2 based on I1 color.
                E12 = self.epiNVS.encodePose(I1_resized*255., F)
            except Exception as e:
                print(e)
                print(f"Issue with Fundamental matrix computation")
                E12 = np.zeros((256, 256, 3)).astype(np.float32)
            

            # Normalize both source/target images and the encoded pose.
            source_images[i] = I1_resized if self.noise else I1_resized / 255.0
            target_images[i] = I2_resized / 255.0
            encoded_poses[i] = E12 / 255.0

        return source_images, target_images, encoded_poses
    
    def get_restricted_batch(self,batch_size):
        # Numpy array data.
        source_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )
        target_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )
        encoded_poses = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )

        
        list_id_obj = L_Car_restrictedTest[:batch_size] if self.name == "car" else L_Chair_restrictedTest[:batch_size]
        
        for i, id_obj in enumerate(list_id_obj):

            # Given the ID object we processing for the batch, we need to retrieve the corresponding idx.
            idx_obj = self.epiNVS.getIdxwithId(id_obj)

            # Sample two random views.
            img_idx1 = np.random.randint(Nviews)
            #img_idx2 = np.random.randint(Nviews) if not self.restrictedCameraTransformation else (img_idx1 + np.random.randint(-4,5))%Nviews
            img_idx2 = np.random.randint(Nviews) if not self.restrictedCameraTransformation else \
                       (img_idx1 + random.choice([x for x in range(-18,19) if (x<=-6 or x>=6)]))%Nviews
            #print(f'Sampled pair: {img_idx1,img_idx2}')
            
            Rt1 = self.epiNVS.get_extrinsicRt(idx_obj, img_idx1)
            Rt2 = self.epiNVS.get_extrinsicRt(idx_obj, img_idx2)

            E, _ = self.epiNVS.computeEssential(Rt1, Rt2)
            F = self.epiNVS.computeFundamental(E, self.epiNVS.Kinv)

            # Read both of the images.
            I1 = self.epiNVS.readImg(id_obj, img_idx1)
            I2 = self.epiNVS.readImg(id_obj, img_idx2)
          
            I1_resized = cv2.resize(
                I1, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )
            I2_resized = cv2.resize(
                I2, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )

            if self.noise:
                I1_resized = DataLoader.add_noise(I1_resized / 255.0)

            try:
                # Encode the pose from I2 based on I1 color.
                E12 = self.epiNVS.encodePose(I1_resized*255., F)
            except Exception as e:
                print(e)
                print(f"Issue with Fundamental matrix computation")
                E12 = np.zeros((256, 256, 3)).astype(np.float32)
            
            # Normalize both source/target images and the encoded pose.
            source_images[i] = I1_resized if self.noise else I1_resized / 255.0
            target_images[i] = I2_resized / 255.0
            encoded_poses[i] = E12 / 255.0

        return source_images, target_images, encoded_poses
        
        
    def get_multiview_same_instance(self, batch_size=8, is_train=False):

        # Numpy array data.
        source_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )
        target_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )
        encoded_poses = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )

        N_MAX = N_MAX_TRAIN if is_train else N_MAX_TEST
        idx_obj = np.random.randint(N_MAX)
        id_obj = self.epiNVS.getIdwithIdx(idx_obj)
        for i in range(batch_size):
            # Given the index object we processing for the batch, we need to retrieve the corresponding id.

            # Sample two random views.
            img_idx1 = 10
            img_idx2 = 4 * i + 1

            Rt1 = self.epiNVS.get_extrinsicRt(idx_obj, img_idx1)
            Rt2 = self.epiNVS.get_extrinsicRt(idx_obj, img_idx2)

            E, _ = self.epiNVS.computeEssential(Rt1, Rt2)
            F = self.epiNVS.computeFundamental(E, self.epiNVS.Kinv)

            # Read both of the images.
            I1 = self.epiNVS.readImg(id_obj, img_idx1)
            I2 = self.epiNVS.readImg(id_obj, img_idx2)

            I1_resized = cv2.resize(
                I1, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )
            I2_resized = cv2.resize(
                I2, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )

            try:
                # Encode the pose from I2 based on I1 color.
                E12 = self.epiNVS.encodePose(I1_resized, F)
            except Exception as e:
                print(e)
                print(f"Issue with Fundamental matrix computation")
                E12 = np.zeros((256, 256, 3)).astype(np.float32)

            # Normalize both source/target images and the encoded pose.
            source_images[i] = I1_resized / 255.0
            target_images[i] = I2_resized / 255.0
            encoded_poses[i] = E12 / 255.0

        return source_images, target_images, encoded_poses


if __name__ == "__main__":
    import numpy as np

    sceneDataSynthia = OursDataObjectLoader(
        name="ShapeNet",
        image_size=256,
        train_or_test="train",
        samplingStrategy={"strategy": "gridSampling", "param": 15},
        extendedTranslationMotion=False,
    )
