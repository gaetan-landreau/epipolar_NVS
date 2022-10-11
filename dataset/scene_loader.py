from .dataset import DataLoader
from epipolar_encoding.scene_encoding import EncodePoseScene

import numpy as np
import pandas as pd
import os
import random
import time

#######################################
#### DATALOADER TO WORK WITH .NPY FILES
#######################################
class SceneDataLoaderNumpy(DataLoader):
    def __init__(self, name, use_pose_matrix=False, image_size=256, maxFrames=10):
        """
        For scene data loader, it both contains train and test dataset.
        """
        super().__init__(name, image_size)

        csvToLoad = (
            "kitti_scene_infos.csv"
            if self.name == "kitti"
            else "synthia_scene_infosReduced.csv"
        )
        df = pd.read_csv(f"/data/datasets/NVS_Skip/{csvToLoad}")

        scene_info = df.set_index("scene_id")["scene_frame_numbers"].to_dict()
        self.image_numbers_per_scene = scene_info
        self.scene_list = list(scene_info.keys())
        self.scene_number = len(self.scene_list)

        self.train_ids = {}
        self.test_ids = {}

        self.all_ids = {}

        np.random.seed(100)

        self.dataset_format = "npy"

        for scene_id, scene_frame_n in self.image_numbers_per_scene.items():
            arr = np.arange(scene_frame_n)
            np.random.shuffle(arr)
            self.train_ids[scene_id] = np.asarray(
                sorted(arr[0 : int(0.8 * scene_frame_n)].tolist())
            )
            self.test_ids[scene_id] = np.asarray(
                sorted(arr[int(0.8 * scene_frame_n) + 1 : -1].tolist())
            )
            self.all_ids[scene_id] = np.asarray(sorted(arr.tolist()))

        self.is_pose_matrix = use_pose_matrix
        self.pose_size = (
            6 if not use_pose_matrix else (12 if self.name == "kitti" else 16)
        )

        self.max_frame_difference = maxFrames

        self.scene_offsets = {}
        offset = 0
        for scene_id in self.scene_list:
            self.scene_offsets[scene_id] = offset
            offset += self.image_numbers_per_scene[scene_id]

        self.all_images = np.load(
            "/data/datasets/NVS_Skip/%s_imageOurs.npy" % self.name
        )
        self.all_poses = np.load("/data/datasets/NVS_Skip/%s_pose.npy" % self.name)
        self.all_pose_matrices = np.load(
            "/data/datasets/NVS_Skip/%s_pose_matrix.npy" % self.name
        )

    def get_image_pose(self, scene_id, frame_n):
        image = self.all_images[self.scene_offsets[scene_id] + frame_n]
        image = image.astype(np.float32)
        image = image / 255
        if self.is_pose_matrix:
            pose = self.all_pose_matrices[self.scene_offsets[scene_id] + frame_n]
        else:
            pose = self.all_poses[self.scene_offsets[scene_id] + frame_n]
        return image, pose

    def get_single_data_tuple(self, scene_id, is_train=True):
        frame_difference = np.random.randint(
            -self.max_frame_difference, self.max_frame_difference
        )
        scene_total_length = self.image_numbers_per_scene[scene_id]
        if is_train:
            input_id = random.choice(self.train_ids[scene_id])
            input_index = self.train_ids[scene_id].tolist().index(input_id)

            scene_frame_n_train = len(self.train_ids[scene_id])

        else:
            input_id = random.choice(self.test_ids[scene_id])
            input_index = (
                self.all_ids[scene_id].tolist().index(input_id)
            )  # self.test_ids[scene_id].tolist().index(input_id)

            scene_frame_n_test = len(self.test_ids[scene_id])

        target_index = input_index + frame_difference
        target_index = (
            max(min(target_index, scene_frame_n_train - 1), 0)
            if is_train
            else max(min(target_index, len(self.all_ids[scene_id]) - 1), 0)
        )
        # print(f'Sampled index: {input_index,target_index}')

        # max(min(target_index, scene_frame_n_test - 1), 0)
        if is_train:
            target_id = self.train_ids[scene_id][target_index]
        else:
            target_id = self.all_ids[scene_id][
                target_index
            ]  # self.test_ids[scene_id][target_index]
        # print(f'Sampled iD: {input_id,target_id}')
        input_image, input_pose = self.get_image_pose(scene_id, input_id)
        target_image, target_pose = self.get_image_pose(scene_id, target_id)

        return (input_image, input_pose, target_image, target_pose), (
            input_id,
            target_id,
        )

    def get_batched_data(
        self,
        batch_size=32,
        single_model=True,
        model_name=None,
        verbose=False,
        return_info=False,
        is_train=True,
    ):

        # load new model
        input_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )
        target_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )

        input_poses = np.zeros((batch_size, self.pose_size), dtype=np.float32)
        target_poses = np.zeros((batch_size, self.pose_size), dtype=np.float32)

        id_info = []
        scene_id = random.choice(self.scene_list)
        for i in range(batch_size):
            if not single_model:
                scene_id = random.choice(self.scene_list)
            single_data, index_info = self.get_single_data_tuple(
                scene_id, is_train=is_train
            )
            input_image, input_pose, target_image, target_pose = single_data
            input_index, target_index = index_info
            # if not is_train:
            # print(f'Sampled index: {input_index,target_index}')
            id_info.append((scene_id, input_index, target_index))
            input_images[i] = input_image
            target_images[i] = target_image
            input_poses[i] = input_pose
            target_poses[i] = target_pose

        if return_info:
            data_tuple = (input_images, target_images, (input_poses, target_poses))
            return data_tuple, id_info
        else:
            return input_images, target_images, (input_poses, target_poses)

    def get_specific_data(self, target_data_infos):
        n = len(target_data_infos)

        input_images = np.zeros(
            (n, self.image_size, self.image_size, 3), dtype=np.float32
        )
        target_images = np.zeros(
            (n, self.image_size, self.image_size, 3), dtype=np.float32
        )

        input_poses = np.zeros((n, self.pose_size), dtype=np.float32)
        target_poses = np.zeros((n, self.pose_size), dtype=np.float32)

        for i in range(n):
            data_info = target_data_infos[i]
            scene_id = data_info[0]
            input_index = data_info[1]
            target_index = data_info[2]
            input_image, input_pose = self.get_image_pose(scene_id, input_index)
            target_image, target_pose = self.get_image_pose(scene_id, target_index)
            input_images[i] = input_image
            target_images[i] = target_image
            input_poses[i] = input_pose
            target_poses[i] = target_pose

        return input_images, target_images, (input_poses, target_poses)

    def get_batched_data_i_j(
        self, scene_id, difference, frame_min_index, frame_max_index
    ):
        n = frame_max_index - frame_min_index
        input_images = np.zeros(
            (n, self.image_size, self.image_size, 3), dtype=np.float32
        )
        target_images = np.zeros(
            (n, self.image_size, self.image_size, 3), dtype=np.float32
        )

        input_poses = np.zeros((n, self.pose_size), dtype=np.float32)
        target_poses = np.zeros((n, self.pose_size), dtype=np.float32)
        scene_total_length = self.image_numbers_per_scene[scene_id]
        for i in range(n):
            input_frame = self.test_ids[scene_id][frame_min_index + i]
            target_frame = input_frame + difference
            target_frame = max(min(target_frame, scene_total_length - 1), 0)

            input_image, input_pose = self.get_image_pose(scene_id, input_frame)
            target_image, target_pose = self.get_image_pose(scene_id, target_frame)
            input_images[i] = input_image
            target_images[i] = target_image
            input_poses[i] = input_pose
            target_poses[i] = target_pose

        return input_images, target_images, (input_poses, target_poses)


###################################
#### DATALOADER TO WORK WITH IMAGES
###################################
class OursDataSceneLoader(DataLoader):
    def __init__(
        self,
        name,
        image_size=256,
        train_or_test="train",
        samplingStrategy={},
        maxFrames=10,
        extendedTranslationMotion=False,
    ):
        super().__init__(name, image_size)

        self.dataset_format = "img"
        self.in_dir = (
            "/data/datasets/Synthia/"
            if self.name == "synthia"
            else f"/data/datasets/KITTI/RGB_rescaled_{image_size}x{image_size}"
        )

        #####################################################
        # Instantiate an EncodePoseScene pose encoding object.
        self.epiNVS = EncodePoseScene(
            self.in_dir,
            image_size,
            train_or_test,
            samplingStrategy,
            extendedTranslationMotion,
        )

        if train_or_test == "test":
            self.epiNVStwo = EncodePoseScene(
                self.in_dir,
                image_size,
                "train",
                samplingStrategy,
                extendedTranslationMotion,
            )

        self.nbChannelEncodedPose = (
            3 if not self.epiNVS.extendedTranslationMotion else 4
        )

        self.max_frame_difference = maxFrames

        self.pose_size = 6

    def get_batched_data(
        self, batch_size=16, single_model=False, is_train=False, withRawPose=False
    ):

        self.epiNVS.fundamentalFs = []

        # Numpy array data.
        source_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )
        target_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )

        encoded_poses = np.zeros(
            (batch_size, self.image_size, self.image_size, self.nbChannelEncodedPose),
            dtype=np.float32,
        )

        scene_id = (
            self.epiNVS.sampleRandomScene()
        )  # Randomly consider a scene at each batch.

        # SetUp intrinsic camera pose based on scene_id
        self.epiNVS.setUpConfig(scene_id)
        if not is_train:
            self.epiNVStwo.setUpConfig(scene_id)

        if withRawPose:
            viewpointTransformation = np.zeros(
                (batch_size, self.pose_size), dtype=np.float32
            )

        for i in range(batch_size):

            if not single_model:
                scene_id = (
                    self.epiNVS.sampleRandomScene()
                )  # Randomly consider a scene at each batch.

                # SetUp intrinsic camera pose based on scene_id
                self.epiNVS.setUpConfig(scene_id)
                if not is_train:
                    self.epiNVStwo.setUpConfig(scene_id)
                    IdFramestwo = self.epiNVStwo.IdFrames
                    nbFramestwo = self.epiNVStwo.nb_frames

            # Get the total number of frames associated to this scene_id and a list of batch_size sample.
            Idframes = self.epiNVS.IdFrames

            nbFrames = self.epiNVS.nb_frames

            # Required since there is an indexing issue on 'SYNTHIA-SEQS-05-FALL' set. (start at idx 1 and not 0)
            clipMin = 1 if scene_id in ["SYNTHIA-SEQS-05-FALL"] else 0
            clipMax = nbFrames - 1 if is_train else nbFrames + nbFramestwo - 1

            # Ensure that the second (target) view we sampled is closed enough to the source frame.
            frame_difference = np.random.randint(
                -self.max_frame_difference, self.max_frame_difference
            )

            if is_train:
                # Consider the source view.
                sampledIdFrame1 = np.random.choice(Idframes)
                sampledIdxFrame1 = Idframes.index(sampledIdFrame1)

                # Consider the target view.
                sampledIdxFrame2 = max(
                    min(sampledIdxFrame1 + frame_difference, clipMax), clipMin
                )  # clip the target index.
                sampledIdFrame2 = Idframes[sampledIdxFrame2]

                # Retrieve the extrinsic matrices given the sampled Idx.
                Rt1 = self.epiNVS.get_extrinsicRt(sampledIdxFrame1)
                Rt2 = self.epiNVS.get_extrinsicRt(sampledIdxFrame2)

            else:
                try:
                    sampledIdFrame1 = np.random.choice(Idframes)
                    sampledIdxFrame1 = Idframes.index(sampledIdFrame1)

                    sampledIdFrame2 = max(
                        min(sampledIdFrame1 + frame_difference, clipMax), clipMin
                    )  # clip the target index.
                    sampledIdxFrame2 = (
                        Idframes.index(sampledIdFrame2)
                        if (sampledIdFrame2 in Idframes)
                        else IdFramestwo.index(sampledIdFrame2)
                    )

                    Rt1 = self.epiNVS.get_extrinsicRt(sampledIdxFrame1)
                    Rt2 = (
                        self.epiNVStwo.get_extrinsicRt(sampledIdxFrame2)
                        if sampledIdFrame2 in IdFramestwo
                        else self.epiNVS.get_extrinsicRt(sampledIdxFrame2)
                    )

                except Exception as e:
                    print(f"Scene concerned: {scene_id} \n")

                    # Read both of the images.
            I1 = self.epiNVS.readImg(
                self.name,
                sampledIdFrame1,
                StereoLR=self.epiNVS.stereoLR,
                camLoc=self.epiNVS.camLoc,
            )
            I2 = self.epiNVS.readImg(
                self.name,
                sampledIdFrame2,
                StereoLR=self.epiNVS.stereoLR,
                camLoc=self.epiNVS.camLoc,
            )

            if withRawPose:

                t1 = Rt1[:, -1]
                t2 = Rt2[:, -1]

                viewpointTransformation[i, 3:] = t2 - t1
                viewpointTransformation[i, 0] = Rt2[-1, 1] - Rt1[-1, 1]
                viewpointTransformation[1] = Rt2[0, 2] - Rt1[0, 2]
                viewpointTransformation[2] = Rt2[1, 0] - Rt1[1, 0]

            # Compute both the Essential and Fundamental matrices.
            E, t = self.epiNVS.computeEssential(Rt1, Rt2)
            F = self.epiNVS.computeFundamental(E, self.epiNVS.Kinv)

            self.epiNVS.fundamentalFs.append(F)

            if self.noise:
                I1 = DataLoader.add_noise(I1 / 255.0)
                
            try:
                # Encode the pose from I2 based on I1 color - on 3 or 4 channels, depending if translation motion is included.
                E12 = (
                    self.epiNVS.encodePose(I1*255., F)
                    if not self.epiNVS.extendedTranslationMotion
                    else self.epiNVS.encodePoseExtended(I1*255., F, t)
                )

            except Exception as e:
                print(e)
                print(f"Issue with Fundamental matrix computation")
                E12 = np.zeros((256, 256, self.nbChannelEncodedPose)).astype(np.float32)

            

            # Normalize both source/target images and the encoded pose.
            source_images[i] = I1 if self.noise else I1 / 255.0
            target_images[i] = I2 / 255.0
            E12[:, :, :-1] /= 255.0
            encoded_poses[i] = E12

        if not withRawPose:
            return source_images, target_images, encoded_poses
        else:
            return (
                source_images,
                target_images,
                encoded_poses,
                viewpointTransformation,
            )

    def get_consecutive_batch(self, batch_size=16):

        self.epiNVS.fundamentalFs = []

        # Numpy array data.
        source_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )
        target_images = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=np.float32
        )

        encoded_poses = np.zeros(
            (batch_size, self.image_size, self.image_size, self.nbChannelEncodedPose),
            dtype=np.float32,
        )

        scene_id = (
            self.epiNVS.sampleRandomScene()
        )  # Randomly consider a scene at each batch.
        print(f"Selected scene: {scene_id}")
        # SetUp intrinsic camera pose based on scene_id
        self.epiNVS.setUpConfig(scene_id)

        viewpointTransformation = np.zeros(
            (batch_size, self.pose_size), dtype=np.float32
        )

        for i in range(batch_size):

            # Get the total number of frames associated to this scene_id and a list of batch_size sample.
            Idframes = self.epiNVS.IdFrames

            if not i:
                # Consider the source view.
                sampledIdFrame1 = np.random.choice(Idframes)
                sampledIdxFrame1 = Idframes.index(sampledIdFrame1)
                print(f"Source view index: {sampledIdxFrame1}")

            # Consider the target view.
            sampledIdxFrame2 = sampledIdFrame1 + (i + 1)
            sampledIdFrame2 = Idframes[sampledIdxFrame2]

            # Retrieve the extrinsic matrices given the sampled Idx.
            Rt1 = self.epiNVS.get_extrinsicRt(sampledIdxFrame1)
            Rt2 = self.epiNVS.get_extrinsicRt(sampledIdxFrame2)

            assert (
                sampledIdFrame1 == sampledIdxFrame1
            ), "Issue on the sampling for the source image."
            assert (
                sampledIdFrame2 == sampledIdxFrame2
            ), "Issue on the sampling for the target image"

            # Read both of the images.
            I1 = self.epiNVS.readImg(
                self.name,
                sampledIdFrame1,
                StereoLR=self.epiNVS.stereoLR,
                camLoc=self.epiNVS.camLoc,
            )
            I2 = self.epiNVS.readImg(
                self.name,
                sampledIdFrame2,
                StereoLR=self.epiNVS.stereoLR,
                camLoc=self.epiNVS.camLoc,
            )

            t1 = Rt1[:, -1]
            t2 = Rt2[:, -1]

            viewpointTransformation[i, 3:] = t2 - t1
            viewpointTransformation[i, 0] = Rt2[-1, 1] - Rt1[-1, 1]
            viewpointTransformation[1] = Rt2[0, 2] - Rt1[0, 2]
            viewpointTransformation[2] = Rt2[1, 0] - Rt1[1, 0]
            
            # Compute both the Essential and Fundamental matrices.
            E, t = self.epiNVS.computeEssential(Rt1, Rt2)
            F = self.epiNVS.computeFundamental(E, self.epiNVS.Kinv)

            self.epiNVS.fundamentalFs.append(F)
            try:
                # Encode the pose from I2 based on I1 color - on 3 or 4 channels, depending if translation motion is included.
                E12 = (
                    self.epiNVS.encodePose(I1, F)
                    if not self.epiNVS.extendedTranslationMotion
                    else self.epiNVS.encodePoseExtended(I1, F, t)
                )

            except Exception as e:
                print(e)
                print(f"Issue with Fundamental matrix computation")
                E12 = np.zeros((256, 256, self.nbChannelEncodedPose)).astype(np.float32)

            # Normalize both source/target images and the encoded pose.
            source_images[i] = I1 / 255.0
            target_images[i] = I2 / 255.0

            E12[:, :, :-1] /= 255.0
            encoded_poses[i] = E12

        return source_images, target_images, encoded_poses, viewpointTransformation


if __name__ == "__main__":
    from tqdm import tqdm
    import time
    import numpy as np

    sceneDataSynthia = OursDataSceneLoader(
        name="kitti",
        image_size=256,
        train_or_test="train",
        samplingStrategy={"strategy": "gridSampling", "param": 15},
        extendedTranslationMotion=False,
    )
    L = []
    for i in tqdm(range(100)):
        t = time.time()
        I1, I2, E12 = sceneDataSynthia.get_batched_data(batch_size=16, is_train=True)
        L.append(time.time() - t)
    print(f"Required average time: {np.mean(L)}")
    print(I1.shape)
    print(I2.shape)
    print(E12.shape)
