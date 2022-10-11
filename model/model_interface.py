from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
import tensorflow as tf
import os
import gc
import time

from tqdm import trange

import wandb
from utils.test_utils import test_few_models_and_export_image
from utils.utils import mae_custom, ssim_custom, psnr
from dataset.dataset import DataLoader
from loss import PixelLoss, PerceptualLoss, CyclicLoss, SpectralLoss

from logs.logs_train import logsTrain


class ModelInterface:
    def __init__(self, name, image_size=256):
        self.name = name
        self.model = Model()
        self.pixel_normalizer = lambda x: 2 * x - 1
        self.pixel_normalizer_reverse = lambda x: 0.5 * x + 0.5
        self.image_size = image_size
        self.decoder_original_features = {}
        self.encoder_original_features = {}
        self.decoder_rearranged_features = {}

    def build_model(self):
        pass

    def get_model(self):
        return self.model

    def save_model(self, filePath):
        self.model.save_weights(f"{filePath}.h5")
        print("Saved model to disk")

    def load_model(self, file_name):
        if not file_name.endswith(".h5"):
            file_name += ".h5"
        self.model.load_weights(file_name)
        print("Loaded model from disk")

    def get_predicted_image(self, sampled_input_data):
        source_images, pose_info = sampled_input_data
        return self.model.predict([source_images, pose_info])

    def process_pose_info(self, data: DataLoader, pose_info):
        return pose_info

    def train(self, train_data: DataLoader, test_data: DataLoader = None, **params):

        ######################################
        ############ Parameters.  ############
        ######################################

        max_iter = params["nbIteration"]
        batch_size = params["batchSize"]
        self.image_size = params["imageSize"]
        lr = params["learningRate"]

        self.dataset_format = params["data"]["format"]

        single_model = params["hyperparameters"]["singleModel"]

        logs = params["logs"]
        resumeTraining = params["resume"]

        ######################################
        ############## Loss . ################
        ######################################

        pixel_mae = PixelLoss("l1")

        losses = [pixel_mae]
        weights = [1.0]


        if params["losses"]["spectral"]:
            spectral = SpectralLoss(kernel_size=5, pixel_loss="l2")
            losses.append(spectral)
            weights.append(1.0)
            print("Spectral loss has been added.")
            time.sleep(3)
        # Build and compile the DL model.
        self.build_model()
        self.model.compile(
            optimizer=Adam(learning_rate=lr, beta_1=0.9),
            loss=losses,
            loss_weights=weights,
        )

        # Logs configuration.
        logFolderName = os.path.join(
            logs["trainFolder"],
            params["model"]["type"] + params["data"]["name"].upper(),
        )
        export_image_per = logs["export_image_per"]
        # saveBestModel = logs["save_best_model_only"]
        writeLossLogFreq = logs["saveLossFrequency"]
        # writeMetricsLogFreq = logs['saveMetricsFrequency']

        started_time_date = time.strftime("%Y%m%d_%H%M%S")

        baseNameFolder = (
            "_".join([self.name, started_time_date])
            if not resumeTraining["do"]
            else "_".join([self.name, resumeTraining["timeId"]])
        )
        folderRunName = logFolderName + "/" + baseNameFolder

        if not os.path.exists(folderRunName):
            os.makedirs(folderRunName)

        # Weight and Biases logs.
        try:
            ours_or_baseline = (
                f"Ours-Grid{train_data.epiNVS.p}"
                if train_data.epiNVS.useGridSampling
                else f"Ours-Random{train_data.epiNVS.p}"
            )
        except AttributeError as e:
            ours_or_baseline = "ICIP Baseline"

        trainConfig = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "dataset": train_data.name,
            "single_model": single_model,
            "Sampling": {
                "useGrid": train_data.epiNVS.useGridSampling,
                "value": train_data.epiNVS.p,
                "extendedPose": train_data.epiNVS.extendedTranslationMotion,
            }
            if self.dataset_format == "img"
            else False,
            "MaxFrame": train_data.max_frame_difference
            if train_data.name in ["synthia", "kitti"]
            else None,
        }

        logs = logsTrain(
            runName=f"{train_data.name.upper()}-{ours_or_baseline}",
            idRun=resumeTraining["runId"],
            config=trainConfig,
            resumeTraining=True if resumeTraining["do"] else False,
        )

        run = logs.getRun()

        started_time = time.time()
        f = None
        wr = None
        f_test = None
        it_restart = 0

        # if Resume training
        if resumeTraining["do"]:
            it_restart = resumeTraining["iteration"]

            weightFile = os.path.join(folderRunName, "modelWeights_best.h5")
            self.load_model(weightFile)

            print(f"--> Training is gonna be resumed from iteration: {it_restart}")

        lossMin = 0.5

        pbar = trange(max_iter, desc="Iterations")
        for i in pbar:

            # Allow to directly go the the iteration where the training need to be resume.
            if i <= it_restart:
                continue

            if self.dataset_format == "img":

                if self.name != "modelICPR_h_attn":
                    (
                        source_images,
                        target_images,
                        encoded_poses,
                    ) = train_data.get_batched_data(
                        batch_size, single_model=single_model, is_train=True
                    )

                    loss_info = self.model.train_on_batch(
                        [source_images, encoded_poses], target_images
                    )

                elif self.name == "modelICPR_h_attn":
                    (
                        source_images,
                        target_images,
                        encoded_poses,
                        pose_info,
                    ) = train_data.get_batched_data(
                        batch_size,
                        single_model=single_model,
                        is_train=True,
                        withRawPose=True,
                    )
                    target_view = self.process_pose_info(pose_info)
                    loss_info = self.model.train_on_batch(
                        [source_images, encoded_poses, target_view], target_images
                    )

            elif self.dataset_format == "npy":
                source_images, target_images, pose_info = train_data.get_batched_data(
                    batch_size, single_model=single_model, is_train=True
                )
                target_view = self.process_pose_info(train_data, pose_info)

                loss_info = self.model.train_on_batch(
                    [source_images, target_view], target_images
                )

            pbar.set_postfix({"Train loss": loss_info})

            # .log({"loss": loss_info})
            run.log({"loss": loss_info})

            # Test and write on some images.
            if i % export_image_per == 0:
                if not os.path.exists(folderRunName):
                    os.makedirs(folderRunName)

                test_few_models_and_export_image(
                    self,
                    train_data,
                    str(i),
                    folderRunName,
                    test_n=5,
                    single_model=False,
                    data_type=self.dataset_format,
                )

            elapsed_time = time.time() - started_time

            # Write log.
            if not i % writeLossLogFreq:
                weightUpdate = False
                # Save the best weight based on the training loss.
                if loss_info < lossMin:
                    self.save_model(os.path.join(folderRunName, "modelWeights_best"))
                    # wandb.run.summary["iterationBestWeight"] = i
                    run.summary["iterationBestWeight"] = i
                    lossMin = loss_info
                    weightUpdate = True

                if wr is None:
                    import csv

                    f = open(f"{folderRunName}/log_loss.csv", "w", encoding="utf-8")
                    wr = csv.writer(f)
                    wr.writerow(
                        ["epoch"]
                        + self.get_model().metrics_names
                        + ["weightBest"]
                        + ["elapsed_time"]
                    )

                wr.writerow(
                    [i]
                    + (loss_info if type(loss_info) is list else [loss_info])
                    + [weightUpdate]
                    + [time.strftime("%H:%M:%S", time.gmtime(elapsed_time))]
                )
                f.flush()

            # Clean up working issue ?
            gc.collect()
            clear_session()

    def evaluate(self, source_images, target_images, encoded_pose):
        if self.prediction_model is None:
            self.prediction_model = self.get_model()  # prediction_model
            self.prediction_model.compile(
                optimizer="adam", loss="mae", metrics=[mae_custom, ssim_custom, psnr]
            )
        return self.prediction_model.evaluate(
            [source_images, encoded_pose], target_images, verbose=False
        )

    def predict(self, source_images, encoded_pose):
        self.prediction_model = self.get_model()

        pred_imgs = self.prediction_model.predict([source_images, encoded_pose])
        return pred_imgs
