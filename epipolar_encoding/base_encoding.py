from epipolarNVS.utils.utils_epipolar import * #remove parent folder epipolarNVS
import numpy as np
import os


class BaseEncode:
    def __init__(
        self,
        in_dir: str,
        img_shape: int,
        train_or_test: str,
        samplingStrategy: str,
        extendedTranslationMotion: bool,
    ):

        ###############
        ## Image shape
        if isinstance(img_shape, list):
            self.H, self.W = img_shape[0], img_shape[1]
        else:
            self.H = img_shape
            self.W = img_shape

        self.ToT = train_or_test

        self.root_path = os.path.join(in_dir)

        self.fileJson_basename = (
            f"transforms_{self.ToT}.json"  # or transforms_test/train.json
        )

        # Attributes related to the pixel sampling strategy used.
        self.p = samplingStrategy["param"]
        self.xmin = 0
        self.xmax = 255
        # Regular grid generation if required.
        self.useGridSampling = (
            True if samplingStrategy["strategy"] == "gridSampling" else False
        )

        self.pixelGrid = self.generateRegularGrid() if self.useGridSampling else None

        # List of Fundamental matrices: use for the Cyclic loss.
        self.fundamentalFs = []

        # Used for OpenCV fundamental matrix computation.
        self.SIFT = cv2.SIFT_create()

        # RGB or RGB-D encoded pose.
        self.extendedTranslationMotion = extendedTranslationMotion

    def get_F(self) -> np.array:
        return self.F

    @staticmethod
    def computeEssential(Rt1: np.array, Rt2: np.array) -> np.array:

        # Camera Extrinsic extraction.
        R1 = Rt1[:, :3]
        t1 = Rt1[:, -1]

        R2 = Rt2[:, :3]
        t2 = Rt2[:, -1]

        # Compute relative camera pose between view 1 and 2.
        R = R2 @ R1.T
        t = t2 - R @ t1

        # print(f'Source translation: {t1}')
        # print(f'Target translation: {t2}')
        # print(f'Difference: {np.abs(t2)-np.abs(t1)}')

        # print(f'Computed relative translation:{t}')

        tX = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])

        E = tX @ R

        return E, np.abs(t2) - np.abs(t1)

    def computeFundamentalCV(self, I1: np.array, I2: np.array) -> np.array:
        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.SIFT.detectAndCompute(I1, None)
        kp2, des2 = self.SIFT.detectAndCompute(I2, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        pts1 = []
        pts2 = []

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

        return F

    def computeFundamental(self, E: np.array, Kinv: np.array) -> np.array:
        self.F = Kinv.T @ E @ Kinv
        return self.F

    def generateRandomly(self, I: np.array) -> list:

        indices = np.where(np.any(I != NULL_COLOR, axis=-1))
        indexes = list(zip(indices[1], indices[0]))

        return random.sample(indexes, int(self.p * len(indexes)))

    def generateRegularGrid(self) -> itertools.product:

        x = np.linspace(self.xmin, self.xmax, self.p)
        y = np.linspace(self.xmin, self.xmax, self.p)

        xx, yy = np.meshgrid(x, y)
        Xrange = [int(x0) for x0 in xx[0]]
        Yrange = [int(yy[i][0]) for i in range(len(yy))]

        return [pix for pix in itertools.product(Xrange, Yrange)]

    @staticmethod
    def get_epipolarline_equation(F: np.array, pix: np.array) -> np.array:
        return F @ pix

    def drawEpiLines(self, I: np.array, r: np.array, colRGB: tuple) -> np.array:
        # Compute two extremum points.
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [self.W, -(r[2] + r[0] * self.W) / r[1]])

        try:
            return cv2.line(
                I, (int(x0), int(y0)), (int(x1), int(y1)), color=colRGB, thickness=1
            )

        except Exception as e:
            print(f"Issue on encoding")
            return I

    def encodePose(self, I: np.array, F: np.array) -> np.array:

        Epose = np.zeros((self.H, self.W, 3)).astype(np.float32)

        # Get the complete list of pixel to sample: either from a regular grid or randomly sampled.
        sampledPix = (
            self.pixelGrid if self.useGridSampling else self.generateRandomly(I)
        )

        for pix in sampledPix:
            # Hom. coordinates.
            pixH = np.array([[pix[0]], [pix[1]], [1.0]])
            colRGB = tuple(I[int(pix[1]), int(pix[0]), :].astype(np.uint8).tolist())
            if colRGB != (0.0, 0.0, 0.0):
                # Compute the epipolar line equation.
                r = BaseEncode.get_epipolarline_equation(F, pixH)

                # Update Itilde.
                Epose = self.drawEpiLines(Epose, r, colRGB)

        return Epose

    def encodePoseExtended(self, I, F, t):

        # First encode the pose through an RGB image.
        encodedPoseRGB = self.encodePose(I, F)

        # Find out the largest motion direction (either along the X or Z direction) and kept the sign of the motion.
        posMaxMotion = np.argmax(np.abs(t))
        delta_t = np.sign(t[posMaxMotion]) * np.abs(t[posMaxMotion])

        # Build a mask on a single channel to get the epilines position, filled up correct pixels with delta_t  and expand it to get a (256,256,1) image.
        additionalChannel = np.where(encodedPoseRGB[:, :, 0] > 0, delta_t, 0.0)
        additionalChannel = np.expand_dims(additionalChannel, axis=-1)

        # Build the 4-channels extended encoded pose information.
        extendedEpose = np.concatenate((encodedPoseRGB, additionalChannel), axis=-1)

        return extendedEpose

    def get_intrinsicK(self, dict):
        pass

    def get_extrinsicRt(self, **kwargs):
        pass

    def loadJson(self, path):
        pass
