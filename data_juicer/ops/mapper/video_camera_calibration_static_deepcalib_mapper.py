import numpy as np

from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_camera_calibration_static_deepcalib_mapper"

cv2 = LazyLoader("cv2", "opencv-python")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCameraCalibrationStaticDeepcalibMapper(Mapper):
    """Compute the camera intrinsics and field of view (FOV)
    for a static camera using DeepCalib."""

    _accelerator = "cuda"

    def __init__(
        self,
        model_path: str = "weights_10_0.02.h5",
        frame_field: str = MetaKeys.video_frames,
        tag_field_name: str = MetaKeys.static_camera_calibration_deepcalib_tags,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param model_path: The path to the DeepCalib Regression model.
        :param frame_field: The field name where the video frames are stored.
        :param tag_field_name: The field name to store the tags. It's
            "static_camera_calibration_deepcalib_tags" in default.
        :param args: extra args
        :param kwargs: extra args

        """

        super().__init__(*args, **kwargs)

        LazyLoader.check_packages(["tensorflow"])
        import keras
        from keras.applications.imagenet_utils import preprocess_input

        self.keras = keras
        self.preprocess_input = preprocess_input

        self.model_key = prepare_model(model_type="deepcalib", model_path=model_path)
        self.frame_field = frame_field
        self.tag_field_name = tag_field_name

        self.INPUT_SIZE = 299
        self.focal_start = 40
        self.focal_end = 500

    def process_single(self, sample=None, rank=None):

        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return []

        # load videos
        videos_frames = sample[self.frame_field]
        model = get_model(self.model_key, rank, self.use_cuda())

        sample[Fields.meta][self.tag_field_name] = []

        for video_idx in range(len(videos_frames)):
            final_k_list = []
            final_xi_list = []
            final_hfov_list = []
            final_vfov_list = []

            for i, frame in enumerate(videos_frames[video_idx]):
                if isinstance(frame, bytes):
                    image_array = np.frombuffer(frame, dtype=np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                else:
                    image = cv2.imread(frame)

                height, width, channels = image.shape

                image = cv2.resize(image, (self.INPUT_SIZE, self.INPUT_SIZE))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image / 255.0
                image = image - 0.5
                image = image * 2.0
                image = np.expand_dims(image, 0)

                image = self.preprocess_input(image)

                prediction = model.predict(image)
                prediction_focal = prediction[0]
                prediction_dist = prediction[1]

                # Scale the focal length based on the original width of the image.
                curr_focal_pred = (
                    (prediction_focal[0][0] * (self.focal_end + 1.0 - self.focal_start * 1.0) + self.focal_start * 1.0)
                    * (width * 1.0)
                    / (self.INPUT_SIZE * 1.0)
                )
                curr_focal_pred = curr_focal_pred.item()

                # Following DeepCalib's official codes
                curr_dist_pred = prediction_dist[0][0] * 1.2
                curr_dist_pred = curr_dist_pred.item()

                temp_k = [[curr_focal_pred, 0, width / 2], [0, curr_focal_pred, height / 2], [0, 0, 1]]
                temp_xi = curr_dist_pred

                temp_hfov = 2 * np.arctan(width / 2 / curr_focal_pred)  # rad
                temp_vfov = 2 * np.arctan(height / 2 / curr_focal_pred)

                temp_hfov = temp_hfov.item()
                temp_vfov = temp_vfov.item()

                final_k_list.append(temp_k)
                final_xi_list.append(temp_xi)
                final_hfov_list.append(temp_hfov)
                final_vfov_list.append(temp_vfov)

        sample[Fields.meta][self.tag_field_name].append(
            {
                CameraCalibrationKeys.intrinsics: final_k_list,
                CameraCalibrationKeys.xi: final_xi_list,
                CameraCalibrationKeys.hfov: final_hfov_list,
                CameraCalibrationKeys.vfov: final_vfov_list,
            }
        )

        return sample
