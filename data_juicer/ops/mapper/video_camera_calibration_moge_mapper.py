import numpy as np

from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_camera_calibration_moge_mapper"

cv2 = LazyLoader("cv2", "opencv-python")
torch = LazyLoader("torch")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCameraCalibrationMogeMapper(Mapper):
    """Compute the camera intrinsics and field of view (FOV)
    for a static camera using Moge-2 (more accurate
    than DeepCalib)."""

    _accelerator = "cuda"

    def __init__(
        self,
        model_path: str = "Ruicheng/moge-2-vitl",
        tag_field_name: str = MetaKeys.camera_calibration_moge_tags,
        frame_field: str = MetaKeys.video_frames,
        output_intrinsics: bool = True,
        output_hfov: bool = True,
        output_vfov: bool = True,
        output_points: bool = True,
        output_depth: bool = True,
        output_mask: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param model_path: The path to the Moge-2 model.
        :param tag_field_name: The field name to store the tags. It's
            "camera_calibration_moge_tags" in default.
        :param frame_field: The field name where the video frames are stored.
        :param output_intrinsics: Determines whether to output camera intrinsics.
        :param output_hfov: Determines whether to output horizontal field of view.
        :param output_vfov: Determines whether to output vertical field of view.
        :param output_points: Determines whether to output point map
            in OpenCV camera coordinate system (x right, y down, z forward).
            For MoGe-2, the point map is in metric scale.
        :param output_depth: Determines whether to output depth maps.
        :param output_mask: Determines whether to output a binary mask for valid pixels.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        self.model_key = prepare_model(model_type="moge", model_path=model_path)
        self.tag_field_name = tag_field_name
        self.frame_field = frame_field
        self.output_points = output_points
        self.output_depth = output_depth
        self.output_mask = output_mask
        self.output_intrinsics = output_intrinsics
        self.output_hfov = output_hfov
        self.output_vfov = output_vfov
        assert (
            self.output_points
            or self.output_depth
            or self.output_mask
            or self.output_intrinsics
            or self.output_hfov
            or self.output_vfov
        ), "At least one type of output info must be True."

    def _need_anything(self, sample) -> bool:
        """Whether this video still needs any requested outputs."""

        existing_tags = sample[Fields.meta].get(self.tag_field_name)
        if not existing_tags:
            return True

        if not isinstance(existing_tags[0], dict):
            raise ValueError(
                f"The existing field {self.tag_field_name} in sample[Fields.meta] should be a sequence of dict, but get {existing_tags}."
            )

        # Map: instance flag -> corresponding tag key
        requirements = {
            "output_intrinsics": CameraCalibrationKeys.intrinsics,
            "output_hfov": CameraCalibrationKeys.hfov,
            "output_vfov": CameraCalibrationKeys.vfov,
            "output_points": CameraCalibrationKeys.points,
            "output_depth": CameraCalibrationKeys.depth,
            "output_mask": CameraCalibrationKeys.mask,
        }

        for tag_dict in existing_tags:
            missing_any = any(getattr(self, flag, False) and key not in tag_dict for flag, key in requirements.items())
            if missing_any:
                return True

        return False

    def process_single(self, sample=None, rank=None):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return sample

        if sample.get(self.frame_field) is None:
            return sample

        if not self._need_anything(sample):
            return sample

        model = get_model(self.model_key, rank, self.use_cuda())

        videos_frames = sample[self.frame_field]
        num_videos = len(videos_frames)

        if self.tag_field_name not in sample[Fields.meta]:
            sample[Fields.meta][self.tag_field_name] = [{} for _ in range(num_videos)]

        tags_list = sample[Fields.meta][self.tag_field_name]

        if len(tags_list) != num_videos:
            raise ValueError(
                f"The field {self.tag_field_name} in sample[Fields.meta] "
                "should be a list of dict with the same length as the number of videos."
            )

        if rank is not None:
            device = f"cuda:{rank}" if self.use_cuda() else "cpu"
        else:
            device = "cuda" if self.use_cuda() else "cpu"

        for video_idx in range(num_videos):
            (final_k_list, final_hfov_list, final_vfov_list, final_points_list, final_depth_list, final_mask_list) = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

            tag_dict = tags_list[video_idx]

            need_K = self.output_intrinsics and CameraCalibrationKeys.intrinsics not in tag_dict
            need_hfov = self.output_hfov and CameraCalibrationKeys.hfov not in tag_dict
            need_vfov = self.output_vfov and CameraCalibrationKeys.vfov not in tag_dict
            need_points = self.output_points and CameraCalibrationKeys.points not in tag_dict
            need_depth = self.output_depth and CameraCalibrationKeys.depth not in tag_dict
            need_mask = self.output_mask and CameraCalibrationKeys.mask not in tag_dict
            need_intrinsics_related = need_K or need_hfov or need_vfov

            for i, frame in enumerate(videos_frames[video_idx]):
                if isinstance(frame, bytes):  # rgb bytes from data-juicer video decoder
                    image_array = np.frombuffer(frame, dtype=np.uint8)
                    input_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                else:
                    input_image = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)

                height, width, channels = input_image.shape
                input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

                output = model.infer(input_image)

                if need_intrinsics_related:
                    intrinsics = output["intrinsics"].cpu().tolist()
                    temp_k = [
                        [intrinsics[0][0] * width, 0, intrinsics[0][2] * width],
                        [0, intrinsics[1][1] * height, intrinsics[1][2] * height],
                        [0, 0, 1],
                    ]
                    if need_K:
                        final_k_list.append(temp_k)
                    if need_hfov:
                        temp_hfov = 2 * np.arctan(1 / 2 / intrinsics[0][0])  # rad
                        final_hfov_list.append(temp_hfov)
                    if need_vfov:
                        temp_vfov = 2 * np.arctan(1 / 2 / intrinsics[1][1])
                        final_vfov_list.append(temp_vfov)

                if need_points:
                    points = output["points"].cpu().tolist()
                    final_points_list.append(points)

                if need_depth:
                    depth = output["depth"].cpu().tolist()
                    final_depth_list.append(depth)

                if need_mask:
                    mask = output["mask"].cpu().tolist()
                    final_mask_list.append(mask)

            if need_K:
                tag_dict[CameraCalibrationKeys.intrinsics] = final_k_list
            if need_hfov:
                tag_dict[CameraCalibrationKeys.hfov] = final_hfov_list
            if need_vfov:
                tag_dict[CameraCalibrationKeys.vfov] = final_vfov_list
            if need_points:
                tag_dict[CameraCalibrationKeys.points] = final_points_list
            if need_depth:
                tag_dict[CameraCalibrationKeys.depth] = final_depth_list
            if need_mask:
                tag_dict[CameraCalibrationKeys.mask] = final_mask_list

        return sample
