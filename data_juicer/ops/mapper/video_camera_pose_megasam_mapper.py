import importlib
import os
import subprocess
import sys

import numpy as np
from loguru import logger

from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_camera_pose_megasam_mapper"

cv2 = LazyLoader("cv2", "opencv-python")
torch = LazyLoader("torch")


def to_standard_list(obj):
    if isinstance(obj, np.ndarray):
        return to_standard_list(obj.tolist())
    elif isinstance(obj, list):
        return [to_standard_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(to_standard_list(item) for item in obj)
    else:
        return obj


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCameraPoseMegaSaMMapper(Mapper):
    """Extract camera poses by leveraging MegaSaM and MoGe-2."""

    _accelerator = "cuda"

    def __init__(
        self,
        tag_field_name: str = MetaKeys.video_camera_pose_tags,
        frame_field: str = MetaKeys.video_frames,
        camera_calibration_field: str = "camera_calibration",
        max_frames: int = 1000,
        *args,
        **kwargs,
    ):
        """
        Initialization method.
        :param tag_field_name: The field name to store the tags. It's "video_camera_pose_tags" in default.
        :param frame_field: The field name where the video frames are stored.
        :param camera_calibration_field: The field name where the camera calibration info is stored.
        :param max_frames: Maximum number of frames to save.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        megasam_repo_path = os.path.join(DATA_JUICER_ASSETS_CACHE, "mega-sam")
        # droid_slam conflict with the VideoCalibrationMapper
        droid_slam_home = os.path.join(megasam_repo_path, "base", "droid_slam")

        self._prepare_env(megasam_repo_path, droid_slam_home)

        droid_module_path = f"{droid_slam_home}/droid.py"
        spec = importlib.util.spec_from_file_location("droid", droid_module_path)
        if spec is None:
            raise ImportError(f"Could not load spec from {droid_module_path}")
        droid = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(droid)

        from lietorch import SE3

        self.SE3 = SE3
        self.Droid = droid.Droid

        self.tag_field_name = tag_field_name
        self.max_frames = max_frames
        self.frame_field = frame_field
        self.camera_calibration_field = camera_calibration_field

    def _prepare_env(self, megasam_repo_path, droid_slam_home):
        for i in range(len(sys.path)):
            if "DroidCalib/droid_slam" in sys.path[i]:
                logger.warning("Removing DroidCalib/droid_slam from sys.path, it maybe conflicting with mega-sam.")
                sys.path.pop(i)
                break

        if droid_slam_home not in sys.path:
            sys.path.insert(1, droid_slam_home)

        if not os.path.exists(megasam_repo_path):
            subprocess.run(
                ["git", "clone", "--recursive", "https://github.com/mega-sam/mega-sam.git", megasam_repo_path],
                check=True,
            )

            with open(os.path.join(megasam_repo_path, "base", "src", "altcorr_kernel.cu"), "r") as f:
                temp_file_content = f.read()
                temp_file_content = temp_file_content.replace(".type()", ".scalar_type()")

            with open(os.path.join(megasam_repo_path, "base", "src", "altcorr_kernel.cu"), "w") as f:
                f.write(temp_file_content)

            with open(os.path.join(megasam_repo_path, "base", "src", "correlation_kernels.cu"), "r") as f:
                temp_file_content = f.read()
                temp_file_content = temp_file_content.replace(".type()", ".scalar_type()")

            with open(os.path.join(megasam_repo_path, "base", "src", "correlation_kernels.cu"), "w") as f:
                f.write(temp_file_content)

            with open(os.path.join(megasam_repo_path, "base", "src", "droid_kernels.cu"), "r") as f:
                temp_file_content = f.read()
                temp_file_content = temp_file_content.replace(".type()", ".scalar_type()")

            with open(os.path.join(megasam_repo_path, "base", "src", "droid_kernels.cu"), "w") as f:
                f.write(temp_file_content)

            with open(
                os.path.join(megasam_repo_path, "base", "thirdparty", "lietorch", "lietorch", "src", "lietorch_gpu.cu"),
                "r",
            ) as f:
                temp_file_content = f.read()
                temp_file_content = temp_file_content.replace(".type()", ".scalar_type()")

            with open(
                os.path.join(megasam_repo_path, "base", "thirdparty", "lietorch", "lietorch", "src", "lietorch_gpu.cu"),
                "w",
            ) as f:
                f.write(temp_file_content)

            with open(
                os.path.join(
                    megasam_repo_path, "base", "thirdparty", "lietorch", "lietorch", "src", "lietorch_cpu.cpp"
                ),
                "r",
            ) as f:
                temp_file_content = f.read()
                temp_file_content = temp_file_content.replace(".type()", ".scalar_type()")

            with open(
                os.path.join(
                    megasam_repo_path, "base", "thirdparty", "lietorch", "lietorch", "src", "lietorch_cpu.cpp"
                ),
                "w",
            ) as f:
                f.write(temp_file_content)

        try:
            import torch_scatter  # noqa F401
        except ImportError:
            """ "Please refer to https://github.com/rusty1s/pytorch_scatter to locate the
            installation link that is compatible with your PyTorch and CUDA versions."""
            # torch_version = "2.6.0"
            # cuda_version = "cu124"
            subprocess.run(
                [
                    "pip",
                    "install",
                    "torch-scatter",
                    # "-f",
                    # f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html",
                ],
                cwd=os.path.join(megasam_repo_path, "base"),
            )

        try:
            import droid_backends  # noqa F401
            import lietorch  # noqa F401
        except ImportError:
            subprocess.run(["pip", "uninstall", "droid_backends", "-y"])
            subprocess.run(["python", "setup.py", "install"], cwd=os.path.join(megasam_repo_path, "base"))

    def _preprocess_stream(self, frames, depth_list, intrinsics_list):
        """Pre-process all frames once and cache the results.

        Returns a list of (t, image, depth, intrinsics, mask) tuples.
        Avoids the cost of repeated image decoding / resize when
        image_stream is consumed multiple times (tracking + terminate).
        """
        cached = []
        for t, (raw_image, raw_depth, raw_intr) in enumerate(zip(frames, depth_list, intrinsics_list)):
            if isinstance(raw_image, bytes):
                image_array = np.frombuffer(raw_image, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            else:
                image = cv2.imread(raw_image)
            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
            image = image[: h1 - h1 % 8, : w1 - w1 % 8]
            image = torch.as_tensor(image).permute(2, 0, 1)
            image = image[None]

            if isinstance(raw_depth, np.ndarray):
                depth = torch.from_numpy(raw_depth.astype(np.float32))
            else:
                depth = torch.as_tensor(raw_depth, dtype=torch.float32)
            depth = torch.nn.functional.interpolate(depth[None, None], (h1, w1), mode="nearest-exact").squeeze()
            depth = depth[: h1 - h1 % 8, : w1 - w1 % 8]

            mask = torch.ones_like(depth)

            if isinstance(raw_intr, np.ndarray):
                intr_arr = raw_intr.astype(np.float32)
                intrinsics = torch.tensor([intr_arr[0, 0], intr_arr[1, 1], intr_arr[0, 2], intr_arr[1, 2]])
            else:
                intrinsics = torch.as_tensor([raw_intr[0][0], raw_intr[1][1], raw_intr[0][2], raw_intr[1][2]])
            intrinsics[0::2] *= w1 / w0
            intrinsics[1::2] *= h1 / h0

            cached.append((t, image, depth, intrinsics, mask))
        return cached

    @staticmethod
    def _iter_cached(cached):
        """Yield from a cached preprocessed list (same interface as image_stream)."""
        for item in cached:
            yield item

    def process_single(self, sample=None, rank=None):
        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return sample

        videos_frames = sample[self.frame_field]

        sample[Fields.meta][self.tag_field_name] = []

        for video_idx in range(len(videos_frames)):
            frames = videos_frames[video_idx]
            cur_video_calibration = sample[Fields.meta][self.camera_calibration_field][video_idx]
            depth_list = cur_video_calibration[CameraCalibrationKeys.depth]
            intrinsics = cur_video_calibration[CameraCalibrationKeys.intrinsics]

            if isinstance(intrinsics, np.ndarray):
                intrinsics = intrinsics.astype(np.float32)
            else:
                intrinsics = np.array(to_standard_list(intrinsics), dtype=np.float32)

            # (3, 3) -> (N, 3, 3)
            if intrinsics.ndim == 2:
                assert intrinsics.shape == (3, 3)
                intrinsics = np.tile(intrinsics[np.newaxis, :, :], (len(frames), 1, 1))
            elif intrinsics.ndim == 3:
                assert len(intrinsics) == len(frames), f"Expected {len(frames)}, got {len(intrinsics)}"
            else:
                raise ValueError(f"Invalid intrinsics shape: {intrinsics.shape}, expected (N, 3, 3) or (3, 3)")

            intrinsics_list = intrinsics.tolist()

            # Pre-process all frames once (avoids double decode + resize)
            cached_stream = self._preprocess_stream(frames, depth_list, intrinsics_list)

            valid_image_list = []
            valid_depth_list = []
            valid_intrinsics_list = []
            valid_mask_list = []

            for t, image, depth, intr, mask in cached_stream:

                valid_image_list.append(image[0])
                valid_depth_list.append(depth)
                valid_mask_list.append(mask)
                valid_intrinsics_list.append(intr)

                if t == 0:
                    args = droid_args(image_size=[image.shape[2], image.shape[3]])
                    droid = self.Droid(args)

                droid.track(t, image, depth, intrinsics=intr, mask=mask)

            droid.track_final(t, image, depth, intrinsics=intr, mask=mask)

            # Reuse cached stream for terminate (no re-decode)
            traj_est, depth_est, motion_prob = droid.terminate(
                self._iter_cached(cached_stream),
                _opt_intr=True,
                full_ba=True,
            )

            t = traj_est.shape[0]
            images = np.array(valid_image_list[:t])
            disps = 1.0 / (np.array(valid_depth_list[:t]) + 1e-6)

            poses = traj_est
            intrinsics = droid.video.intrinsics[:t].cpu().numpy()

            intrinsics = intrinsics[0] * 8.0
            poses_th = torch.as_tensor(poses, device="cpu")
            cam_c2w = self.SE3(poses_th).inv().matrix().numpy()

            K = np.eye(3)
            K[0, 0] = intrinsics[0]
            K[1, 1] = intrinsics[1]
            K[0, 2] = intrinsics[2]
            K[1, 2] = intrinsics[3]

            max_frames = min(self.max_frames, images.shape[0])

            # return_images = np.uint8(images[:max_frames, ::-1, ...].transpose(0, 2, 3, 1))
            return_depths = np.float32(1.0 / disps[:max_frames, ...])
            return_cam_c2w = cam_c2w[:max_frames]

            sample[Fields.meta][self.tag_field_name].append(
                {
                    CameraCalibrationKeys.depth: return_depths,
                    CameraCalibrationKeys.intrinsics: K,
                    CameraCalibrationKeys.cam_c2w: return_cam_c2w,
                }
            )

        return sample


class droid_args:
    def __init__(self, image_size):
        self.weights = os.path.join(DATA_JUICER_ASSETS_CACHE, "mega-sam", "checkpoints", "megasam_final.pth")
        self.disable_vis = True
        self.image_size = image_size
        self.buffer = 1024
        self.stereo = False
        self.filter_thresh = 2.0

        self.warmup = 8
        self.beta = 0.3
        self.frontend_nms = 1
        self.keyframe_thresh = 2.0
        self.frontend_window = 25
        self.frontend_thresh = 12.0
        self.frontend_radius = 2

        self.upsample = False
        self.backend_thresh = 16.0
        self.backend_radius = 2
        self.backend_nms = 3
