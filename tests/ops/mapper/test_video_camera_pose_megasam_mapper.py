import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_camera_pose_megasam_mapper import VideoCameraPoseMegaSaMMapper
from data_juicer.utils.constant import Fields, MetaKeys, CameraCalibrationKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE


@unittest.skip(
    'Requires mega-sam conda environment with CUDA compiled extensions '
    '(droid_backends, lietorch). Run with: '
    'conda activate mega-sam && python -m pytest <this_file>'
)
class VideoCameraPoseMegaSaMMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid12_path = os.path.join(data_path, 'video12.mp4')

    def _build_ds_list(self, num_frames_vid3=49, num_frames_vid12=3):
        """Build dataset with pre-populated camera calibration (depth + intrinsics)
        to simulate output from VideoCameraCalibrationMogeMapper."""
        h3, w3 = 584, 328
        h12, w12 = 328, 584

        # Generate dummy frame paths
        frames_vid3 = [os.path.join(self.data_path, f'frame_{i}.jpg')
                       for i in range(num_frames_vid3)]
        frames_vid12 = [os.path.join(self.data_path, f'frame_{i}.jpg')
                        for i in range(num_frames_vid12)]

        # Dummy intrinsics (3x3)
        K3 = [[465.47, 0, w3 / 2.0], [0, 465.47, h3 / 2.0], [0, 0, 1]]
        K12 = [[1227.37, 0, w12 / 2.0], [0, 1227.37, h12 / 2.0], [0, 0, 1]]

        # Dummy depth maps
        depth3 = np.random.rand(num_frames_vid3, h3, w3).tolist()
        depth12 = np.random.rand(num_frames_vid12, h12, w12).tolist()

        ds_list = [{
            'videos': [self.vid3_path],
            Fields.meta: {
                MetaKeys.video_frames: [frames_vid3],
                'camera_calibration': [{
                    CameraCalibrationKeys.depth: depth3,
                    CameraCalibrationKeys.intrinsics: K3,
                }],
            }
        }, {
            'videos': [self.vid12_path],
            Fields.meta: {
                MetaKeys.video_frames: [frames_vid12],
                'camera_calibration': [{
                    CameraCalibrationKeys.depth: depth12,
                    CameraCalibrationKeys.intrinsics: K12,
                }],
            }
        }]

        tgt_list = [{
            "depths_ndim": 3,
            "intrinsic_shape": [3, 3],
            "cam_c2w_last_dim": 4,
        }, {
            "depths_ndim": 3,
            "intrinsic_shape": [3, 3],
            "cam_c2w_last_dim": 4,
        }]

        return ds_list, tgt_list

    def _run_and_assert(self, num_proc):
        ds_list, tgt_list = self._build_ds_list()

        op = VideoCameraPoseMegaSaMMapper(
            tag_field_name=MetaKeys.video_camera_pose_tags,
            frame_field=MetaKeys.video_frames,
            camera_calibration_field='camera_calibration',
            max_frames=1000,
        )
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, tgt_list):
            tag_list = sample[Fields.meta][MetaKeys.video_camera_pose_tags]
            self.assertIsInstance(tag_list, list)
            self.assertGreater(len(tag_list), 0)

            for video_result in tag_list:
                # Check output keys
                self.assertIn(CameraCalibrationKeys.depth, video_result)
                self.assertIn(CameraCalibrationKeys.intrinsics, video_result)
                self.assertIn(CameraCalibrationKeys.cam_c2w, video_result)

                # Check shapes
                depths = np.array(video_result[CameraCalibrationKeys.depth])
                intrinsic = np.array(video_result[CameraCalibrationKeys.intrinsics])
                cam_c2w = np.array(video_result[CameraCalibrationKeys.cam_c2w])

                self.assertEqual(depths.ndim, target["depths_ndim"])
                self.assertEqual(list(intrinsic.shape), target["intrinsic_shape"])
                self.assertEqual(cam_c2w.shape[-1], target["cam_c2w_last_dim"])
                self.assertEqual(cam_c2w.shape[-2], 4)  # (N, 4, 4)

    def test(self):
        self._run_and_assert(num_proc=1)

    def test_mul_proc(self):
        self._run_and_assert(num_proc=2)


if __name__ == '__main__':
    unittest.main()
