import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_hand_reconstruction_hawor_mapper import VideoHandReconstructionHaworMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields, MetaKeys, CameraCalibrationKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE


@unittest.skip('Users need to download MANO_RIGHT.pkl and MANO_LEFT.pkl.')
class VideoHandReconstructionHaworMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid4_path = os.path.join(data_path, 'video4.mp4')

    def _build_ds_list(self):
        """Build dataset with pre-extracted frames and camera calibration in meta."""
        ds_list = [{
            'videos': [self.vid3_path],
            Fields.meta: {
                MetaKeys.video_frames: [
                    [self.vid3_path]  # placeholder; real test needs extracted frames
                ],
                'camera_calibration': [{
                    CameraCalibrationKeys.hfov: [0.76] * 49,
                }],
            }
        }, {
            'videos': [self.vid4_path],
            Fields.meta: {
                MetaKeys.video_frames: [
                    [self.vid4_path]
                ],
                'camera_calibration': [{
                    CameraCalibrationKeys.hfov: [0.66] * 22,
                }],
            }
        }]
        return ds_list

    def _run_and_assert(self, num_proc):
        ds_list = self._build_ds_list()

        op = VideoHandReconstructionHaworMapper(
            hawor_model_path="hawor.ckpt",
            hawor_config_path="model_config.yaml",
            hawor_detector_path="detector.pt",
            mano_right_path="path_to_mano_right_pkl",
            mano_left_path="path_to_mano_left_pkl",
            frame_field=MetaKeys.video_frames,
            camera_calibration_field='camera_calibration',
            thresh=0.2,
        )
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        res_list = dataset.to_list()

        for sample in res_list:
            tag = sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]
            self.assertIsInstance(tag, list)
            self.assertGreater(len(tag), 0)

            for video_result in tag:
                # Check top-level keys
                self.assertIn('fov_x', video_result)
                self.assertIn('img_focal', video_result)
                self.assertIn('left', video_result)
                self.assertIn('right', video_result)

                # Check hand output structure (axis-angle format)
                for hand_type in ['left', 'right']:
                    hand = video_result[hand_type]
                    self.assertIn('frame_ids', hand)
                    self.assertIn('global_orient', hand)
                    self.assertIn('hand_pose', hand)
                    self.assertIn('betas', hand)
                    self.assertIn('transl', hand)

                    n_frames = len(hand['frame_ids'])
                    if n_frames > 0:
                        # global_orient: list of (3,) axis-angle
                        self.assertEqual(
                            np.array(hand['global_orient']).shape,
                            (n_frames, 3))
                        # hand_pose: list of (45,) axis-angle
                        self.assertEqual(
                            np.array(hand['hand_pose']).shape,
                            (n_frames, 45))
                        # betas: list of (10,)
                        self.assertEqual(
                            np.array(hand['betas']).shape,
                            (n_frames, 10))
                        # transl: list of (3,)
                        self.assertEqual(
                            np.array(hand['transl']).shape,
                            (n_frames, 3))

    def test(self):
        self._run_and_assert(num_proc=1)

    def test_mul_proc(self):
        self._run_and_assert(num_proc=2)


if __name__ == '__main__':
    unittest.main()
