import json
import os
import shutil
import tempfile
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.export_to_lerobot_mapper import ExportToLeRobotMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ExportToLeRobotMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid4_path = os.path.join(data_path, 'video4.mp4')

    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix='lerobot_test_')

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def _make_sample(self, video_path, num_frames=10, task_desc="pick up the cup"):
        """Create a synthetic sample with hand action data."""
        states = (np.random.randn(num_frames, 8).astype(np.float32)).tolist()
        actions = (np.random.randn(num_frames, 7).astype(np.float32)).tolist()

        sample = {
            'videos': [video_path],
            'text': task_desc,
            Fields.meta: {
                'hand_action_tags': [{
                    'right': {
                        'states': states,
                        'actions': actions,
                        'valid_frame_ids': list(range(num_frames)),
                        'hand_type': 'right',
                    },
                    'left': {
                        'states': [],
                        'actions': [],
                        'valid_frame_ids': [],
                        'hand_type': 'left',
                    }
                }],
                MetaKeys.video_frames: [
                    [f'/tmp/frame_{i}.jpg' for i in range(num_frames)]
                ],
            }
        }
        return sample

    def test_process_single(self):
        """Test processing a single sample."""
        sample = self._make_sample(self.vid3_path, num_frames=10)
        ds_list = [sample]

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            hand_action_field='hand_action_tags',
            frame_field=MetaKeys.video_frames,
            fps=10,
            robot_type='egodex_hand',
        )

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        self.assertEqual(len(res_list), 1)
        export_info = res_list[0][Fields.meta].get('lerobot_export', [])
        self.assertGreater(len(export_info), 0)

        ep = export_info[0]
        self.assertIn('uuid', ep)
        self.assertIn('parquet_path', ep)
        self.assertEqual(ep['num_frames'], 10)

        # Verify staging files exist
        self.assertTrue(os.path.exists(ep['parquet_path']))

    def test_finalize_dataset(self):
        """Test the full pipeline: process + finalize."""
        samples = [
            self._make_sample(self.vid3_path, num_frames=10, task_desc="pick up cup"),
            self._make_sample(self.vid4_path, num_frames=8, task_desc="place cup"),
        ]

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            hand_action_field='hand_action_tags',
            frame_field=MetaKeys.video_frames,
            fps=10,
            robot_type='egodex_hand',
        )

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)

        # Finalize
        ExportToLeRobotMapper.finalize_dataset(
            output_dir=self.output_dir,
            fps=10,
            robot_type='egodex_hand',
        )

        meta_dir = os.path.join(self.output_dir, 'meta')

        # Check info.json
        info_path = os.path.join(meta_dir, 'info.json')
        self.assertTrue(os.path.exists(info_path))
        with open(info_path, 'r') as f:
            info = json.load(f)
        self.assertEqual(info['codebase_version'], 'v2.0')
        self.assertEqual(info['robot_type'], 'egodex_hand')
        self.assertEqual(info['total_episodes'], 2)
        self.assertEqual(info['total_frames'], 18)  # 10 + 8
        self.assertEqual(info['fps'], 10)
        self.assertEqual(info['total_tasks'], 2)

        # Check features
        self.assertIn('observation.state', info['features'])
        self.assertEqual(info['features']['observation.state']['shape'], [8])
        self.assertIn('action', info['features'])
        self.assertEqual(info['features']['action']['shape'], [7])

        # Check episodes.jsonl
        episodes_path = os.path.join(meta_dir, 'episodes.jsonl')
        self.assertTrue(os.path.exists(episodes_path))
        with open(episodes_path, 'r') as f:
            episodes = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(episodes), 2)

        # Check tasks.jsonl
        tasks_path = os.path.join(meta_dir, 'tasks.jsonl')
        self.assertTrue(os.path.exists(tasks_path))
        with open(tasks_path, 'r') as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(tasks), 2)

        # Check modality.json
        modality_path = os.path.join(meta_dir, 'modality.json')
        self.assertTrue(os.path.exists(modality_path))
        with open(modality_path, 'r') as f:
            modality = json.load(f)
        self.assertIn('state', modality)
        self.assertIn('action', modality)

        # Check data directory
        data_dir = os.path.join(self.output_dir, 'data', 'chunk-000')
        self.assertTrue(os.path.exists(data_dir))
        parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        self.assertEqual(len(parquet_files), 2)

        # Check staging is cleaned up
        staging_dir = os.path.join(self.output_dir, 'staging')
        self.assertFalse(os.path.exists(staging_dir))

    def test_empty_action_data(self):
        """Test with empty action data - should not export anything."""
        sample = {
            'videos': [self.vid3_path],
            'text': 'test',
            Fields.meta: {
                'hand_action_tags': [],
                MetaKeys.video_frames: [[]],
            }
        }

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            hand_action_field='hand_action_tags',
        )

        dataset = Dataset.from_list([sample])
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        export_info = res_list[0][Fields.meta].get('lerobot_export', [])
        self.assertEqual(len(export_info), 0)

    def test_same_task_deduplication(self):
        """Test that episodes with the same task share a task_index."""
        samples = [
            self._make_sample(self.vid3_path, num_frames=5, task_desc="pick up cup"),
            self._make_sample(self.vid4_path, num_frames=5, task_desc="pick up cup"),
        ]

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            hand_action_field='hand_action_tags',
            fps=10,
        )

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)

        ExportToLeRobotMapper.finalize_dataset(
            output_dir=self.output_dir, fps=10,
        )

        with open(os.path.join(self.output_dir, 'meta', 'info.json'), 'r') as f:
            info = json.load(f)
        self.assertEqual(info['total_tasks'], 1)  # same task

    def test_mul_proc(self):
        """Test with multiple processes."""
        samples = [
            self._make_sample(self.vid3_path, num_frames=5),
            self._make_sample(self.vid4_path, num_frames=5),
        ]

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            hand_action_field='hand_action_tags',
            fps=10,
        )

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        for sample in res_list:
            export_info = sample[Fields.meta].get('lerobot_export', [])
            self.assertGreater(len(export_info), 0)


if __name__ == '__main__':
    unittest.main()
