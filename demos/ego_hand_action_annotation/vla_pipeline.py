import os
import json
import sys
sys.path.insert(0, os.path.dirname(__file__))

from ray.data import ActorPoolStrategy

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.ops.mapper import (
    VideoCameraCalibrationMogeMapper,
    VideoHandReconstructionHaworMapper,
    VideoCameraPoseMegaSaMMapper,
    VideoExtractFramesMapper,
    VideoHandActionComputeMapper,
    ExportToLeRobotMapper)

from custom_ops.video_action_captioning_mapper import VideoActionCaptioningMapper


def save_to_jsonl(samples, output_file):
    total_samples = len(samples[video_key])
    all_keys = list(samples.keys())
    
    all_samples = []
    for sample_idx in range(total_samples):
        sample = {}
        for k in all_keys:
            sample[k] = samples[k][sample_idx]
        all_samples.append(sample)
    
    try:
        with open(output_file, 'w') as f:
            lines= [json.dumps(sample) for sample in all_samples]
            f.write('\n'.join(lines))
    except Exception as e:
        print(e)

    return all_samples


if __name__ == '__main__':
    import ray

    from ray.data import DataContext
    DataContext.get_current().enable_fallback_to_arrow_object_ext_type = True

    ray.init(address='auto')

    output_dir = "./output/"

    os.makedirs(output_dir, exist_ok=True)
    LEROBOT_OUTPUT_DIR = os.path.join(output_dir, "lerobot_dataset")

    video_paths = [
        "./data/1018.mp4",
        "./data/1034.mp4",
    ]
    samples = [
        {
            "videos": [video],
            "text": "",
            # Fields.stats: {},
            Fields.meta: {}
        } for video in video_paths
    ]
    
    ds = ray.data.from_items(samples)

    video_key = "videos"

    extract_frames_op = VideoExtractFramesMapper(
        frame_sampling_method="all_keyframes",
        output_format='path',  #'bytes',
        frame_dir=os.path.join(output_dir, 'frames'),
        frame_field=MetaKeys.video_frames,
        legacy_split_by_text_token=False,
        batch_mode=True,
        skip_op_error=False,
        video_key=video_key,
        video_backend='ffmpeg'
    )

    action_op = VideoHandActionComputeMapper(
        hand_reconstruction_field=MetaKeys.hand_reconstruction_hawor_tags,
        camera_pose_field=MetaKeys.video_camera_pose_tags,
        tag_field_name=MetaKeys.hand_action_tags,
        hand_type="both",
        batch_mode=True,
        skip_op_error=False,
    )
    caption_op = VideoActionCaptioningMapper(
        api_or_hf_model='qwen-vl-max',
        is_api_model=True,
        hand_type='right',
        frame_field=MetaKeys.video_frames,
        tag_field_name="hand_action_caption",
        batch_mode=True,
        skip_op_error=False,
    )

    export_op = ExportToLeRobotMapper(
        output_dir=LEROBOT_OUTPUT_DIR,
        hand_action_field=MetaKeys.hand_action_tags,
        frame_field=MetaKeys.video_frames,
        video_key=video_key,
        task_description_key="text",
        fps=10,
        robot_type="egodex_hand",
        batch_mode=True,
        skip_op_error=False,
    )

    ds = ds.map_batches(
            extract_frames_op,
            batch_size=1,
            num_cpus=2,
            batch_format="pyarrow",
            runtime_env=None,
        ).map_batches(
            VideoCameraCalibrationMogeMapper,
            fn_constructor_kwargs=dict(
                tag_field_name=MetaKeys.camera_calibration_moge_tags,
                frame_field=MetaKeys.video_frames,
                output_depth=True,
                output_points=False,
                output_mask=False,
                batch_mode=True,
                skip_op_error=False
            ),
            batch_size=1,
            num_gpus=0.1,
            batch_format="pyarrow",
            compute=ActorPoolStrategy(min_size=1, max_size=2),
            runtime_env=None,
        ).map_batches(
            VideoHandReconstructionHaworMapper,
            fn_constructor_kwargs=dict(
                batch_mode=True,
                skip_op_error=False,
                camera_calibration_field=MetaKeys.camera_calibration_moge_tags,
                tag_field_name=MetaKeys.hand_reconstruction_hawor_tags,
                mano_right_path='/path/to/MANO_RIGHT.pkl',
                mano_left_path='/path/to/MANO_LEFT.pkl',
                frame_field=MetaKeys.video_frames,
            ),
            batch_size=1,
            num_gpus=0.1,
            batch_format="pyarrow",
            compute=ActorPoolStrategy(min_size=1, max_size=2),
            runtime_env=None,
        ).map_batches(
            VideoCameraPoseMegaSaMMapper,
            fn_constructor_kwargs=dict(
                tag_field_name=MetaKeys.video_camera_pose_tags,
                camera_calibration_field=MetaKeys.camera_calibration_moge_tags,
                batch_mode=True,
                skip_op_error=False,
            ),
            batch_size=1,
            num_gpus=0.1,
            batch_format="pyarrow",
            compute=ActorPoolStrategy(min_size=1, max_size=2),
            runtime_env={"conda": "mega-sam"},
        ).map_batches(
            action_op,
            batch_size=1,
            num_cpus=2,
            batch_format="pyarrow",
            runtime_env=None,
        ).map_batches(
            caption_op,
            batch_size=1,
            num_cpus=2,
            batch_format="pyarrow",
            runtime_env=None,
        ).map_batches(
            export_op,
            batch_size=1,
            num_cpus=2,
            batch_format="pyarrow",
            runtime_env=None,
        )
    
    ds.write_parquet(output_dir)

    ExportToLeRobotMapper.finalize_dataset(
        output_dir=LEROBOT_OUTPUT_DIR,
        fps=10,
        robot_type="egodex_hand",
    )
    print(f"LeRobot exported to: {LEROBOT_OUTPUT_DIR}")
