import unittest
import os

from data_juicer.core import NestedDataset as Dataset
from data_juicer.ops.mapper.lidar_segmentation_mapper import LiDARSegmentationMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.constant import Fields, MetaKeys


class LiDARSegmentationMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    lidar_test1 = os.path.join(data_path, 'lidar_test1.bin')
    lidar_test2 = os.path.join(data_path, 'lidar_test2.bin')
    lidar_test3 = os.path.join(data_path, 'lidar_test3.bin')

    model_cfg_name = "cylinder3d_8xb2-laser-polar-mix-3x_semantickitti"
    model_path = "cylinder3d_8xb2-amp-laser-polar-mix-3x_semantickitti_20230425_144950-372cdf69.pth"
 

    def test_cpu(self):
        source = [
            {
                'lidar': self.lidar_test1
            },
            {
                'lidar': self.lidar_test2
            },
            {
                'lidar': self.lidar_test3
            }
        ]
        
        op = LiDARSegmentationMapper(
            model_name="cylinder3d",
            model_cfg_name=self.model_cfg_name,
            model_path=self.model_path,
        )
        
        dataset = Dataset.from_list(source)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=False)
        res_list = dataset.to_list()

        self.assertEqual(len(res_list), 3)
        self.assertEqual(list(res_list[0][Fields.meta][MetaKeys.lidar_segmentation_tags].keys()), ['box_type_3d', 'pts_semantic_mask'])
        self.assertEqual(len(res_list[0][Fields.meta][MetaKeys.lidar_segmentation_tags]["pts_semantic_mask"]), 17238)
        self.assertEqual(res_list[0][Fields.meta][MetaKeys.lidar_segmentation_tags]["box_type_3d"], "LiDAR")


    def test_cuda(self):
        source = [
            {
                'lidar': self.lidar_test1
            },
            {
                'lidar': self.lidar_test2
            },
            {
                'lidar': self.lidar_test3
            }
        ]
        
        op = LiDARSegmentationMapper(
            model_name="cylinder3d",
            model_cfg_name=self.model_cfg_name,
            model_path=self.model_path,
        )
        
        dataset = Dataset.from_list(source)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        self.assertEqual(len(res_list), 3)
        self.assertEqual(list(res_list[0][Fields.meta][MetaKeys.lidar_segmentation_tags].keys()), ['box_type_3d', 'pts_semantic_mask'])
        self.assertEqual(len(res_list[0][Fields.meta][MetaKeys.lidar_segmentation_tags]["pts_semantic_mask"]), 17238)
        self.assertEqual(res_list[0][Fields.meta][MetaKeys.lidar_segmentation_tags]["box_type_3d"], "LiDAR")


    
    def test_cpu_mul_proc(self):
        source = [
            {
                'lidar': self.lidar_test1
            },
            {
                'lidar': self.lidar_test2
            },
            {
                'lidar': self.lidar_test3
            }
        ]
        
        op = LiDARSegmentationMapper(
            model_name="cylinder3d",
            model_cfg_name=self.model_cfg_name,
            model_path=self.model_path,
        )
        
        dataset = Dataset.from_list(source)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=2, with_rank=False)
        res_list = dataset.to_list()

        self.assertEqual(len(res_list), 3)
        self.assertEqual(list(res_list[0][Fields.meta][MetaKeys.lidar_segmentation_tags].keys()), ['box_type_3d', 'pts_semantic_mask'])
        self.assertEqual(len(res_list[0][Fields.meta][MetaKeys.lidar_segmentation_tags]["pts_semantic_mask"]), 17238)
        self.assertEqual(res_list[0][Fields.meta][MetaKeys.lidar_segmentation_tags]["box_type_3d"], "LiDAR")


    def test_cuda_mul_proc(self):
        source = [
            {
                'lidar': self.lidar_test1
            },
            {
                'lidar': self.lidar_test2
            },
            {
                'lidar': self.lidar_test3
            }
        ]
        
        op = LiDARSegmentationMapper(
            model_name="cylinder3d",
            model_cfg_name=self.model_cfg_name,
            model_path=self.model_path,
        )
        
        dataset = Dataset.from_list(source)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        self.assertEqual(len(res_list), 3)
        self.assertEqual(list(res_list[0][Fields.meta][MetaKeys.lidar_segmentation_tags].keys()), ['box_type_3d', 'pts_semantic_mask'])
        self.assertEqual(len(res_list[0][Fields.meta][MetaKeys.lidar_segmentation_tags]["pts_semantic_mask"]), 17238)
        self.assertEqual(res_list[0][Fields.meta][MetaKeys.lidar_segmentation_tags]["box_type_3d"], "LiDAR")


if __name__ == "__main__":
    unittest.main()