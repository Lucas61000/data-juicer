"""
Branch Executor for parallel branch execution.

This executor supports executing multiple branches in parallel from a common
checkpoint, automatically deduplicating the common processing steps.

Example usage:
    common_process:
      - miner_u_mapper: {...}
      - document_deduplicator: {...}

    branches:
      - name: "data_augmentation"
        process:
          - data_augmentation_mapper: {...}
        export_path: "./outputs/branch1.jsonl"
      - name: "html_enhancement"
        process:
          - html_enhancement_mapper: {...}
        export_path: "./outputs/branch2.jsonl"
"""

import os
from copy import deepcopy
from typing import Any, Dict, Optional

from loguru import logger

from data_juicer.core.data import DatasetBuilder
from data_juicer.core.executor.base import ExecutorBase
from data_juicer.core.executor.default_executor import DefaultExecutor


class BranchExecutor(ExecutorBase):
    """
    Executor that supports parallel branch execution from a common checkpoint.

    This executor:
    1. Executes common_process once and saves a checkpoint
    2. Executes each branch in parallel from the checkpoint
    3. Each branch has its own export_path
    """

    def __init__(self, cfg: Optional[Any] = None):
        """Initialize the branch executor."""
        super().__init__(cfg)
        self.executor_type = "branch"
        self.work_dir = self.cfg.work_dir
        self.job_id = getattr(self.cfg, "job_id", None)

        # Parse branch configuration
        self.common_process = getattr(self.cfg, "common_process", [])
        self.branches = getattr(self.cfg, "branches", [])

        if not self.common_process:
            raise ValueError("common_process must be specified for branch executor")
        if not self.branches:
            raise ValueError("branches must be specified for branch executor")

        # Setup checkpoint directory for common process
        self.common_ckpt_dir = os.path.join(self.work_dir, ".branch_ckpt", "common")
        os.makedirs(self.common_ckpt_dir, exist_ok=True)

        # Initialize dataset builder
        self.datasetbuilder = DatasetBuilder(self.cfg, executor_type="default")

    def run(self, load_data_np: Optional[int] = None, skip_export: bool = False, skip_return: bool = False):
        """
        Run the branch execution pipeline.

        :param load_data_np: number of workers when loading the dataset
        :param skip_export: whether to export the results
        :param skip_return: skip return for API calls
        :return: dict of branch names to processed datasets
        """
        logger.info("Starting branch execution...")

        # Step 1: Execute common process and save checkpoint
        logger.info("Executing common process...")
        common_dataset = self._execute_common_process(load_data_np)

        # Step 2: Execute branches in parallel
        logger.info(f"Executing {len(self.branches)} branches in parallel...")
        branch_results = {}
        for branch in self.branches:
            branch_name = branch.get("name", f"branch_{len(branch_results)}")
            logger.info(f"Executing branch: {branch_name}")
            branch_dataset = self._execute_branch(branch, common_dataset, load_data_np, skip_export)
            branch_results[branch_name] = branch_dataset

        logger.info("Branch execution completed")
        return branch_results

    def _execute_common_process(self, load_data_np: Optional[int] = None):
        """Execute the common process and save checkpoint."""
        # Create a temporary config for common process
        common_cfg = deepcopy(self.cfg)
        common_cfg.process = self.common_process
        common_cfg.export_path = None  # Don't export common process
        common_cfg.work_dir = self.work_dir
        common_cfg.checkpoint_dir = self.common_ckpt_dir

        # Use default executor to run common process
        common_executor = DefaultExecutor(common_cfg)
        common_executor.datasetbuilder = self.datasetbuilder

        # Run common process
        common_dataset = common_executor.run(load_data_np=load_data_np, skip_export=True, skip_return=True)

        logger.info(f"Common process completed. Dataset size: {len(common_dataset)}")
        return common_dataset

    def _execute_branch(
        self, branch: Dict[str, Any], common_dataset: Any, load_data_np: Optional[int] = None, skip_export: bool = False
    ):
        """Execute a single branch from the common checkpoint."""
        branch_name = branch.get("name", "unknown")
        branch_process = branch.get("process", [])
        branch_export_path = branch.get("export_path")

        if not branch_process:
            logger.warning(f"Branch {branch_name} has no process, returning common dataset")
            return common_dataset

        # Create a temporary config for this branch
        branch_cfg = deepcopy(self.cfg)
        branch_cfg.process = branch_process
        branch_cfg.export_path = branch_export_path if not skip_export else None
        branch_cfg.work_dir = self.work_dir
        branch_cfg.checkpoint_dir = os.path.join(self.work_dir, ".branch_ckpt", branch_name)

        # Use default executor to run branch
        branch_executor = DefaultExecutor(branch_cfg)

        # Run branch process starting from common dataset
        branch_dataset = branch_executor.run(
            dataset=common_dataset, load_data_np=load_data_np, skip_export=skip_export, skip_return=True
        )

        logger.info(f"Branch {branch_name} completed. Dataset size: {len(branch_dataset)}")
        return branch_dataset
