#!/usr/bin/env python3
"""
Operation Prober for sampling-based cost estimation.

This module provides functionality to estimate operation costs by running
them on a small sample of data. The measured costs are used by optimization
strategies (like op_reorder) to make better decisions.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class ProbeResult:
    """Result of probing a single operation."""

    op_name: str
    total_time_seconds: float
    samples_processed: int
    time_per_sample_ms: float
    # Selectivity for filters (ratio of samples that pass)
    selectivity: Optional[float] = None
    # Any errors encountered during probing
    error: Optional[str] = None

    @property
    def cost(self) -> float:
        """Get the cost score (higher = more expensive)."""
        return self.time_per_sample_ms


@dataclass
class ProbeResults:
    """Collection of probe results for multiple operations."""

    results: Dict[str, ProbeResult] = field(default_factory=dict)
    sample_size: int = 0
    total_probe_time_seconds: float = 0.0

    def get_cost(self, op_name: str, default: float = 10.0) -> float:
        """Get the cost for an operation, with fallback to default."""
        if op_name in self.results:
            result = self.results[op_name]
            if result.error is None:
                return result.cost
        return default

    def get_selectivity(self, op_name: str, default: float = 1.0) -> float:
        """Get the selectivity for a filter operation."""
        if op_name in self.results:
            result = self.results[op_name]
            if result.selectivity is not None:
                return result.selectivity
        return default

    def get_effective_cost(self, op_name: str, default: float = 10.0) -> float:
        """Get effective cost considering selectivity.

        For filters, the effective cost considers how many samples pass through.
        A highly selective filter (low selectivity) that runs early reduces
        the data volume for subsequent operations.
        """
        cost = self.get_cost(op_name, default)
        selectivity = self.get_selectivity(op_name, 1.0)
        # Effective cost = actual cost / (1 - selectivity)
        # This favors filters that remove more data (lower selectivity)
        if selectivity < 1.0:
            # Bonus for selective filters
            return cost * selectivity
        return cost


class OpProber:
    """
    Probes operations to estimate their costs using actual execution.

    This class runs each operation on a small sample of data and measures
    the execution time. The results can be used by optimization strategies
    to make data-driven reordering decisions.
    """

    def __init__(
        self,
        sample_size: int = 100,
        timeout_per_op_seconds: float = 30.0,
        min_samples: int = 10,
    ):
        """
        Initialize the operation prober.

        Args:
            sample_size: Number of samples to use for probing (default: 100)
            timeout_per_op_seconds: Maximum time to spend probing each op
            min_samples: Minimum samples required for valid probing
        """
        self.sample_size = sample_size
        self.timeout_per_op_seconds = timeout_per_op_seconds
        self.min_samples = min_samples

    def probe_operations(
        self,
        ops: List[Any],
        dataset: Any,
    ) -> ProbeResults:
        """
        Probe all operations to estimate their costs.

        Args:
            ops: List of operation objects to probe
            dataset: Dataset to sample from for probing

        Returns:
            ProbeResults containing cost estimates for each operation
        """
        probe_results = ProbeResults()
        probe_start = time.time()

        # Get a sample of the dataset
        sample_dataset = self._get_sample(dataset)
        if sample_dataset is None:
            logger.warning("Failed to get sample dataset for probing")
            return probe_results

        probe_results.sample_size = len(sample_dataset)
        logger.info(f"Probing {len(ops)} operations on {probe_results.sample_size} samples")

        # Probe each operation
        for op in ops:
            op_name = self._get_op_name(op)
            try:
                result = self._probe_single_op(op, sample_dataset)
                probe_results.results[op_name] = result
                logger.debug(
                    f"  {op_name}: {result.time_per_sample_ms:.2f} ms/sample"
                    + (f", selectivity={result.selectivity:.2%}" if result.selectivity else "")
                )
            except Exception as e:
                logger.warning(f"Failed to probe {op_name}: {e}")
                probe_results.results[op_name] = ProbeResult(
                    op_name=op_name,
                    total_time_seconds=0,
                    samples_processed=0,
                    time_per_sample_ms=0,
                    error=str(e),
                )

        probe_results.total_probe_time_seconds = time.time() - probe_start
        logger.info(f"Probing completed in {probe_results.total_probe_time_seconds:.2f}s")

        # Clean up any state that might interfere with later runs
        # This helps prevent "subprocess abruptly died" errors
        import gc

        del sample_dataset
        gc.collect()

        return probe_results

    def _get_sample(self, dataset: Any) -> Any:
        """Get a sample from the dataset for probing."""
        try:
            # Get dataset length
            if hasattr(dataset, "__len__"):
                dataset_len = len(dataset)
            elif hasattr(dataset, "num_rows"):
                dataset_len = dataset.num_rows
            elif hasattr(dataset, "count"):
                dataset_len = dataset.count()
            else:
                logger.warning("Cannot determine dataset length")
                return None

            # Determine sample size
            actual_sample_size = min(self.sample_size, dataset_len)
            if actual_sample_size < self.min_samples:
                logger.warning(f"Dataset too small for probing ({dataset_len} < {self.min_samples})")
                return None

            # Take a sample - convert to list of dicts to completely isolate from original
            if hasattr(dataset, "select"):
                # HuggingFace datasets - convert to list and back to create completely
                # independent copy that won't interfere with multiprocessing
                from datasets import Dataset

                indices = list(range(actual_sample_size))
                sample_rows = [dataset[i] for i in indices]
                return Dataset.from_list(sample_rows)

            elif hasattr(dataset, "take"):
                # Ray datasets or similar
                return dataset.take(actual_sample_size)
            else:
                # Try slicing
                return dataset[:actual_sample_size]

        except Exception as e:
            logger.warning(f"Failed to sample dataset: {e}")
            return None

    def _probe_single_op(self, op: Any, sample_dataset: Any) -> ProbeResult:
        """Probe a single operation and return the result."""
        op_name = self._get_op_name(op)

        # Get initial sample count
        initial_count = self._get_dataset_length(sample_dataset)

        # Temporarily disable multiprocessing for probing to avoid subprocess crashes
        # on small sample sizes
        original_num_proc = getattr(op, "num_proc", None)
        if hasattr(op, "num_proc"):
            op.num_proc = 1

        try:
            # Time the operation
            start_time = time.time()

            # Run the operation
            # We need to handle different op types (Filter, Mapper, etc.)
            if hasattr(op, "run"):
                # Standard Data-Juicer operation
                result_dataset = op.run(sample_dataset)
            elif hasattr(op, "process"):
                result_dataset = op.process(sample_dataset)
            else:
                raise ValueError(f"Operation {op_name} has no run or process method")
        finally:
            # Restore original num_proc setting
            if hasattr(op, "num_proc") and original_num_proc is not None:
                op.num_proc = original_num_proc

        elapsed_time = time.time() - start_time

        # Calculate metrics
        final_count = self._get_dataset_length(result_dataset)
        samples_processed = initial_count

        # Calculate selectivity for filters
        selectivity = None
        if "filter" in op_name.lower() and initial_count > 0:
            selectivity = final_count / initial_count

        # Time per sample in milliseconds
        time_per_sample_ms = (elapsed_time * 1000) / max(samples_processed, 1)

        return ProbeResult(
            op_name=op_name,
            total_time_seconds=elapsed_time,
            samples_processed=samples_processed,
            time_per_sample_ms=time_per_sample_ms,
            selectivity=selectivity,
        )

    def _get_op_name(self, op: Any) -> str:
        """Get the name of an operation."""
        if hasattr(op, "_name"):
            return op._name
        elif hasattr(op, "name"):
            return op.name
        else:
            return type(op).__name__

    def _get_dataset_length(self, dataset: Any) -> int:
        """Get the length of a dataset."""
        if dataset is None:
            return 0
        if hasattr(dataset, "__len__"):
            return len(dataset)
        elif hasattr(dataset, "num_rows"):
            return dataset.num_rows
        elif hasattr(dataset, "count"):
            return dataset.count()
        return 0
