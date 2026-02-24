#!/usr/bin/env python3
"""
Performance Test: Baseline vs Optimized Pipeline Comparison

This script runs performance benchmarks comparing baseline pipeline execution
(without optimizer) vs optimized pipeline execution (with optimizer enabled).

Features:
- Separate process execution for isolation
- Support for recipe path and dataset path
- Comprehensive metrics collection
- Result validation and comparison
- Detailed reporting
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_juicer.config import init_configs  # noqa: E402
from data_juicer.core import DefaultExecutor  # noqa: E402
from data_juicer.core.data.dataset_builder import DatasetBuilder  # noqa: E402

# Default strategies to test when none specified
DEFAULT_STRATEGIES = ["op_reorder", "filter_fusion"]


class PipelinePerformanceTester:
    """Performance tester for comparing baseline vs optimized pipeline execution."""

    def __init__(
        self, output_dir: str = "./outputs/pipeline_perf_test", strategies: list = None, executor_type: str = "default"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.strategies = strategies or DEFAULT_STRATEGIES
        self.executor_type = executor_type

        # Setup logging
        log_file = self.output_dir / "perf_test.log"
        logger.add(log_file, rotation="10 MB", level="INFO")

        self.results = {"individual": {}, "optimized": {}, "comparison": {}, "metadata": {}}

    def load_dataset(self, dataset_path: str) -> Any:
        """Load dataset from path using DatasetBuilder."""
        logger.info(f"Loading dataset from: {dataset_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        # Build a minimal config Namespace for DatasetBuilder
        cfg = Namespace(dataset_path=dataset_path)
        builder = DatasetBuilder(cfg)
        dataset = builder.load_dataset()
        dataset_length = len(dataset.to_list()) if dataset is not None else 0
        logger.info(f"Loaded dataset with {dataset_length} samples")
        return dataset

    def create_temp_config(self, recipe_path: str, dataset_path: str, mode: str) -> str:
        """Create a temporary config file for execution."""
        # Load the original recipe
        with open(recipe_path, "r") as f:
            recipe_config = yaml.safe_load(f)

        # Create temp config
        temp_config = {
            "project_name": f"perf-test-{mode}",
            "dataset_path": dataset_path,
            "export_path": str(self.output_dir / f"result_{mode}.jsonl"),
            "np": recipe_config.get("np", 4),  # Use recipe's np or default to 4
            "use_cache": False,
            # Use the new optimizer config instead of legacy op_fusion
            "enable_optimizer": mode == "optimized",
            "optimizer_strategies": self.strategies,
            "executor_type": self.executor_type,
            "process": recipe_config.get("process", []),
        }

        # Write temp config
        temp_config_path = self.output_dir / f"temp_config_{mode}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(temp_config, f, default_flow_style=False)

        return str(temp_config_path)

    def run_individual_pipeline(self, recipe_path: str, dataset_path: str) -> Dict[str, Any]:
        """Run baseline pipeline execution (optimizer disabled) in separate process."""
        logger.info("Running baseline pipeline (optimizer disabled)...")

        temp_config_path = self.create_temp_config(recipe_path, dataset_path, "individual")

        # Run in separate process
        start_time = time.time()
        result = self._run_in_process(temp_config_path, "individual")
        end_time = time.time()

        result["wall_time"] = end_time - start_time
        result["config_path"] = temp_config_path

        return result

    def run_optimized_pipeline(self, recipe_path: str, dataset_path: str) -> Dict[str, Any]:
        """Run optimized pipeline execution (optimizer enabled) in separate process."""
        logger.info("Running optimized pipeline (optimizer enabled)...")

        temp_config_path = self.create_temp_config(recipe_path, dataset_path, "optimized")

        # Run in separate process
        start_time = time.time()
        result = self._run_in_process(temp_config_path, "optimized")
        end_time = time.time()

        result["wall_time"] = end_time - start_time
        result["config_path"] = temp_config_path

        return result

    def _run_in_process(self, config_path: str, mode: str) -> Dict[str, Any]:
        """Run pipeline execution in a separate process."""
        # Create process and run
        result_queue = mp.Queue()
        process = mp.Process(target=_worker_process, args=(config_path, mode, result_queue))

        process.start()
        process.join(timeout=3600)  # 1 hour timeout

        if process.is_alive():
            process.terminate()
            process.join()
            return {"execution_time": 0, "output_samples": 0, "success": False, "error": "Process timeout"}

        if not result_queue.empty():
            return result_queue.get()
        else:
            return {"execution_time": 0, "output_samples": 0, "success": False, "error": "No result from process"}

    def validate_results(self, individual_result: Dict, optimized_result: Dict) -> Dict[str, Any]:
        """Validate that both executions produced similar results."""
        logger.info("Validating results...")

        validation = {
            "samples_match": False,
            "individual_samples": individual_result.get("output_samples", 0),
            "optimized_samples": optimized_result.get("output_samples", 0),
            "sample_difference": 0,
            "validation_passed": False,
        }

        if individual_result.get("success") and optimized_result.get("success"):
            individual_samples = individual_result["output_samples"]
            optimized_samples = optimized_result["output_samples"]

            validation["samples_match"] = individual_samples == optimized_samples
            validation["sample_difference"] = abs(individual_samples - optimized_samples)
            validation["validation_passed"] = validation["samples_match"]

            if validation["validation_passed"]:
                logger.info("Validation passed: Both executions produced the same number of samples")
            else:
                logger.warning(
                    f"Validation failed: Sample count mismatch "
                    f"(baseline: {individual_samples}, optimized: {optimized_samples})"
                )
        else:
            logger.error("Validation failed: One or both executions failed")

        return validation

    def compare_performance(self, individual_result: Dict, optimized_result: Dict) -> Dict[str, Any]:
        """Compare performance metrics between baseline and optimized execution."""
        logger.info("Comparing performance metrics...")

        comparison = {
            "individual_time": individual_result.get("wall_time", 0),
            "optimized_time": optimized_result.get("wall_time", 0),
            "speedup": 0,
            "improvement_percent": 0,
            "faster_mode": "none",
        }

        if individual_result.get("success") and optimized_result.get("success"):
            individual_time = individual_result["wall_time"]
            optimized_time = optimized_result["wall_time"]

            if optimized_time > 0:
                comparison["speedup"] = individual_time / optimized_time
                comparison["improvement_percent"] = ((individual_time - optimized_time) / individual_time) * 100

            if optimized_time < individual_time:
                comparison["faster_mode"] = "optimized"
            elif individual_time < optimized_time:
                comparison["faster_mode"] = "baseline"
            else:
                comparison["faster_mode"] = "equal"

        return comparison

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive performance report."""
        logger.info("Generating performance report...")

        report_path = self.output_dir / "performance_report.md"

        with open(report_path, "w") as f:
            f.write("# Pipeline Optimizer Performance Report\n\n")

            # Summary
            f.write("## Summary\n\n")
            comparison = results["comparison"]
            validation = results["validation"]

            f.write(f"- **Test Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Recipe**: {results['metadata']['recipe_path']}\n")
            f.write(f"- **Dataset**: {results['metadata']['dataset_path']}\n")
            f.write(f"- **Executor**: {results['metadata']['executor_type']}\n")
            f.write(f"- **Strategies**: {', '.join(results['metadata']['strategies'])}\n")
            f.write(f"- **Validation**: {'PASSED' if validation['validation_passed'] else 'FAILED'}\n")
            f.write(f"- **Faster Mode**: {comparison['faster_mode'].title()}\n")
            f.write(f"- **Speedup**: {comparison['speedup']:.2f}x\n")
            f.write(f"- **Improvement**: {comparison['improvement_percent']:.1f}%\n\n")

            # Detailed Results
            f.write("## Detailed Results\n\n")

            f.write("### Baseline (Optimizer Disabled)\n")
            f.write(f"- Execution Time: {results['individual']['wall_time']:.2f}s\n")
            f.write(f"- Output Samples: {results['individual']['output_samples']:,}\n")
            f.write(f"- Success: {results['individual']['success']}\n")
            if not results["individual"]["success"]:
                f.write(f"- Error: {results['individual']['error']}\n")
            f.write("\n")

            f.write("### Optimized (Optimizer Enabled)\n")
            f.write(f"- Execution Time: {results['optimized']['wall_time']:.2f}s\n")
            f.write(f"- Output Samples: {results['optimized']['output_samples']:,}\n")
            f.write(f"- Success: {results['optimized']['success']}\n")
            if not results["optimized"]["success"]:
                f.write(f"- Error: {results['optimized']['error']}\n")
            f.write("\n")

            # Performance Comparison
            f.write("### Performance Comparison\n")
            f.write(f"- Baseline Time: {comparison['individual_time']:.2f}s\n")
            f.write(f"- Optimized Time: {comparison['optimized_time']:.2f}s\n")
            f.write(f"- Speedup: {comparison['speedup']:.2f}x\n")
            f.write(f"- Improvement: {comparison['improvement_percent']:.1f}%\n")
            f.write(f"- Faster Mode: {comparison['faster_mode'].title()}\n\n")

            # Validation Results
            f.write("### Validation Results\n")
            f.write(f"- Samples Match: {validation['samples_match']}\n")
            f.write(f"- Baseline Samples: {validation['individual_samples']:,}\n")
            f.write(f"- Optimized Samples: {validation['optimized_samples']:,}\n")
            f.write(f"- Sample Difference: {validation['sample_difference']}\n")
            f.write(f"- Validation Passed: {validation['validation_passed']}\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            if validation["validation_passed"]:
                if comparison["faster_mode"] == "optimized":
                    f.write(
                        "**Use Optimized Pipeline**: The optimizer improves performance and produces correct results.\n"
                    )
                elif comparison["faster_mode"] == "baseline":
                    f.write(
                        "**Consider Baseline**: The baseline is faster for this workload. "
                        "Optimizer may still help for larger datasets or different operation mixes.\n"
                    )
                else:
                    f.write("**Both Modes Similar**: Performance is similar. Optimizer has minimal overhead.\n")
            else:
                f.write("**Investigation Required**: Results don't match between baseline and optimized modes.\n")

        return str(report_path)

    def save_results(self, results: Dict[str, Any]) -> str:
        """Save results to JSON file."""
        results_path = self.output_dir / "results.json"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return str(results_path)

    def run_test(self, recipe_path: str, dataset_path: str) -> Dict[str, Any]:
        """Run the complete performance test."""
        logger.info("Starting pipeline performance test...")
        logger.info(f"Recipe: {recipe_path}")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Strategies: {self.strategies}")
        logger.info(f"Executor: {self.executor_type}")

        # Store metadata
        self.results["metadata"] = {
            "recipe_path": recipe_path,
            "dataset_path": dataset_path,
            "strategies": self.strategies,
            "executor_type": self.executor_type,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Run baseline (no optimization)
        individual_result = self.run_individual_pipeline(recipe_path, dataset_path)
        self.results["individual"] = individual_result

        # Run optimized pipeline
        optimized_result = self.run_optimized_pipeline(recipe_path, dataset_path)
        self.results["optimized"] = optimized_result

        # Validate results
        validation = self.validate_results(individual_result, optimized_result)
        self.results["validation"] = validation

        # Compare performance
        comparison = self.compare_performance(individual_result, optimized_result)
        self.results["comparison"] = comparison

        # Save results
        results_path = self.save_results(self.results)
        logger.info(f"Results saved to: {results_path}")

        # Generate report
        report_path = self.generate_report(self.results)
        logger.info(f"Report generated: {report_path}")

        # Print summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print a summary of the test results."""
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE TEST SUMMARY")
        logger.info("=" * 60)

        comparison = self.results.get("comparison", {})
        validation = self.results.get("validation", {})

        logger.info(f"Baseline Time:  {comparison.get('individual_time', 0):.2f}s")
        logger.info(f"Optimized Time: {comparison.get('optimized_time', 0):.2f}s")
        logger.info(f"Speedup:        {comparison.get('speedup', 0):.2f}x")
        logger.info(f"Improvement:    {comparison.get('improvement_percent', 0):.1f}%")
        logger.info(f"Validation:     {'PASSED' if validation.get('validation_passed') else 'FAILED'}")
        logger.info(f"Faster Mode:    {comparison.get('faster_mode', 'none').title()}")
        logger.info("=" * 60)


def _worker_process(config_path: str, mode: str, result_queue: mp.Queue):
    """Worker function for running pipeline execution in separate process."""
    try:
        # Add the project root to the path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))

        # Initialize config
        args = ["--config", config_path]
        cfg = init_configs(args=args)

        # Create executor based on executor_type
        executor_type = getattr(cfg, "executor_type", "default")
        if executor_type == "ray":
            from data_juicer.core import RayExecutor

            executor = RayExecutor(cfg)
        else:
            executor = DefaultExecutor(cfg)

        # Run and collect metrics
        start_time = time.time()
        dataset = executor.run()
        end_time = time.time()

        # Collect results - handle both Dataset and RayDataset
        if dataset is not None:
            if hasattr(dataset, "data") and hasattr(dataset.data, "count"):
                # RayDataset
                dataset_length = dataset.data.count()
            elif hasattr(dataset, "__len__"):
                dataset_length = len(dataset)
            else:
                dataset_length = 0
        else:
            dataset_length = 0
        result = {
            "execution_time": end_time - start_time,
            "output_samples": dataset_length,
            "success": True,
            "error": None,
        }

        result_queue.put(result)

    except Exception as e:
        import traceback

        result = {
            "execution_time": 0,
            "output_samples": 0,
            "success": False,
            "error": f"{str(e)}\n{traceback.format_exc()}",
        }
        result_queue.put(result)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pipeline Performance Test: Compare baseline vs optimized execution")
    parser.add_argument("--recipe-path", type=str, required=True, help="Path to the recipe YAML file")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/pipeline_perf_test",
        help="Output directory for results and reports",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated list of optimization strategies to test. "
        "Available: op_reorder, filter_fusion, mapper_fusion. "
        "Default: op_reorder,filter_fusion. "
        "Example: --strategies filter_fusion (test only filter fusion)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--executor",
        type=str,
        default="default",
        choices=["default", "ray"],
        help="Executor type: 'default' (local) or 'ray' (distributed). Default: default",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Parse strategies
    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]
        logger.info(f"Testing strategies: {strategies}")

    logger.info(f"Using executor: {args.executor}")

    # Create tester and run test
    tester = PipelinePerformanceTester(args.output_dir, strategies=strategies, executor_type=args.executor)

    try:
        results = tester.run_test(args.recipe_path, args.dataset_path)

        # Exit with appropriate code
        validation = results.get("validation", {})
        if validation.get("validation_passed"):
            logger.info("Test completed successfully")
            sys.exit(0)
        else:
            logger.error("Test failed validation")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
