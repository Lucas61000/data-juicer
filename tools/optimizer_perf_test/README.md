# Optimizer Performance Test

Benchmark tool for comparing pipeline execution with and without the optimizer.

## Usage

```bash
python tools/optimizer_perf_test/run_test.py \
  --recipe-path tools/optimizer_perf_test/configs/optimizer_benchmark.yaml \
  --dataset-path test_data/benchmark_10k.jsonl \
  --output-dir outputs/optimizer_bench \
  --verbose
```

## Options

- `--recipe-path PATH`: Path to the recipe YAML file (required)
- `--dataset-path PATH`: Path to the dataset file (required)
- `--output-dir PATH`: Output directory (default: `./outputs/pipeline_perf_test`)
- `--strategies LIST`: Comma-separated list of strategies to test (default: `op_reorder,filter_fusion`)
- `--executor TYPE`: Executor type: `default` (local) or `ray` (distributed). Default: `default`
- `--verbose`: Enable verbose logging

## Testing Specific Strategies

Test only filter fusion:
```bash
python tools/optimizer_perf_test/run_test.py \
  --recipe-path tools/optimizer_perf_test/configs/optimizer_benchmark.yaml \
  --dataset-path test_data/benchmark_10k.jsonl \
  --strategies filter_fusion
```

Test only operation reordering:
```bash
python tools/optimizer_perf_test/run_test.py \
  --recipe-path tools/optimizer_perf_test/configs/optimizer_benchmark.yaml \
  --dataset-path test_data/benchmark_10k.jsonl \
  --strategies op_reorder
```

Available strategies: `op_reorder`, `filter_fusion`, `mapper_fusion`

## Testing with Ray Executor

Test with Ray distributed executor:
```bash
python tools/optimizer_perf_test/run_test.py \
  --recipe-path tools/optimizer_perf_test/configs/optimizer_benchmark.yaml \
  --dataset-path test_data/benchmark_10k.jsonl \
  --executor ray
```

## Generating Test Data

```bash
python tools/optimizer_perf_test/generate_test_data.py \
  --output test_data/benchmark_10k.jsonl \
  --samples 10000
```

## Output

The test generates:

- `results.json`: Raw performance metrics
- `performance_report.md`: Human-readable report
- `perf_test.log`: Detailed execution log

## Example Results

With 10k samples and 5 filter operations:

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Time | 12.37s | 9.94s |
| Speedup | - | **1.24x** |
| Improvement | - | **19.6%** |

The optimizer fuses 4 compatible filters into a single `FusedFilter` operation, reducing overhead and enabling parallel execution of filter predicates.
