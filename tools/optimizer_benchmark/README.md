# Optimizer Benchmark

Benchmark tool for comparing pipeline execution with and without the optimizer.

## Usage

```bash
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark.yaml \
  --dataset-path test_data/benchmark_10k.jsonl \
  --output-dir outputs/optimizer_bench \
  --verbose
```

## Options

- `--recipe-path PATH`: Path to the recipe YAML file (required)
- `--dataset-path PATH`: Path to the dataset file (required)
- `--output-dir PATH`: Output directory (default: `./outputs/optimizer_benchmark`)
- `--strategies LIST`: Comma-separated list of strategies to test (default: `op_reorder,filter_fusion`)
- `--executor TYPE`: Executor type: `default` (local) or `ray` (distributed). Default: `default`
- `--verbose`: Enable verbose logging

## Testing Specific Strategies

Test only filter fusion:
```bash
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark.yaml \
  --dataset-path test_data/benchmark_10k.jsonl \
  --strategies filter_fusion
```

Test only operation reordering:
```bash
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark.yaml \
  --dataset-path test_data/benchmark_10k.jsonl \
  --strategies op_reorder
```

Available strategies: `op_reorder`, `filter_fusion`, `mapper_fusion`

## Testing with Ray Executor

Test with Ray distributed executor:
```bash
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark.yaml \
  --dataset-path test_data/benchmark_10k.jsonl \
  --executor ray
```

## Generating Test Data

```bash
python tools/optimizer_benchmark/generate_test_data.py \
  --output test_data/benchmark_10k.jsonl \
  --samples 10000
```

## Output

The benchmark generates:

- `results.json`: Raw performance metrics
- `performance_report.md`: Human-readable report
- `benchmark.log`: Detailed execution log

## Example Results

With 356K samples (C4 dataset) and 4 INTER_WORDS filters:

| Executor | Baseline | Optimized | Speedup | Improvement |
|----------|----------|-----------|---------|-------------|
| Default  | 88.98s   | 75.69s    | 1.18x   | **+14.9%**  |
| Ray      | 110.00s  | -         | -       | Skipped     |

The optimizer fuses filters sharing intermediate variables (like tokenized words) into a single `FusedFilter` operation, reducing overhead by computing expensive operations once.

**Note:** Filter and mapper fusion are automatically skipped in Ray mode since Ray already parallelizes operations efficiently.
