# Filter and Mapper Fusion Strategies

This document details the behavior and performance characteristics of Data-Juicer's fusion optimization strategies.

## Overview

Fusion strategies combine multiple operations into a single pass through the data, reducing overhead and enabling intermediate variable sharing.

| Strategy | Default Executor | Ray Executor |
|----------|------------------|--------------|
| **Filter Fusion** | +15% speedup | Skipped (auto) |
| **Mapper Fusion** | +14% speedup | Skipped (auto) |

## Filter Fusion

### How It Works

Filter fusion combines multiple filters that share **intermediate variables** into a single fused operation. This allows expensive computations (like tokenization) to be performed once and shared across filters.

**Intermediate Variable Registries:**
- `INTER_WORDS` - Tokenized words (word_repetition_filter, words_num_filter, stopwords_filter, flagged_words_filter, perplexity_filter)
- `INTER_LINES` - Split lines
- `LOADED_IMAGES` - Loaded image data
- `LOADED_VIDEOS` - Loaded video data
- `LOADED_AUDIOS` - Loaded audio data
- `INTER_SAMPLED_FRAMES` - Sampled video frames

### Example

```yaml
# These 4 filters all tokenize text into words
# Without fusion: tokenize 4 times
# With fusion: tokenize once, share across all filters

process:
  - word_repetition_filter:
      rep_len: 10
      max_ratio: 0.5

  - words_num_filter:
      min_num: 5
      max_num: 1000

  - stopwords_filter:
      lang: en
      max_ratio: 0.8

  - flagged_words_filter:
      lang: en
      max_ratio: 0.01
```

### Performance Results

**Dataset:** C4 (356K samples)
**Filters:** 4 INTER_WORDS filters

| Executor | Baseline | Optimized | Speedup | Improvement |
|----------|----------|-----------|---------|-------------|
| Default  | 88.98s   | 75.69s    | 1.18x   | **+14.9%**  |
| Ray      | 110.00s  | 125.77s   | 0.87x   | -14.3%      |

### Why Ray Is Different

Ray already parallelizes operations efficiently across workers. When filters are fused:
- **Default executor**: Fusion reduces iteration overhead and shares intermediate variables
- **Ray executor**: Fusion forces sequential execution within each worker, losing Ray's parallelization benefits

**Result:** Filter fusion is automatically skipped in Ray mode.

## Mapper Fusion

### How It Works

Mapper fusion combines consecutive mappers into a single pass through the data, reducing dataset iteration overhead.

### Example

```yaml
# These 6 mappers can be fused into a single pass
process:
  - clean_html_mapper: {}
  - clean_links_mapper: {}
  - clean_email_mapper: {}
  - clean_copyright_mapper: {}
  - punctuation_normalization_mapper: {}
  - whitespace_normalization_mapper: {}
```

### Performance Results

**Dataset:** C4 (356K samples)
**Mappers:** 6 text cleaning mappers

| Executor | Baseline | Optimized | Speedup | Improvement |
|----------|----------|-----------|---------|-------------|
| Default  | 45.07s   | 38.59s    | 1.17x   | **+14.4%**  |
| Ray      | 65.18s   | 90.37s    | 0.72x   | -38.6%      |

### Why Ray Is Different

Same as filter fusion - Ray's built-in parallelization is more efficient than sequential fusion within workers.

**Result:** Mapper fusion is automatically skipped in Ray mode.

## Configuration

### Enable Fusion (Default Executor)

```yaml
enable_optimizer: true
optimizer_strategies:
  - filter_fusion
  - mapper_fusion

executor_type: default  # Fusion provides benefit here
```

### Ray Executor (Fusion Auto-Skipped)

```yaml
enable_optimizer: true
optimizer_strategies:
  - filter_fusion   # Will be skipped automatically
  - mapper_fusion   # Will be skipped automatically

executor_type: ray  # Ray parallelizes better without fusion
```

## When to Use Fusion

### Use Filter Fusion When:
- Using **default executor**
- Pipeline has **3+ filters** sharing intermediate variables
- Filters perform expensive computations (tokenization, model inference)

### Use Mapper Fusion When:
- Using **default executor**
- Pipeline has **3+ consecutive mappers**
- Mappers are simple text transformations

### Don't Use Fusion When:
- Using **Ray executor** (auto-skipped anyway)
- Only 1-2 operations (overhead outweighs benefit)
- Operations have complex dependencies

## Benchmarking

Test fusion performance on your data:

```bash
# Filter fusion benchmark
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark_words.yaml \
  --dataset-path your_data.jsonl \
  --strategies filter_fusion \
  --executor default

# Mapper fusion benchmark
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark_mappers.yaml \
  --dataset-path your_data.jsonl \
  --strategies mapper_fusion \
  --executor default
```

## Technical Details

### Filter Fusion Implementation

1. **Grouping**: Filters are grouped by their intermediate variable registry
2. **Context sharing**: A shared context dictionary passes intermediate variables between filters
3. **Sequential execution**: Filters run sequentially within the fused operation
4. **Stats initialization**: Stats field is initialized once for all fused filters

### Mapper Fusion Implementation

1. **Grouping**: Consecutive mappers without dependencies are grouped
2. **Single pass**: All mappers process each batch in sequence
3. **No intermediate variables**: Mappers don't share intermediate computations

### Files

- `data_juicer/core/optimizer/filter_fusion_strategy.py` - Filter fusion strategy
- `data_juicer/core/optimizer/mapper_fusion_strategy.py` - Mapper fusion strategy
- `data_juicer/core/optimizer/fused_op.py` - FusedFilter and FusedMapper implementations
- `data_juicer/ops/op_fusion.py` - Intermediate variable registries
