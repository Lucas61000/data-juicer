# Branch Execution

Branch execution allows you to execute multiple processing branches in parallel from a common checkpoint, automatically deduplicating the common processing steps.

分支执行允许您从公共检查点并行执行多个处理分支，自动去重公共处理步骤。

## Overview 概述

The branch executor:
1. Executes `common_process` once and saves a checkpoint
2. Executes each branch in parallel from the checkpoint
3. Each branch has its own `export_path`

分支执行器：
1. 执行一次 `common_process` 并保存检查点
2. 从检查点并行执行每个分支
3. 每个分支都有自己的 `export_path`

## Configuration 配置

```yaml
executor_type: 'branch'

# Common process executed once
common_process:
  - miner_u_mapper:
      # configuration
  - document_deduplicator:
      # configuration

# Branches executed in parallel
branches:
  - name: "branch1"
    process:
      - data_augmentation_mapper:
          # configuration
    export_path: './outputs/branch1.jsonl'
  
  - name: "branch2"
    process:
      - html_enhancement_mapper:
          # configuration
    export_path: './outputs/branch2.jsonl'
```

## Use Cases 使用场景

### Example: Multiple Enhancement Paths

You have a common preprocessing pipeline (minerU → deduplication), and then want to apply different enhancements in parallel:

您有一个公共的预处理流程（minerU → 去重），然后想要并行应用不同的增强：

```yaml
common_process:
  - miner_u_mapper: {...}
  - document_deduplicator: {...}

branches:
  - name: "data_augmentation"
    process:
      - data_augmentation_mapper: {...}
    export_path: './outputs/augmented.jsonl'
  
  - name: "html_enhancement"
    process:
      - html_enhancement_mapper: {...}
    export_path: './outputs/html_enhanced.jsonl'
```

This will:
1. Execute minerU and deduplication once
2. Use the result as input for both branches
3. Execute data augmentation and HTML enhancement in parallel
4. Export each branch to its own file

这将：
1. 执行一次 minerU 和去重
2. 使用结果作为两个分支的输入
3. 并行执行数据增强和 HTML 增强
4. 将每个分支导出到自己的文件

## Benefits 优势

1. **Automatic Deduplication**: Common steps are executed only once
   自动去重：公共步骤只执行一次

2. **Parallel Execution**: Branches run in parallel for better performance
   并行执行：分支并行运行以获得更好的性能

3. **Clear Configuration**: Branch structure is explicit and self-documenting
   清晰的配置：分支结构明确且自文档化

4. **Flexible Export**: Each branch can export to different paths
   灵活的导出：每个分支可以导出到不同的路径

## Implementation Details 实现细节

- Common process checkpoints are saved in `{work_dir}/.branch_ckpt/common/`
- Branch checkpoints are saved in `{work_dir}/.branch_ckpt/{branch_name}/`
- Each branch uses a separate DefaultExecutor instance
- Branches can be executed sequentially or in parallel (depending on executor implementation)

- 公共流程检查点保存在 `{work_dir}/.branch_ckpt/common/`
- 分支检查点保存在 `{work_dir}/.branch_ckpt/{branch_name}/`
- 每个分支使用单独的 DefaultExecutor 实例
- 分支可以顺序或并行执行（取决于执行器实现）
