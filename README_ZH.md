# HumanVBench

本仓库包含论文 **HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks (CVPR'26)** 的评估脚本与数据操作算子。

---

## 1. HumanVBench 下载与评估

您可以从 [HuggingFace](https://huggingface.co/datasets/datajuicer/HumanVBench) 或 [ModelScope](https://www.modelscope.cn/datasets/Data-Juicer/HumanVBench) 下载 HumanVBench 数据集。

要评估模型在 HumanVBench 上的表现，请使用 `Evaluation.py` 脚本。您可以通过修改 `eval_goal` 参数来指定评估类型：

* **`'all'`**：评估所有任务。返回每个问题的结果、各任务准确率、随机基线准确率，以及四个维度的平均准确率。
* **特定维度**：从 `'Emotion Perception'`（情绪感知）、`'Person Recognition'`（人物识别）、`'Human Behavior Analysis'`（人体行为分析）或 `'Cross-Modal Speech-Visual Alignment'`（跨模态语音 - 视觉对齐）中选择。
* **`'Self Selection'`**：通过指定 `all_task_names` 中的任务名称，评估特定任务。

### 评估步骤

1. **数据准备**：
   下载 JSONL 文件和视频数据。将 `task_jsonls_path` 设置为包含 16 个 JSONL 文件的目录路径，将 `videos_save_path` 设置为视频根目录。
   ```text
   (videos_save_path)
   ├── Action_Temporal_Analysis
   │   └── (... .mp4)
   ├── Active_speaker_detection
   │   └── (... .mp4)
   └── ...
   ```

2. **模型加载**：
   在 `if __name__ == "__main__":` 代码块中完成模型加载部分，初始化您的模型、processor 和 tokenizer。

3. **推理逻辑**：
   在 `eval_on_one_task` 函数中实现推理逻辑以生成输出文本。如果使用 API 调用，可修改此部分直接返回 API 响应结果。

## 2. Data-Juicer-HumanVbench-ops（数据标注流水线）
本部分提供由 Data-Juicer 支持的、用于以人为中心的视频过滤与标注的数据操作算子。

### 相关文件
- 示例配方：`data-juicer/demos/video_humanvbench_simple/analyzer.yaml`
- 算子定义：`data-juicer/data_juicer/config/config_all.yaml`

### 快速开始
由于这些算子需要对外部模型进行特定修改，调整后的代码库存储在 `data-juicer/thirdparty/humanvbench_models` 中。

#### 选项 1：自动模式（推荐）
算子内置逻辑可自动处理依赖项的 `git clone` 和 `merge diff` 操作。您可以直接运行流水线：

```shell
cd data-juicer
dj-process --config demos/video_humanvbench_simple/analyzer.yaml
```

#### 选项 2：手动模式
如果您倾向于手动配置，请按照 `thirdparty/humanvbench_models/README.md` 中的说明克隆仓库并应用 `.diff` 补丁，然后运行：

```shell
cd data-juicer
dj-process --config demos/video_humanvbench_simple/analyzer.yaml
```

**注意**：HumanVBench 中的大多数任务均可基于此标注流水线的结果构建。详细构建流程请参阅论文附录。