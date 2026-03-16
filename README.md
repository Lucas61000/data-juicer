# HumanVBench

This repository contains the evaluation scripts and data operators for the paper: **HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks (CVPR'26)**.

---

## 1. HumanVBench Download and Evaluation

You can download HumanVBench from [HuggingFace](https://huggingface.co/datasets/datajuicer/HumanVBench) or [ModelScope](https://www.modelscope.cn/datasets/Data-Juicer/HumanVBench).

To evaluate a model on HumanVBench, use the `Evaluation.py` script. You can modify the `eval_goal` parameter to specify the evaluation type:

* **'all'**: Evaluate on all tasks. Returns results for each question, task-specific accuracy, random accuracy, and average accuracy across four dimensions.
* **Specific Dimension**: Choose from `'Emotion Perception'`, `'Person Recognition'`, `'Human Behavior Analysis'`, or `'Cross-Modal Speech-Visual Alignment'`.
* **'Self Selection'**: Evaluate on specific tasks by specifying names from `all_task_names`.

### Steps to Complete the Evaluation

1. **Data Preparation**: 
   Download JSONL files and videos. Set `task_jsonls_path` to the directory containing the 16 JSONL files and `videos_save_path` to the video root.
   ```text
   (videos_save_path)
   ├── Action_Temporal_Analysis
   │   └── (... .mp4)
   ├── Active_speaker_detection
   │   └── (... .mp4)
   └── ...
   ```
2. **Model Loading**:
Complete the model loading section in the if __name__ == "__main__": block to initialize your model, processor, and tokenizer.

3. **Inference Logic**:
Implement the logic in eval_on_one_task to generate the outputs text. If using an API, modify this section to return the API response directly.

## 2. Data-Juicer-HumanVbench-ops (Data Annotation Pipeline)
This section provides the data operators used for human-centric video filtering and annotation, powered by Data-Juicer.

### Relevant Files
- Example Recipe: `data-juicer/demos/video_humanvbench_simple/analyzer.yaml`
- Operator Definitions: `data-juicer/data_juicer/config/config_all.yaml`

### Quick Start
Since these operators require specific modifications to external models, the adjusted codebases are stored in `data-juicer/thirdparty/humanvbench_models`.

#### Option 1: Automatic Mode (Recommended)
The operators include logic to automatically handle git clone and merge diff for dependencies. You can run the pipeline directly:

```shell
cd data-juicer
dj-process --config demos/video_humanvbench_simple/analyzer.yaml
```

#### Option 2: Manual Mode
If you prefer manual setup, follow the instructions in `thirdparty/humanvbench_models/README.md` to clone repositories and apply `.diff` patches, then run:

```shell
cd data-juicer
dj-process --config demos/video_humanvbench_simple/analyzer.yaml
```

**Note:** Most tasks in HumanVBench can be constructed using the results from this annotation pipeline. Detailed construction processes are in the paper's appendix. 