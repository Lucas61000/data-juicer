# Data-Juicer-HumanVbench-ops

这是论文：**HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks (CVPR'26)** 的算子贡献页。

## 相关算子介绍文件位置

* **范例 Recipe：** `demos/video_humanvbench_simple/analyzer.yaml`
* **算子定义：** `data_juicer/config/config_all.yaml`

## 快速开始

由于 HumanVBench 算子涉及外部仓库的修改，这些经过调整的仓库目前存储在：
`thirdparty/humanvbench_models`

为了使用这些算子，你可以选择：

1. **手动模式：** 按照 `thirdparty/humanvbench_models/README.md` 下的指引手动完成 `git clone` 和 `.diff` 补丁合并，然后运行：
```shell
cd data-juicer
dj-process --config demos/video_humanvbench_simple/analyzer.yaml

```


2. **自动模式（推荐）：** 直接开始运行：
```shell
cd data-juicer
dj-process --config demos/video_humanvbench_simple/analyzer.yaml

```
我们在相关算子已经涵盖了自动 `git clone` 和 `merge diff` 的逻辑，手动干预是非必须的。
