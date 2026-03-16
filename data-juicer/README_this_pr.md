# Data-Juicer-HumanVbench-ops

This is the operator contribution page for the paper: **HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks (CVPR'26)**.

## Related Operator Documentation Locations

* **Example Recipe:** `demos/video_humanvbench_simple/analyzer.yaml`
* **Operator Definition:** `data_juicer/config/config_all.yaml`

## Quick Start

As HumanVBench operators involve modifications to external repositories, these adjusted repositories are currently stored in:
`thirdparty/humanvbench_models`

To use these operators, you can choose:

1. **Manual Mode:** Follow the instructions in `thirdparty/humanvbench_models/README.md` to manually complete the `git clone` and `.diff` patch merging, then run:

```shell
dj-process --config demos/video_humanvbench_simple/analyzer.yaml

```

2. **Automatic Mode (Recommended):** Start running directly:

```shell
dj-process --config demos/video_humanvbench_simple/analyzer.yaml

```
The relevant operators already cover the logic for automatic `git clone` and `merge diff`, making manual intervention non-essential.
