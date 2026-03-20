# KenLM / `perplexity_filter` 在 macOS 上的问题

`pip install kenlm` / `uv pip install kenlm` 会从源码编译，常见报错：

- `fatal error: 'cstddef' file not found` / `'climits' file not found`  
  → 本机 C++ 标准库头文件对当前编译器不可见（未装 Command Line Tools、或只用裸 clang 无 SDK）。
- `Could NOT find Eigen3`  
  → 缺 Eigen（KenLM 部分路径需要）。

## 推荐做法

### A. 先不跑 perplexity（默认菜谱）

全量 **`agent_interaction_quality_analysis.yaml`** 默认 **注释掉 `perplexity_filter`**，并把 **`signal_on_high_perplexity: false`**。流水线不依赖 KenLM，其它 bad-case 信号仍可用。

需要 perplexity 时再按下面 B/C 装好 KenLM，在 YAML 里打开 filter + `signal_on_high_perplexity: true`。

### B. 用 Conda 装二进制包（省事）

```bash
conda install -c conda-forge kenlm
# 再在装好的环境里 pip install 本项目
```

### C. 在 Homebrew 环境下调编译（需本机工具链）

1. 安装 Xcode Command Line Tools：  
   `xcode-select --install`
2. 安装依赖：  
   `brew install cmake eigen`
3. 再试：  
   `uv pip install kenlm`  
   若仍失败，确认 `clang++ -v` 能找 到 macOS SDK。

### D. 仍失败时

- 在项目 issue / 团队内网查是否提供 **预编译 wheel** 或 **Linux CI** 跑带 perplexity 的流水线。
- 或使用 **`llm_perplexity_filter`**（HF 模型算 ppl，更重、非 KenLM）——需另改菜谱。
