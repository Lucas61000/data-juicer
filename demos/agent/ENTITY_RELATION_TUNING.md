# 实体 / 关系字段：常见现象与配置建议

**全量菜谱默认已关掉**第 7 步实体/关系/关键词（省 LLM、且对 bad-case 主链路增益有限）。若要启用，在 `agent_interaction_quality_analysis.yaml` 中取消注释并参考 **`minimal_configs/07_entity_keyword.yaml`**。

对应算子：`extract_entity_attribute_mapper`、`relation_identity_mapper`（写入 `__dj__meta__` 的 `main_entities`、`attributes`、`attribute_descriptions`、`role_relation` 等）。

## 1. `main_entities` 里名字重复出现

**这是算子设计**：对 `query_entities` × `query_attributes` **嵌套循环**，结果放进 **四条平行数组**（`main_entities`、`attributes`、`attribute_descriptions`、`attribute_support_texts`），第 `i` 条对应第 `i` 次 `(实体, 属性)` 请求。

- 例：`["用户","用户","助手","助手"]` + `["A","B","A","B"]` = 2 实体 × 2 属性。

若要「看起来不重复」，应 **减少实体或属性个数**（如只抽 `用户` + `助手` 各 1～2 个维度），而不是指望算子自动去重。

## 2. `attribute_descriptions` 空泛、像任务说明、或复述 prompt

常见原因：

1. **`require_support_demos: true`（历史默认）**  
   只有同时解析出 **描述 + ``` 代码块摘录** 才算成功；agent 场景下模型经常只写段落、不写代码块 → **整格为空或反复 retry**。  
   → 菜谱已改为 **`require_support_demos: false`**（仍鼓励模型写摘录，但不作为硬门槛）。

2. **输入是整段 `text`**  
   `text` 含多轮 + 工具 JSON，模型更易跑题、复述 system。可尝试：
   - 在菜谱里为算子设 **`text_key: "query"`**（仅用户最后问句，信息少但干净），或
   - 后续可加自定义 mapper 先拼 `query + "\n---\n" + response` 到新字段再指定 `text_key`（需配置扩展）。

3. **实体/属性过于抽象**  
   「目标 / 行为」在客服/agent 日志里边界模糊，易产出「无明确目标」类句。  
   → 改成业务可定义的短语（如「主要诉求」「助手回应要点」「是否调用工具」等）。

4. **温度偏高**  
   → 对这两个算子加 **`sampling_params: { temperature: 0.2 }`**（菜谱已示例）。

若仍差，可在 YAML 里覆盖 **`system_prompt_template` / `input_template`**（需保留输出里的 `#实体`、`##属性：` 结构，否则原正则解析会失败）。

## 3. `role_relation` 多为「未知关系」或接近空

- **「未知关系」** 多半是 **模型在结论里写出来的**，不是 Data-Juicer 硬编码。
- 算子依赖 **固定格式**（`分析推理：…` + `所以助手是用户的：…`）；`qwen-turbo` 等若换用「因此」「故」或中英文冒号，旧正则可能抽不到 → 会得到空字符串（代码里已加 **宽松兜底**，从含 `所以/因此/…：` 的行里再抽一截）。
- 优化方向：
  - 降低 **`temperature`**；
  - 在 **system / input** 里明确「最后一行必须严格：`所以{助手}是{用户}的：具体关系`」；
  - 确认 **`text`** 里真的同时出现与用户、助手相关的内容（否则模型只能答未知）。

## 4. 性能与成本（实体×属性）

每对 `(实体, 属性)` **至少 1 次 API**。全量 `3×2` 已比 `2×2` 多 50% 调用；生产上请按需要收缩列表。

## 5. 与 `07_entity_keyword.yaml` 的关系

最小示例仍可用「用户/助手 + 目标/行为」演示 API；**全量 agent 菜谱**更偏向「少实体、具体属性名 + require_support_demos=false + 低温」，以减少空泛与解析失败。
