# SFT 常见问题

这一节整理 SFT 中最容易混淆的点。很多 SFT 问题表面上是训练问题，实际根源都在数据格式、chat template 或 label mask。

## SFT 后模型为什么更像助手？

因为训练数据从普通文本变成了对话示范。

SFT 并不是给模型增加了一个“服从用户”的特殊模块，而是让模型反复看到这样的模式：

```text
<|im_start|>user
问题<|im_end|>
<|im_start|>assistant
回答<|im_end|>
```

当这种格式大量出现后，模型在推理时看到 user 片段，就更倾向于续写 assistant 片段。

## 为什么 prompt 部分通常不算 loss？

因为 prompt 是条件，不是希望模型学习生成的目标。

以一条样本为例：

```text
user: 什么是 SFT？
assistant: SFT 是监督微调...
```

训练时我们希望模型学会“在这个问题之后回答什么”，而不是学会“如何生成用户的问题”。所以 user 和 system 片段进入 `input_ids`，但在 `labels` 中通常被设置为 `-100`。

如果 prompt 也参与 loss，模型会把一部分训练能力花在复述模板、用户问题和系统提示上。这不是 SFT 的主要目标。

## 为什么 assistant 的结束标记也要算 loss？

MiniMind 的 `generate_labels` 会把 assistant 内容一直标记到 `<|im_end|>\n`：

```python
for j in range(start, min(end + len(self.eos_id), self.max_length)):
    labels[j] = input_ids[j]
```

这意味着模型不只学习“回答什么”，还学习“什么时候结束回答”。

如果结束标记没有被充分学习，模型在推理时可能更容易出现回答停不下来、继续编造下一轮对话、把 user/assistant 标记也生成出来等问题。

## 为什么训练模板和推理模板必须一致？

因为 SFT 学到的是模板条件下的输出分布。

如果训练时模型看到的是：

```text
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！<|im_end|>
```

推理时却输入：

```text
User: 你好
Assistant:
```

那么模型看到的上下文格式就变了。即使语义接近，token 分布也不同，小模型尤其容易受影响。

所以对 instruction model 或 chat model 来说，`apply_chat_template` 不是可有可无的格式化工具，而是训练和推理之间的协议。

## SFT loss 下降是否等于模型更好？

不一定。

SFT loss 下降说明模型更擅长拟合当前 SFT 数据分布，但这不完全等价于真实使用体验更好。可能出现几种情况：

- 训练数据回答很短，模型也变得偏短。
- 训练数据风格单一，模型回答变得套路化。
- 训练数据有噪声，模型学到错误格式。
- 训练数据领域很窄，模型通用能力下降。
- eval loss 下降，但开放问答的事实性没有提升。

所以 SFT 评估不能只看 loss，还应该看人工问答、格式稳定性、指令遵循、拒答边界、工具调用格式等。

## 为什么 SFT 可能让模型“变笨”？

常见原因是 SFT 数据太窄、太少、质量不稳定，或者训练过度。

Pretrain 阶段给模型建立了比较宽的语言分布；SFT 阶段如果只用很窄的任务数据继续训练，模型会向这个窄分布靠拢。结果可能是某些格式变好了，但开放问题、事实性或者表达多样性下降。

这也是为什么全参数 SFT 通常要更谨慎。MiniMind 的 Full SFT 默认学习率比 Pretrain 小很多，就是为了减少对基座能力的破坏。

## 多轮对话中是否只训练最后一轮 assistant？

不一定，取决于 dataset 如何生成 labels。

MiniMind 的 `generate_labels` 会扫描整段 `input_ids`，只要找到 assistant 片段，就把这个 assistant 片段标成可训练目标。因此在一条多轮对话中，多个 assistant 回答都可能参与 loss。

以前面的样本为例：

```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "再见"},
    {"role": "assistant", "content": "再见！"}
  ]
}
```

两个 assistant 回答都会参与训练：

```text
你好！<|im_end|>
再见！<|im_end|>
```

这样做可以充分利用多轮对话里的每一轮 assistant 回复。

## Tool Calling 为什么也能放进 SFT？

因为 tool call 最终也是一种文本输出格式。

在 MiniMind 中，tool use 样本会通过 chat template 展开成：

```text
<tool_call>
{"name": "translate_text", "arguments": {"text": "你好世界", "target_language": "english"}}
</tool_call>
```

对模型来说，它仍然是在预测 assistant 片段中的下一个 token。区别只是这段回答不再是普通自然语言，而是一个结构化调用协议。

因此，SFT 可以教模型“什么时候输出自然语言回答”，也可以教模型“什么时候输出 tool call 格式”。

## SFT 和 instruction tuning 是什么关系？

instruction tuning 通常可以看成 SFT 的一种重要形式。

更宽泛地说，SFT 指用有监督样本继续训练模型；instruction tuning 强调这些样本是“指令 -> 回答”的形式，目标是提升指令遵循能力。

在现代 LLM 语境里，很多人说 SFT 时，实际说的就是 instruction tuning 或 chat fine-tuning。

## SFT 和 DPO/RLHF 是什么关系？

SFT 通常是后训练的第一步。它给模型一个基本可用的助手行为，让模型学会按模板回答。

DPO、RLHF、RLAIF 等方法通常接在 SFT 后面，用偏好数据、奖励模型或者规则反馈继续优化模型行为。它们关心的不只是“标准答案是什么”，还包括“多个回答中哪个更好”。

可以简单理解为：

```text
Pretrain: 学会语言和知识
SFT: 学会按指令回答
DPO/RLHF/RLAIF: 学会更偏好哪些回答方式
```

如果没有 SFT，模型可能连稳定对话格式都没有；这时直接做偏好优化或者强化学习，训练会更不稳定。
