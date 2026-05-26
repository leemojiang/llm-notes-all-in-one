# SFT 和 Pretrain 的异同

SFT 和 Pretrain 很容易被理解成两个完全不同的训练任务，但从语言模型的角度看，它们的底层形式非常接近：都是给模型一串 token，然后让模型预测下一个 token。

它们的区别主要来自训练数据、loss mask 和训练目的。

## 相同点：都是 next token prediction

无论 Pretrain 还是 SFT，模型前向时都接收 `input_ids` 和 `labels`：

```python
res = model(input_ids, labels=labels)
loss = res.loss + res.aux_loss
```

如果当前序列是 \\(z_1, z_2, \cdots, z_T\\)，标准语言模型训练目标是让模型根据前文预测后文：

\\[
p_\theta(z_t \mid z_1, z_2, \cdots, z_{t-1})
\\]

其中 \\(z_t\\) 表示第 \\(t\\) 个 token，\\(\theta\\) 表示模型参数。

如果所有非 padding token 都参与 loss，那么目标可以写成：

\\[
\mathcal{L} = - \sum_{t=1}^{T} m_t \log p_\theta(z_t \mid z_{<t})
\\]

这里 \\(m_t\\) 是第 \\(t\\) 个位置的 loss mask。若 \\(m_t=1\\)，这个位置参与 loss；若 \\(m_t=0\\)，这个位置不参与 loss。

Pretrain 和 SFT 的共同点就是：它们都可以被放进这个公式里。

## 不同点一：数据分布不同

Pretrain 数据通常是普通文本：

```json
{"text": "大语言模型是一种基于深度学习的生成模型..."}
```

它的目标是让模型吸收尽可能多的语言模式、事实知识和通用表达能力。

SFT 数据通常是对话或者指令：

```json
{
  "conversations": [
    {"role": "user", "content": "什么是 SFT？"},
    {"role": "assistant", "content": "SFT 是监督微调，用高质量问答数据训练模型按指令回答。"}
  ]
}
```

它的目标不是重新学习世界知识，而是把已有能力组织成更符合用户期望的交互形式。

## 不同点二：模板不同

Pretrain 文本一般只需要加 BOS、EOS 和 padding：

```text
<bos>大语言模型是一种基于深度学习的生成模型...<eos><pad><pad>
```

SFT 数据需要 chat template：

```text
<|im_start|>user
什么是 SFT？<|im_end|>
<|im_start|>assistant
SFT 是监督微调，用高质量问答数据训练模型按指令回答。<|im_end|>
```

这一步非常关键。SFT 训练的不是孤立答案，而是“在某种对话协议下，assistant 应该如何输出”。因此推理时也必须使用相同或兼容的 chat template。

## 不同点三：loss mask 不同

Pretrain 的 `labels` 更简单。MiniMind 的 `PretrainDataset` 中，代码是：

```python
labels = input_ids.clone()
labels[input_ids == self.tokenizer.pad_token_id] = -100
```

也就是说，除了 padding 之外，普通文本 token 都参与训练。

SFT 的 `labels` 更挑剔。它会先把所有位置设成 `-100`，再只打开 assistant 回答的位置：

```python
labels = [-100] * len(input_ids)
```

然后在找到 assistant 片段后：

```python
for j in range(start, min(end + len(self.eos_id), self.max_length)):
    labels[j] = input_ids[j]
```

因此两者的 mask 可以对比如下：

| 位置 | Pretrain | SFT |
|---|---|---|
| 普通文本 | 参与 loss | 不一定存在 |
| system prompt | 不适用 | 不参与 loss |
| user prompt | 不适用 | 不参与 loss |
| assistant 回答 | 不适用 | 参与 loss |
| assistant 结束标记 | 不适用 | 参与 loss |
| padding | 不参与 loss | 不参与 loss |

SFT 中 user prompt 不参与 loss，并不表示它不重要。它仍然进入 `input_ids`，作为 assistant 生成时的条件上下文。

## 不同点四：能力变化不同

Pretrain 更像是在学习“语言和世界”：

- 词语和句法关系。
- 事实知识。
- 上下文关联。
- 长文本续写。
- 基础推理模式。

SFT 更像是在学习“如何作为助手表达这些能力”：

- 识别用户问题。
- 按指令回答。
- 使用稳定的对话格式。
- 在合适位置停止。
- 输出工具调用或思考标签等结构。

因此，SFT 通常会显著改善对话体验，但它不一定凭空增加模型知识。模型答得更像助手，不等于它知道了更多事实。

## 一个直观类比

Pretrain 像是让模型读大量书、网页、代码和对话，从中形成语言和知识基础。

SFT 像是给模型看一批标准示范：

```text
当用户这样问时，助手应该这样答。
当需要工具时，助手应该这样写 tool call。
当回答结束时，应该输出结束标记。
```

所以 SFT 更接近“行为格式和任务接口的训练”。它把 Pretrain 学到的能力接到用户可用的交互方式上。

## 小结

Pretrain 和 SFT 的关系可以概括为：

| 维度 | Pretrain | SFT |
|---|---|---|
| 训练目标 | next token prediction | next token prediction |
| 数据来源 | 大规模普通文本 | 指令、问答、多轮对话、tool call |
| 数据模板 | 普通文本模板 | chat template |
| loss 位置 | 大多数非 padding token | assistant 回答 token |
| 主要作用 | 学语言、知识和基础能力 | 学指令跟随、对话格式和输出协议 |
| 初始化 | 可以从头训练 | 通常从 pretrained 权重开始 |

一句话总结：SFT 没有改变语言模型的基本训练范式，它改变的是模型看到的数据分布，以及我们希望它在哪些 token 上接受监督。
