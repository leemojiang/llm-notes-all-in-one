# SFT 目标和操作

SFT 是 Supervised Fine-Tuning，也就是监督微调。它通常接在 Pretrain 之后，用整理好的指令、问答或者多轮对话数据，让一个主要学会“续写文本”的基座模型，进一步学会按照用户输入给出回答。

有了 Pretrain 的基础，SFT 的训练形式会简单很多：从训练目标上看，SFT 并没有发明一个全新的 loss。它仍然是 next token prediction，也就是给定前面的 token，预测下一个 token。真正变化的是训练数据的组织方式，以及哪些 token 会参与 loss。

## SFT 要解决什么问题？

Pretrain 阶段的模型主要学习语言规律、事实知识和上下文统计关系。它看到的是大规模普通文本，训练目标是把每个位置的下一个 token 预测对。

但是一个完成 Pretrain 的模型并不天然知道“用户问一句，我应该以助手身份回答一句”。如果输入是：

```text
什么是 SFT？
```

预训练模型可能会继续补全文档、百科、论坛帖子，也可能输出另一个问题。它的目标只是“像训练文本一样续写”，而不是“作为助手完成用户请求”。

SFT 的作用，就是把训练分布改成更接近真实交互的形式：

```text
用户提出问题 -> 助手给出回答
```

经过这样的训练后，模型仍然在做 next token prediction，但它预测的对象变成了“在当前对话模板下，assistant 接下来应该说什么”。

> 对话式案例只是 SFT 的一种形式。由于训练目标没有变化，如果从似然损失函数的角度理解，SFT 更像是在把模型的输出分布推向某个特定的数据分布。这个过程通常体现在风格迁移和输出结构上：例如原模型很少见到对话数据，SFT 之后就会学习一问一答的结构；如果 SFT 数据包含数学、代码、推理等结构化任务，模型也会学习这些任务中“按步骤输出”的模式。这就是常说的行为对齐和分布迁移。沿着这个角度，更容易理解 SFT 在整个训练链路中的位置.

## SFT 的基本样本

参考 MiniMind，一个最小的 Chat SFT 样本可以抽象成两部分：

- prompt：用户问题、系统提示、历史对话、工具说明等上下文。
- response：希望模型学习生成的 assistant 回答。

如果把 prompt token 记为 \\(x_1, x_2, \cdots, x_m\\)，把 assistant 回答 token 记为 \\(y_1, y_2, \cdots, y_n\\)，那么完整输入序列可以写成：

\\[
s = [x_1, x_2, \cdots, x_m, y_1, y_2, \cdots, y_n]
\\]

其中，\\(s\\) 表示送入模型的一整段 token 序列，\\(x_i\\) 表示 prompt 中第 \\(i\\) 个 token，\\(y_t\\) 表示回答中第 \\(t\\) 个 token。

SFT 通常只希望模型学习回答部分，因此 loss 可以写成：

\\[
\mathcal{L}_{\text{SFT}} = - \sum_{t=1}^{n} \log p_\theta(y_t \mid x_1,\cdots,x_m,y_1,\cdots,y_{t-1})
\\]

这里 \\(p_\theta\\) 表示参数为 \\(\theta\\) 的模型给出的条件概率。这个公式表达的含义是：prompt 只是条件，真正被监督学习的是 assistant 的回答 token。具体实现中，通常会在 prompt 部分的 token 上设置 loss mask，让它们不参与损失计算。除此之外，SFT 的 loss 和 Pretrain 的语言模型 loss 是同一种东西。

## MiniMind 中的 SFT 入口

在 MiniMind 的 README 中，当前主线 SFT 数据是：

- `sft_t2t_mini.jsonl`：适合快速训练对话模型。
- `sft_t2t.jsonl`：适合完整复现主线版本。

README 还特别强调，当前版本的 SFT 数据统一使用多轮对话格式，并且 Tool Calling 能力已经混入主线 SFT 数据。这意味着 SFT 不只是普通问答微调，还承担了让模型学习对话模板、思考标签和工具调用格式的职责。

MiniMind 中和本章关系最密切的文件有：

- `minimind_upstream/minimind/README.md`：说明 SFT 数据来源、格式和训练入口。
- `src/minimind_learning/dataset/lm_dataset.py`：实现 `SFTDataset`，负责把对话数据转换成 `input_ids` 和 `labels`。
- `src/minimind_learning/trainer/train_full_sft.py`：实现全参数 SFT 训练脚本。
- `src/minimind_learning/trainer/train_pretrain.py`：用于和 SFT 脚本对比。

## 本章结构

这一组笔记会按下面的顺序展开：

1. [SFT 数据和 Chat Template](data-and-chat-template.md)：一条 SFT 样本如何从 JSONL 变成训练张量。
2. [Full SFT 和 Pretrain 脚本的区别](full-sft-vs-pretrain-script.md)：只关注和 Pretrain 不同的地方，跳过重复训练细节。
3. [SFT 和 Pretrain 的异同](sft-vs-pretrain.md)：从目标、数据、loss mask、能力变化几个角度对比。
4. [SFT Eval 应该怎么做](eval.md)：讨论 SFT loss eval、生成式评估和任务评估各自该看什么。
5. [SFT 常见问题](pitfalls.md)：整理 SFT 中容易混淆的点。
