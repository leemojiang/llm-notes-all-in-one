# SFT Eval 应该怎么做

和 Pretrain 一样，SFT 阶段也需要 eval。但 SFT 的 eval 更容易让人误解：loss 能告诉我们训练是否稳定，却不能充分说明模型是否获得了我们想要的“风格、格式和任务行为”。

贯穿这一章的应该是两个核心问题：

- SFT 阶段的训练 loss 应该怎么看？它如何观察训练稳定性以及是否该停止训练？它与 Pretrain 阶段的区别是什么？
- SFT 强调风格化和行为对齐，单纯的 loss 不一定能说明模型是否获得了目标能力。那么应该如何设计评价实验，评估模型在 SFT 阶段是否学到了我们想要的能力？(评价实验目标以及流程设计)

## Loss 用来看训练状态：SFT loss eval 和 Pretrain loss eval 一样吗？

SFT loss eval 和 Pretrain loss eval 的代码形式基本一样：

```python
with torch.no_grad():
    res = model(input_ids, labels=labels)
    loss = res.loss
```

在 SFT 中，loss 的特殊之处来自 label mask。Pretrain 通常是除了 padding 之外，大多数 token 都参与 loss；SFT 通常只让 assistant 回答参与 loss，因此它的 loss mask 范围要大得多。

所以 SFT eval loss 衡量的是：在当前 chat template 和 prompt 条件下，模型对标准 assistant 回答的拟合程度。

我们希望 loss 主要回答三个训练问题：

- 训练是否正常收敛。
- 是否出现发散、异常震荡或者数值问题。
- 是否开始过拟合当前 SFT 数据。

## SFT 阶段的训练稳定性

SFT 阶段是不是不是很容易出现训练不稳定? 
它的LR调度是否需要和Pretrain不同,比如我是否还需要从头warmup吗?
SFT 阶段的数据量与 Pretrain一般如何搭配?

SFT 阶段一般比 Pretrain 更容易训练不稳定，而且学习率调度通常必须和 Pretrain 不同。尤其是 warmup 仍然需要，但规模要小得多。

### 学习率要比 Pretrain 小

SFT 是“微调行为”，不是“重新学习知识”，所以学习率通常要比 Pretrain 小 10 到 100 倍。

典型范围可以这样理解：

```text
Pretrain: 1e-4 ~ 3e-4
SFT:      5e-6 ~ 5e-5
```

对于更大的模型，例如 70B 量级，SFT 学习率甚至可能使用：

```text
1e-6 ~ 2e-6
```

学习率过大时，SFT 很容易把模型已经学到的通用能力冲坏。表现上可能不是 loss 立刻爆炸，而是模型变得模板化、啰嗦，或者开放问答能力下降。

### Warmup 仍然需要，但要很短

SFT 仍然需要 warmup，原因是 SFT 数据分布和 Pretrain 数据分布差异很大。如果一开始就使用完整学习率，容易出现 loss spike，甚至导致梯度异常。

但 SFT 的 warmup 不需要像 Pretrain 那么长。Pretrain 可能使用 2k 到 10k steps 的 warmup，而 SFT 通常可以短很多：

```text
小模型 SFT: 50 ~ 200 steps
大模型 SFT: 100 ~ 500 steps
```

这里 warmup 的目的不是让长训练逐渐进入稳定区，而是避免训练刚开始时因为分布切换太突然导致梯度爆炸。

### Decay 策略可以更简单

Cosine decay 仍然是常见选择，但 SFT 的总训练 steps 往往很少，可能只有几千到几万步，所以 cosine decay 会下降得很快。

因此 SFT 也可以使用：

- short warmup + cosine decay。
- short warmup + linear decay。
- short warmup + constant learning rate。

如果 SFT 数据少于 100k 条，`constant LR + short warmup` 往往是比较稳的选择。

## SFT 数据量和 Pretrain 通常如何搭配？

SFT 数据量通常只占 Pretrain 的 0.001% 到 0.1%，但对模型行为的影响却可能是压倒性的。也就是说，SFT 数据量极小，但权重很大。

原因是 SFT 和 Pretrain 学的东西不同。

Pretrain 学的是：

- 世界知识。
- 语言规律。
- 语义结构。
- 推理能力。

SFT 学的是：

- 希望模型“怎么表现”。
- 风格、格式、礼貌和结构。
- 遵循指令。
- 输出格式，例如 JSON、Markdown、CoT 或 Tool Calling。

行为比知识更容易被改变，所以少量 SFT 数据就能强烈改变模型输出分布。

### SFT loss 是尖锐监督

这里需要先澄清一点：Pretrain 和 SFT 都是 token-level 的监督，本质上都是 next-token prediction，也都可以使用 cross entropy loss。二者的差异不在于“是不是 token-level”，而在于目标分布是否唯一、是否尖锐、是否容错。

Pretrain 的监督相对宽松。自然语言的下一个 token 往往有很多合理选择。例如：

```text
他走进了房间，看见了一只 ___
```

这里可以接“猫”“狗”“椅子”“人”“灯”等很多词。模型预测其中某个合理 token，即使没有完全命中数据中的原始 token，也通常仍然是在向合理语言分布靠近。因此 Pretrain 的目标分布更像是多峰的，梯度方向相对分散，loss landscape 也更平滑。

SFT 的监督更尖锐。SFT 数据里通常只有一个标注答案，训练时每个参与 loss 的 assistant token 都必须匹配这个标注答案。例如：

```text
Q: 请解释 Transformer 的注意力机制
A: Transformer 使用自注意力机制来捕捉序列中不同位置之间的依赖关系……
```

如果模型生成的是：

```text
Transformer 使用注意力机制来捕捉序列中不同位置之间的关系……
```

这句话在语义上可能完全可以接受，但 token-level CE 仍然会认为“自注意力机制”和“注意力机制”、“依赖关系”和“关系”等位置没有匹配标注。也就是说，SFT 在数学形式上仍然是语言模型 loss，但目标序列被当作唯一答案来监督，容错率很低。

所以，SFT 的目标分布更像是单峰的：梯度方向更集中、更明确，模型会迅速向标注数据的表达方式靠拢。这就是为什么少量高质量 SFT 数据就能显著改变模型行为，而 Pretrain 往往需要海量数据才能改变模型的基础分布。

可以把两者的差异概括为：

| 维度 | Pretrain | SFT |
|---|---|---|
| 数学形式 | token-level CE | token-level CE |
| 目标分布 | 多峰，多种合理续写 | 单峰，标注答案唯一 |
| 容错性 | 相对高 | 低 |
| 梯度方向 | 平滑、分散 | 强烈、集中 |
| 风格影响 | 较弱 | 很强 |
| 行为改变 | 需要大量数据 | 少量数据即可 |

一句话总结：Pretrain 是宽松的 token-level 监督，SFT 是严格的 token-level 监督。两者的数学形式相似，但目标分布不同，训练动力学也不同。

### SFT 数据分布高度集中

SFT 数据通常具有这些特点：

- 高质量。
- 风格一致。
- 结构统一。
- 任务明确。

这会让模型很快偏向这种分布。对于行为对齐来说，这是优点；但如果数据过窄，也会变成风险。

### SFT 数据太多会怎样？

SFT 数据太多，或者训练轮数太多，可能导致：

1. 模型变啰嗦、变模板化。

因为 SFT 数据风格太强，模型会越来越像训练样本里的固定表达。

2. 推理能力下降。

尤其是数学、代码和逻辑能力，可能因为过度拟合 SFT 分布而出现 catastrophic forgetting。

3. 模型变得过度对齐。

例如太礼貌、太安全、不敢回答、不敢推理，或者在不需要拒答的地方也拒答。

### 几个搭配原则

原则 1：SFT tokens 通常约为 Pretrain tokens 的 0.001% 到 0.05%。

这是一个常见经验范围。SFT 不靠 token 数量取胜，而靠样本质量、覆盖范围和格式一致性取胜。

原则 2：模型越大，SFT 占比通常越小。

大模型更容易被少量 SFT 数据改变行为，因此不一定需要大量 SFT 数据。

原则 3：能力提升更多依赖 mid-train，行为对齐更多依赖 SFT。

不要指望用 SFT 大幅提升基础能力。用 SFT 去“灌知识”或者“硬提能力”，很容易把模型训傻。

原则 4：SFT 数据必须高质量、风格一致。

数量不是第一位的，质量决定一切。

## Loss 如何判断是否该停止训练？

这部分和 Pretrain 类似，但 SFT 更要警惕“过拟合风格”。通俗地说，模型大体不会突然坏掉，但要小心把它训到某个狭窄的风格沟里。

如果 train loss 和 eval loss 都在稳定下降，说明训练还在有效拟合数据，可以继续观察。

如果 train loss 继续下降，但 eval loss 开始上升，通常说明模型开始记住训练集表达，而不是泛化到验证集，可以考虑停止训练、降低 epoch、降低学习率，或者增加更有代表性的验证集。

如果 loss 很低，但生成回答变差，例如回答变短、模板味变重、重复更多、开放问答变差，说明 loss 已经不能代表目标能力了。这时不应该只根据 loss 继续训练，而应该转向生成式 eval 和 task-level eval。

如果 eval loss 很高，但人工观察回答不错，也不一定说明模型差。开放式问答有很多合理表达，标准答案只是一种写法；模型用不同措辞答对了，loss 仍然可能不低。

一句话：SFT loss 适合判断训练是否稳定，不适合单独决定模型是否“好用”。

## 什么是生成式 Eval 和 Task-level Eval？

SFT 的目标通常不是单纯降低 loss，而是让模型形成某种可用行为。所以除了 loss eval，还需要生成式 eval 和 task-level eval。

### 生成式 Eval

生成式 eval 是把固定 prompt 输入模型，让模型用真实推理方式生成回答，然后观察输出是否符合预期。

它回答的问题是：

- 模型是否按 chat template 正常回答？
- 是否会生成多余的 `user`、`assistant`、`<|im_start|>` 等角色标记？
- 是否能正常结束，而不是重复或停不下来？
- 回答是否过短、过长、太模板化？
- 风格是否符合 SFT 目标？

生成式 eval 更接近真实使用，但它不一定有明确的自动分数。它适合做人类抽样检查，也适合配合一些简单规则检查。

例如，对于普通对话 SFT，可以固定一组 prompts：

```text
解释一下什么是 SFT。
给我一个 Python 快速排序示例。
用三句话总结注意力机制。
```

然后观察不同 checkpoint 的回答是否越来越符合助手风格，是否更稳定，是否更少出现角色污染。

### Task-level Eval

Task-level eval 是围绕具体任务设计评价指标。它回答的问题是：模型是否真的完成了这次 SFT 想要它学会的任务。

不同任务应该使用不同评价方式：

- 选择题：看准确率，或者比较候选项条件概率。
- 数学题：抽取最终答案，做 exact match 或规则校验。
- 代码题：运行单元测试。
- 摘要题：检查长度、覆盖率、事实错误，必要时人工评估。
- Tool Calling：检查工具名、参数 JSON、schema 合法性和最终执行结果。
- JSON 输出：检查 JSON 是否可解析，字段是否齐全，类型是否正确。

这类 eval 的核心是：先明确“能力目标”，再设计能检验这个目标的评价方法。不要用一个平均 loss 去解释所有能力。

## 一个实用的 Eval 流程

比较稳妥的 SFT eval 流程可以这样设计：

1. 先准备 held-out SFT 验证集，用和训练一致但固定的 chat template 计算 eval loss。

这一步只用于判断训练状态：是否收敛、是否发散、是否过拟合。

2. 固定一组生成式 eval prompts，每个 checkpoint 使用相同解码参数生成回答。

这一步用于检查模型真实输出，包括格式、风格、重复、结束标记和角色污染。

3. 根据 SFT 目标设计 task-level eval。

如果训练目标是 Tool Calling，就检查工具调用；如果训练目标是数学，就检查答案；如果训练目标是代码，就运行测试；如果训练目标是 JSON 格式输出，就检查 JSON schema。

4. 把 loss、生成样例和任务指标放在一起判断。

如果 loss 下降，但生成质量和任务指标变差，说明模型可能过拟合了 SFT 风格；如果 loss 不低，但任务指标更好，说明标准答案的 token 拟合不是当前任务的核心指标。

5. 用固定流程比较 checkpoint。

如果要比较两个 SFT checkpoint，必须固定 prompts、chat template、解码参数和评价脚本，否则对比不干净。

如果只是快速调参，`eval loss + 少量生成样例` 就够了。如果要决定最终模型，必须加入 task-level eval。

## 小结

SFT eval 可以分工理解：

- loss eval：看训练状态，判断收敛、发散和过拟合。
- 生成式 eval：看模型真实输出的格式、风格和稳定性。
- task-level eval：看模型是否获得目标能力。

所以 SFT 阶段不要只问“loss 降了吗”，还要问“模型是否按我希望的方式完成了任务”。这也是 SFT 和 Pretrain 在 eval 思路上最重要的差别。
