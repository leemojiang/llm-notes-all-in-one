# Full SFT 和 Pretrain 脚本的区别

MiniMind 的 `train_full_sft.py` 和 `train_pretrain.py` 在训练主循环上几乎一致：都是加载模型、构造 dataset 和 dataloader、前向计算 loss、反向传播、梯度累积、保存权重。

所以这一节不重复 optimizer、checkpoint、batch size、混合精度这些已经在 Pretrain 章节讲过的内容，只看 SFT 脚本相对 Pretrain 真正改变了什么。

## 区别一：Dataset 从 PretrainDataset 变成 SFTDataset

Pretrain 脚本使用的是 `PretrainDataset`：

```python
from minimind_learning.dataset.lm_dataset import PretrainDataset
```

对应构造代码：

```python
train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
```

Full SFT 脚本使用的是 `SFTDataset`：

```python
from minimind_learning.dataset.lm_dataset import SFTDataset
```

对应构造代码：

```python
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
```

这个改动看起来只换了一个类，但它实际改变了训练样本的含义。

`PretrainDataset` 读的是：

```json
{"text": "一段普通文本..."}
```

`SFTDataset` 读的是：

```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
  ]
}
```

因此训练循环虽然仍然拿到 `(input_ids, labels)`，但 `labels` 的 mask 方式已经完全不同。

## 区别二：默认数据路径不同

Pretrain 默认读取：

```python
parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
```

Full SFT 默认读取：

```python
parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
```

这对应 MiniMind README 中的阶段式训练组合：

```text
pretrain_t2t_mini.jsonl -> sft_t2t_mini.jsonl
```

前者让模型学习语言建模，后者让模型学习指令和对话格式。

## 区别三：默认加载权重不同

Pretrain 脚本默认从头训练：

```python
parser.add_argument("--from_weight", default="none", type=str, help="基于哪个权重训练，none表示从头开始")
```

Full SFT 脚本默认加载 pretrain 权重：

```python
parser.add_argument("--from_weight", default="pretrain", type=str, help="基于哪个权重训练")
```

这说明 SFT 在 MiniMind 的训练链路里不是从零开始，而是在 Pretrain 已经学到的语言能力和基础知识上继续训练。

如果把模型参数记为 \\(\theta\\)，Pretrain 得到的参数记为 \\(\theta_{\text{pretrain}}\\)，那么 SFT 的初始化可以写成：

\\[
\theta_0 = \theta_{\text{pretrain}}
\\]

然后 SFT 在此基础上继续优化：

\\[
\theta_{\text{sft}} = \operatorname{Train}_{\text{SFT}}(\theta_0)
\\]

其中 \\(\theta_0\\) 是 SFT 开始时的参数，\\(\theta_{\text{sft}}\\) 是 SFT 结束后的参数。

## 区别四：学习率更小

Pretrain 默认学习率是：

```python
parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
```

Full SFT 默认学习率是：

```python
parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
```

这是一个很常见的设置：SFT 是在已有模型上做后续调整，不希望用过大的学习率破坏 Pretrain 阶段学到的语言能力。尤其是小模型和小数据场景下，学习率过大很容易让模型变得更会套模板，但通用生成能力下降。

## 区别五：训练序列长度不同

Pretrain 默认最大长度是：

```python
parser.add_argument("--max_seq_len", default=340, type=int, help="训练的最大截断长度")
```

Full SFT 默认最大长度是：

```python
parser.add_argument("--max_seq_len", default=768, type=int, help="训练的最大截断长度")
```

SFT 样本通常包含 role 标记、system prompt、多轮历史、assistant 回答，甚至还可能包含 tool call 和 tool response，所以同样一条样本会比普通文本带有更多结构信息。更长的 `max_seq_len` 可以减少重要对话上下文被截断的概率。

## 区别六：训练循环基本不变

Full SFT 的核心训练循环仍然是：

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)

    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    with autocast_ctx:
        res = model(input_ids, labels=labels)
        loss = res.loss + res.aux_loss
        loss = loss / args.accumulation_steps

    scaler.scale(loss).backward()
```

这段代码和 Pretrain 的本质逻辑是一样的。模型并不知道自己正在做 “Pretrain” 还是 “SFT”，它只接收 `input_ids` 和 `labels`，然后根据 label 位置计算交叉熵。

真正让训练目标发生变化的地方，是 dataset 生成的 `labels`：

- Pretrain：除了 padding，文本中的大多数 token 都参与 loss。
- SFT：只有 assistant 回答片段参与 loss。

因此，SFT 的关键不在训练 loop，而在数据构造和 label mask。

## 小结

`train_full_sft.py` 可以理解成复用了 Pretrain 的训练框架，只替换了训练阶段最关键的几个入口：

| 对比项 | Pretrain | Full SFT |
|---|---|---|
| Dataset | `PretrainDataset` | `SFTDataset` |
| 默认数据 | `pretrain_t2t_mini.jsonl` | `sft_t2t_mini.jsonl` |
| 默认初始权重 | `none` | `pretrain` |
| 默认学习率 | `5e-4` | `1e-5` |
| 默认最大长度 | `340` | `768` |
| loss 位置 | 普通文本 token | assistant 回答 token |

所以这部分代码阅读的重点不是“训练循环又写了一遍”，而是理解：同一套 next-token 训练框架，只要换掉数据和 labels，就能从预训练切换到监督微调。
