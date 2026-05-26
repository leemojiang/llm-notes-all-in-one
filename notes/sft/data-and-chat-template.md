# SFT 数据和 Chat Template

SFT 数据处理最核心的问题是：一条人类可读的对话样本，如何变成模型可以训练的 `input_ids` 和 `labels`。

MiniMind 的这条链路可以概括为：

```text
jsonl 对话样本
-> conversations
-> pre_processing_chat
-> tokenizer.apply_chat_template
-> post_processing_chat
-> tokenizer(prompt).input_ids
-> generate_labels(input_ids)
-> input_ids, labels
```

下面用 README 中的 SFT 数据格式作为例子，逐步看每一步之后数据变成了什么样。

## 原始 JSONL 数据

MiniMind README 中给出的普通多轮 SFT 样本如下：

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

这个阶段的数据还是结构化 JSON。每条消息都有 `role` 和 `content`，但是模型不能直接读 JSON 对象，必须先把它展开成一段带角色标记的文本。

## 第一步：读取 conversations

在 `SFTDataset.__getitem__` 中，代码先从样本里取出 `conversations`：

```python
sample = self.samples[index]
conversations = pre_processing_chat(sample["conversations"])
```

此时数据可以理解成 Python list：

```python
[
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "再见"},
    {"role": "assistant", "content": "再见！"},
]
```

## 第二步：pre_processing_chat

`pre_processing_chat` 会在一部分非 tool use 样本前面随机插入 system prompt。MiniMind 这样做，是为了让模型在训练中见过“有 system 消息”和“没有 system 消息”两种情况。

对应代码来自 `src/minimind_learning/dataset/lm_dataset.py`：

```python
def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留，不额外插入 system prompt。
    if any(conv.get("tools") or conv.get("functions") for conv in conversations):
        return conversations

    system_prompts = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    if conversations and conversations[0].get("role") != "system" and random.random() < add_system_ratio:
        return [{"role": "system", "content": random.choice(system_prompts)}] + conversations
    return conversations
```

如果这次没有插入 system prompt，数据不变：

```python
[
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "再见"},
    {"role": "assistant", "content": "再见！"},
]
```

如果随机插入了 system prompt，则会变成：

```python
[
    {"role": "system", "content": "你是minimind，请尽力帮助用户解决问题。"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "再见"},
    {"role": "assistant", "content": "再见！"},
]
```

对于 tool use 数据，代码会直接返回原始 `conversations`，因为工具定义通常挂在 system 消息上，随意插入 system prompt 可能破坏工具调用格式。

## 第三步：apply_chat_template

接下来，`create_chat_prompt` 会把结构化消息交给 tokenizer 的 `apply_chat_template`：

```python
def create_chat_prompt(self, conversations: list):
    messages = []
    tools = None
    for message in conversations:
        message = dict(message)
        if message.get("role") == "system" and (message.get("tools") or message.get("functions")):
            raw_tools = message.get("tools") or message.get("functions")
            tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        messages.append(message)

    return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )
```

`chat_template` 的作用，是把不同 role 的消息转换成模型训练时统一使用的文本格式。MiniMind tokenizer 中的特殊 token 是：

```json
{
  "bos_token": "<|im_start|>",
  "eos_token": "<|im_end|>"
}
```

如果不插入 system prompt，上面的对话会被展开成近似下面的文本：

```text
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
<think>

</think>

你好！<|im_end|>
<|im_start|>user
再见<|im_end|>
<|im_start|>assistant
<think>

</think>

再见！<|im_end|>
```

注意，SFT 引入 chat template 之后，模型学习的并不是裸文本“你好！”，而是在固定对话协议中的 assistant 片段。模型会同时学到：

- 看到 `<|im_start|>user` 后，后面是用户输入。
- 看到 `<|im_start|>assistant` 后，后面是助手回答。
- assistant 回答通常以 `<|im_end|>` 结束。
- 当前模板还可能包含 `<think>`、`<tool_call>`、`<tool_response>` 等结构。

这也是为什么训练时的模板和推理时的模板必须一致。如果训练用一种格式，推理用另一种格式，模型看到的条件分布就变了。

## 第四步：post_processing_chat

MiniMind 的 chat template 会给 assistant 片段加上空 thinking 标签：

```text
<think>

</think>
```

`post_processing_chat` 会以一定概率删除这个空 thinking 标签：

```python
def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以一定概率移除空 thinking 标签。
    if "<think>\n\n</think>\n\n" in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content
```

如果删除空 thinking 标签，prompt 可能变成：

```text
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！<|im_end|>
<|im_start|>user
再见<|im_end|>
<|im_start|>assistant
再见！<|im_end|>
```

如果保留，则模型会见到带 `<think></think>` 的回答格式。这个设计让同一个模型可以接触“显式思考标签”和“直接回答”两类格式。

## 第五步：tokenize 成 input_ids

文本 prompt 会被 tokenizer 转成 token id：

```python
input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
```

假设简化显示，不写真实 token id，而用 token 文本表示，`input_ids` 大致对应：

```text
[
  "<|im_start|>", "user", "\n", "你好", "<|im_end|>", "\n",
  "<|im_start|>", "assistant", "\n", "你好！", "<|im_end|>", "\n",
  "<|im_start|>", "user", "\n", "再见", "<|im_end|>", "\n",
  "<|im_start|>", "assistant", "\n", "再见！", "<|im_end|>", "\n",
  "<pad>", "<pad>", ...
]
```

真实情况中，一个中文词或者特殊标记可能被切成一个或多个 token id。这里的关键不是具体 id，而是序列结构：所有角色、换行、回答内容、结束标记都会进入 `input_ids`。

## 第六步：generate_labels

最后一步是生成 `labels`。SFT 不希望模型对 user prompt、system prompt 和 padding 计算 loss，只希望它学习 assistant 回答。因此 MiniMind 会扫描 `input_ids`，找到所有 assistant 片段，然后只把这些位置的 label 设为真实 token id，其余位置设为 `-100`。

对应代码：

```python
def generate_labels(self, input_ids: list):
    """
    仅 assistant 回复部分参与 loss 计算；其他位置设置为 -100。
    labels 和 input_ids 同长度，shift 由模型内部完成。
    """
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i : i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels
```

初始化时，`SFTDataset` 定义了 assistant 段落的起止模式：

```python
self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
```

也就是说，代码会寻找：

```text
<|im_start|>assistant\n
```

然后一直标记到：

```text
<|im_end|>\n
```

简化后的 `labels` 可以理解为：

| 区域 | input_ids 中的内容 | labels |
|---|---|---|
| user 片段 | `<|im_start|>user\n你好<|im_end|>\n` | 全部 `-100` |
| assistant 角色头 | `<|im_start|>assistant\n` | `-100` |
| assistant 回答 | `你好！<|im_end|>\n` | 对应 token id |
| user 片段 | `<|im_start|>user\n再见<|im_end|>\n` | 全部 `-100` |
| assistant 角色头 | `<|im_start|>assistant\n` | `-100` |
| assistant 回答 | `再见！<|im_end|>\n` | 对应 token id |
| padding | `<pad>` | `-100` |

这里有一个容易忽略的细节：MiniMind 把 assistant 回答后面的 `<|im_end|>\n` 也纳入 label。这样模型不只学习回答内容，还学习什么时候结束回答。

## 最终返回的训练样本

`__getitem__` 最终返回两个张量：

```python
return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

其中：

- `input_ids` 是完整对话模板的 token id。
- `labels` 和 `input_ids` 等长。
- 非 assistant 回答位置是 `-100`。
- assistant 回答和结束标记位置是真实 token id。

完整的 `SFTDataset.__getitem__` 是：

```python
def __getitem__(self, index):
    sample = self.samples[index]
    conversations = pre_processing_chat(sample["conversations"])
    prompt = self.create_chat_prompt(conversations)
    prompt = post_processing_chat(prompt)
    input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
    input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
    labels = self.generate_labels(input_ids)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

这就是 SFT 数据处理的关键：输入仍然是一整段对话，但 loss 只看 assistant 应该学习的部分。

## Tool Calling 样本如何进入模板

README 中还给了 tool use 样本：

```json
{
  "conversations": [
    {"role": "system", "content": "# Tools ...", "tools": "[...]"},
    {"role": "user", "content": "把'你好世界'翻译成english"},
    {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"translate_text\",\"arguments\":{\"text\":\"你好世界\",\"target_language\":\"english\"}}]"},
    {"role": "tool", "content": "{\"translated_text\":\"Hello World\"}"},
    {"role": "assistant", "content": "Hello World"}
  ]
}
```

`create_chat_prompt` 会把 system 消息上的 `tools` 解析成 JSON，把 assistant 消息上的 `tool_calls` 也解析成 JSON，然后一起交给 `apply_chat_template`。

展开后，模板会生成类似这样的结构：

```text
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

<tools>
...
</tools>
...
<|im_end|>
<|im_start|>user
把'你好世界'翻译成english<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "translate_text", "arguments": {"text": "你好世界", "target_language": "english"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"translated_text":"Hello World"}
</tool_response><|im_end|>
<|im_start|>assistant
Hello World<|im_end|>
```

这说明 SFT 数据中的 tool call 并不是额外走一条训练管线，而是被 chat template 统一展开成文本协议。模型最终还是在做 next token prediction，只是 assistant 需要学习的目标里包含了 `<tool_call>` 这种结构化输出。
