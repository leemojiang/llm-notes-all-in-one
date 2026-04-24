# MiniMind 学习笔记

这个项目是对 [MiniMind](https://github.com/jingyaogong/minimind/tree/master?tab=readme-ov-file) 的学习整理。仓库中既保留了上游参考代码，也加入了我自己的代码阅读、实验记录和笔记。

## 项目结构

- `minimind_upstream`：上游 MiniMind 代码，作为对照参考
- `src`：当前仓库的核心代码
- `tests`：测试代码
- `notes`：学习笔记，按 mdBook 方式组织
- `datasets`：数据集文件
- `notebooks`：实验用 Jupyter Notebook

## 项目使用

```bash
# 同步项目依赖
uv sync --extra cpu

# 如果你使用 CUDA 环境，也可以切到 cuda 依赖
uv sync --extra cuda

# 同步 submodule
git submodule update --init --recursive

# 以开发模式安装当前项目
uv pip install -e .

# 登录 swanlab
swanlab login
```

## 笔记入口

- [notes/SUMMARY.md](notes/SUMMARY.md)
- [导读](notes/2.LLM_Pretrain.md)
- [环境准备与项目结构](notes/0.GettingStarted.md)
- [预训练快速开始](notes/1.Pretrain.md)

如果本地安装了 `mdbook`，可以直接在仓库根目录运行：

```bash
mdbook serve
```
    

