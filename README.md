# MiniMind的学习笔记

## 项目结构
这个项目是 [MiniMind](https://github.com/jingyaogong/minimind/tree/master?tab=readme-ov-file)项目的学习笔记.对它的代码进行了整理,补充了注释以及相应的笔记.

使用src layout并使用UV进行环境管理,项目结构更加清晰.

    - minimind_upstream/minimind 原始minimind项目,作为参考.
    - src 代码所在位置
    - tests 单元测试
    - docs 文档
    - datasets 数据集文件
    - notebooks 放jupyter文件

## 项目使用
```bash
    # 同步项目
    uv sync --extra cpu
    # 或者如果你有GPU支持,可以使用以下命令安装GPU版本的依赖
    uv sync --extra gpu
    
    # git submodule 同步文件
    git submodule update --init --recursive

    # 以开发模式安装包
    uv pip install -e .

    # 登录swanlab
    swanlab login
```

## 目录
- [0. 项目配置 ](docs/0.GettingStarted.md)
- [2. Pretrain训练 快速开始](docs/Pretrain.md)
    

