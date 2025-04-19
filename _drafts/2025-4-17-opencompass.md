---
title: OpenCompass
date: 2025-4-17 14:00:00 +0800
categories: [AI]
tags: [AI, AI实践]
---

# OpenCompass

## 1. Introduction


## 环境准备

新建 conda 环境
```bash
conda create --name opencompass python=3.10 -y
# conda create --name opencompass_lmdeploy python=3.10 -y
conda activate opencompass
```

安装 OpenCompass
```bash
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```

数据集准备

```bash
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

**OpenCompass 配置准备**

可以提前在配置文件中设置好模型路径。模型和数据集的配置文件预存于 configs/models 和 configs/datasets 中。用户可以使用
tools/list_configs.py 查看或过滤当前可用的模型和数据集配置。

```bash
# 列出所有配置
python tools/list_configs.py
# 列出与llama和mmlu相关的所有配置
python tools/list_configs.py llama mmlu
```

## 配置评估任务

### 命令行（自定义HF模型）

对于 HuggingFace 模型，用户可以通过命令行直接设置模型参数，无需额外的配置文件。例如，对于
`internlm/internlm2-chat-1_8b` 模型，可以使用以下命令进行评估：

```bash
python run.py \
--datasets demo_gsm8k_chat_gen demo_math_chat_gen \
--hf-path internlm/internlm2-chat-1_8b \
--debug
```

请注意，通过这种方式，OpenCompass 一次只评估一个模型，而其他方式可以一次评估多个模型。