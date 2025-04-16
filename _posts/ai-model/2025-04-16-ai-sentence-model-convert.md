---
title: AI - sentence transformer 模型转换
date: 2025-04-16 12:00:00 +0800
categories: [AI]
tags: [AI, AI实践]
---


一般情况下，下载的 Embedding 模型有三层，分别是：Transformer 层、池化层、归一化层。

但是在ModelScope上下载模型 `sungw111/text2vec-base-chinese-sentence` 只有两层，分别是：Transformer 层、池化层。
缺少了归一化层。因此，需要手动添加归一化层，进行模型转换。


## 1. 下载模型

```python
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('sungw111/text2vec-base-chinese-sentence', cache_dir="./models/")
```

## 2. 模型结构

**查看模型结构**

- 可以通过 `model_dir` 目录结构查看。

    从下载的模型目录结构上，可以看出，缺少了归一化层。

  ```plain
  models
  ├─._____temp
  │  └─sungw111
  │      └─text2vec-base-chinese-sentence
  │          └─1_Pooling
  └─sungw111
      └─text2vec-base-chinese-sentence
          └─1_Pooling

  ```

- 也可以通过 `model_dir` 目录下的 `modules.json` 文件查看模型结构。

    modules.json 文件内容：
    ```json
    [
      {
        "idx": 0,
        "name": "0",
        "path": "",
        "type": "sentence_transformers.models.Transformer"
      },
      {
        "idx": 1,
        "name": "1",
        "path": "1_Pooling",
        "type": "sentence_transformers.models.Pooling"
      }
    ]
    ```

- 或者直接用代码打印输出。


  ```plain
  > SentenceTransformer(
    (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: ErnieModel
    (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  )
  ```

## 3. 转换模型

  **转换模型代码：**

```python
def convert_model(source_model_path, save_path):
    # 原始 Bert 模型
    bert = models.Transformer(source_model_path)
    pooling = models.Pooling(bert.get_word_embedding_dimension(),
                            pooling_mode='mean')
    # 添加缺失的归一化层
    normalize = models.Normalize()
    # 组合完整模型
    full_model = SentenceTransformer(modules=[bert, pooling, normalize])

    print(full_model)

    full_model.save(save_path)
    return

```

  **调用转换函数：**
``` python
import numpy as np
from sentence_transformers import SentenceTransformer,models

def cal_norm(model_path, text):
    model = SentenceTransformer(model_path)
    vec = model.encode(text)
    return np.linalg.norm(vec)


source_model_path = r"源模型路径"
text = "验证向量归一化"

norm_result = cal_norm(source_model_path, text)
print(norm_result) # 输出： 17.602589

target_model_path=r"转换后模型路径"
convert_model(source_model_path, target_model_path)

norm_result = cal_norm(target_model_path, text)
print(norm_result)  # 输出： 1.0

```

**查看转换后的模型结构：**

  modules.json 文件内容：
```json
[
  {
    "idx": 0,
    "name": "0",
    "path": "",
    "type": "sentence_transformers.models.Transformer"
  },
  {
    "idx": 1,
    "name": "1",
    "path": "1_Pooling",
    "type": "sentence_transformers.models.Pooling"
  },
  {
    "idx": 2,
    "name": "2",
    "path": "2_Normalize",
    "type": "sentence_transformers.models.Normalize"
  }
]
```
