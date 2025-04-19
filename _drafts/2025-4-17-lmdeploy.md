---
title: "LM Deploy"
date: 2025-04-17
categories: [Tech]
tags: [Tech]
---

# LM Deploy

## 1. 简介


## 离线量化

```bash
lmdeploy lite auto_awq
/root/models/internlm2_5-7b-chat
--calib-dataset 'ptb'
--calib-samples 128
--calib-seqlen 2048
--work-dir /root/models/internlm2_5-7b-chat-w4a16-4bit
```

简化命令

```bash
lmdeploy lite auto_awq internlm/internlm2_5-7b-chat --work-dir internlm2_5-7b-chat-4bit
```
