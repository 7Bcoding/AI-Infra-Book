# 阅读与发布说明

《AI Infra 由浅入深》由弦宇撰写。可在 [GitHub 目录](README.md) 逐章阅读，也可从[阅读站](https://7bcoding.github.io/AI-Infra-Book/) 下载带书签的 PDF、离线 HTML 或 Markdown 与实验包。

本版分为八篇三十五章。第二篇共三章，第 6 章跟着一次前向计算走进 GPU，第 7 章沿硬件代际讨论显存、互联和网卡，第 8 章用同一个投影比较 GPU、TPU 与昇腾。上一版的第 10 章及以后各章在本版提前一章。使用旧书签的读者可以按章名从当前目录进入，也可对照[上一版目录](https://github.com/7Bcoding/AI-Infra-Book/tree/022c53d62be2f6ea7a6c89ac50f9cbfa876d3a84)。

## 在线阅读站

运行 `python tools/build_public_site.py --repo 7Bcoding/AI-Infra-Book`，将生成 `_site/`。
本地预览：`python -m http.server 8000 --directory _site`。
上传仓库后，在 Settings → Pages 将 Source 设置为 GitHub Actions，运行 Publish reading site。

## 版本与反馈

读者可以在仓库的提交记录中查看版本修订。
Issue 请注明章节、图号或 PDF 页码，以及复现方式和建议修改。
第三方字体、软件与数据的说明见 THIRD_PARTY_NOTICES.md。
