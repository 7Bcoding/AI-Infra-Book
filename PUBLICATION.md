# 阅读与发布说明

《AI Infra 由浅入深》由弦宇撰写。可在 [GitHub 目录](README.md) 逐章阅读，也可从[阅读站](https://7bcoding.github.io/AI-Infra-Book/) 下载带书签的 PDF、离线 HTML 或 Markdown 与实验包。

本书分为八篇三十五章。第二篇从单卡的执行和存储讲到代际演进、系统互联，再比较 GPU、TPU 与昇腾。随书算例可用 Python 标准库复算，包括分块覆盖率、显存传输时间和分块注意力。阅读过三十六章版的读者可对照[当时的目录](https://github.com/7Bcoding/AI-Infra-Book/tree/022c53d62be2f6ea7a6c89ac50f9cbfa876d3a84)，该版第 10 章起的内容在现行目录提前一章。

## 在线阅读站

运行 `python tools/build_public_site.py --repo 7Bcoding/AI-Infra-Book`，将生成 `_site/`。
本地预览：`python -m http.server 8000 --directory _site`。
上传仓库后，在 Settings → Pages 将 Source 设置为 GitHub Actions，运行 Publish reading site。

## 版本与反馈

读者可以在仓库的提交记录中查看版本修订。
Issue 请注明章节、图号或 PDF 页码，以及复现方式和建议修改。
第三方字体、软件与数据的说明见 THIRD_PARTY_NOTICES.md。
