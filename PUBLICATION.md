# 阅读与发布说明

《AI Infra 由浅入深》由残差宇撰写。可在 [GitHub 目录](README.md) 逐章阅读，也可从[阅读站](https://7bcoding.github.io/AI-Infra-Book/) 下载带书签的 PDF、离线 HTML 或 Markdown 与实验包。

本版分为八篇三十五章。旧书签若指向分布式训练、推理或平台，请按主题在新目录里定位。分布式训练在第 18 至 22 章，在线推理在第 23 至 26 章，后训练与应用在第 27 至 31 章，平台工程在第 32 至 35 章。两版之间的修订可在 GitHub 提交记录中查阅。

## 在线阅读站

运行 `python tools/build_public_site.py --repo 7Bcoding/AI-Infra-Book`，将生成 `_site/`。
本地预览：`python -m http.server 8000 --directory _site`。
上传仓库后，在 Settings → Pages 将 Source 设置为 GitHub Actions，运行 Publish reading site。

## 版本与反馈

读者可以在仓库的提交记录中查看版本修订。
Issue 请注明章节、图号或 PDF 页码，以及复现方式和建议修改。
第三方字体、软件与数据的说明见 THIRD_PARTY_NOTICES.md。
