# 阅读与发布说明

Markdown 正文位于 book/chapters；PDF 和离线 HTML 位于 dist。
本目录只收录六篇二十五章定稿，不含旧卷稿、研究快照及本地校样。

## 在线阅读站

运行 `python tools/build_public_site.py --repo 7Bcoding/AI-Infra-Book`，将生成 `_site/`。
本地预览：`python -m http.server 8000 --directory _site`。
上传仓库后，在 Settings → Pages 将 Source 设置为 GitHub Actions，运行 Publish reading site。

## 版本与反馈

首次公开发布拟使用 v1.0.0。PDF 为 481 页校订版；本版定位为首次公开版，后续通过勘误继续修订。
Issue 请注明章节、图号或 PDF 页码，以及复现方式和建议修改。
第三方字体、软件与数据的说明见 THIRD_PARTY_NOTICES.md。
