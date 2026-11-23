# 印刷版字体

PDF 正文使用 Noto Serif CJK SC Regular，标题、图注、表格及插图文字使用 Noto Sans CJK SC Regular / Bold。它们以 `BookSerif`、`BookSans` 的 CSS 名称载入，不需要读者在自己的电脑上安装字体。数学公式保留 KaTeX 配套字体，代码保留导出时嵌入的等宽字体。

这里的 WOFF2 字体子集收录了本版各章正文、插图文字和作者署名用到的字。字体内部保留版权信息，许可证随文件提供。上游版本、下载地址及校验值见 [print-fonts.json](print-fonts.json)。

- [Noto CJK 官方项目](https://github.com/notofonts/noto-cjk)
- [Noto Serif 许可证](NotoSerif-LICENSE.txt)
- [Noto Sans 许可证](NotoSans-LICENSE.txt)

添加当前子集之外的汉字后，先安装 `fonttools` 和 `brotli`，再运行 `python tools/prepare_print_fonts.py`。也可以用 `--python-libs` 指定本地依赖目录。完整上游字体缓存在 `local/print-fonts/`；发布文件仅保留这三个子集及许可证。脚本会检查正文中的汉字是否为三套字体共同覆盖，避免悄悄退回系统字体。
