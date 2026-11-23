# 第三方材料说明

本书的原创内容授权见 LICENSE。引用的论文与实现链接保留在各章及附录中；其版权归各自权利人所有。本仓库不发布研究期间收集的文章正文或第三方仓库快照。

- Noto Serif CJK SC 和 Noto Sans CJK SC：字体来源、固定版本及许可证见 assets/fonts/README.md 和该目录中的 OFL 许可证。
- KaTeX：离线 HTML 使用 KaTeX 排版与字体，许可证为 MIT；完整声明见 assets/licenses/KaTeX-LICENSE.txt。
- Microsoft Azure 公开会话长度数据：转换数据中的来源、署名、CC BY 4.0 许可及转换说明见 experiments/part06/conversation_lengths.json。
- Mahimahi 网络轨迹：取自 ravinet/mahimahi 固定提交 `1a775777d2f8e667c59447bf0ff268178b6ff4ad`，上游仓库附 GNU GPL v3 许可证，全文见 assets/licenses/Mahimahi-COPYING.txt。原始 Verizon-LTE-short.up 与 10 ms 分箱转换数据一并提供于 experiments/part05；转换由本书完成，来源与原文件 SHA-256 保留在 JSON 中，可使用 radio_replay.py --source 验证。该第三方材料及其转换版本按上游 GPL v3 条款提供，不受本书保留版权条款限制。版权归 Mahimahi 项目贡献者；此处为数据汇编收录，不表示上游作者认可本书。
- 第 19 章图 19-8 依据 FALCON 论文图 4 在三个时刻的近似读数重新绘制。队列对照用时来自 NVIDIA 的 Exemplar 案例三。取数方法、论文版本和原始链接见 [观测数据说明](experiments/part04/collective_field_measurements.json)。
