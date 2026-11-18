# AI Infra 由浅入深

从数学模型的表达与学习，走进计算设备、算子与编译，再到分布式训练、在线服务和完整 AI 系统。本书面向具备数学基础的读者，在需要时建立计算机与深度学习基础，沿具体问题、推导、算例和图解逐步深入。

[在线阅读](https://7Bcoding.github.io/AI-Infra-Book/) · [读全书 PDF](dist/AI-Infra-由浅入深-CQY.pdf) · [下载离线 HTML](dist/AI-Infra-全书.html) · [前言](book/前言.md) · [Markdown 分章与合订本](dist/AI-Infra-由浅入深.md) · [按问题查阅](book/appendices/六篇查阅与复算.md) · [下载合订稿、插图和实验](dist/AI-Infra-六篇实验.zip)

在 GitHub 上点下方章名即可开始，每章末尾可翻到相邻章节。PDF 带印刷页码、分篇书签和可点击目录；下载 HTML 后，可断网放大插图、查看公式并取得实验程序。

按图查找可用[全书图解索引](docs/全书图解索引.md)，按题复习可用[各章练习与解答](docs/练习答案核对.md)。

## 目录

### 第一篇　模型表示与学习原理

从数学表达和神经网络训练出发，了解 Transformer 如何处理上下文。 [连续阅读第一篇](dist/AI-Infra-第一篇.html)

- [第 1 章　AI Infra 解决什么问题](book/chapters/ch01.md) · [离线](dist/AI-Infra-第1章.html)
- [第 2 章　数学表达怎样变成程序](book/chapters/ch02.md) · [离线](dist/AI-Infra-第2章.html)
- [第 3 章　神经网络怎样学会一项任务](book/chapters/ch03.md) · [离线](dist/AI-Infra-第3章.html)
- [第 4 章　一个 Transformer 的两种运行方式](book/chapters/ch04.md) · [离线](dist/AI-Infra-第4章.html)
- [第 5 章　长上下文：模型怎样记住过去](book/chapters/ch05.md) · [离线](dist/AI-Infra-第5章.html)

### 第二篇　计算架构与存储系统

以矩阵乘法为例，理解处理器的计算能力、存储层次与性能限制。 [连续阅读第二篇](dist/AI-Infra-第二篇.html)

- [第 6 章　从 CPU 到 GPU：怎样让许多计算同时进行](book/chapters/ch06.md) · [离线](dist/AI-Infra-第6章.html)
- [第 7 章　GPU 存储系统：数据如何到达计算单元](book/chapters/ch07.md) · [离线](dist/AI-Infra-第7章.html)
- [第 8 章　从 GPU 到 TPU 与 NPU：架构的取舍和性能边界](book/chapters/ch08.md) · [离线](dist/AI-Infra-第8章.html)

### 第三篇　算子优化与模型编译

逐级改变数据复用、精度与程序布局，检验结果与执行代价。 [连续阅读第三篇](dist/AI-Infra-第三篇.html)

- [第 9 章　矩阵乘法：从三重循环到高性能算子](book/chapters/ch09.md) · [离线](dist/AI-Infra-第9章.html)
- [第 10 章　注意力计算与归约优化](book/chapters/ch10.md) · [离线](dist/AI-Infra-第10章.html)
- [第 11 章　少用几个比特：低精度计算与量化](book/chapters/ch11.md) · [离线](dist/AI-Infra-第11章.html)
- [第 12 章　框架与运行时：一个张量的生命周期](book/chapters/ch12.md) · [离线](dist/AI-Infra-第12章.html)
- [第 13 章　编译器：把通用模型变成适合机器的程序](book/chapters/ch13.md) · [离线](dist/AI-Infra-第13章.html)

### 第四篇　分布式训练系统

把一次训练步拆给许多设备，再在一次失败后恢复到一致的进度。 [连续阅读第四篇](dist/AI-Infra-第四篇.html)

- [第 14 章　从服务器到集群：数据怎样跨设备移动](book/chapters/ch14.md) · [离线](dist/AI-Infra-第14章.html)
- [第 15 章　集合通信：让每张卡得到恰好需要的数据](book/chapters/ch15.md) · [离线](dist/AI-Infra-第15章.html)
- [第 16 章　大模型并行训练：容量、通信与气泡的交换](book/chapters/ch16.md) · [离线](dist/AI-Infra-第16章.html)
- [第 17 章　混合专家：条件计算与系统协同设计](book/chapters/ch17.md) · [离线](dist/AI-Infra-第17章.html)
- [第 18 章　数据与容错：从最快一步到可靠地完成训练](book/chapters/ch18.md) · [离线](dist/AI-Infra-第18章.html)

### 第五篇　大模型推理与在线服务

让不同长度的请求共用权重与显存，分别核算首词、逐词和整单的时延。 [连续阅读第五篇](dist/AI-Infra-第五篇.html)

- [第 19 章　推理引擎：变长请求怎样共享设备](book/chapters/ch19.md) · [离线](dist/AI-Infra-第19章.html)
- [第 20 章　生成策略：额外计算何时换来更快或更好的回答](book/chapters/ch20.md) · [离线](dist/AI-Infra-第20章.html)
- [第 21 章　分布式推理：计算与状态应该放在哪里](book/chapters/ch21.md) · [离线](dist/AI-Infra-第21章.html)

### 第六篇　后训练、模型应用与平台工程

介绍后训练、多模态应用、智能体及 AI 平台的建设与评估。 [连续阅读第六篇](dist/AI-Infra-第六篇.html)

- [第 22 章　后训练：训练系统与生成系统怎样形成闭环](book/chapters/ch22.md) · [离线](dist/AI-Infra-第22章.html)
- [第 23 章　当输入输出不再只是文字：多模态与生成式计算](book/chapters/ch23.md) · [离线](dist/AI-Infra-第23章.html)
- [第 24 章　智能体、检索与执行环境：模型之外的时间和状态](book/chapters/ch24.md) · [离线](dist/AI-Infra-第24章.html)
- [第 25 章　建设一套 AI 平台：资源、可靠性与成本的共同设计](book/chapters/ch25.md) · [离线](dist/AI-Infra-第25章.html)

## 动手读

初次学习可先用 [第三章的小网络](experiments/ch03/verify.py) 核对手算的梯度。进入工程章节后，可以在普通电脑上[从中断点恢复一个训练步](experiments/part04/recover_step.py)，或[回放同一批交接任务](experiments/part06/evidence_replay.py)，看看高负载下哪一项承诺最先失守。把实验包解压后，在解压目录运行：

```sh
python experiments/ch03/verify.py
python experiments/part04/recover_step.py
python experiments/part06/evidence_replay.py --arrival-ms 20
python experiments/part06/service_budget.py --workers 8 16 24
```

各章附有与正文对应的程序和运行条件。可选的 [PyTorch 执行跟踪](experiments/part03/trace_layer.py)、[双进程专家训练](experiments/part04/distributed_step.py) 与 [Triton 矩阵乘法](experiments/part03/triton_projection.py) 供有相应环境的读者实测。

最后一条命令重放 19,366 次公开对话请求。读到[第 25 章](book/chapters/ch25.md)时，可以先算出无排队时的达标上限，再看加副本能接近它多少。各章正文和附录保留了参考论文与实现链接。

[第 25.8 节](book/chapters/ch25.md)汇总算子、训练、服务与平台的评估方法。运行 `python experiments/part06/evaluation_protocol.py`，可以复算请求计数、配对收益区间、达标方案与单位任务能耗，再将[教学记录](experiments/part06/evaluation_case.json)替换为自己的实验输入。

## 版权与反馈

© 2026 CQY。保留版权，免费阅读和下载；使用范围见 [LICENSE](LICENSE)，第三方材料见 [说明](THIRD_PARTY_NOTICES.md)。发现问题可在 [Issues](https://github.com/7Bcoding/AI-Infra-Book/issues) 提交章节、页码与建议。
