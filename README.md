# AI Infra 由浅入深

**残差宇 著**

一次模型请求，从数学公式到用户屏幕，中间要经过哪些计算？为什么换了更快的 GPU，有时仍要等同样久？本书从可手算的小模型讲起，沿着数据走过算子、加速器、集群和在线服务。具备线性代数、微积分与概率基础就可以开始读。

[在线阅读](https://7bcoding.github.io/AI-Infra-Book/) · [下载 PDF](dist/AI-Infra-由浅入深-残差宇.pdf) · [下载离线 HTML](dist/AI-Infra-全书.html) · [前言](book/前言.md) · [Markdown 合订本与实验包](dist/AI-Infra-全书实验.zip)

在 GitHub 上点章名可以直接读正文。需要找一幅图、一道题或某个工程问题时，可以翻到[图解索引](docs/全书图解索引.md)、[练习与解答索引](docs/练习答案核对.md)或[查阅与复算附录](book/appendices/全书查阅与复算.md)。PDF 的目录和书签均可跳转。离线 HTML 内置公式、插图和实验包。

## 目录

### 第一篇　模型表示与学习原理

一条请求进来，用户等的是什么？从张量和梯度算起，在纸上跑一次 Transformer，再算保存一段历史需要多少显存。 [连续阅读第一篇](dist/AI-Infra-第一篇.html)

- [第 1 章　AI Infra 解决什么问题](book/chapters/ch01.md) · [离线](dist/AI-Infra-第1章.html)
- [第 2 章　从数学表达式到程序](book/chapters/ch02.md) · [离线](dist/AI-Infra-第2章.html)
- [第 3 章　神经网络怎样学会一项任务](book/chapters/ch03.md) · [离线](dist/AI-Infra-第3章.html)
- [第 4 章　一个 Transformer 的两种运行方式](book/chapters/ch04.md) · [离线](dist/AI-Infra-第4章.html)
- [第 5 章　长上下文与模型记忆](book/chapters/ch05.md) · [离线](dist/AI-Infra-第5章.html)

### 第二篇　计算架构与存储系统

跟随一块权重从显存进入矩阵单元，看指令调度、缓存与片上复用如何影响算力。再把同一层模型放上昇腾和 TPU，对照三种设计的数据流。 [连续阅读第二篇](dist/AI-Infra-第二篇.html)

- [第 6 章　GPU 的由来与执行结构](book/chapters/ch06.md) · [离线](dist/AI-Infra-第6章.html)
- [第 7 章　GPU 存储层次与矩阵乘法的数据流](book/chapters/ch07.md) · [离线](dist/AI-Infra-第7章.html)
- [第 8 章　GPU、昇腾与 TPU 的架构取舍](book/chapters/ch08.md) · [离线](dist/AI-Infra-第8章.html)

### 第三篇　算子、运行时与编译系统

逐步改变矩阵乘的访存、精度和布局，用设备计时检验预期。随后把镜头移到相邻算子，找出单个内核跑快以后仍留在请求里的等待。 [连续阅读第三篇](dist/AI-Infra-第三篇.html)

- [第 9 章　矩阵乘法的实现与测量](book/chapters/ch09.md) · [离线](dist/AI-Infra-第9章.html)
- [第 10 章　归约与注意力的分块计算](book/chapters/ch10.md) · [离线](dist/AI-Infra-第10章.html)
- [第 11 章　低精度计算与量化](book/chapters/ch11.md) · [离线](dist/AI-Infra-第11章.html)
- [第 12 章　张量的生命周期与框架运行时](book/chapters/ch12.md) · [离线](dist/AI-Infra-第12章.html)
- [第 13 章　从计算图到机器指令](book/chapters/ch13.md) · [离线](dist/AI-Infra-第13章.html)
- [第 14 章　性能剖析与实验设计](book/chapters/ch14.md) · [离线](dist/AI-Infra-第14章.html)

### 第四篇　预训练数据与训练预算

同样一千万个词元，可以让模型见到很不一样的材料。先算抽样次数和训练预算，再跟一批数据从文件走到设备，看训练会在哪里停下来。 [连续阅读第四篇](dist/AI-Infra-第四篇.html)

- [第 15 章　训练语料、分词与数据配方](book/chapters/ch15.md) · [离线](dist/AI-Infra-第15章.html)
- [第 16 章　模型规模、训练预算与收敛](book/chapters/ch16.md) · [离线](dist/AI-Infra-第16章.html)
- [第 17 章　输入流水线与分布式存储](book/chapters/ch17.md) · [离线](dist/AI-Infra-第17章.html)

### 第五篇　集群互联与分布式训练

模型放不下一张卡时，沿着通信链路追一次完整的训练步。分片、路由和恢复都要回答同一个问题：下一步能否拿到正确的权重和数据？ [连续阅读第五篇](dist/AI-Infra-第五篇.html)

- [第 18 章　服务器互联与集群网络](book/chapters/ch18.md) · [离线](dist/AI-Infra-第18章.html)
- [第 19 章　集合通信与梯度同步](book/chapters/ch19.md) · [离线](dist/AI-Infra-第19章.html)
- [第 20 章　并行训练与状态分片](book/chapters/ch20.md) · [离线](dist/AI-Infra-第20章.html)
- [第 21 章　混合专家的路由与执行](book/chapters/ch21.md) · [离线](dist/AI-Infra-第21章.html)
- [第 22 章　训练恢复与模型交付](book/chapters/ch22.md) · [离线](dist/AI-Infra-第22章.html)

### 第六篇　生成推理与在线服务

从一台机器上的变长请求出发，看多步验证、跨机传输或弱网回退何时能让用户早一点读到结果。 [连续阅读第六篇](dist/AI-Infra-第六篇.html)

- [第 23 章　单机推理与变长请求调度](book/chapters/ch23.md) · [离线](dist/AI-Infra-第23章.html)
- [第 24 章　生成策略、推测解码与约束生成](book/chapters/ch24.md) · [离线](dist/AI-Infra-第24章.html)
- [第 25 章　分布式推理与状态调度](book/chapters/ch25.md) · [离线](dist/AI-Infra-第25章.html)
- [第 26 章　端侧计算与弱网协作](book/chapters/ch26.md) · [离线](dist/AI-Infra-第26章.html)

### 第七篇　后训练与典型 AI 应用

训练数据开始来自模型自己的输出，系统还要等采样和判分。接着用视频、工具调用和商品推荐，追踪三种不同的任务怎样检验一次交付。 [连续阅读第七篇](dist/AI-Infra-第七篇.html)

- [第 27 章　后训练数据、微调与偏好学习](book/chapters/ch27.md) · [离线](dist/AI-Infra-第27章.html)
- [第 28 章　强化学习的采样与训练系统](book/chapters/ch28.md) · [离线](dist/AI-Infra-第28章.html)
- [第 29 章　多模态表示与生成计算](book/chapters/ch29.md) · [离线](dist/AI-Infra-第29章.html)
- [第 30 章　检索、智能体与可靠执行](book/chapters/ch30.md) · [离线](dist/AI-Infra-第30章.html)
- [第 31 章　推荐系统的召回、排序与在线特征](book/chapters/ch31.md) · [离线](dist/AI-Infra-第31章.html)

### 第八篇　AI 平台与容量工程

算清集群里谁能先用哪几张卡，追一次发布中的长输入退化，再按供电、故障余量和用户时限做最后的容量决策。 [连续阅读第八篇](dist/AI-Infra-第八篇.html)

- [第 32 章　多租户集群调度与资源隔离](book/chapters/ch32.md) · [离线](dist/AI-Infra-第32章.html)
- [第 33 章　模型发布、可观测性与事故处置](book/chapters/ch33.md) · [离线](dist/AI-Infra-第33章.html)
- [第 34 章　数据中心的供电、散热与故障域](book/chapters/ch34.md) · [离线](dist/AI-Infra-第34章.html)
- [第 35 章　AI 系统的性能评估与容量决策](book/chapters/ch35.md) · [离线](dist/AI-Infra-第35章.html)

## 动手读

只需 Python 标准库就能从小网络开始。可以先在纸上算出一次梯度，再运行程序核对。到了输入流水线一章，同样的 2 MiB 数据有两套读取记录可以比较。训练结束后，还可以试着故意转错一块权重，看交付测试能否抓住它。解压实验包后，在根目录运行相应命令。

```sh
python experiments/ch03/verify.py
python experiments/worked/storage_batch.py --replay
python experiments/worked/model_handoff.py
python experiments/worked/edition_cases.py
python experiments/part06/service_budget.py --workers 8 16 24
```

最后一项实验可重放 19,366 次公开对话请求，检验副本数和时限对达标率的影响。带有相应硬件和软件环境的读者还可以运行 [PyTorch 执行跟踪](experiments/part03/trace_layer.py)、[双进程训练](experiments/part04/distributed_step.py)和 [Triton 矩阵乘法](experiments/part03/triton_projection.py)。

## 版权与反馈

© 2026 残差宇。保留版权，免费阅读和下载。使用范围见 [LICENSE](LICENSE)，第三方材料见[说明](THIRD_PARTY_NOTICES.md)。发现问题时，欢迎在 [Issues](https://github.com/7Bcoding/AI-Infra-Book/issues) 留下章节、图号或页码。
