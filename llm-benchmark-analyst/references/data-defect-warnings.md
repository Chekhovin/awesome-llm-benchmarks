# 来源中的缺陷与解释边界

以下内容仅从 [benchmark-source.md](benchmark-source.md) 提炼。它记录源文档的风险主张，不代表本 skill 已重新审计数据集，也不代表旧问题必然影响现行版本。引用时区分“源文档提示”“已核对原始依据”“当前版本受影响”和“是否修复未知”。

不沿用旧版缺陷目录、替代基准建议或固定严重性等级。风险应贴近受影响证据；若决定性结论依赖存在争议的数据，明确限制结论，并寻找新清单内测量互补能力的证据。

## 有明确条目的风险

| 对象与原文位置 | 源文档提示 | 报告中的处理 |
| --- | --- | --- |
| PostTrainBench，175–179 行 | 训练后优化可通过蒸馏更强模型走捷径，与递归自我改进目标不一致 | 先核对参评方案和允许资源；能证明任务内优化表现，不直接证明前沿模型自主递归改进。原文示例型号不是最新榜单证据 |
| AA-Omniscience，345–349 行 | 源文档称社区发现大量问题题目，未在该段给出具体审计链接或影响比例 | 作为待核实的题目质量警示，不编造错误率或断言已证实全榜失效；准确率、幻觉率与全知指数分别分析 |
| Context Arena（MRCR v2），371–378 行 | 源文档提到少量样本即可带来很大得分提升 | 提示对任务适配的敏感性；召回任务成绩不足以独立证明广泛长上下文推理。未核对技术报告时不将该观察升级为已证实训练污染 |
| GPQA Diamond，420–425 行 | 源文档引用对 OCR、录入和题目质量的批评 | 核对审计适用的数据版本与范围；不以微小分差决定科学能力排名，不臆造受影响题目数 |
| Humanity's Last Exam，426–438 行 | 源文档引用同一份题目质量审计，并强调结构化学术问题不等于自主研究 | 分开 HLE 的工具模式和 HLE-Diamond 子集；审计结论不能未经核实直接推广至筛选后的子集，也不能假定子集已完全修复 |

GPQA/HLE 的原文依据：[Humanity's Last Hallucination: A Forensic Audit of the Scientific Insolvency in GPQA and HLE](https://zenodo.org/records/18293568)。该引用在报告中应保留为可追溯来源；引用存在不等于所有指控都已获独立确认。

## 通用方法论风险

### 基准相关性与证据重复

原文前言指出跨领域基准也可能高度相关，引用 [Benchmark scores are well correlated, even across domains](https://epoch.ai/data-insights/benchmark-correlations)。多个相关成绩不是等量独立证据。综合指数和组成项、同题库的不同复测、同一结果的转载，需要辨认重叠。

### 基准设计缺陷

原文前言引用 [Fantastic Bugs and Where to Find Them in AI Benchmarks](https://arxiv.org/abs/2511.16842) 和 [Why benchmarking is hard](https://epoch.ai/gradient-updates/why-benchmarking-is-hard)。前言中的举例只用作方法论背景，不自动扩充当前可选基准，也不能从“基准会有缺陷”推导出任意具体基准存在已证实缺陷。

### 编码基础设施噪声

原文引用 [Quantifying infrastructure noise in agentic coding evals](https://www.anthropic.com/engineering/infrastructure-noise)，说明基础设施配置可带来数个百分点波动。涉及智能体编码时检查资源、超时、工具、运行环境和重试设置；不要把小差距全归因于模型。

### 视觉评测不稳定

原文视觉章节（532–535 行）提示小样本、标记样式与 JPEG 压缩等因素可改变排名，引用 [VPBench](https://lisadunlap.github.io/vpbench/)。应按实际数据、渲染和版本核验影响，不能笼统否定所有视觉任务。

### OCR 的任务与语言差异

原文 OCR 章节（560–565 行）引用 [Supercharge your OCR Pipelines with Open Models](https://huggingface.co/blog/ocr-open-models)，指出不同文档类型与语言表现差异很大。英语、公式、复杂表格、古文字、扫描/拍照失真任务应分别解释，不外推到整个文档处理场景。

### 经济价值的外推边界

原文白领任务章节（257–259 行）引用 [What do “economic value” benchmarks tell us?](https://epoch.ai/blog/what-do-economic-value-benchmarks-tell-us)。具体任务结果不能直接换算就业替代率、现实收入或完整职业胜任度；要说明成果物、任务选择、工具和验证标准。

## 来源自身的歧义

- AA Intelligence 的原文同时出现“七项”和“10项”，且列出具体组成。不要擅自选择一个作为当前事实，先核对官网版本与权重；无法核对时明确保留矛盾。
- 带版本的条目与正文样本量可能分属不同时间。引用 Terminal-Bench 4.0 等版本时，名称、题量与现行结果必须一起核验，不从旧数字推断配置。
- “人工设计”“人工审核”“私有”“无污染”等描述记录为来源声明，不能据此断言完全没有泄漏、偏差或可利用漏洞。
- 原文空栏目、入口网站和评测工具不能被补成独立榜单；源文档本身也不是一份经过实时复核的排名数据库。

## 采集问题与基准缺陷分开

无法加载网页、截图读数不清、模型别名不确定、结果时间未知，是本次取证的局限，不是基准本身有缺陷。保留原始数值，标注不确定字段；不能用猜测补全，也不能仅凭截图提取方式自动否定可靠的可见结果。

没有公开误差区间时，只说“差异是否稳定尚不能判断”，不要给虚构置信度或任意显著性阈值。对是否修复、是否受影响有新证据时，在本次报告说明适用版本，不静默修改源快照或扩大清单。
