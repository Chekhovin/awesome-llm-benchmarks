# 任务维度与新清单路由

本文件完全依据 [benchmark-source.md](benchmark-source.md) 重建，用于选择候选与定位原文，不代替原文或实时榜单。下列行号对应本包快照，名称是检索锚点；只读取本次任务涉及的部分。

条目的原始栏目与实际能力可能不同。优先按任务含义路由：例如 UNO-Bench 需要音视频联合理解，WBench 测交互式世界模型，LibraryDesignBench 测软件库设计。标题中的“生成”“特殊场景”不能代替具体定义。

## 1. 综合能力、偏好与行业指数

原文：综合评测，29–77 行。

| 候选 | 适合回答的问题与边界 |
| --- | --- |
| Artificial Analysis → AA Intelligence | 综合能力信号；核对当前组成和权重，不将组成项重复计为独立支持 |
| Artificial Analysis Capability Indices | 职业任务导向的行业能力；按原文所列行业选择，保留行业权重口径 |
| Vals Index | 金融、法律与软件工程任务的经济权重综合指标 |
| Epoch AI → Epoch 能力指数（ECI） | 跨基准、跨时间能力变化；同一尺度和拟合版本才可比较 |
| LiveBench | 推理、编码、数学、数据分析；记录题目批次和时间 |
| LMArena | 人类偏好及细分榜；不是客观任务正确率 |
| OpenCompass | 原文明示的大语言模型闭源榜、学术榜、多模态榜 |
| SEAL LLM Leaderboards、CAIS AI Dashboard | 原文明示的综合视图及本清单已列出的子榜；避免重复计算 |
| llm-stats、vals.ai、Artificial Analysis、Epoch AI | 发现与定位入口；不能据此无限扩展至网站全部基准 |

NeMo Evaluator SDK 是评测工具，列于本栏目但没有可直接当成模型能力的统一分数。

## 2. 软件工程与编码智能体

原文：Coding Benchmarks、AI编码智能体，78–147 行。

| 子任务 | 候选条目 |
| --- | --- |
| 真实代码库、多轮任务、生产代码质量 | SWE-Together、CursorBench、FrontierCode、APEX-SWE、Ramp SWE-Bench、Real-SWE |
| 通用编码工作流与框架下的表现 | Kilo Bench、OpenHands Index、BridgeBench、Terminal-Bench 4.0 |
| 专门语言、平台与代码迁移 | Android Bench、Kotlin Benchmark、Next.js AI Agent Evaluations |
| 理解代码、编写测试、重构、审查 | SWE Atlas 的代码库问答/测试编写/重构三榜、Code Review Bench |
| 视觉到网站、动画行为复现 | Vision2Web、Animation Bench；需要视觉审美时补充 DesignArena 的对应代码类别 |
| 算法优化、系统诊断与工程设计 | ALE-Bench、ITBench、CADBench |

FrontierCode 的 Extended/Main/Diamond 是包含关系不同的任务集；SWE Atlas 的三榜测不同能力；Kilo Bench 与 Terminal-Bench 4.0 不能因有共同任务渊源而合并。ITBench 聚焦 Kubernetes 故障根因分析，不等于通用代码生成。CADBench 聚焦原生机械设计。

软件库设计另查 LibraryDesignBench（原文 526 行）；智能体遇到信息缺口时的求助行为另查 HiL-Bench（511 行）。两者均不能单独代表整体编程能力。

## 3. 长程编程、科研工程与自我改进

原文：长程任务、科研场景，148–199 行。

- 长程软件工程：DeepSWE、SWE-Marathon、MirrorCode。
- 开放技术挑战与科研代码：FrontierSWE v2、WeirdML、MLS-Bench。
- GPU 内核与服务优化：kernelbench hard、InferenceBench。记录硬件、时间预算、正确性门槛和具体性能目标。
- 训练后优化与自我改进：PostTrainBench、Reward Hacking Bench、RSI-Exam。分开观察目标成绩、未见数据泛化与是否利用捷径。

MirrorCode 强调在隔离条件下复现程序行为；PostTrainBench 的蒸馏捷径限制了对递归自我改进的解释。长任务得分不能直接换算成人类可被替代的工作时长。

## 4. 服务表现与使用中漂移

原文：模型使用，200–211 行。

- Claude Code Opus Performance Tracker：特定产品、任务子集和日期下的变化。
- AI STUPID LEVEL：持续编码任务监控；核对重复试验、置信区间和评分轴。
- 平台Coding Plan测评：首字时间、平均 TPS、完整响应耗时。

区分能力变化与服务性能变化；日期、入口、套餐、负载和测试任务不同，不能直接归因为同一个模型“变笨”。

## 5. 工具使用、协作和多服务智能体

原文：Agentic Benchmarks、AI智能体，212–256、300–305 行。

| 子任务 | 候选条目 |
| --- | --- |
| 长期且结果可验证的现实任务 | Agents' Last Exam |
| MCP 多工具、多步骤工作流 | MCPMark、MCP Atlas |
| 助手任务与工作空间操作 | PinchBench、Claw-Eval、WildClawBench、ClawsBench |
| 多工作日、多会话协作 | ClawMark、ClawArena |
| 拆分模型与框架贡献 | PawBench |
| 知识、搜索、图像事实性、文档依据 | FACTS Benchmark 对应四条轨道 |

PawBench 要记录具体 Model × Harness 组合；Claw-Eval 的安全项是乘法门控，不能把总分低简单解释成完成度低。真实服务环境与模拟服务环境的证据分开。

## 6. 白领工作与经济价值

原文：白领经济价值任务，257–299 行。

- 职业成果物与真实工作：GDPval-AA、$OneMillion-Bench、Remote Labor Index (RLI)。
- 连续项目与定量分析：AA-Briefcase、AA-AnalystAgent。
- 专业服务与消费任务：APEX-Agents、APEX、The AI Consumer Index (ACE)。
- SaaS 自动化与企业流程：AutomationBench、AutomationBench-AA、EnterpriseOps-Gym-AA。

APEX-Agents 与 APEX 是不同对象；AutomationBench-AA 的主指标加入安全限制，不与原版得分直接混算。成果物评测、模拟经营收入和真实商业收入分开，不能从任务通过率直接推导就业替代率。

## 7. 深度研究、检索与电脑操作

原文：DeepResearch、视觉定位与Gui智能体，306–324 行；按需跨读 FACTS、prinzbench。

- FutureSearch-Deep Research Bench (DRB)：研究任务中的检索与推理，原文说明使用离线存储网页；不推断为实时联网搜索能力。
- FACTS Benchmark 搜索轨道与 Grounding：分别观察检索整合和基于给定文档回答。
- prinzbench：法律研究和难检索公开信息；属于社区评测。
- Cua-Bench：原文介绍的 KiCad 专家任务；不要仅凭名称推断覆盖所有桌面应用。
- cua-speedrun：电脑操作的时间与模型调用成本；速度必须连同任务成功情况解读。

研究涉及长文档时补充第 9 节；涉及专业内容时补充第 11 节。医学或法律研究得分不直接推广到所有搜索任务。

## 8. 推理、知识、情绪与决策

原文：Intelligence Benchmarks，325–359 行；决策，615–622 行。

- 概念推理指数（CRI）：LMCA、ACCoRD、DTBench capabilities；分别注意论据评价、逻辑一致性与决策理论任务。
- ARC-AGI-3、dig.bench：新任务适应、交互实验与规则发现。
- SimpleBench、Pencil Puzzle Bench：常识/对抗性文本推理与符号谜题。
- AA-Omniscience、Bullshit Benchmark：知识、拒答校准与识别错误前提。
- AttuneBench：真实多轮对话中的情绪理解与互动。
- Jev Decision Index：类型化选择、标签、工具、排序或概率决策。

AA-Omniscience 的准确率、幻觉率、全知指数不是同一个指标。dig.bench 的文本交互实验不能当成视觉推理；Jev 的固定请求集结果不能代替开放式长任务表现。

## 9. 长上下文、幻觉与信息依据

原文：幻觉与上下文召回，360–382 行。

- HalluHard：多轮回答的事实主张及来源支撑。
- AA-LCR：长文档间的信息整合与推理。
- Context Arena（MRCR v2）：对话中的特定实例召回，保留上下文长度和针数。
- LOCA-bench：上下文受控增长时的智能体表现。

跨栏目候选：FACTS Grounding、AA-Briefcase、Medical Long Context Reasoning (MLCR) benchmark、MLCR-AA。精确召回、长文档推理和长期项目记忆是不同能力；召回得分高不等于三者都强。

## 10. 数学、科学与科研自动化

原文：AI4S（特化领域），383–451 行。

| 子任务 | 候选条目 |
| --- | --- |
| 数学竞赛、证明、近期研究题 | MathArena、FrontierMath、MathScienceBench、MathDuels、PutnamBench |
| 学术知识与高难科学问答 | GPQA Diamond、Humanity's Last Exam、HLE-Diamond |
| 物理研究与物理竞赛 | CritPt、PhyArena（HiPhO） |
| 科研工作流与生物数据分析 | Terminal-Bench-Science、BioMysteryBench |

MathArena 下仅使用原文明示的 ArXivMath、IMProofBench、MathArenaApex、Visual Math、Final-Answer Comps、Proof-Based Comps、Project Euler、BrokenArXiv、ArXivLean；记录赛题批次、运行次数、工具和费用口径。

保留 FrontierMath 难度层、PutnamBench 形式化系统、MathDuels 解题/出题双评分。HLE 的有工具/无工具结果和 HLE-Diamond 子集各自记录。结构化学术题高分不能直接证明自主科研能力；代码密集科研另查第 3 节。

## 11. 医学、法律、金融与商业

原文：特定行业基准，452–505 行。

| 行业 | 候选条目与任务边界 |
| --- | --- |
| 医学与健康 | CHI-Bench、Medical Long Context Reasoning (MLCR) benchmark、MLCR-AA、HealthBench Professional；分别回原文核对测试任务 |
| 药物发现 | DrugDiscoveryBench；早期发现阶段的计算与检索，不代表临床疗效或整个制药流程 |
| 法律 | Harvey LAB-AA；需要研究与检索维度时补充 prinzbench |
| 模拟经营 | Vending-Bench 2、YC-Bench、CEO-Bench；分别保留初始资金、模拟周期、随机种子与期末余额口径 |
| 财务、税务与尽调 | Finance Agent、TaxCalcBench、Diligence Stack Agent Bench |
| 商业工作流 | Commerce Agent Bench |

MLCR-AA 是从 MLCR 的高难类别选出的私有子集，评分还包括完整性、准确性和简洁性；两者不能互换。TaxCalcBench 的美国税务场景不能泛化成所有国家和复杂税制的能力证明。

## 12. 视觉、文档、嵌入与音视频

原文：视觉理解与推理、OCR与嵌入评测、生图等栏目，532–614 行。

| 子任务 | 候选条目 |
| --- | --- |
| 视觉理解与感知 | MMMU-Pro、ZeroBench、BabyVision、PerceptionBench |
| 空间重建与视觉规划 | Blueprint-Bench 2、MazeBench |
| 文档识别、版面与公式 | OmniDocBench、olmOCR-Bench、Real5-OmniDocBench、OCRVerse、Chronicles-OCR、PDF Parse Bench |
| 嵌入与检索表示 | Embedding Leaderboard（MTEB Leaderboard） |
| 图像生成与设计偏好 | GenExam、DesignArena 的对应类别 |
| 音视频联合理解 | UNO-Bench |
| 交互式视频世界模型 | WBench |
| 语音交互 | τ-voice |

OCRVerse 的原文将其介绍为方法，使用前确认结果比较的是解析方案、模型还是基准任务；不要臆造统一榜单。DesignArena 的图像、视频、代码和幻灯片子榜分别使用。τ-voice 保留响应率、延迟、打断率和选择性，不用单一语音识别数值替代交互质量。

“世界模型”“Omni”在原文中的小标题下没有独立条目，但相关实质内容可按 UNO-Bench、WBench 路由；不得从空标题增补其他榜单。

## 13. 特殊行为、社区与游戏

原文：特殊场景，506–531 行；社区评测、游戏，623–678 行。

- 边界与安全行为：SpeechMap、DecodingTrust Bench、Political Manipulation。
- 求助判断、体素生成、翻译、软件库设计：HiL-Bench、Voxelbench、Last Translation Benchmark、LibraryDesignBench。
- 社区综合与约束任务：nao老师的LLM Benchmark、XSCT Bench、LisanBench。
- 知识截止点：knowledge-cutoff。估计的事件知识边界不同于厂商声明的训练截止日期。
- 研究/社区入口：Kaggle Benchmarks。平台存在不等于其所有用户评测均获纳入。
- 法律研究：prinzbench。
- lechmazur 条目明确列出六项：Creative Story‑Writing Benchmark、Elimination Game Benchmark、NYT Connections puzzles、Sycophancy Benchmark、Thematic Generalization Benchmark、Persuasion Benchmark。分别观察创作、社交策略、字谜、谄媚、主题泛化与说服，不合并成一个分数。
- 游戏：AI Poker Leaderboard、RuneBench。

社区评测要注明题库公开性、更新频率与样本规模，不能因属个人项目直接否定，也不能与大样本受控评测等权。原文“预测”栏目为空，不补入清单外项目。

## 14. 开放、采纳、数据、工具与基础设施

原文：前言 1–28 行、开放指数 41–43 行、数据质量评估及 AI Infra性能 679–711 行。

| 对象 | 观察内容 |
| --- | --- |
| Relative Adoption Metric (RAM) | 按模型规模和发布后时间归一的下载采纳情况 |
| Artificial Analysis Openness Index | 权重、许可、数据与方法的开放程度 |
| OpenDataArena | 固定模型与训练配置下的数据集质量及数据血缘 |
| Modded-NanoGPT Optimization Benchmark | 优化算法减少训练步数的能力；不等同于墙钟时间竞速 |
| InferenceMAX | 模型、硬件、并行和并发条件下的吞吐与延迟 |
| MLPerf Training | 系统达到指定训练质量目标所需时间 |
| AA-AgentPerf（AI Hardware Benchmarking & Performance Analysis） | 满足服务指标时每兆瓦可支持的智能体数量 |
| GPU Benchmark | 测试平台及不同数值精度下的计算卡性能 |

前言中的 inspect-ai、lighteval、Open Benchmark Index 与第三方质量检索，以及综合栏目中的 NeMo Evaluator SDK，是执行或发现资源，不是新增的模型能力基准。仅当用户要求实际评测时再规划执行、环境和费用。

## 跨栏目选择原则

按真实工作流补充证据，不机械抓取整个维度。例如前端任务可连接软件工程、Vision2Web、Animation Bench 与 DesignArena 对应子榜；科研任务可连接数学/物理知识、科研工作流与科研编程；医学文档任务可连接医学长文档、OCR 与事实依据。只纳入会影响结论的子能力，并检查是否共享题库或重复展示同一结果。
