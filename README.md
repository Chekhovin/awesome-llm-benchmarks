# awesome-llm-benchmarks

[English](./README-en.md) | [日本語](./README-jp.md)

大模型评测榜单汇总：收录综合评测、编码、智能体、推理、多模态、行业基准等大模型评测榜单与基准测试资源，持续更新。

Tips：

同一领域甚至不同领域的不同基准测试分数均具有很高的相关性，参见*[Benchmark scores are well correlated, even across domains](https://epoch.ai/data-insights/benchmark-correlations)*

基准测试（Benchmarks）构成了评价模型性能的事实标准，但测试本身并非准确无误，甚至可能存在系统性缺陷，典型案例τ²-bench、MMLU-Pro、GPQA、HLE。（参见论文：*[Fantastic Bugs and Where to Find Them in AI Benchmarks](https://arxiv.org/abs/2511.16842)*）（案例说明见下面子项介绍）

当前对LLM进行评测的难点和缺陷参见epoch.ai的博客*[Why benchmarking is hard](https://epoch.ai/gradient-updates/why-benchmarking-is-hard)*。Anthropic发现基础设施配置可以对智能体编程基准测试产生数个百分点的波动（参见*[Quantifying infrastructure noise in agentic coding evals](https://www.anthropic.com/engineering/infrastructure-noise)*）。

自部署模型测试：参考Hugging Face官方教程：使用inspect-ai和lighteval（[huggingface.co/docs/inference-providers/guides/evaluation-inspect-ai](https://huggingface.co/docs/inference-providers/guides/evaluation-inspect-ai)）（[github.com/huggingface/lighteval](https://github.com/huggingface/lighteval)）（[huggingface.co/docs/lighteval/main/en/index](https://huggingface.co/docs/lighteval/main/en/index)）

Hugging Face官方推出的开放Benchmark汇总，可以按语言、标签浏览任务，并搜索任务描述。（[huggingface.co/spaces/OpenEvals/open_benchmark_index](https://huggingface.co/spaces/OpenEvals/open_benchmark_index)）

epoch.ai对benchmark进行第三方质量检测（[epoch.ai/benchmarks/search?reviewed=verified&reviewed=flawed&reviewed=not-enough-info](https://epoch.ai/benchmarks/search?reviewed=verified&reviewed=flawed&reviewed=not-enough-info)）

- Relative Adoption Metric (RAM)（[atomproject.ai/relative-adoption-metric](https://atomproject.ai/relative-adoption-metric)）

  相对采纳指标（Relative Adoption Metric, RAM）。“RAM 得分”：这是一个更合适的指标，可用于评估不同规模的新开源模型的下载情况。  
  得分 = （模型的下载量） / （同一规模类别中排名前十的模型下载量的中位数）得分为1意味着该模型有望成为其所属规模中下载量排名前十的模型之一。

  数据收集（选用中位数）  
  按总下载量排名的每个尺寸分组前 10 大模型（来自 HuggingFace）  
  里程碑时刻的累计下载量：发布后第7天、14天、30天、60天、90天、180天、365天  
  各模型随时间推移的累计 HuggingFace 下载总量

## 综合评测

- Artificial Analysis（[AI Model & API Providers Analysis | Artificial Analysis](https://artificialanalysis.ai/)）

  - AA Intelligence（[Artificial Analysis Intelligence Index | Artificial Analysis](https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index)）

    一个综合基准，整合七项具有挑战性的评估，全面衡量人工智能在数学、科学、编程和推理方面的能力。综合了10项评估的表现：GDPval-AA v2, Terminal-Bench 2.1, τ³-Bench Banking, HLE, AA-Omniscience Accuracy, SciCode, GPQA Diamond, AA-LCR, CritPt, AA-Omniscience Non-Hallucination（按权重高低排序）。
  - Artificial Analysis Capability Indices（[artificialanalysis.ai/models/capabilities](https://artificialanalysis.ai/models/capabilities)）

    用于比较各模型在关键行业领域中的表现，这些新推出的行业指数涵盖了编程、金融与会计、法律、医疗与健康、策略与运营、工程以及经济学领域。

    每个指数的设定都基于 O*NET 职业分类体系中的常见工作任务。这些任务包括金融建模、法律研究与合同审查，以及临床决策支持和患者病历记录等。从每项任务中提炼出相应的能力指标，挑选出最能代表该领域工作的基准项，并根据该能力在领域内出现的频率赋予相应权重。
  - Artificial Analysis Openness Index（[artificialanalysis.ai/evaluations/artificial-analysis-openness-index](https://artificialanalysis.ai/evaluations/artificial-analysis-openness-index)）

    一项标准化且独立评估的指标，用于衡量人工智能模型在可用性和透明度方面的开放程度。开放性不仅仅指能够下载模型权重。它还涉及许可协议、数据和方法论。在开放指数中获得 100 分的模型将具备开放权重、采用宽松许可协议，并完整发布训练代码、预训练数据和训练后数据——这不仅允许用户使用模型，还能完全复现其训练过程，或从模型创建者的部分或全部方法中汲取灵感来构建自己的模型。
- llm-stats（[llm-stats.com/](https://llm-stats.com/)）

  一个综合性评测网站。包含benchmark汇总（[llm-stats.com/benchmarks](https://llm-stats.com/benchmarks)）
- vals.ai（[www.vals.ai/home](https://www.vals.ai/home)）

  一个综合性评测网站，分为综合、法律、金融、医疗、数学、学术、教育、编码、游戏等类别。

  - Vals Index

    衡量 AI 模型在金融、法律与软件工程领域执行真实任务的能力。通过计算模型在关键行业中的表现加权平均值来实现这一目标，权重则对应各行业对美国经济的贡献。
- SEAL LLM Leaderboards（[scale.com/leaderboard](https://scale.com/leaderboard)）

  评估最新 LLM 的智能体能力、前沿性能、安全性及公众情绪。
- Epoch AI（[epoch.ai/benchmarks](https://epoch.ai/benchmarks)）

  有多个基准测试。

  Epoch 能力指数（ECI）将多个不同 AI 基准的分数综合为一个“通用能力”尺度，即使在单个基准已达到饱和的长时间跨度内，也能对模型进行比较。
- LMArena（[arena.ai/leaderboard/](https://arena.ai/leaderboard/)）

  一个基于真实用户盲测投票和 Elo 排名的众包评测平台，用来比较不同大模型在真实交互中的综合表现。
- OpenCompass（[OpenCompass司南 - 评测榜单](https://rank.opencompass.org.cn/home)）

  面向大模型开发者和使用者的开源开放评测平台，提供多能力维度、多个评测集和榜单的统一评测。包括大语言模型闭源榜、学术榜、多模态榜。
- LiveBench（[LiveBench](https://livebench.ai/#/)）

  一个专为避免测试集污染和实现客观评估而设计的 LLM 基准测试，包括推理、编码、数学、数据分析。
- CAIS AI Dashboard（[dashboard.safe.ai/](https://dashboard.safe.ai/)）

  包括文本、视觉、风险、自动化、远程劳动指数（对于远程工作的自动化）五个指标
- NeMo Evaluator SDK（[NVIDIA-NeMo/Evaluator: Open-source library for scalable, reproducible evaluation of AI models and benchmarks.](https://github.com/NVIDIA-NeMo/Evaluator)）

  NVIDIA 开源的大模型评测 SDK，可在统一框架下对任意兼容 API 的模型进行大规模、可复现的 benchmark 评测。

## Coding Benchmarks

- SWE-Together（[togetherbench.com/](https://togetherbench.com/)）

  一个基于真实用户与智能体编程会话重构而成的多回合基准测试集，包含109 个代码仓库级别的任务。
- ITBench（[artificialanalysis.ai/evaluations/itbench-aa](https://artificialanalysis.ai/evaluations/itbench-aa)）

  用于在站点可靠性工程（SRE）场景下对智能体开展评测，具体方向为 Kubernetes 故障根因分析。本次评测共设置 59 项 Kubernetes 故障任务，其中 40 项来自 IBM 公开版本，另外 19 项由 ITBench 团队私下提供，每项任务均重复测试 3 次。在各个测试场景中，智能体会获取一份 Kubernetes 故障的离线快照，包含告警信息、系统事件、调用链路、性能指标以及应用拓扑结构，并且需要输出结构化的 JSON 诊断报告，明确引发故障的各类核心对象，包括部署项、服务、容器组、命名空间、网络策略等。
- CursorBench（[cursor.com/cn/cursorbench](https://cursor.com/cn/cursorbench)）

  基于工程团队真实 Cursor 会话构建的评测，任务来自真实的 Cursor 用量，而不是公开代码仓库。许多任务来自内部代码库和受控来源，从而降低了模型在训练阶段见过这些任务的风险。
- FrontierCode（[cognition.com/blog/frontier-code-1.1](https://cognition.com/blog/frontier-code-1.1)）（人工编写任务）

  一个用于衡量模型能否真正达到高质量生产代码库标准的基准测试。首个衡量代码可合并性的基准，评估标准涵盖端到端代码质量——包括正确性、测试质量、范围规范、代码风格以及对代码库标准的遵守情况。难度分为三个等级：Extended, Main, and Diamond。Diamond包含难度最高的 50 个任务，Main则包含难度最高的 100 个任务（含钻石版中的任务），而Extended则是全部 150 个任务。
- Kilo Bench（[kilo.ai/leaderboard](https://kilo.ai/leaderboard)）

  基于Terminal-Bench 的终端密集型任务数据构建——包含 89 个真实世界任务，涵盖 git 操作、密码分析、QEMU 自动化等各类场景。模型测试专门通过 Kilo 测试框架执行。
- OpenHands Index（[index.openhands.dev/home](https://index.openhands.dev/home)）

  一种用于评估人工智能编码代理在真实世界软件工程任务中表现的综合基准，同时呈现模型的性能和成本效益。从五个类别对模型进行评估： 问题解决 （修复漏洞）、 全新项目开发 （构建新应用）、 前端开发 （用户界面设计）、 测试 （测试用例生成）以及信息收集 。
- APEX-SWE（[www.mercor.com/apex/apex-swe-leaderboard/](https://www.mercor.com/apex/apex-swe-leaderboard/)）

  用于评估软件工程师的实际日常工作，与单元级和单一仓库的bug修复基准测试不同，它包含200个案例，涵盖两个互补的场景：(1)集成任务，需要在异构服务之间进行端到端的系统构建和部署，评估模型编排端到端工作流以及在异构服务间同步数据的能力；(2)可观测性任务，需要使用生产级别的遥测技术进行调试，评估模型诊断和修复现实世界软件工程生产故障的能力。
- Ramp SWE-Bench（[labs.ramp.com/swebench#explore](https://labs.ramp.com/swebench#explore)）（人工审核任务）

  一个基于生产环境的私有编码基准，源自 Ramp 后端的工程工作打造。参考SWE-Bench，将该基准测试作为行为工具，用于研究编码智能体如何处理Ramp工程师已经委托给他们的工作。包含80项任务，涵盖信用卡授权、账单支付、报销、会计、采购、资金管理、欺诈、智能体等业务。
- BridgeBench（[www.bridgemind.ai/bridgebench](https://www.bridgemind.ai/bridgebench)）

  BridgeMind 推出的 vibe coding 基准，用标准化任务评测模型在调试、算法、重构、生成、UI 和安全等编程场景下的表现。
- ALE-Bench（[sakanaai.github.io/ALE-Bench-Leaderboard/](https://sakanaai.github.io/ALE-Bench-Leaderboard/)）

  一个用于评估 AI 系统在基于分数的算法编程竞赛中表现的基准测试。该基准借鉴了AtCoder Heuristic Contest (AHC)中的现实任务，提出了计算上困难且尚无已知精确解法的优化问题（例如路由和调度问题）。
- Android Bench（[developer.android.com/bench](https://developer.android.com/bench)）（人工筛选任务）

  评估了LLMs在解决现实世界Android开发问题方面的能力，包含100项任务。Android Bench 更偏向于库（58%）。
- Kotlin Benchmark（[kotlinlang.org/benchmark/](https://kotlinlang.org/benchmark/)）

  JetBrains 推出的官方基准测试，用于评估 AI 编码智能体在 Kotlin 软件工程任务上的表现。基于 SWE-bench 方法论，聚焦于仓库级别的 Kotlin 软件工程任务。包含105个来自活跃开源代码仓库的工程任务。每个任务都要求AI智能体解读真实的问题描述、梳理项目上下文并生成可用的补丁。所有解决方案都会在容器化环境中进行严格验证，只有生成的方案通过了规定的测试验证，该任务才会被标记为已解决。
- Vision2Web（[vision2web-bench.github.io/](https://vision2web-bench.github.io/)）

  一个用于评估多模态编码智能体是否具备根据视觉原型和结构化需求构建真实网站能力的基准测试，包含193项任务，旨在衡量真实环境中的端到端网页开发能力。每个任务提供多模态输入，如 UI 原型图像、需求描述和开发资源，智能体需要生成满足功能和视觉保真度的可执行网站。
- CADBench（[www.seldon.global/blog/cadbench](https://www.seldon.global/blog/cadbench)）

  CADBench 用于衡量前沿 AI 模型是否能够在 Autodesk Fusion 中完成原生的、需要长期规划的机械设计任务，以及生成的几何结构、特征历史记录和约束条件能否通过确定性核实。
- Animation Bench（[www.physera.ai/research/animation](https://www.physera.ai/research/animation)）

  旨在测试编码智能体是否能够还原网页的行为逻辑，而不仅仅是其外观表现。包括48个任务，模型会接收到 12 到 24 张从动画中采样得到的静态帧，若有时间戳信息也会一并提供；此外还会收到一个 HAR 文件，其中包含了该网页加载时所需的 HTML、CSS、JavaScript、字体及图片资源。模型不会收到网站的源项目文件、视频，或除帧及其时间戳之外的任何时间描述信息。

### AI编码智能体

- Terminal-Bench 4.0（[www.tbench.ai/](https://www.tbench.ai/)）（人工设计和验证）

  一个包含89个精心筛选的任务的数据集，涵盖软件工程、系统管理、数据科学、安全、科学计算等多个领域，用于评估 AI 智能体在终端环境中完成复杂任务的表现。Terminal-Bench 中的每个任务包括一段任务指令，一个 Docker 环境，一个测试套件，一个参考解决方案（人类编写）。代理需要通过命令行工具（如运行Bash命令、编辑文件）与环境交互来探索和解决问题。结果驱动，只关注任务最终完成的状态（通过测试验证），不限制代理的具体实现方法。
- SWE Atlas（[labs.scale.com/leaderboard/sweatlas-refactoring](https://labs.scale.com/leaderboard/sweatlas-refactoring)）

  SWE Atlas 是一套基准测试套件，用于在各类专业软件工程任务中评估人工智能编码智能体。该套件并非只衡量单一技能，而是包含三个排行榜，分别针对软件开发生命周期中不同且互补的能力。任务均来自4种编程语言（Go、Python、C和TypeScript）的11个生产代码仓库，仓库选自SWE-Bench Pro。

  1. [代码库问答](https://labs.scale.com/leaderboard/sweatlas-qna) - 通过运行时分析和多文件推理问题理解复杂代码库。包含124项任务，智能体访问 Docker 容器内代码仓库，且必须回答一系列关于该系统运行原理的深度技术问题。这些问题在设计上要求具备智能体推理能力——需要运行软件、追踪多个文件中的执行流程并综合分析得出结论。
  2. [测试编写](https://labs.scale.com/leaderboard/sweatlas-tw) - 为代码仓库中的指定功能编写有实际意义的生产级测试。包含90个任务，智能体访问 Docker 容器内代码仓库，但该仓库中某一重要工作流的关键测试存在缺失。这些任务具有智能体设计的特性——提示词仅通过相关描述，从宏观层面说明了待测试的工作流或行为。智能体需要自主探索代码库，确定需要编写的具体测试内容及存放位置，执行测试后再提交成果。测试需覆盖提示词中描述的行为，同时限定测试范围仅针对与该行为相关的代码，并编写符合代码库规范与最佳实践、清晰且可维护的代码。
  3. [重构](https://labs.scale.com/leaderboard/sweatlas-refactoring) - 重构代码以在保留功能的同时提升性能和可读性。包含70项任务，智能体访问 Docker 容器内代码仓库，提示词中会给出重构任务。这些任务在设计上具备智能体特性；提示词会从宏观层面描述理想的重构结构，明确需要提取、整合或重组的代码。智能体需要自主探索代码库，理解现有架构，对分布在代码库多个文件中的内容执行重构操作，并确保所有现有测试仍能正常通过。重构后的代码需能够正确重组指定组件，清理无效代码和过时产物，更新文档以反映变更，同时避免引入回归问题或破坏性变更。
- Next.js AI Agent Evaluations（[nextjs.org/evals](https://nextjs.org/evals)）

  各类 AI 编程智能体在 Next.js 代码生成与迁移任务上的表现数据，包括成功率与执行时间等指标。
- Code Review Bench（[codereview.withmartian.com/](https://codereview.withmartian.com/)）

  用于评估 AI 代码审查工具的基准
- Real-SWE（[realswe.withspecific.com/](https://realswe.withspecific.com/)）

  一个用于评估前沿 AI 模型处理真实企业代码库能力的基准测试。每个测试任务均源自我们从某家真实企业获得授权使用的私有生产环境代码库。

### 长程任务

- DeepSWE（[deepswe.datacurve.ai/blog/deepswe-v1-1](https://deepswe.datacurve.ai/blog/deepswe-v1-1)）（人工审核任务）

  一个长时程软件工程基准测试，包含四个特性：

  - 无污染：任务均为全新编写，并非基于现有提交或拉取请求改编。
  - 高多样性：任务涵盖5种语言（TypeScript、JavaScript、Python、Go 和 Rust）的91个代码仓库，范围广泛。
  - 真实世界的复杂性：提示词长度仅为 SWE-bench Pro 的一半，但解决方案所需的代码量是其 5.5 倍，输出标记量约为其 2 倍。
  - 可靠验证：验证人员通过手动编写代码来测试软件的行为，而非实现细节。
- SWE-Marathon（[www.swe-marathon.org/](https://www.swe-marathon.org/)）

  面向超长周期（Ultra-Long-Horizon）软件工程任务的AI智能体基准测试，包含20个覆盖4类软件工程领域的长周期任务，每个任务配备独立可执行环境、人工参考解、多层验证套件
- MirrorCode（[epoch.ai/MirrorCode](https://epoch.ai/MirrorCode)）

  旨在检验人工智能模型处理长期编码任务的能力。在 MirrorCode 测试中，人工智能模型需要在无法获取原始源代码的情况下，从头开始重新实现整个程序。人工智能生成的解决方案必须在包括预留测试集在内的所有端到端测试中，与原始程序的输出结果完全一致。MirrorCode 所设定的 25 个目标程序涵盖了计算领域的多个方向：Unix 实用工具、数据序列化与查询工具、生物信息学、解释器、静态分析、密码学以及压缩技术。

  对 AI 模型进行了沙盒隔离处理，要求它们无法访问互联网、无法接触原始代码库，从而杜绝了任何作弊的可能性。此外，还有许多端到端测试是模型在编写代码时根本无法看到的，因此它们无法简单地通过创建查找表来模拟原始程序的输出结果。

### 科研场景

- FrontierSWE v2（[www.frontierswe.com/](https://www.frontierswe.com/)）

  一个包含超长期、开放式技术挑战的编程基准，包含34个任务，例如优化编译器或训练用于蛋白质预测的最先进模型。平均而言，智能体每个任务运行时长为11小时，且几乎所有任务都无法完成。其中的`granite_inf` 任务用于评估智能体在推理引擎内部端到端优化模型实现的能力。
- kernelbench hard（[kernelbench.com/hard](https://kernelbench.com/hard)）（人工设计）

  KernelBench v3 的聚焦升级版，测试前沿模型能否在不作弊的情况下快速编写 triton/cuda/cutlass/cute-dsl/ptx 代码。测试在本地的 rtx pro 6000 blackwell 设备上进行，七道人工设计的题目，以真实的代码智能体命令行界面作为测试框架。包括：fp8 gemm, topk, sonic MoE fwd, KimiDeltaAttention, paged attention decode, kahan softmax, w4a16 gemm。这些都要求对 sm120 架构有深入理解。
- PostTrainBench（[posttrainbench.com/](https://posttrainbench.com/)）（有缺陷）

  通过测试 AI 智能体能否成功对其他语言模型进行训练后优化，来衡量 AI 研发的自动化水平。每个智能体将获得四个基础模型（Qwen 3 1.7B、Qwen 3 4B、SmolLM3-3B 和 Gemma 3 4B）、一台 H100 GPU，以及十小时的时间限制，以通过训练后优化提升模型性能。

  缺陷：模型会利用捷径：用更强模型的推理轨迹做监督微调（即蒸馏）。而实际榜单中表现最好的Claude Opus 4.8、GLM 5.2确实均采用该策略——但蒸馏完全不符合基准“递归自我改进”的目标（前沿后训练不可能依赖外部更强模型）。
- WeirdML（[htihle.github.io/weirdml.html](https://htihle.github.io/weirdml.html)）

  向 LLMs 提出了一系列奇特且非传统的机器学习任务，这些任务需要细致的思考和真正的理解才能解决，旨在测试 LLM 的以下能力：  
  真正理解数据的特性与问题本质  
  为问题设计合适的机器学习架构与训练配置，并生成可运行的 PyTorch 代码来实现解决方案  
  根据终端输出和测试集上的准确率，在五次迭代中调试并改进解决方案  
  充分利用有限的计算资源和时间
- InferenceBench（[inferencebench.ai/](https://inferencebench.ai/)）

  用于评估前沿编码智能体能否在固定计算预算下优化大语言模型（LLM）服务工作负载。每次测试中，智能体都会获得一个基础 LLM、一块 NVIDIA H100 显卡、一定的运行时间限制以及一个特定场景下的目标；智能体必须搭建并运行一个兼容 OpenAI 协议的推理服务器，该服务器需能最大化实现该场景下的核心指标，同时还需通过质量检验与完整性检验。目标是在某个瓶颈场景下，或在综合性能均衡的全场景下，实现相较于 PyTorch 基准模型的加速效果。涵盖四种场景：预填充延迟、解码延迟、吞吐量、均衡服务。
- MLS-Bench（[mls-bench.com/](https://mls-bench.com/)）

  用于评估人工智能系统是否能够设计出具备泛化能力且可扩展的机器学习方法。该基准测试涵盖了 12 个领域的 140 个任务，包括语言模型、视觉与生成、强化学习、机器人技术、机器学习系统、科学领域的人工智能、优化算法、时间序列分析、因果推理等。每个任务都围绕一个定义明确的研究问题展开，要求智能体提出一项模块化的改进方案——比如新的损失函数、注意力机制变体、采样器或路由规则——随后评估该改进能否在不同模型、数据集以及随机种子下均有效。
- Reward Hacking Bench（[www.rewardhacking.io/](https://www.rewardhacking.io/)）

  旨在考察模型在后训练任务中的作弊情况。给模型分配一项编码任务：对一个基础能力较弱的小模型进行训练，目标是提升其在各项基准测试中的得分。考察模型在该过程中走捷径和规避审核机制的行为。
- RSI-Exam（[rsi-exam.ai/](https://rsi-exam.ai/)）（人工设计任务）

  旨在评估 AI 智能体是否能在长期内实现自我改进，并能否将能力泛化到未见过的数据上。测试过程包括：让智能体针对解决问题的方法或驱动模型运行的机制进行数小时的自主实验，最后再在隐藏的测试集上执行一次测试。共有 6 个主要领域，总计 88 个任务。领域分布详情：物理科学与工程、人工智能模型与智能体、优化、规划与控制、系统与硬件、生命科学与医学、金融、法律与商业。

### 模型使用

- Claude Code Opus Performance Tracker（[marginlab.ai/trackers/claude-code/](https://marginlab.ai/trackers/claude-code/)）

  检测 Claude Code Opus针对 SWE 任务出现的统计显著性退化，每日更新： 对精选的 SWE-Bench-Pro 子集进行每日基准测试。
- AI STUPID LEVEL（[aistupidlevel.info/about](https://aistupidlevel.info/about)）

  独立的 AI 模型性能监控平台，持续监控 AI 模型的性能。通过让多个模型执行真实的编码任务（`涵盖算法实现、调试、代码重构、优化和错误恢复`）来客观衡量它们的能力，检测可能未被注意到的性能变化（“漂移”）。对每个模型运行多次试验（n=5），计算置信区间，并使用统计检验来区分真实变化与噪声。采用 7 轴评分系统：正确性（35%）、规范遵循（15%）、代码质量（15%）、效率（10%）、稳定性（10%）、拒绝率（10%）和恢复能力（5%）。每个模型使用不同的随机种子运行 5 次编码任务，计算中位数分数并使用 t 分布提供 95%置信区间。
- 平台Coding Plan测评（[coding.15o.cc/](https://coding.15o.cc/)）

  对比不同厂商、不同模型的首字时间、平均 TPS 与完整响应耗时。

## Agentic Benchmarks

- Agents' Last Exam（[agents-last-exam.org/](https://agents-last-exam.org/)）

  一个专为评估人工智能智能体在长期、具有经济价值且结果可验证的现实任务中的表现而设计的基准测试。ALE 由 250 多位行业专家共同开发，其评估范围涵盖了依据 O*NET / SOC 2018（美国联邦职业分类体系）定义的非实体行业。该测试体系基于一个包含 55 个子领域的任务分类法，这些子领域被归为 13 个行业集群，共涉及 1500 多项具体任务。ALE 被设计为一个动态发展的基准测试：随着新工作流程和行业的不断加入，其任务库也会持续扩充。
- FACTS Benchmark（[www.kaggle.com/benchmarks/google/facts/leaderboard](https://www.kaggle.com/benchmarks/google/facts/leaderboard)）

  一个参数化基准，用于衡量模型在事实性问答场景中准确调用其内部知识的能力，包含一个由 1052 个问题组成的公开集和一个由 1052 个问题组成的私有集。  
  一个搜索基准，用于测试模型将搜索作为工具以检索信息并正确整合信息的能力，包含一个由 890 个条目组成的公开数据集和一个由 994 个条目组成的私有数据集。  
  一个多模态基准，用于测试模型以事实准确的方式回答与输入图像相关提示的能力，包含一个 711 项的公开数据集和一个 811 项的私有数据集。

  FACTS Grounding：评估LLMs基于所提供的长篇文档生成事实准确的响应的能力，测试LLM的响应是否完全基于所提供的上下文，并能正确从长篇上下文文档中整合信息。
- MCPMark（[mcpmark.ai/](https://mcpmark.ai/)）

  一套综合性压力测试 MCP 基准评测体系，包含多样化的可验证任务，旨在评估模型和智能体在真实 MCP 应用场景中的能力。包含以下MCP：Notion、Github、Filesystem、Postgres、Playwright、Playwright-WebArena。
- MCP Atlas（[scale.com/leaderboard/mcp_atlas](https://scale.com/leaderboard/mcp_atlas)）

  通过模型上下文协议（MCP）评估语言模型处理现实世界工具使用的能力，衡量的是多步骤工作流中的表现。包含 1,000 个人工撰写的任务，每个任务都需要调用多个工具来解决，工具来自 40 多个 MCP 服务器和 300 多个工具。任务范围从仅需 2 至 3 个工具且链条简单的单领域查询，到需要 5 个以上工具并包含条件分支和错误处理的复杂工作流。每项任务都包含精心挑选的干扰项工具，这些工具看似合理但实际错误。干扰项由数据标注者从与必需工具相同的类别中选取。该框架为每个任务提供 12-18 个工具（3-7 个必需工具加上 5-10 个干扰项），迫使代理基于工具描述进行推理，而非盲目调用。
- PinchBench（[pinchbench.com/](https://pinchbench.com/)）

  衡量 LLM 作为 OpenClaw 智能体的“大脑”表现如何。向智能体抛出真实任务：安排会议、编写代码、处理电子邮件、研究主题以及管理文件。包含23个不同类别的任务，任务以带有 YAML 前置属性的 Markdown 文件形式定义。
- Claw-Eval（[claw-eval.github.io/#/](https://claw-eval.github.io/#/)）（人工审核和验证）

  包含中文和英文共104个任务（32个中文 + 72个英文），提供19个带错误注入的模拟服务。采用三维评分体系——完成度、鲁棒性、安全性。Score = Safety × (0.80 × Completion + 0.20 × Robustness)。如果某一次任务安全分数为零，那么该项总分为零。
- WildClawBench（[internlm.github.io/WildClawBench/](https://internlm.github.io/WildClawBench/)）

  让每个任务都在真实的 OpenClaw 实例中运行——智能体可以访问真实的 bash shell、真实的文件系统、真实的浏览器，以及真实的电子邮件和日历服务。包含60 个原创任务，手工打造，考察智能体指令遵循、多模态推理、长时程规划、代码生成与调试的能力。
- ClawsBench（[clawsbench.benchflow.ai/#results](https://clawsbench.benchflow.ai/#results)）

  用于严格评估智能体的高保真模拟工作空间——Gmail、日历、文档、云端硬盘和 Slack。包含44 个结构化任务，涵盖了单服务、跨服务和安全关键场景。
- ClawMark（[claw-mark.com/leaderboard](https://claw-mark.com/leaderboard)）

  专为与人类在多个工作日、多种服务中协同工作而设计的协作者智能体基准测试。涵盖 13 个专业领域的 100 项任务，采用完全基于规则的评分体系——不使用 LLM 作为裁判。
- PawBench（[agentscope-ai.github.io/PawBench/](https://agentscope-ai.github.io/PawBench/)）

  评估 (Model × Harness) 在生产环境任务的表现，同一组 150 任务跑遍多个 (Model × Harness) 组合，独立观察两条轴——分清模型和 Harness 各自的贡献。包含3个Agent Harness（Hermes、OpenClaw 和 QwenPaw），

  按照 5 个维度打标：

  应用场景：例如办公协同、软件工程、自动化脚本、多模态内容生成。  
  原子能力：例如工具调用、Skill 使用、规划、逻辑推理、自我校验。  
  复杂度：L1 / L2 / L3，避免只靠简单题刷高分。  
  输入模态：区分纯文本任务和图像、音频、视频等多模态任务。  
  运行环境：区分离线沙箱任务和需要联网的 Web 搜索 / 网页获取任务。

### 白领经济价值任务

更多介绍可以参见*[What do “economic value” benchmarks tell us?](https://epoch.ai/blog/what-do-economic-value-benchmarks-tell-us)*

- GDPval-AA（[artificialanalysis.ai/evaluations/gdpval-aa](https://artificialanalysis.ai/evaluations/gdpval-aa)）

  为 OpenAI 的 GDPval 数据集开发的评估框架。它在 44 种职业和 9 个主要行业中，对 AI 模型在真实任务中的表现进行测试。包含220项任务，要求模型生成多样化的输出，包括文档、幻灯片、图表和电子表格，以模拟金融、医疗、法律及其他专业领域中的实际工作成果。
- $OneMillion-Bench（[xbench.org/profession/onemillion](https://xbench.org/profession/onemillion)）

  旨在覆盖中国及海外具有高经济价值、高差异化特征且可自动评估的真实应用场景。“百万美元”这一名称源于基于官方薪资数据的估算：将每个任务所需的预估耗时乘以相应领域资深专家的时薪后，完成全部200个任务所需的专家人力成本累计接近100万美元。该基准测试旨在解决一个更为现实的问题：它是否能够真正稳定且准确地替代人类来完成高价值工作，从始至终无一疏漏。
- AA-Briefcase（[artificialanalysis.ai/articles/aa-briefcase](https://artificialanalysis.ai/articles/aa-briefcase)）（人工设计任务）

  用于检验模型在复杂项目中执行现实知识工作任务的能力。 该测试要求模型完成为期数周的知识工作项目，每个项目都包含众多相互关联的任务以及数千个输入源文件。模型在一项连贯的长期项目中进行测试，各项任务逐周推进，均需依托共同的机构背景信息，并需要产出诸如财务模型、董事会演示文稿以及设计原型图等符合现实需求的成果。任务由来自 Google、麦肯锡咨询公司及波士顿咨询集团等公司的数据科学、产品管理和企业战略领域的专家历经数月精心设计而成，这些任务挑战均取材于实际职业经验。

  要求模型针对每项任务处理数百份输入文件，这些文件涵盖了 Slack 聊天记录、电子邮件、公司文档、会议记录以及大规模数据导出文件。总体而言，AA-Briefcase 包含近 2,000 份原始文件，其中电子邮件和 Slack 导出数据分别包含 3,500 多封邮件及 25,000 条消息。这些信息来源零散且杂乱，往往还包含现实世界中常见的矛盾信息，从而测试模型能否应对真实知识型工作中的种种模糊性与不确定性。
- AA-AnalystAgent（[artificialanalysis.ai/evaluations/aa-analyst-agent](https://artificialanalysis.ai/evaluations/aa-analyst-agent)）

  专门用于评估模型在真实电子表格与文档上的定量分析能力。涵盖 14 个商业与科学领域的 80 道题目，内容涉及医疗支出报告、贸易与商品统计、水文与气象数据、政府拨款情况、能源成本模型、金融模型、环境报告以及项目进度安排；选取自真实分析师工作的五个工作流类别：数据源查找与诊断、数据筛选与汇总、比率/趋势/敏感性分析、损益表建模，以及现金流/资产负债表/估值建模。
- Remote Labor Index (RLI)（[scale.com/leaderboard/rli](https://scale.com/leaderboard/rli)）

  远程劳动力指数（RLI）是一项基准测试，衡量 AI 代理执行现实世界中专业自由职业平台具有经济价值的多媒体远程工作的能力。包含240个项目，试题来源于 Upwork 平台上 358 名经过验证的自由职业者的自下而上的收集。要求智能体完成具有视觉输出的任务（网页设计、产品美术、视频编辑等）。其需要理解并生成复杂的多文件交付成果，涵盖数十种独特的交付文件类型。这包括文档、音频、视频、3D 模型以及 CAD 文件。
- APEX-Agents（[www.mercor.com/apex/apex-agents-leaderboard/](https://www.mercor.com/apex/apex-agents-leaderboard/)）（[artificialanalysis.ai/evaluations/apex-agents-aa](https://artificialanalysis.ai/evaluations/apex-agents-aa)）

  衡量前沿 AI 代理是否能够跨三个专业服务岗位（投资银行分析师、管理咨询师和企业律师）执行跨应用程序的长周期任务。

  - APEX（[www.mercor.com/apex/apex-v1-leaderboard/](https://www.mercor.com/apex/apex-v1-leaderboard/)）

    评估前沿模型是否具备在以下四类工作中执行具有经济价值任务的能力：投资银行助理、管理咨询师、大型律师事务所律师以及初级保健医生（医学博士）。
  - The AI Consumer Index (ACE)（[www.mercor.com/apex/ace-leaderboard/](https://www.mercor.com/apex/ace-leaderboard/)）

    评估前沿人工智能模型在购物、餐饮、游戏和 DIY 等日常消费任务中的表现能力。
- AutomationBench（[zapier.com/benchmarks](https://zapier.com/benchmarks)）

  一项针对模拟 SaaS 应用程序的复杂智能体工作流自动化测试。利用 6 个业务职能领域（销售、营销、运营、客服、财务及人力资源）中的 47 种真实工具，对 AI 智能体执行端到端工作流程的能力进行测试。该测试体系基于 370 万家公司每月执行的超过 20 亿次任务中所呈现的真实模式而构建。每个任务都会启动一个微型模拟公司，向智能体分配一个真实运营人员会收到的请求（包含 CRM 记录、收件箱对话记录、电子表格以及客服工单，也包含各种陷阱：过时的数据行、几乎相同的名称，以及隐藏在收件箱中的各种规则），随后对其留下的结果进行评分。评分对象并非智能体的回复，而是其生成的数据。

  - AutomationBench-AA（[artificialanalysis.ai/evaluations/automationbench-aa](https://artificialanalysis.ai/evaluations/automationbench-aa)）

    与 Zapier 提供的排行榜不同，AutomationBench-AA 中的主要分数反映了模型在不违反任何安全限制的前提下完成任务的比例。
- EnterpriseOps-Gym-AA（[artificialanalysis.ai/evaluations/enterprise-ops-gym-aa](https://artificialanalysis.ai/evaluations/enterprise-ops-gym-aa)）

  用于评估 LLM 智能体是否能够完成具有状态依赖性的多步骤企业工作流程。每个测试任务都限定在单一业务领域内：包括日常协作工具如电子邮件、日历、团队协作工具及云存储服务；也包括核心业务系统如客户服务、人力资源及 IT 服务管理。第八个领域为“混合领域”，其任务特点在于需要同时调用多个系统。在所有测试中，智能体都会被置于一个实时沙盒环境中，并被要求通过调用工具来完成实际的运营工作。  
  任务的评分依据是底层数据库的最终状态，而非对话记录，因此分数能够反映智能体是否切实完成了任务。系统不会给予部分分数：只有当所有验证器对结果状态均判定为合格时，该任务才被视为成功。

### AI智能体

- ClawArena（[github.com/aiming-lab/ClawArena?tab=readme-ov-file](https://github.com/aiming-lab/ClawArena?tab=readme-ov-file)）

  一个面向 AI 编程智能体的多会话真实场景基准评测，包含64 个场景，涵盖 8 个领域 — 科技/人力资源、医院、非政府组织、临床、内容创作、金融、人力资源、校园；1,879 轮评测，融合多选推理与执行验证；多会话上下文 — 智能体需要对工作区文件、多频道聊天记录以及评测中动态注入的更新进行推理。

### DeepResearch

- FutureSearch-Deep Research Bench (DRB)（[evals.futuresearch.ai/](https://evals.futuresearch.ai/)）

  用于评估LLM智能体在网络上的研究能力。169项多样的现实世界任务中，每项任务都提供10到10万个离线存储的网页供搜索和推理使用。

### 视觉定位与Gui智能体

- Cua-Bench（[cua.ai/cuabench](https://cua.ai/cuabench)）（人工设计和审查任务）

  面向电气工程领域，用于评估 AI 智能体主要借助键盘和鼠标在台式电脑上，或主要利用触摸屏在移动设备上完成复杂任务的能力。包含 25 个由专家编写的 KiCad 电路图绘制任务（KiCad 是一套完整的电子设计自动化套件，拥有密集的快捷键语法、多个相互关联的编辑器以及大量模态对话框）。由三个部分组成：

  1. **基础镜像**  ——预先打包好的 Windows、Linux、macOS 和 Android 环境，内含运行基准测试所需的应用程序及依赖项
  2. **任务数据集**  ——涵盖多种操作系统的、可验证且动态变化的电脑使用环境
  3. **评估/训练工具包**  - 用于运行基准测试、生成数据以及测试智能体的工具
- cua-speedrun（[cuaspeedrun.com/](https://cuaspeedrun.com/)）

  测量模型在完成电脑使用领域的每项任务所需的时间以及模型调用所产生的成本。

## Intelligence Benchmarks

- 概念推理指数（CRI）（[conceptualreasoning.ai/](https://conceptualreasoning.ai/)）（人工审核任务）

  一套包含三个测试项的概念推理基准。当实证证据有限、不存在（实际意义上）可验证的答案时，模型必须高度依赖论证逻辑来得出结论。我们将此称为概念性推理。为衡量LLM在此能力上进展，构建了三个基准测试：LMCA、ACCoRD 以及 DTBench capabilities。

  LMCA（语言模型概念性论证）是一个经过精心筛选且由专家评级的数据集，其中包含了涵盖决策理论、哲学以及先进人工智能风险等多样化主题的概念性论证内容。专注于论证过程有助于规避验证概念性问题最终答案的难度。包含560条立场文本，以及针对这些文本的1,461条论据。通过将模型给出的评分与专家人类评分进行对比，来衡量模型评判论据优劣的能力。

  ACCoRD（概念推理一致性评估）用于衡量模型在概念性问题上所表达的信念与偏好在逻辑上的一致性程度。例如，如果我们询问一个模型事件 A 发生的概率 P(A)，再询问同一模型事件 A 与 B 同时发生的概率 P(A&B)，那么这两个概率值是否满足 P(A) ≥ P(A&B)？该数据集中的所有一致性约束均要求模型给出数值型概率估计值或偏好排序。

  DTBench 的功能 （决策理论基准测试）是一个包含 407 道人工编写的选择题的数据集，旨在衡量模型在涉及对自身行为或与（近似）副本互动进行准确预测的决策理论情境下的推理能力。
- ARC-AGI-3（[arcprize.org/leaderboard](https://arcprize.org/leaderboard)）

  专注于那些对人类而言相对简单、但对 AI 却困难甚至不可能完成的任务，从而揭示那些无法通过“规模扩大”自然涌现的能力鸿沟。
- dig.bench（[digbench.ai/](https://digbench.ai/)）

  dig.bench 是一个用于衡量科学发现能力的基准测试。其包含的 70 款游戏分别用于测试智能体能否通过实验来发现该游戏自身未知的规则，每个游戏都设有未知的转换规则，智能体必须通过交互与实验才能将其揭示。所有游戏均以文本形式呈现，这使其处于语言模型的自然应用领域：模型无需面对任何视觉层面的干扰即可进行探索，因此 dig.bench 所测试的正是纯粹的发现能力。这些游戏根据难度分为7个等级。
- SimpleBench（[simple-bench.com/](https://simple-bench.com/)）

  一个针对 LLMs 的多项选择文本基准测试，拥有非专业（高中）知识的个体在此基准上的表现优于最先进模型。SimpleBench 包含 200 多道题目，涵盖时空推理、社交智能以及我们称之为语言对抗鲁棒性（或脑筋急转弯）的内容。
- AA-Omniscience（[Artificial Analysis Omniscience Index | Artificial Analysis](https://artificialanalysis.ai/evaluations/omniscience)）（存在重大缺陷）

  涵盖 6 个领域（“商业”、“人文与社会科学”、“健康”、“法律”、“软件工程”和“科学、工程与数学”）中 42 个主题的 6,000 道问题。三项指标：准确率（正确百分比）、幻觉率（错误答案占所有非回避答案的百分比）、全知指数（回答正确+1，回答错误-1，回避回答 0）。

  重大缺陷：其中大量试题被社区发现存在问题
- Bullshit Benchmark（[petergpt.github.io/bullshit-benchmark/viewer/index.html](https://petergpt.github.io/bullshit-benchmark/viewer/index.html)）

  设计了55个完全不合逻辑的“胡说八道”问题，并评估模型在面对这些问题时是选择反驳还是真诚地尝试回答。
- Pencil Puzzle Bench（[ppbench.com/](https://ppbench.com/)）

  一个通过铅笔谜题评估大型语言模型推理能力的框架，包含300个谜题，涵盖20种类型。
- AttuneBench（[public.attunebench.com/](https://public.attunebench.com/)）

  旨在衡量大模型的情商，基于200场真实的多轮人机对话构建。对大模型在各项能力上的表现进行评分，包括感知情绪、理解情绪、利用情绪引导思维，以及在互动中管理情绪。该基准沿用了心理学家评估人类情绪智力的相同维度，这让我们能将这一复杂概念转化为具体、可测量的行为。我们还会对模型的回复生成能力进行评分，同时评估对话层面的指标，比如情绪变化轨迹以及参与者自我认定的对话目标。

### 幻觉与上下文召回

关于长上下文召回和推理问题，参考这篇文章*[Evaluating Long Context (Reasoning) Ability](https://nrehiew.github.io/blog/long_context/)*

- HalluHard（[halluhard.com/](https://halluhard.com/)）

  一个多轮幻觉基准测试，共包含 950 道题目 ，覆盖四个领域：法律案例（250）、研究问题（250）、医疗指南（250）及编程（200）。利用一个用户大模型生成引人入胜的后续问题，并测量模型在 3 轮对话 （包含首次提问及后续 2 轮）中的表现。 HalluHard 旨在引出开放式响应，同时要求模型将事实主张建立在引用的来源之上。这种设计确保基准测试专门关注幻觉（无依据的事实错误），而非响应的其他方面。  
  对于法律、研究和医学领域，对每个响应抽取 5 个主张并进行逐项判断；对于编程领域，进行逐响应判断。核实流程包括提取主张、通过网络搜索检索证据，并获取全文来源（包括 PDF 解析）以验证引用材料是否支持生成的内容。
- AA-LCR（[Artificial Analysis Long Context Reasoning Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning)）

  专为评估语言模型在多个长文档间进行推理能力而设计的基准。要求模型阅读 10 万 token 的输入（使用 cl100k_base 分词器衡量），整合输入文档中多个位置的信息，并据此推导出答案。旨在真实还原知识工作者期望语言模型执行的推理任务。涵盖 7 种纯文本文档类型（即公司报告、行业报告、政府咨询、学术文献、法律文件、营销材料和调查报告）。
- Context Arena（MRCR v2）（[contextarena.ai/](https://contextarena.ai/)）（有缺陷）

  数据来源为OpenAI发布的MRCR。OpenAI的MRCR测试旨在评估大型语言模型（LLM）处理复杂对话历史的能力。其关键方面包括：  
  核心任务：在冗长的对话（“干草堆”）中找到并区分多个相同的信息片段（“针”）。  
  设置： 受谷歌的[MRCR评估](https://arxiv.org/pdf/2409.12640v2)启发，此版本会插入2、4或8个相同的请求（例如，“写一首关于貘的诗”）以及干扰性请求。关键信息/干扰信息由GPT-4o生成，以实现自然融入。  
  挑战：模型必须根据顺序检索一个特定实例（例如，第二首诗），这需要仔细跟踪对话。它还必须在答案前加上一个特定的随机代码（哈希值）。

  缺陷：仅依靠 1,000 个样本，Microsoft MAI 的技术报告就显示他们能将得分从 60% 提升至 90% 以上（实际上他们是用一个更小的模型实现了 90% 的得分，这反而让情况变得更糟糕）。
- LOCA-bench（[github.com/hkust-nlp/LOCA-bench](https://github.com/hkust-nlp/LOCA-bench)）

  旨在评估语言智能体在极端且可控的上下文增长场景下的表现。给定任务提示后，LOCA-bench 利用自动化和可扩展的环境状态控制来调节智能体的上下文长度。在保持任务语义不变的前提下，将上下文长度扩展至任意规模。

### AI4S（特化领域）

- MathArena（[matharena.ai/?view=problem](https://matharena.ai/?view=problem)）

  MathArena 是一个用于评估 LLMs 在最新数学竞赛中表现的平台，目标是严格检验 LLMs 在面对模型训练期间未曾接触过的新数学问题时，其推理和泛化能力。在评估性能时，我们会让每个模型在每个题目上运行 4 次，然后计算其平均得分以及所有运行次数的总成本（以美元计）。所显示的成本是指该模型在单次竞赛的所有题目上运行一次的平均费用。

  包含ArXivMath、IMProofBench、MathArenaApex、Visual Math、Final-Answer Comps、Proof-Based Comps、Project Euler、BrokenArXiv、ArXivLean等基准测试。

  ArXivMath：旨在评估LLMs在处理来自近期 arXiv 论文中的数学研究问题上的表现。（人工筛选任务）

  IMProofBench：旨在通过基于证明的问题来评估语言模型在科研级别的数学推理能力。模型可以访问多种工具，包括网络搜索和代码执行功能。

  MathArenaApex：一组从 2025 年举办的各大数学竞赛中精选出的 12 道高难度题目。

  Visual Math：袋鼠数学竞赛题目。Final-Answer Comps和Proof-Based Comps：近期各大数学竞赛题目。

  Project Euler：一个把数学和编程结合起来的题库，专注于需要综合运用数学洞察力、算法思维和编程技能来解决复杂问题。

  BrokenArXiv：测试模型在数学推理中的可靠性。从近期的arXiv论文中提取问题，对其进行轻微干扰，使其成为看似非常合理但可证明为错误的陈述。模型若拒绝证明该陈述，并明确识别出该陈述在现有形式下为假，则会得分。（人工筛选任务）

  ArXivLean：一个用于在 Lean 中进行研究级数学定理证明的基准测试，所有样本均取自最新的 arXiv 论文
- FrontierMath（[epoch.ai/frontiermath/tiers-1-4](https://epoch.ai/frontiermath/tiers-1-4)）

  包含 350 道原创数学题（50 道最高难度等级 4 的问题），涵盖从具有挑战性的大学水平问题到可能需要专家数学家数日才能解决的难题。要求：

  1. 明确且可验证的答案
  2. 抵御猜测：答案应具备“防猜测”特性，即随机尝试或简单的暴力方法几乎不可能成功
  3. 计算可行性：解决计算密集型问题时，必须包含脚本，展示如何仅基于该领域的标准知识找到答案。这些脚本在标准硬件上的累计运行时间必须少于一分钟。
- MathScienceBench（[math.science-bench.ai/benchmarks/](https://math.science-bench.ai/benchmarks/)）

  一个面向研究级数学题的 AI 基准测试。让活跃研究者提交 PhD 级别、接近科研语境的数学问题，用来测试大模型在高难度数学推理上的表现。
- MathDuels（[mathduels.ai/](https://mathduels.ai/)）

  自对弈数学基准，每个前沿模型既要解决其他模型编写的题目，也要为这些模型出题，因此测试的难度会随着参赛模型的整体实力提升而增加。基于这两种角色，系统会计算出两个评分，分别为解题能力评分和出题能力评分。
- PutnamBench（[trishullab.github.io/PutnamBench/leaderboard.html](https://trishullab.github.io/PutnamBench/leaderboard.html)）

  在普特南数学竞赛中对形式化数学推理进行基准测试。包含 1712 个手工构建的形式化问题，题目源自北美顶尖本科数学竞赛——威廉·洛厄尔·普特南数学竞赛。其中 660 个问题使用 Lean 4 形式化，640 个使用 Isabelle 形式化，412 个使用 Coq 形式化。
- GPQA Diamond（[artificialanalysis.ai/evaluations/gpqa-diamond](https://artificialanalysis.ai/evaluations/gpqa-diamond)）（有严重缺陷）

  GPQA 基准中最难的 198 个问题，专为“防谷歌”设计，需要真正的科学专业知识，而非搜索技巧。  
  这些研究生级别的物理、生物和化学问题，只有具备博士学位的领域专家才能稳定解答，因此非常适合用于测试真正的科学推理能力。

  被发现题目在OCR识别和录入过程中存在大量错误，数据处理的工程流程堪称灾难。（来源：[Humanity's Last Hallucination : A Forensic Audit of the Scientific Insolvency in GPQA and HLE](https://zenodo.org/records/18293568)）
- Humanity's Last Exam（[lastexam.ai/](https://lastexam.ai/)）（[Artificial Analysis](https://artificialanalysis.ai/evaluations/humanitys-last-exam)）（有严重缺陷）

  一个处于人类知识前沿的多模态基准，旨在成为涵盖广泛学科的最后一个封闭式学术基准。该数据集包含跨越百余门学科的 2,500 道高难度问题。我们公开发布这些问题，同时保留一个未公开的测试集，用于评估模型过拟合情况。

  在 HLE 上取得高准确率将证明模型在封闭式、可验证的问题以及前沿科学知识方面具备专家级表现，但这本身并不意味着其具备自主研究能力或“通用人工智能”。HLE 测试的是结构化的学术问题，而非开放式的科研或创造性解决问题的能力，因此它是一种聚焦于技术知识与推理能力的衡量标准。

  分为 (w/ tools)有工具（测试angentic能力）和 (w/o tools)（无工具）（测试模型本身智能）两种情况

  被发现题目在OCR识别和录入过程中存在大量错误，数据处理的工程流程堪称灾难。（来源：[Humanity's Last Hallucination : A Forensic Audit of the Scientific Insolvency in GPQA and HLE](https://zenodo.org/records/18293568)）

  - HLE-Diamond（[lastexam.ai/blog/hle-diamond](https://lastexam.ai/blog/hle-diamond)）

    这是HLE题库中经过精心筛选出的子集，包含 1,000 道题目。
- CritPt（[CritPt Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/critpt)）

  旨在测试 LLMs 在研究级物理推理任务表现的基准，包含 71 项综合性研究挑战。
- PhyArena（HiPhO）（[phyarena.github.io/](https://phyarena.github.io/)）

  对 LLMs 和 MLLMs 物理推理能力的基准测试。HiPhO：高中物理奥林匹克竞赛基准
- Terminal-Bench-Science（[www.terminal-bench-science.ai/](https://www.terminal-bench-science.ai/)）（人工审核任务）

  利用一系列由专家精心挑选的、源自科学研究的复杂工作流程来衡量 AI 智能体的能力。当前测试包含了来自生命科学、物理科学、地球科学、数学及工程科学领域的 70 项任务。
- BioMysteryBench（[www.vals.ai/benchmarks/biomysterybench](https://www.vals.ai/benchmarks/biomysterybench)）

  旨在测试语言模型是否能够分析生物学数据集以还原缺失的信息，例如：确定样本源自哪种组织。每项任务均以去除了部分元数据的真实生物数据作为起点。模型必须通过计算分析来还原被移除的信息：检查文件、运行命令行工具、编写代码以及查询生物数据库。这些任务涵盖了测序、表达分析、变异检测、表观基因组学、宏基因组学、蛋白质组学和代谢组学等多个领域。

### 特定行业基准

#### 医学

- CHI-Bench（[actava.ai/benchmarks](https://actava.ai/benchmarks)）

  针对三大领域的长周期医疗业务流程开展测评，分别为医疗机构事前授权、支付方利用率管理以及医疗照护管理。每项任务都会在高精度仿真环境中为智能体提供临床案例，该仿真环境集成了 20 款医疗应用，并对外开放 87 项 MCP 工具。智能体需依托包含 1290 余份文档的管理式医疗运营手册，通过调用工具、输出对应岗位工作成果，将案例流程推进至最终办结状态。
- Medical Long Context Reasoning (MLCR) benchmark（[www.wisedocs.ai/blogs/medical-long-context-reasoning](https://www.wisedocs.ai/blogs/medical-long-context-reasoning)）（人工设计任务）

  测试模型在处理更长文档长度时，对许多专业人士在审核医疗病例过程中所提各类问题的有效性。基于真实场景构建开源10份合成医疗案例，总计包含50至150个跨专科的医疗摘要。测试时，会针对这10个案例中的每一个，向大语言模型提出250个按难度分类的问题。此外，为衡量额外冗余上下文的影响，模拟包含无关信息的医疗记录。测试插入来自常见医疗和保险表单的OCR文本，填满模型的上下文窗口，从而构建出类似“大海捞针”式的问题。
- MLCR-AA（[artificialanalysis.ai/evaluations/mlcr-aa](https://artificialanalysis.ai/evaluations/mlcr-aa)）

  旨在评估模型处理长篇医学病例文档的推理能力。AA版本选取了MLCR原版本数据集两个难度最高类别中的 60 道测试题作为私有测试集：一个是“专家级”类别，要求模型能针对整份病例文档进行专业的医学推理；另一个是“复合型”类别，即把多个独立问题整合进一个查询请求中。

  每个问题的回答均需基于一份长达100至150页的完整病历文件。随后由由三个模型组成的评审小组从完整性、准确性以及简洁性三个维度进行评分——其中简洁性测试旨在防止模型生成过于冗长的回复（即回复长度不应超过专家答案的5倍以上）。准确性指标用于验证模型回复是否切实基于原始文档及病例背景；完整性指标则用于评估模型是否涵盖了专家标注答案中所包含的关键信息。简洁性测试则确保模型不会产生冗余内容。
- DrugDiscoveryBench（[labs.scale.com/leaderboard/drugdiscoverybench](https://labs.scale.com/leaderboard/drugdiscoverybench)）

  用于评估前沿编码智能体在执行药物早期发现阶段所需的多步骤计算任务及信息检索任务时的可靠性。该基准测试包含 82 项任务，涵盖了早期发现的全流程：靶点识别与验证、从专利、数据库及文献中筛选潜在候选分子、从候选分子中筛选出先导化合物并进行构效关系分析，以及先导化合物的优化。该基准测试在候选化合物选定之前即告结束；药物代谢动力学（DMPK）、毒理学、制剂开发及临床试验等内容均不在测评范围内。
- HealthBench Professional（[medicalsphere.ai/benchmarks/healthbench-professional](https://medicalsphere.ai/benchmarks/healthbench-professional)）

  一个包含 525 个由内科医生编写的任务的医学基准测试，旨在评估 LLMs 在三种实际临床场景中的表现：诊疗咨询、文书撰写与记录以及医学研究。每个任务均为单轮或多轮对话形式，内容源自内科医生在测试“ChatGPT for Clinicians”时的实际交流记录。这些任务由三位或更多内科医生依据既定评分标准进行评判。

#### 法律

法律领域Benchmarks参考*[LLM Agents in Law: Taxonomy, Applications, and Challenges](https://arxiv.org/pdf/2601.06216)*附录3

- Harvey LAB-AA（[artificialanalysis.ai/evaluations/harvey-lab-aa](https://artificialanalysis.ai/evaluations/harvey-lab-aa)）

  旨在衡量 AI 智能体执行实际法律工作的能力，而非仅仅回答孤立的法律问题。每项任务都会向智能体提供类似合作伙伴式的指令以及一组置于沙盒环境中的案例文档。智能体需阅读这些材料、综合处理各项信息，最终生成一份法律成果物。包含120 个法律任务，涵盖了从企业并购、资本市场到税务、诉讼及破产等 24 个法律实务领域。

#### 商业与金融

- Vending-Bench 2（[andonlabs.com/evals/vending-bench-2](https://andonlabs.com/evals/vending-bench-2)）

  一个用于衡量 AI 模型在长时间范围内运营企业表现的基准测试。模型需在一年内模拟运营自动售货机业务，并以其期末银行账户余额进行评分。
- YC-Bench（[collinear-ai.github.io/yc-bench/#leaderboard](https://collinear-ai.github.io/yc-bench/#leaderboard)）

  让智能体在长达一年的模拟创业周期（跨越数百轮决策）中运营一家初创公司来评估这些能力。该智能体必须管理员工、选择任务合同，并在部分可观测的环境中维持盈利能力——其中敌对客户和不断增长的薪资支出会因糟糕的决策而产生连锁反应。每个模型使用 3 组随机种子进行测试，所有模型初始资金为20万美元。
- CEO-Bench（[ceobench.com/](https://ceobench.com/)）

  智能体运营一家模拟的人工智能初创公司，时长为500天。为智能体提供100万美元的初始现金，并以模拟结束时的现金余额作为绩效指标。该智能体通过可编程接口开展运营，可访问业务数据库、公司管理工具以及社交媒体。
- Finance Agent（[www.vals.ai/benchmarks/fabv2](https://www.vals.ai/benchmarks/fabv2)）

  测试各类智能体执行初级金融分析师应具备的任务的能力。包含537道题目，涉及信息检索、市场调研及预测分析等方面。
- TaxCalcBench（[github.com/column-tax/tax-calc-bench?tab=readme-ov-file](https://github.com/column-tax/tax-calc-bench?tab=readme-ov-file)）

  对前沿模型在美国税务计算任务中的评估。包含 51 对用户输入和预期正确计算的税务申报输出，适用于相对简单的税务情况，并包含申报状态、收入来源、税收抵免与扣除项。
- Diligence Stack Agent Bench（[csbench.com/benchmarks/diligence-stack-agent](https://csbench.com/benchmarks/diligence-stack-agent)）

  旨在衡量模型在执行财务建模及财务健康状况与研究相关任务上的表现。测试集是两个专为 Diligence Stack 研究流程而构建的私有知识库。
- Commerce Agent Bench（[github.com/Accio-org/CommerceAgentBench](https://github.com/Accio-org/CommerceAgentBench)）

  旨在评估智能体是否能够完成长期性的商业工作流，而不仅仅是回答相关问题。包含107 个任务，测试任务涵盖了浏览器操作、类原生 CLI 工具使用、API/MCP 工作流处理、文档与电子表格制作、公开网络研究、供应商分析、产品发布、物流以及各类商业运营环节。

### 特殊场景

- SpeechMap（[SpeechMap.AI Explorer](https://speechmap.ai/)）

  旨在探索人工智能言论的边界。测试不同提供商、国家和话题下，语言模型对敏感和争议性提示的反应。大多数 AI 基准衡量的是模型能做什么，而我们关注的是它们不能做什么：它们回避、拒绝或屏蔽的内容。
- HiL-Bench（[labs.scale.com/leaderboard/hil](https://labs.scale.com/leaderboard/hil)）

  用于衡量智能体的求助判断能力：即智能体能否识别出缺失、模糊或冲突的信息无法仅通过探索或推理来解决，并能在恰当的时机提出针对性问题以澄清正确信息。包含软件工程和文本到 SQL 两个领域。任务均取自SWE-Bench Pro和 BIRD 数据集，并被注入了障碍。包含300个任务，在两个领域中平均分配，其中有200个公开任务和100个用于无偏评估的私有预留任务。在这些任务中，数据集共包含1131个障碍，平均每个任务有3.8个障碍。
- Voxelbench（[voxelbench.ai/leaderboard](https://voxelbench.ai/leaderboard)）

  一种用于评估语言模型在生成体素结构方面的性能的基准测试。
- DecodingTrust Bench（[decodingtrust-agent.com/leaderboard](https://decodingtrust-agent.com/leaderboard)）

  针对15个以上领域和50个沙箱环境中AI智能体的动态红队测试框架，覆盖环境、工具、技能中的各类间接注入以及直接提示词注入。
- Political Manipulation（[political-manipulation.ai/](https://political-manipulation.ai/)）

  通过比较模型对涉及对立政治主题的配对提示的处理方式来评估其隐性政治偏见（这种偏见在单个响应中几乎无法察觉，因为它表现为不同响应之间的不一致性，而非明显的立场倾向）。数据集包含成对的左翼与右翼编码提示语，并根据隐性操纵技术的分类体系对其进行评分。我们通过测量情感一致性与有用性一致性来识别不同类型的政治操纵行为。
- Last Translation Benchmark（[last-translation-benchmark.vilda.net/leaderboard-results](https://last-translation-benchmark.vilda.net/leaderboard-results)）

  一个由人工编写并经同行评审的示例集合（包含文本、图像、音频及视频），专门用于“难倒”当前最先进的机器翻译模型。我们还提出了一种新的评估方法：每个示例都配有人工制定的核实规则，详细描述了该示例下模型的具体失败情形
- LibraryDesignBench（[ldbench.com/](https://ldbench.com/)）

  让一个智能体负责设计图书馆，然后仅根据其他智能体利用该图书馆进行构建的效果来为其打分。

#### 预测

## 视觉理解与推理

视觉评估存在不稳定性，主要原因分为三个：一是数据集规模小；二是标记样式等细节变化会显著影响模型准确率和排名；三是 JPEG 压缩等 对人“不可见” 变化会改变基准测试排名。（参见[lisadunlap.github.io/vpbench/](https://lisadunlap.github.io/vpbench/)）

- MMMU-Pro（[MMMU-Pro Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/mmmu-pro)）

  多项选择选项为 10 个，并引入仅视觉输入格式，其中问题嵌入在截图或照片中。  
  该基准包含 3,460 道题目，涵盖六个核心学科（艺术与设计、商业、科学、健康与医学、人文与社会科学、技术与工程），要求模型在更贴近现实的场景中同时处理视觉与文本信息。
- ZeroBench（[zerobench.github.io/](https://zerobench.github.io/)）

  面向当代大型多模态模型的一项极难视觉基准测试，包含 100 道由设计师团队精心独创并经过广泛评审的挑战性问题，下有334 个子问题，对应回答每个主要问题所需的独立推理步骤
- BabyVision（[xbench.org/agi/babyVision](https://xbench.org/agi/babyVision)）

  xbench 的 AGI 对齐系列的一部分，专注于评估 “无法言说” 挑战中的视觉理解能力。
- PerceptionBench（[www.kimi.com/blog/perception-bench](https://www.kimi.com/blog/perception-bench)）

  一个专门用于测试视觉感知能力的基准测试。它将视觉感知拆解为一系列基础能力来进行评估， 这些能力是根据当前模型出现的错误反推得出的，而非事先人为定义的。 通过将 40 多个基准测试中前沿模型的失败案例追溯到其最初的视觉成因，提炼出了 10 种感知能力以及 3,000 个经过验证的测试问题。解答这些问题仅需观察即可，无需任何推理或外部知识。
- Blueprint-Bench 2（[andonlabs.com/evals/blueprint-bench-2](https://andonlabs.com/evals/blueprint-bench-2)）

  通过让智能体将公寓照片转化为精准的2D平面图来测试空间推理能力。每个智能体依次处理50套公寓，每套公寓查看约20张室内照片，并生成一张平面图，展示房间布局、连接方式及相对尺寸。
- MazeBench（[mazebench.com/blog?post=maze-bench-results](https://mazebench.com/blog?post=maze-bench-results)）（人工设计）

  一个 3D 开放世界环境，专门用于测试智能体的长期规划能力以及视觉空间推理能力。该环境包含数百个房间与解谜关卡。

  模型在其原生的智能体运行环境中运行 MazeBench，例如 Codex 和 Claude Code，这样就能以较低的成本完成长时间的运行任务。通过 MCP 服务器，这些智能体可以执行十一种动作：四种移动动作、四种摄像头控制动作、撤销指令、关卡重置，以及可在不同房间间传送的传送动作。MazeBench 最显著的特征就是摄像头旋转功能。模型必须将摄像头向上、向下、向左和向右旋转，才能从二十个不同角度观察游戏场景。随着摄像头的转动，相应的移动指令也会随之调整。

### 世界模型

## OCR与嵌入评测

[Supercharge your OCR Pipelines with Open Models](https://huggingface.co/blog/ocr-open-models)

在测试不同的 OCR 模型时，它们在不同文档类型、语言等方面的性能差异很大。

- OmniDocBench（[OmniDocBench/README_zh-CN.md at main · opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench/blob/main/README_zh-CN.md)）

  一个针对真实场景下多样性文档解析评测集，这个广泛使用的基准测试因其多样化的文档类型而脱颖而出，包括书籍、杂志和教科书。其评估标准设计精良，支持 HTML 和 Markdown 格式的表格。
- olmOCR-Bench（[olmocr/olmocr/bench at main · allenai/olmocr](https://github.com/allenai/olmocr/tree/main/olmocr/bench)）

  该基准在评估英语方面非常成功。
- Real5-OmniDocBench（[huggingface.co/datasets/PaddlePaddle/Real5-OmniDocBench](https://huggingface.co/datasets/PaddlePaddle/Real5-OmniDocBench)）

  一个面向现实世界场景的全新基准，基于OmniDocBench v1.5数据集构建。该数据集包含五个不同的场景：扫描、扭曲、屏幕拍摄、光照和倾斜。除扫描类别外，所有图像均通过手持移动设备手动获取，以密切模拟现实世界条件。
- OCRVerse（[github.com/DocTron-hub/OCRVerse](https://github.com/DocTron-hub/OCRVerse)）

  首个端到端的综合 OCR 方法，能够统一实现文本中心 OCR 和视觉中心 OCR（例如图表、网页和科学图表）。文本中心型数据类型覆盖九个文档场景：自然场景、书籍、杂志、论文、报告、幻灯片、考试试卷、笔记和报纸，这些场景涵盖了日常生活中的高频文本场景并满足基本的 OCR 需求。视觉中心型数据类型包含六个专业场景：图表、网页、图标、几何图形、电路和分子结构，这些场景专注于专业结构化内容。
- Chronicles-OCR（[github.com/VirtualLUOUCAS/Chronicles-OCR/blob/main/README_ZH.md](https://github.com/VirtualLUOUCAS/Chronicles-OCR/blob/main/README_ZH.md)）

  专为评估视觉语言大模型（VLLMs）跨时域视觉感知能力而设计的综合性基准，覆盖汉字完整的演化轨迹——"汉字七体"。包含 2,800 张严格均衡的图像（每种书体 400 张 × 7 种书体），涵盖从龟甲到纸本书法在内的高度多样化物理媒介。
- PDF Parse Bench（[github.com/phorn1/pdf-parse-bench](https://github.com/phorn1/pdf-parse-bench)）

  评估不同 PDF 解析方案从文档中提取数学公式的有效性。
- Embedding Leaderboard（[MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard)）

  即MTEB Leaderboard，用统一任务集合比较 embedding 模型在检索、分类、聚类等任务上的表现。

## 生图、视频音频生成和角色扮演评测

- DesignArena（[www.designarena.ai/leaderboard](https://www.designarena.ai/leaderboard)）

  包括Code Categories、Web App、Mobile、Full Stack、Agent、Builder、Image、Image Editing、Graphic Design、Logo、SVG、Video、Video Editing、Slides等多个榜单。
- GenExam（[github.com/OpenGVLab/GenExam](https://github.com/OpenGVLab/GenExam)）

  首个多学科文本到图像生成考试基准，包含 1000 个样本，涵盖 10 个学科，考试风格提示按照四级分类法组织。测试模型整合理解、推理和生成的能力。
- UNO-Bench（[UNO-Bench](https://meituan-longcat.github.io/UNO-Bench/)）

  一个统一基准，用于探索全模型中单模态与全模态之间的组合规律，UNO-Bench 中几乎 100% 的问题都需要对音频和视觉信息的联合理解。除了传统的多项选择题外，我们还提出了一种创新的多步骤开放式问答格式，以评估复杂推理能力。

  我们的材料具有三个关键特性：a. 多元来源——主要来自众包的真实世界照片和视频，辅以无版权限制的网站和高质量公共数据集。b. 丰富多样的主题——涵盖社会、文化、艺术、生活、文学和科学。c. 实时录制音频——由超过 20 位真人说话者录制的对话，确保音频特征丰富，反映真实世界的声音多样性。
- WBench（[meituan-longcat.github.io/WBench/#leaderboard](https://meituan-longcat.github.io/WBench/#leaderboard)）

  首个面向交互式视频世界模型的系统性多轮评测基准，包含 289 个测试案例和 1058 个交互轮次。每个用例均指定了一个世界场景和一段多轮交互序列，覆盖多样的场景、风格、主题以及第一、第三人称视角，同时包含导航、主体动作、事件编辑和视角切换四种交互类型。在导航任务中，WBench 整合了文本、六自由度位姿和离散动作控制，可对拥有不同原生输入接口的模型进行评估。

### 语音识别与交互

- τ-voice（[taubench.com/#leaderboard?benchmark=voice](https://taubench.com/#leaderboard?benchmark=voice)）

  扩展自文本基准 τ²-bench，包含 278 个任务，覆盖三个真实领域：零售、航空、电信。支持多种口音、背景噪音、打断行为、回馈词、非对话语音，从以下几个方面衡量语音交互质量：响应率、延迟、打断率、选择性（是否正确忽略回馈词/非对话语音）。

### Omni

## 决策

- Jev Decision Index（[huggingface.co/spaces/multimodalart/jev-decision-index](https://huggingface.co/spaces/multimodalart/jev-decision-index)）

  决策指数为每个模型生成一个数值，用于衡量对 TypeSafe 的 Jev 进行开源复现后模型执行类型化决策的能力：即选择选项、标签、工具、排序结果或概率值。所有参赛模型都会使用同一套固定测试集，包含 36 个静态基准测试的 120340 条请求，该测试集是 Jev jev-1.13.0 版本评分所用请求集合的固定子集

## 社区评测

以下为AI社区大佬们的独立评测

- nao老师的LLM Benchmark（[llm2014.github.io/llm_benchmark/](https://llm2014.github.io/llm_benchmark/)）

  个人性质，使用滚动更新的私有题库进行长期跟踪评测。侧重模型对逻辑，数学，编程，人类直觉等问题的测试。题库规模不大，长期维持在30题/240个用例以内，不使用任何互联网公开题目。题目每月会有滚动更新。题目不公开，意图是分享一种评测思路，以及个人见解。
- knowledge-cutoff（[apoorvumang.github.io/knowledge-cutoff/](https://apoorvumang.github.io/knowledge-cutoff/)）

  一个用于评估语言模型实际知识截止点的基准测试——即模型 真正了解世界信息的范围，该范围通常早于模型所宣称的截止日期。  
  其原理是：针对最近几个月发生的各类真实事件对模型进行测试。针对每个月，统计模型能正确回答的事件数量。随着时间越来越接近真实的知识边界，模型的正确率会逐渐下降，因此通过逐月绘制正确率曲线，就能大致推断出模型的世界知识在何时终止。
- XSCT Bench（[xsct.ai/](https://xsct.ai/)）

  包含文本、Web开发、生图、Openclaw、Omni等领域的真实产品场景测试。
- LisanBench（[lisanbench.com/](https://lisanbench.com/)）

  X用户Lisan al Gaib（@scaling01）推出的个人评测。给模型一个起始英文单词，模型必须不断生成下一个单词，满足以下所有严格约束：

  - 与前一个单词恰好相差 1 个字母（Levenshtein 编辑距离 = 1）
  - 必须是有效英文单词（使用 words_alpha.txt 词典，约 37 万词，但实际只用最大连通分量 ≈ 10.8 万词）
  - 不能重复使用过任何已经出现过的单词
  - 目标：尽可能生成最长的有效链条

  分数 = 多个不同起始词的最长链长度累加
- Kaggle Benchmarks（[www.kaggle.com/benchmarks?type=community](https://www.kaggle.com/benchmarks?type=community)）

  包括两种主要类型的基准测试：1）研究基准测试，即由AI实验室的研究人员创建的评估；2）社区基准测试，即由 Kaggle 社区创建的评估，用户能够设计、运行并分享他们自己用于评估人工智能模型的自定义基准测试。指导：[www.kaggle.com/docs/benchmarks#How%20to%20create%20a%20benchmark](https://www.kaggle.com/docs/benchmarks#How%20to%20create%20a%20benchmark)
- prinzbench（[github.com/prinz-ai/prinzbench/](https://github.com/prinz-ai/prinzbench/)）

  一种私有的评估工具，它根据 LLMs 进行美国法律研究和分析的能力（“法律推理”），以及他们在网上查找难以找到的公开信息的能力（“大海捞针”），对 LLMs 进行排名。包括25 道法律研究题目和8 道搜索题目。
- Creative Story‑Writing Benchmark、Elimination Game Benchmark、NYT Connections puzzles、Sycophancy Benchmark、Thematic Generalization Benchmark、Persuasion Benchmark（[github.com/lechmazur](https://github.com/lechmazur)）

  Creative Story‑Writing Benchmark：评估LLM在遵循创作要求的同时创作引人入胜小说的能力。每篇故事都必须有意义地融入十个必需元素 ：角色、物品、概念、属性、动作、方法、背景、时间框架、动机和基调。

  Elimination Game Benchmark：“淘汰游戏”是一项多人锦标赛，用于测试大语言模型（LLMs）的社交推理、策略制定和欺骗能力。玩家进行公开和私下对话，结成联盟，并逐轮投票淘汰其他玩家，直到只剩下两人。然后，由被淘汰的玩家组成的陪审团进行决定性投票，选出获胜者。

  NYT Connections puzzles：使用940个《纽约时报》Connections字谜游戏对大型语言模型（LLMs）进行评估

  Sycophancy Benchmark：当同一争议以相反的第一人称视角呈现时，模型是否会保持相同的判断，还是会倾向于支持说话的一方？该基准测试直接衡量这种矛盾。  
  核心指标故意设置得较为严格。只有当模型在双方均以第一人称讲述故事时，都对同一争议的两方表示认同，才被视为具有谄媚性。每例包含 5 种视角：一种中立的第三人称版本，两种删减的第一人称版本，以及两种情感化的第一人称版本。

  Thematic Generalization Benchmark：用于检验大型语言模型是否能够通过少量示例推断出特定的潜在主题，利用反例排除更广泛但错误的模式，然后从相似的干扰项中识别出唯一正确的匹配项。每个测试项目为模型提供：3 个正例，3个反例符合更广泛或相邻的模式，但不完全匹配，8个候选项，其中恰好有1个隐藏的真实匹配项。

  Persuasion Benchmark：衡量一个语言模型在多次对话中能够多大程度地改变另一个模型的立场。每次运行都会将一个模型指定为说服者，另一个模型作为目标，针对同一命题展开讨论。

### 游戏

- AI Poker Leaderboard（[benchmark.gtowizard.com/](https://benchmark.gtowizard.com/)）

  模型与GTO Wizard AI（当前最先进的AI扑克智能体）对战的实时排名
- RuneBench（[maxbittker.github.io/runebench/](https://maxbittker.github.io/runebench/)）

  评估 AI 在玩《RuneScape》方面的能力，并需要在游戏世界中完成各种任务。测量AI在“观察、决策、行动”循环中的行为。

## 数据质量评估

- OpenDataArena（[opendataarena.github.io/](https://opendataarena.github.io/)）

  让每个训练后数据集都具备可测量性、可比性和可验证性，评估多个领域（通用、数学、代码、科学和长链推理）和多种模态（文本、图像）的训练后数据。通过使用固定模型规模（Llama3 / Qwen2 / Qwen3 / Qwen3-VL 7-8B）和一致的训练配置来控制变量。数据血缘分析现代数据集通常存在高度冗余和隐藏依赖的问题。ODA推出了业内首个数据血缘分析工具，用于可视化开源数据的“谱系”。结构建模：映射数据集之间的关系，包括继承、混合和蒸馏。

## AI Infra性能

- Modded-NanoGPT Optimization Benchmark（[github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization)）

  通过协作与竞争的方式找到高效的神经网络优化器。与主要的 NanoGPT 速通挑战不同，后者旨在不择手段地缩短实际运行时间，而本基准的目标是通过优化算法来减少步数（这意味着实际运行时间较长的方法是完全可行的）。
- InferenceMAX（[inferencemax.semianalysis.com/](https://inferencemax.semianalysis.com/)）

  通过在主流硬件平台上对热门模型进行基准测试，并在新软件版本发布时更新测试标准。  
  对于每种模型与硬件组合，InferenceMAX 都会遍历不同的张量并行规模和最大并发请求数，生成一张完整的吞吐量与延迟对比图。
- MLPerf Training（[mlcommons.org/benchmarks/training/](https://mlcommons.org/benchmarks/training/)）

  MLPerf 训练基准套件衡量系统训练模型达到目标质量指标的速度。
- AA-AgentPerf（AI Hardware Benchmarking & Performance Analysis）（[artificialanalysis.ai/benchmarks/hardware](https://artificialanalysis.ai/benchmarks/hardware)）

  面向智能体推理的基准测试：它模拟真实的编程智能体执行流程，并衡量系统在满足生产级服务指标的前提下能够同时支持多少个智能体。 其核心指标是“每兆瓦支持的智能体数量”，即每个兆瓦电力下加速器平台在满足市场设定的性能指标的前提下所能承载的最大智能体数量。涵盖实际生产环境的优化措施，如KV 缓存复用、推测性解码以及预填充/解码过程的分离。测试对象包括单个加速器到整个机柜的各种系统。

  本基准固定服务水平，然后测试系统在维持该水平的前提下还能扩展到何种程度。其性能指标源自 Artificial Analysis 的无服务器 API 基准测试数据——也就是当前市场上实际存在的各种服务层级。速度与延迟均按请求进行测量：包括 P25 输出速度以及 P95 首个令牌生成时间，这些数据均基于测试阶段内的所有请求计算得出。
- GPU Benchmark（[perf.svcfusion.com/](https://perf.svcfusion.com/)）

  - 支持查看不同计算卡的 FP32、FP16、BF16 性能
  - 每一条数据都由人工同 benchmark 脚本跑出来的，不直接搬运纸面数据，并且支持所有人上传自己跑出来的数据
  - 标注测试平台名称，可以对比不同平台显卡性能的差距
