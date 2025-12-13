# awesome-llm-benchmarks
[English](./README-en.md) [Japanese](./README-jp.md)

# 大模型评测榜单汇总

Tips：基准测试（Benchmarks）构成了评价模型性能的事实标准，但测试本身并非准确无误，甚至可能存在系统性缺陷，典型案例τ²-bench。（参见[mp.weixin.qq.com/s/mrr2oDR2V8OvTT7JD918RQ](https://mp.weixin.qq.com/s/mrr2oDR2V8OvTT7JD918RQ)）（论文：[arxiv.org/abs/2511.16842](https://arxiv.org/abs/2511.16842)）
收录规则为：1、近三个月内榜单有更新；2、所纳入新模型已更新至GPT-5、Qwen3-VL这一最新世代

# 综合评测

- Artificial Analysis（[AI Model & API Providers Analysis | Artificial Analysis](https://artificialanalysis.ai/)）

  - AA Intelligence（[Artificial Analysis Intelligence Index | Artificial Analysis](https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index)）

    一个综合基准，整合七项具有挑战性的评估，全面衡量人工智能在数学、科学、编程和推理方面的能力。综合了七项评估的表现：MMLU-Pro、GPQA Diamond、HLE、LCB、SciCode、AIME 2025、IFBench、LCR、Terminal-Bench Hard、𝜏²-Bench Telecom。

  - Artificial Analysis Openness Index（[https://artificialanalysis.ai/evaluations/artificial-analysis-op...](https://artificialanalysis.ai/evaluations/artificial-analysis-openness-index)）

    一项标准化且独立评估的指标，用于衡量人工智能模型在可用性和透明度方面的开放程度。开放性不仅仅指能够下载模型权重。它还涉及许可协议、数据和方法论。在开放指数中获得 100 分的模型将具备开放权重、采用宽松许可协议，并完整发布训练代码、预训练数据和训练后数据——这不仅允许用户使用模型，还能完全复现其训练过程，或从模型创建者的部分或全部方法中汲取灵感来构建自己的模型。

- SEAL LLM Leaderboards（[scale.com/leaderboard](https://scale.com/leaderboard)）

  评估最新 LLM 的智能体能力、前沿性能、安全性及公众情绪。

- Epoch AI（[epoch.ai/benchmarks](https://epoch.ai/benchmarks)）

  有多个基准测试。

  Epoch 能力指数（ECI）将多个不同 AI 基准的分数综合为一个“通用能力”尺度，即使在单个基准已达到饱和的长时间跨度内，也能对模型进行比较。

- LMArena（[Overview Leaderboard | LMArena](https://lmarena.ai/leaderboard/)）

- OpenCompass（[OpenCompass司南 - 评测榜单](https://rank.opencompass.org.cn/home)）

- LiveBench（[LiveBench](https://livebench.ai/#/)）

  一个专为避免测试集污染和实现客观评估而设计的 LLM 基准测试，包括推理、编码、数学、数据分析。

- NeMo Evaluator SDK（[NVIDIA-NeMo/Evaluator: Open-source library for scalable, reproducible evaluation of AI models and benchmarks.](https://github.com/NVIDIA-NeMo/Evaluator)）

- LRM-Eval（[LRM-Eval](https://flageval-baai.github.io/LRM-Eval/)）

  文本任务包括以下子任务：

  - 问题解决

    - 大学课程问题、文字谜题和解码
  - 算法编码

    - 近期发布的编程问题
  - 任务完成

    - 指令遵循、多轮指令遵循、长上下文理解
  - 事实性与回避

    - 长尾知识
  - 安全性

    - 有害内容生成与越狱

# Coding Benchmarks

- SWE-bench（[SWE-bench Leaderboards](https://www.swebench.com/index.html)）

  SWE-bench 是软件工程领域最受欢迎的评估套件之一——这是一个用于评估大语言模型（LLMs）解决来自 GitHub 的真实软件问题能力的基准测试。该基准测试要求代理接收一个代码仓库和问题描述，并挑战其生成一个能解决该问题的补丁。

  - 细节

    SWE-bench 测试集中的每个样本均来自 GitHub 上 12 个开源 Python 仓库中已解决的 GitHub 问题。每个样本都关联一个拉取请求（PR），其中包含解决方案代码和用于验证代码正确性的单元测试。这些单元测试在 PR 中添加解决方案代码之前会失败，但在添加之后会通过，因此被称为 FAIL_TO_PASS 测试。每个样本还关联有 PASS_TO_PASS 测试，这些测试在 PR 合并前后均能通过，用于确保代码库中现有的无关功能未因 PR 而被破坏。

    对于 SWE-bench 中的每个样本，智能体将获得 GitHub 问题的原始文本（称为问题描述），并可访问代码库。基于这些信息，智能体必须编辑代码库中的文件以解决问题。测试用例不会向智能体展示。

    提出的修改通过运行 FAIL_TO_PASS 和 PASS_TO_PASS 测试来评估。如果 FAIL_TO_PASS 测试通过，说明该修改解决了问题；如果 PASS_TO_PASS 测试通过，则表明修改未意外破坏代码库中其他无关部分。只有两组测试全部通过，才能完全解决原始的 GitHub 问题。

  - **SWE-bench Verified**

    openai推出的一个经过人工筛选的包含 500 个实例的子集。（为解决原始 SWE-bench 数据集低估了智能体的能力）

    现有的代理框架通常依赖于 Python 特有的工具，导致对 SWE-bench Verified 过度拟合。

  - SWE-Bench Pro (Public Dataset)（[scale.com/leaderboard/swe_bench_pro_public](https://scale.com/leaderboard/swe_bench_pro_public)）

    SWE-Bench Verified的升级版。

    一个专为对软件工程领域的 AI 代理进行严格且真实评估而设计的基准测试。它旨在通过应对四大关键挑战，解决现有基准测试中的若干局限性：

    1. 数据污染：模型在训练期间可能已接触过评估代码，因此难以判断其是真正解决问题，还是在复现记忆中的答案。
    2. 任务多样性不足：许多基准测试未能涵盖现实世界软件挑战的完整范围，而仅聚焦于简单的工具库。
    3. 问题过于简化：模糊或未明确说明的问题常被从基准测试中剔除，而这与真实开发者的实际工作流程不符。
    4. 不可靠且无法复现的测试：不一致的环境设置使得难以判断解决方案是否真正有效，还是仅仅因为环境配置错误。
   
   - SWE-bench Multilingual

     旨在评估 LLMs 在多种编程语言中的软件工程能力。SWE-bench Multilingual 包含 300 个精心挑选的软件工程任务，这些任务源自 42 个 GitHub 仓库和 9 种编程语言的真实拉取请求（涵盖 9 种流行编程语言（C、C++、Go、Java、JavaScript、TypeScript、PHP、Ruby 和 Rust））。这些仓库涵盖广泛的应用领域，包括 Web 框架、数据存储与处理工具、核心工具以及常用库。

  - Multi-SWE-Bench（[Multi-SWE-bench --- Multi-SWE-bench](https://multi-swe-bench.github.io/#/)）

    字节seed推出的修改版，用于评估 LLMs 在多种编程语言中解决问题能力。该数据集包含 1,632 个跨七种编程语言的问题解决任务：Java、TypeScript、JavaScript、Go、Rust、C 和 C++。评估通过验证项目的内置测试套件结果进行，并以 PR 后的行为作为参考解决方案。

  - 局限性

    基于静态数据集的评估 inherently 有限，SWE-bench 也不例外。由于该基准由对公共 GitHub 仓库的抓取组成，预训练于互联网文本的大型基础模型很可能在这些任务上受到污染。此外，SWE-bench 仅涵盖模型自主性中等风险水平的狭窄分布，因此必须辅以其他评估方式。

- terminal-bench（[Terminal-Bench --- Terminal-Bench](https://www.tbench.ai/)）（[Artificial Analysis](https://artificialanalysis.ai/evaluations/terminalbench-hard)）

  是一组任务和评估框架，用于评估 AI 智能体在终端环境中完成复杂任务的表现。任务示例包括：编译和打包代码仓库，下载数据集并在其上训练分类器，设置服务器。Terminal-Bench 中的每个任务包括一段英文描述，一个 Docker 环境，用于验证代理是否成功完成任务的测试脚本，解决该任务的参考（“理想”）方案。

- ArtifactsBench（[ArtifactsBench: Bridging the Visual-Interactive Gap in LLM Code Generation Evaluation](https://artifactsbenchmark.github.io/)）

  腾讯推出的首个用于评估 LLM 生成视觉产物的自动化多模态评估基准，它能够渲染动态输出，并借助由细粒度检查清单指导的 MLLM 评估器，对保真度和交互性进行评估。包含 1,825 个高质量且具有挑战性的查询，分为九个不同类别：游戏开发、SVG 生成、Web 应用、模拟、数据科学、管理系统、多媒体编辑、快速工具和其他类别。将 ArtifactsBench 的创建组织为八个阶段的流程：提取与过滤、人工与 LLM 驱动的重写与润色、分类与难度筛选、小样本标注、检查清单生成、模型生成、人工 QA 检查与质量控制，以及最终数据整合。

- SWE‑Dev（[DorothyDUUU/SWE-Dev: Official code space for "SWE-Dev: Evaluating and Training Autonomous Feature-Driven Software Development"](https://github.com/DorothyDUUU/SWE-Dev)）

  首个面向功能驱动开发（FDD）的大规模基准与训练语料库——FDD 是向现有代码库添加新功能的实际任务。

- LiveCodeBench（[LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code](https://livecodebench.github.io/index.html)）（[Artificial Analysis](https://artificialanalysis.ai/evaluations/livecodebench)）

  从 LeetCode、AtCoder 和 Codeforces 平台的定期竞赛中收集问题，并用于构建一个全面的基准，以持续评估 LLMs 在各种代码相关场景中的表现。关注更广泛的代码相关能力，例如自我修复、代码执行和测试输出预测，而不仅仅是代码生成。

  - GSO Benchmark

    用于评估软件工程智能体的具有挑战性的软件优化任务。通过 10 个代码库中的 102 项具有挑战性的优化任务，评估语言模型开发高性能软件的能力。该基准测试衡量模型在运行时效率上的提升，与专家开发者优化结果进行对比。

- Aider’s polyglot benchmark（[Aider LLM Leaderboards | aider](https://aider.chat/docs/leaderboards/)）

  在 C++、Go、Java、JavaScript、Python 和 Rust 中的 225 个具有挑战性的 Exercism 编程练习上对 LLM 进行测试。

- SciCode（[SciCode - SciCode 基准测试 --- SciCode - SciCode Benchmark](https://scicode-bench.github.io/)）（[Artificial Analysis](https://artificialanalysis.ai/evaluations/scicode)）

  旨在评估语言模型（LMs）生成代码以解决真实科学研问题的能力。它涵盖了来自物理、数学、材料科学、生物和化学六大领域的 16 个子领域。展现了科学家日常工作的真实流程：识别关键的科学概念与事实，然后将其转化为计算与模拟代码。

  主要聚焦于：1. 数值方法；2. 系统模拟；3. 科学计算。

- OJBench（[He-Ren/OJBench](https://github.com/He-Ren/OJBench)）

  旨在评估大语言模型（LLMs）在竞赛级别的代码推理能力。我们的数据集专注于人类编程竞赛，包含 232 道经过严格筛选的竞赛题目，来源为中国全国信息学奥林匹克竞赛（NOI）和国际大学生程序设计竞赛（ICPC）。每道题目均根据参赛者投票和真实提交数据，细致划分为三个难度等级：简单、中等和困难。OJBench 支持 Python 和 C++ 双语评估。

- Roo Code evals（[Evals | Roo Code](https://roocode.com/evals)）

- CodeClash（[CodeClash](https://codeclash.ai/)）

  一个用于评估人工智能系统在**目标导向软件工程**方面表现的基准。如今的人工智能编程评估以任务为导向，模型会收到明确的指令，然后我们通过单元测试来验证其正确性。但软件开发从根本上是由目标驱动的（“提高用户留存率”“降低成本”“增加收入”）。通过代码实现目标是一个自主导向、迭代式且往往具有竞争性的过程。

  在CodeClash中，2个及以上的大模型智能体在多轮锦标赛中于一个代码竞技场展开竞争。在锦标赛期间，每个智能体都在不断迭代改进自己的代码库，以实现一个高级别的竞争性目标（例如，积累资源、存活最长时间等）。

- Codeforces

  全世界最大的算法练习和竞赛平台之一

## AI编码智能体

- LiveSWEBench（[LiveSWEBench](https://liveswebench.ai/)）

  用于评估 AI 智能体应用程序软件工程能力的基准测试，

  评估了每个助手在三种任务类型上的表现：

  - 代理编程：助手获得一个高级任务，并需完全自主地完成它。
  - 目标编辑：助手获得更直接的指令和待编辑的文件（仍以代理方式运行）。
  - 自动补全：助手获得部分代码片段，并需完成它们。

- DPAI Arena（[DPAI — Developer Productivity AIrena](https://dpaia.dev/)）

  DPAI Arena是业界首个开放的、多语言、多框架、多工作流的基准测试平台，旨在衡量AI编码智能体在实际软件工程任务中的有效性。它围绕灵活的、基于赛道的架构构建，能够在各种工作流（如补丁制作、漏洞修复、PR审核、测试生成、静态分析等）中进行公平且可复现的比较。

# Agentic Benchmarks

- GAIA（[GAIA Leaderboard - a Hugging Face Space by gaia-benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard)）

  全能型基准，通过涉及现实世界问题的基准测试，评估通用人工智能助手在推理、多模态处理、网页浏览以及工具使用方面的能力。

- Gaia2 Leaderboard（[Gaia2 Agents Evaluation Leaderboard - a Hugging Face Space by meta-agents-research-environments](https://huggingface.co/spaces/meta-agents-research-environments/leaderboard)）

  旨在衡量通用智能体能力的基准。与传统的搜索和执行任务不同，Gaia2 异步运行，要求智能体处理模糊性与噪声、适应动态环境、与其他智能体协作，并在时间约束下运作。从以下维度评估智能体：执行（指令遵循、多步工具使用）、搜索（信息检索）、模糊性（处理不明确或不完整的指令）、适应性（应对动态环境变化）、时间（管理时间约束与调度）、噪声（在无关信息和随机工具故障下仍能有效运作）以及智能体间协作（与其他智能体的协作与协调）。

- BrowseComp（[BrowseComp：面向浏览代理的基准测试 | OpenAI --- BrowseComp: a benchmark for browsing agents | OpenAI](https://openai.com/index/browsecomp/)）

  openai推出的面向浏览代理的基准测试，用于衡量 AI 代理定位互联网上复杂、隐蔽信息的能力。专注于那些答案简短、且原则上只有一个正确答案的问题。

  - BrowseComp-ZH（[BrowseComp-ZH/README-ZH.md at main · PALIN2018/BrowseComp-ZH](https://github.com/PALIN2018/BrowseComp-ZH/blob/main/README-ZH.md)）

    首个专为评估大型语言模型（LLMs）在中文网络生态中检索与推理能力而设计的高难度基准测试。受 BrowseComp (Wei 等, 2025) 启发，本项目针对中文信息环境构建了复杂的多跳检索与推理任务，模型需应对平台碎片化、语言特性及内容审查等多重挑战。

- FACTS Benchmark（[www.kaggle.com/benchmarks/google/facts/leaderboard](https://www.kaggle.com/benchmarks/google/facts/leaderboard)）

  一个参数化基准，用于衡量模型在事实性问答场景中准确调用其内部知识的能力，包含一个由 1052 个问题组成的公开集和一个由 1052 个问题组成的私有集。
  一个搜索基准，用于测试模型将搜索作为工具以检索信息并正确整合信息的能力，包含一个由 890 个条目组成的公开数据集和一个由 994 个条目组成的私有数据集。
  一个多模态基准，用于测试模型以事实准确的方式回答与输入图像相关提示的能力，包含一个 711 项的公开数据集和一个 811 项的私有数据集。

  Grounding Benchmark - v2，这是一个扩展基准，用于测试模型在给定提示的上下文中提供基于事实依据的答案的能力。

- DeepSearchQA（[www.kaggle.com/benchmarks/google/dsqa/leaderboard](https://www.kaggle.com/benchmarks/google/dsqa/leaderboard)）

  谷歌推出的一个包含 900 个提示的基准测试，用于评估代理在 17 个不同领域中完成困难的多步信息检索任务的能力。与传统仅针对单答案检索或广泛事实准确性的基准不同，DeepSearchQA 包含一组具有挑战性的手工构建任务，旨在评估代理执行复杂搜索计划以生成详尽答案列表的能力。
  每个任务都构造成一个“因果链”，其中完成某一步的信息发现依赖于前一步的成功完成，强调长期规划和上下文保留。

- DeepResearch Bench（[https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leader...](https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leaderboard)）

  面向深度研究代理的综合性基准测试，包含 100 个博士级研究任务，每个任务均由 22 个不同领域的领域专家精心设计，其中 50 个为中文任务，50 个为英文任务。

- xbench（[xbench --- xbench](https://xbench.org/)）

  国内机构合作推出的基准测试。包含两条互补的赛道，旨在衡量人工智能系统的智能前沿与实际应用价值。  
  AGI 跟踪：衡量核心模型能力，如推理、工具使用和记忆  
  与专业对齐：一类基于工作流、环境和业务关键绩效指标的新评估体系，与领域专家共同设计

  AGI 追踪评估衡量前沿能力，专业对齐评估则通过与业务关键绩效指标和运营规模对齐的动态、领域特定任务，反映真实世界中的实用性。

  xbench 构建为一个动态评估系统，两个赛道均持续更新。

  - xbench-ScienceQA

    专注于评估科学领域内的基础知识能力。

  - xbench-DeepSearch

    专注于评估搜索和信息检索场景中的工具使用能力，贴合中文语境。

  - xbench-Profession-Recruiting

    专注于真实的招聘工作流程和行业标准，涵盖职位需求分解、人才画像定位、候选人体验补充、简历筛选、人脉关系理解以及开放人才搜索等任务，用以评估 Agent 在行业认知、人才搜寻能力与评估技能等多方面的表现。

  - xbench-Profession-Marketing

    专注于真实的营销流程和行业标准，主要聚焦于 KOL 搜索任务。在客户需求咨询与 KOL 匹配中的沟通交互，以及内容分发中的监控与策略调整

- IFBench（[IFBench 基准测试排行榜 | Artificial Analysis --- IFBench Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/ifbench)）

  通过 58 种多样且可验证的域外约束，评估模型在遵循精确指令方面的泛化能力，检验模型是否能遵守特定的输出要求。

- Humanity's Last Exam（[人类的最后一次考试 --- Humanity's Last Exam](https://lastexam.ai/)）（[Artificial Analysis](https://artificialanalysis.ai/evaluations/humanitys-last-exam)）

  一个处于人类知识前沿的多模态基准，旨在成为涵盖广泛学科的最后一个封闭式学术基准。该数据集包含跨越百余门学科的 2,500 道高难度问题。我们公开发布这些问题，同时保留一个未公开的测试集，用于评估模型过拟合情况。

  在 HLE 上取得高准确率将证明模型在封闭式、可验证的问题以及前沿科学知识方面具备专家级表现，但这本身并不意味着其具备自主研究能力或“通用人工智能”。HLE 测试的是结构化的学术问题，而非开放式的科研或创造性解决问题的能力，因此它是一种聚焦于技术知识与推理能力的衡量标准。

  分为 (w/ tools)有工具（测试angentic能力）和 (w/o tools)（无工具）（测试模型本身智能）两种情况

- τ²-bench（[τ-bench --- τ-bench](https://taubench.com/#home)）（[Artificial Analysis](https://artificialanalysis.ai/evaluations/tau2-bench)）（被发现存在重大缺陷）

  用于评估跨多个领域客户服务代理的模拟框架，在协作的现实场景中对 AI 代理进行基准测试。τ-bench 要求代理**在复杂的企业领域中协调、引导并协助用户**实现共同目标。分为总体、零售、电信、航空。

  通过同时模拟智能体与用户，主动修改共享的全局状态，开创了评估对话式 AI 的新范式。电信领域测试智能体引导用户完成技术故障排除的能力，以检验其问题解决与有效沟通技巧。

  重大缺陷：

  在对原始的τ²-bench进行验证时，我们发现了几类问题：

  1. 政策合规问题：预期行动违反了规定的领域政策的任务（例如，在政策不允许的情况下提供补偿、取消已起飞的航班）
  2. 数据库准确性问题：存在物品ID、乘客信息或支付方式参考不正确且与实际数据库不匹配的任务
  3. 逻辑一致性问题：存在不可能场景的任务（例如，兑换相同物品，这是政策所禁止的）
  4. 评估模糊性问题：任务说明过于模糊，导致评估结果不一致

- τ²-Bench-Verified（[github.com/amazon-agi/tau2-bench-verified](https://github.com/amazon-agi/tau2-bench-verified)）

  τ²-Bench-Verified是原始τ²-bench基准测试的修正版和人工验证版。此版本解决了在原始数据集中发现的问题，即任务定义、预期操作和评估标准与所述政策或数据库内容未能正确对齐。

- FinSearchComp（[FinSearchComp 基准测试 --- FinSearchComp Benchmark](https://randomtutu.github.io/FinSearchComp/)）

  首个专为开放式金融搜索设计的基准。真实的决策任务需要三种核心能力：发现正确的信号、核查并整合信息来源，以及在时间压力下形成基于证据的判断。我们提供了一个基础性的端到端评估基础设施——一个开放的金融基准。涵盖全球及大中华区两个子集的三个子任务（时效性数据获取、简单历史查询、复杂历史调查）以及多源调查。

- GDPval-AA（[artificialanalysis.ai/evaluations/gdpval-aa](https://artificialanalysis.ai/evaluations/gdpval-aa)）

  为 OpenAI 的 GDPval 数据集开发的评估框架。它在 44 种职业和 9 个主要行业中，对 AI 模型在真实任务中的表现进行测试。包含220项任务，要求模型生成多样化的输出，包括文档、幻灯片、图表和电子表格，以模拟金融、医疗、法律及其他专业领域中的实际工作成果。

- TheAgentCompany（[代理公司 --- The Agent Company](https://the-agent-company.com/)）

  衡量 LLM 智能体在执行现实世界专业任务中的表现，该基准评估 AI 代理以类似数字员工的方式与世界互动：浏览网页、编写代码、运行程序以及与同事沟通。

- VitaBench（[VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications](https://vitabench.github.io/)）

  以外卖点餐、餐厅就餐、旅游出行等高频生活场景为载体，构建了包含66个工具的交互式评测环境，设计了跨场景综合任务，从深度推理、工具使用与用户交互三大维度衡量智能体表现。

- Toolathlon（[Tool Decathlon - Toolathlon](https://toolathlon.xyz/introduction)）

  一个评估语言智能体在真实环境中通用工具使用能力的基准。它涵盖基于现实世界软件环境的32 个软件应用和 604 个工具。每项任务都需要通过长程工具调用才能完成，共包含 108 个手动设计或编写的任务，平均每个任务需跨越约 20 轮交互。

- MCP-Universe（[mcp-universe.github.io/](https://mcp-universe.github.io/)）

  基于真实世界模型上下文协议服务器的语言模型基准测试，

- MCPMark（[mcpmark.ai/](https://mcpmark.ai/)）

  一套综合性压力测试 MCP 基准评测体系，包含多样化的可验证任务，旨在评估模型和智能体在真实 MCP 应用场景中的能力。包含以下MCP：Notion、Github、Filesystem、Postgres、Playwright、Playwright-WebArena。

- MCP Atlas（[scale.com/leaderboard/mcp_atlas](https://scale.com/leaderboard/mcp_atlas)）

  通过模型上下文协议（MCP）评估语言模型处理现实世界工具使用的能力，衡量的是多步骤工作流中的表现。包含 1,000 个人工撰写的任务，每个任务都需要调用多个工具来解决，工具来自 40 多个 MCP 服务器和 300 多个工具。任务范围从仅需 2 至 3 个工具且链条简单的单领域查询，到需要 5 个以上工具并包含条件分支和错误处理的复杂工作流。每项任务都包含精心挑选的干扰项工具，这些工具看似合理但实际错误。干扰项由数据标注者从与必需工具相同的类别中选取。该框架为每个任务提供 12-18 个工具（3-7 个必需工具加上 5-10 个干扰项），迫使代理基于工具描述进行推理，而非盲目调用。

- SCONE-bench（[red.anthropic.com/2025/smart-contracts/](https://red.anthropic.com/2025/smart-contracts/)）

  Anthropic推出的首个通过模拟盗取资金总价值来衡量智能体利用智能合约能力的基准测试。针对每个目标合约，智能体需识别漏洞并生成利用该漏洞的攻击脚本，使得脚本执行时执行者的原生代币余额至少增加预设阈值。SCONE-bench 摒弃漏洞赏金或推测性模型，直接采用链上资产量化损失。该基准测试提供以下功能：

  1. 一份包含 405 个智能合约的基准测试集，涵盖 2020 至 2025 年间在三条以太坊兼容区块链（以太坊、币安智能链与 Base）上被实际利用的真实漏洞，源自 DefiHackLabs 代码仓库。
  2. 在每个沙盒环境中运行的基础代理，它会在规定时间（60 分钟）内通过模型上下文协议（MCP）提供的工具，尝试利用给定的合约。
  3. 一种使用 Docker 容器进行沙盒化和可扩展执行的评估框架，每个容器都在指定区块号处运行一个分叉的本地区块链，以确保结果的可重现性。
  4. 即插即用支持，可在智能合约部署到实时区块链之前，使用代理对其漏洞进行审计。我们相信此功能能帮助智能合约开发者出于防御目的对其合约进行压力测试。

## AI智能体

- Online-Mind2Web（[huggingface.co/spaces/osunlp/Online_Mind2Web_Leaderboard](https://huggingface.co/spaces/osunlp/Online_Mind2Web_Leaderboard)）

  一个旨在评估网络智能体在真实网站性能的基准测试，涵盖 136 个热门网的 300项任务，涉及多个领域，并采用可靠的 LLM 即法官（WebJudge）自动评估机制。根据人工标注所需步骤数，任务分为三个难度等级：简单（1-5 步）、中等（6-10 步）和困难（11 步以上）。

## LLM记忆与个性化

- LoCoMo（[snap-research.github.io/locomo/](https://snap-research.github.io/locomo/)）

  评估 LLM 智能体的超长期对话记忆，用于衡量模型的长时记忆能力，涵盖问答、事件总结和多模态对话生成任务。

  - 问答任务。智能体需要准确“回忆”过往语境，将相关信息整合到未来回应中。我们通过问答任务直接检验其记忆能力，并将问题划分为五种推理类型以多维度评估记忆表现：单步推理、多步推理、时序推理、常识与世界知识推理、以及对抗性推理。
  - 事件图摘要生成。智能体还需识别对话中的长程因果与时序关联，才能生成具有共情力且贴合语境的回应。我们通过事件图摘要任务来衡量其因果与时序理解能力：以每位 LLM 发言者关联的事件图谱作为标准答案，要求模型从对话历史中提取该信息。
  - 多模态对话生成。对话智能体需要运用从过往对话中提取的相关上下文，生成与当前叙事逻辑一致的回答。我们通过多模态对话生成任务来评估这项能力。

- PERSONAMEM（[zhuoqunhao.github.io/PersonaMem.github.io/](https://zhuoqunhao.github.io/PersonaMem.github.io/)）

  大规模动态用户画像与个性化响应的 LLMs 基准测试，包含超过 180 个模拟用户与 LLM 的互动历史，涵盖多达 60 个多轮对话（约 100 万词元），涉及 15 个不同的个性化场景和 7 类实时用户查询任务。每个基准实例都包含一个具有静态属性（如人口统计信息）和动态属性（如不断演变的偏好）的用户角色。用户通过多轮会话与聊天机器人进行互动，涵盖诸如美食推荐、旅行规划和心理咨询等多种主题。随着用户偏好随时间演变，该基准提供了带标注的问题，用于评估 LLMs 能否针对用户的现场查询提供最合适的响应——这些查询是用户以第一人称视角向 LLMs 发出的。

- Personalized Deep Research（[arxiv.org/abs/2509.25106](https://arxiv.org/abs/2509.25106)）

  首个用于评估深度研究智能体个性化能力的基准体系。该平台将涵盖 10 个领域的 50 项多样化研究任务，与 25 个真实用户画像进行配对（这些画像结合了结构化人物属性与动态现实情境），最终生成 250 个真实用户任务查询。为评估系统性能，我们提出 PQR 三维评估框架，从(P)个性化契合度、(Q)内容质量与(R)事实可靠性三个维度进行综合度量。

## 视觉定位与Gui智能体

- AndroidDaily（[opengelab.github.io/index_zh.html](https://opengelab.github.io/index_zh.html)）

  面向真实世界场景的多维动态基准测试。 我们专注于现代生活六个核心维度（食品、交通、购物、住房、信息消费、娱乐）的实证分析， 优先考虑主导这些类别的热门应用。这确保了基准测试任务具有真实世界的交互结果（如交易支付、服务预订）， 具有紧密的线上线下集成特征。

  静态测试方法包含3146个操作。提供任务描述和逐步截图，要求智能体预测每一步的动作类型和值 （如点击坐标、输入文本）。主要评估数值准确性。

  端到端基准测试方法在功能完整的测试环境（如真实设备或模拟器）中进行，智能体必须自主从头到尾执行任务， 以整体任务成功率作为评估指标。这种设置提供了最高的生态有效性，真实地反映了智能体在复杂环境中的综合能力。

- AndroidWorld（[google-research.github.io/android_world/](https://google-research.github.io/android_world/)）

  一个功能完整的安卓环境，为 20 款真实安卓应用中的 116 项编程任务提供奖励信号。与提供静态测试集的现有交互环境不同，AndroidWorld 能动态构建参数化且以自然语言无限表达的任务，从而在更庞大、更贴近现实的任务套件上进行测试。

- ScreenSpot-V2（[https://gui-agent.github.io/grounding-leaderboard/screenspot.htm...](https://gui-agent.github.io/grounding-leaderboard/screenspot.html)）

  一个图形用户界面定位基准，能够将自然语言指令明确映射到屏幕上的像素级目标。

- ScreenSpot-Pro（[gui-agent.github.io/grounding-leaderboard/](https://gui-agent.github.io/grounding-leaderboard/)）

  面向专业高分辨率计算机使用的图形用户界面定位，专注于高分辨率专业软件图表

- OSWorld-G（[osworld-grounding.github.io/](https://osworld-grounding.github.io/)）

  用于评估模型在细粒度功能组件上性能，包含 564 个精细标注样本，涵盖文本匹配、元素识别、布局理解和精确操作等多种任务类型。

- MMBench-GUI（[github.com/open-compass/MMBench-GUI](https://github.com/open-compass/MMBench-GUI)）

  旨在全面评测GUI Agent在Windows、macOS、Linux、iOS、Android和Web等六大主流平台上的综合能力。它不仅关注任务是否成功，更创新性地提出了 效率-质量面积（Efficiency-Quality Area, EQA） 指标，用于衡量 Agent完成任务的效率。包含四个评估层级：图形用户界面内容理解、图形用户界面元素定位、图形用户界面任务自动化以及图形用户界面任务协同。

‍

# Intelligence Benchmarks

- ARC-AGI-2（[arcprize.org/leaderboard](https://arcprize.org/leaderboard)）

  专注于那些对人类而言相对简单、但对 AI 却困难甚至不可能完成的任务，从而揭示那些无法通过“规模扩大”自然涌现的能力鸿沟。

  1. 所有评估集（公开、半私有、私有）现在均包含 120 个任务（从 100 个增加而来）。
  2. 已从评估集中移除容易受到暴力搜索影响的任务（即 2020 年 Kaggle 竞赛中的所有已解决任务）。
  3. 进行了受控的人类测试，以校准评估集的难度，确保 IDD，并验证至少两名人类能够通过 pass@2 解决（以符合 AI 规则）。
  4. 基于研究（符号解释、组合推理、上下文规则等），设计了新的任务以挑战 AI 推理系统。

- AIME25（[AIME 2025 Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/aime-2025)）

  数据集是一个包含数学问题的答案的数据集，具体来源于2025年美国数学邀请赛（AIME）第一部分的考试题目。该数据集适用于问题回答任务，数据集大小小于1000条记录，语言为英语。

- HMMT 2025

  用于MathArena Leaderboard的2025年2月HMMT数学竞赛（哈佛-麻省理工大学数学竞赛）的问题

- AMO-Bench（[AMO-Bench: Large Language Models Still Struggle in High School Math Competitions](https://amo-bench.github.io/)）

  美团 LongCat 团队发布，该评测集共包含 50 道竞赛专家原创试题，所有题目均对标甚至超越 IMO 竞赛难度。

- IMO Bench（[imobench.github.io/](https://imobench.github.io/)）

  该基准测试集经过了一组IMO奖牌获得者和数学家的审核（他们总共获得了10枚IMO金牌和5枚IMO银牌）。由于IMO的题目难度极高，既需要严谨的多步骤推理，又需要超越简单套用已知公式的创造力，因此IMO-Bench专门以IMO的难度水平为目标。IMO-Bench由三个基准测试组成，用于评估模型的多种能力：IMO-AnswerBench——一项大规模的正确答案测试，IMO-ProofBench——用于证明写作的更高层次评估，以及IMO-GradingBench——旨在推动对长篇答案的自动评估取得进一步进展。

- FrontierMath（[epoch.ai/frontiermath](https://epoch.ai/frontiermath)）

  包含 350 道原创数学题（50 道最高难度等级 4 的问题），涵盖从具有挑战性的大学水平问题到可能需要专家数学家数日才能解决的难题。要求：

  1. 明确且可验证的答案
  2. 抵御猜测：答案应具备“防猜测”特性，即随机尝试或简单的暴力方法几乎不可能成功
  3. 计算可行性：解决计算密集型问题时，必须包含脚本，展示如何仅基于该领域的标准知识找到答案。这些脚本在标准硬件上的累计运行时间必须少于一分钟。

- MMLU-Pro（[MMLU-Pro Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/mmlu-pro)）

  多任务理解数据集，旨在严格评估大型语言模型。它包含来自各个学科领域的1.2万个复杂问题。每个问题有10个答案选项，整合了更多以推理为核心的问题

- GPQA Diamond（[GPQA 钻石基准排行榜 | Artificial Analysis --- GPQA Diamond Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/gpqa-diamond)）

  GPQA 基准中最难的 198 个问题，专为“防谷歌”设计，需要真正的科学专业知识，而非搜索技巧。  
  这些研究生级别的物理、生物和化学问题，只有具备博士学位的领域专家才能稳定解答，因此非常适合用于测试真正的科学推理能力。

- CritPt（[CritPt Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/critpt)）

  旨在测试 LLMs 在研究级物理推理任务表现的基准，包含 71 项综合性研究挑战。

- Frames

  是一个综合评估数据集，旨在测试检索增强生成（RAG）系统在事实性、检索准确性和推理能力方面的表现。824 个具有挑战性的多跳问题，需要从 2-15 篇维基百科文章中获取信息，问题涵盖历史、体育、科学、动物、健康等多样主题。评估 RAG 系统的性能，评估语言模型的事实准确性与推理能力。

- SealQA（[vtllms/sealqa · Datasets at Hugging Face](https://huggingface.co/datasets/vtllms/sealqa)）

  Seal-0，用于评估在事实查询问题上使用网络搜索时，搜索结果存在冲突、噪声或无帮助情况下的搜索增强型语言模型。聚焦于最具挑战性的问题——在这些问题上，聊天模型（如 GPT-4.1）通常准确率接近零。

- Stanford HELM（[Capabilities - Holistic Evaluation of Language Models (HELM)](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard)）

  ‍

- Zenmux（[Benchmark - ZenMux](https://zenmux.ai/benchmark)）

  分为性价比曲线和评分榜两部分。为每个模型独立测试所有可用的提供商通道。使用 Scale AI 公开发布的数据集《人类最后的考试（文本版）》作为主要评估基准。

- AA-LCR（[Artificial Analysis Long Context Reasoning Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning)）

  专为评估语言模型在多个长文档间进行推理能力而设计的基准。要求模型阅读 10 万 token 的输入（使用 cl100k\_base 分词器衡量），整合输入文档中多个位置的信息，并据此推导出答案。旨在真实还原知识工作者期望语言模型执行的推理任务。涵盖 7 种纯文本文档类型（即公司报告、行业报告、政府咨询、学术文献、法律文件、营销材料和调查报告）。

- Fiction-liveBench（[Fiction.liveBench Sept 29 2025](https://fiction.live/stories/Fiction-liveBench-Sept-06-2025/oQdzQvKHw8JyXbN87/home)）

  评估 AI 模型的长上下文理解能力（针对故事写作），基于一组精选的十几个非常长且复杂的故事情节以及大量经过验证的测验，我们根据这些故事的精简版本生成了测试题。

- Context Arena（[contextarena.ai/](https://contextarena.ai/)）

- Context-Bench（[Letta Leaderboard](https://leaderboard.letta.com/)）

  评估语言模型在链式文件操作、追踪实体关系以及管理多步骤信息检索方面的能力。

- Needle in a Haystack（[LLM-NeedleInAHaystack/README_CN.md at main · Lianues/LLM-NeedleInAHaystack](https://github.com/Lianues/LLM-NeedleInAHaystack/blob/main/README_CN.md)）

  大语言模型召回率测试，通过一种类似于"大海捞针"的方法，对现有主流大语言模型进行召回率的测试。

  本测试方法通过以下步骤进行：

  1. **构造测试文本**：在一个固定token长度的上下文中，随机插入多个四位数（1000-9999）
  2. **模型任务**：要求模型从文本中提取所有四位数，并按出现顺序输出为JSON格式
  3. **评分算法**：使用基于编辑距离（Levenshtein Distance）的算法对模型回答进行评分

- Hallucination Leaderboard（[LLM Hallucination Leaderboard - a Hugging Face Space by vectara](https://huggingface.co/spaces/vectara/leaderboard)）

  评估 LLM 在总结文档时引入幻觉的频率。输入短文档，并要求它们仅依据文档中呈现的事实对每篇短文进行摘要。评估的是摘要的事实一致性率，而非整体事实准确性

- AA-Omniscience（[Artificial Analysis Omniscience Index | Artificial Analysis](https://artificialanalysis.ai/evaluations/omniscience)）

  涵盖 6 个领域（“商业”、“人文与社会科学”、“健康”、“法律”、“软件工程”和“科学、工程与数学”）中 42 个主题的 6,000 道问题。三项指标：准确率（正确百分比）、幻觉率（错误答案占所有非回避答案的百分比）、全知指数（回答正确+1，回答错误-1，回避回答 0）。

- R-HORIZON（[R-HORIZON: How Far Can Your Large Reasoning Model Really Go in Breadth and Depth?](https://reasoning-horizon.github.io/)）

  首个系统性评估与增强 LRMs 长链推理能力的评测框架与训练方法。提出了问题组合（Query Composition）方法，通过构建问题间的依赖关系，将孤立任务转化为复杂的多步骤推理链。

  该方法包含三个步骤：

  1. **信息提取：** 从独立问题中提取核心数值、变量等关键信息
  2. **依赖构建：** 将前序问题的答案嵌入到后续问题的条件中
  3. **链式推理：** 模型必须顺序解决所有子问题才能获得最终答案

- TRUEBench（[TRUEBench - a Hugging Face Space by SamsungResearch](https://huggingface.co/spaces/SamsungResearch/TRUEBench)）

  一个用于评估 LLMs 指令遵循能力的基准测试，评估 LLMs 作为人类工作生产力助手的基准。

- SpeechMap（[SpeechMap.AI Explorer](https://speechmap.ai/)）

  旨在探索人工智能言论的边界。测试不同提供商、国家和话题下，语言模型对敏感和争议性提示的反应。大多数 AI 基准衡量的是模型能做什么，而我们关注的是它们不能做什么：它们回避、拒绝或屏蔽的内容。

- SimpleQA

  OpenAI推出的基准测试，用在评估大型语言模型回答简短、寻求事实问题的能力。SimpleQA包含4326个问题，每个问题设计为只有一个正确答案，易于评分。

## AI4S（特化领域）

- PutnamBench（[trishullab.github.io/PutnamBench/leaderboard.html](https://trishullab.github.io/PutnamBench/leaderboard.html)）

  在普特南数学竞赛中对形式化数学推理进行基准测试。包含 1712 个手工构建的形式化问题，题目源自北美顶尖本科数学竞赛——威廉·洛厄尔·普特南数学竞赛。其中 660 个问题使用 Lean 4 形式化，640 个使用 Isabelle 形式化，412 个使用 Coq 形式化。

# 视觉理解与推理

- MMMU（[mmmu-benchmark.github.io/](https://mmmu-benchmark.github.io/)）

  面向专家级通用人工智能的大规模多学科多模态理解与推理基准测试。包含从大学考试、测验和教材中精心收集的 11.5 万个多模态问题，涵盖艺术与设计、商业、科学、健康与医学、人文与社会科学、技术与工程六大核心学科。这些问题横跨 30 个学科和 183 个子领域，包含 30 种高度异质的图像类型，如图表、示意图、地图、表格、乐谱和化学结构等。与现有基准不同，MMMU 聚焦于结合领域专业知识的进阶感知与推理能力，挑战模型完成类似专家所面临的任务。

- MMMU-Pro（[MMMU-Pro Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/mmmu-pro)）

  多项选择选项为 10 个，并引入仅视觉输入格式，其中问题嵌入在截图或照片中。  
  该基准包含 3,460 道题目，涵盖六个核心学科（艺术与设计、商业、科学、健康与医学、人文与社会科学、技术与工程），要求模型在更贴近现实的场景中同时处理视觉与文本信息。

- MATH-Vision（[mathllm.github.io/mathvision/#leaderboard](https://mathllm.github.io/mathvision/#leaderboard)）

  衡量多模态数学推理能力，包含 3,040 道源自真实数学竞赛的高质量视觉情境数学题。该数据集横跨 16 个不同数学学科并按 5 个难度等级划分。

- CharXiv（[charxiv.github.io/#leaderboard](https://charxiv.github.io/#leaderboard)）

  一个包含 2,323 个来自科学论文的自然、具有挑战性且多样化的图表的综合评估套件。CharXiv 包含两类问题：（1）关于识别基本图表元素的描述性问题；（2）要求综合图表中复杂视觉元素信息的推理问题。为确保质量，所有图表和问题均由人工专家精心挑选、整理并验证。

- ROME（[BAAI/ROME · Datasets at Hugging Face](https://huggingface.co/datasets/BAAI/ROME)）

  视觉推理基准。ROME 包含 8 个子任务（共计 281 道高质量问题）。每个样本均经过验证，确保图像对于正确作答是必需的：  
  学术-来自大学课程的问题  
  图表-从近期科学论文、报告或博客文章中收集的图表和图示  
  谜题和游戏-瑞文渐进矩阵、文字谜题和游戏玩法  
  表情包-重新制作的迷因  
  Geo-地理位置推断  
  识别-细粒度识别  
  多图像-找不同任务或视频帧重排序。  
  空间=相对位置、深度/距离、高度等

- ZeroBench（[zerobench.github.io/](https://zerobench.github.io/)）

  面向当代大型多模态模型的一项极难视觉基准测试，包含 100 道由设计师团队精心独创并经过广泛评审的挑战性问题，下有334 个子问题，对应回答每个主要问题所需的独立推理步骤

- VisuLogic（[visulogic-benchmark.github.io/VisuLogic/](https://visulogic-benchmark.github.io/VisuLogic/)）

  国内机构推出。包含六大类别（如数量变化、空间关系、属性比较）的 1000 道人工验证题目，从多维度考察多模态大模型的视觉推理能力。

- OCRBench v2（[99franklin.github.io/ocrbench_v2/](https://99franklin.github.io/ocrbench_v2/)）

  一个用于评估大型多模态模型在视觉文本定位与推理方面的改进基准。包含 10000 组人工验证的问答，且高难度样本占比很高。覆盖31 个不同场景，包括街景、收据、公式、图表等

- MMLongBench-Doc（[huggingface.co/spaces/OpenIXCLab/mmlongbench-doc](https://huggingface.co/spaces/OpenIXCLab/mmlongbench-doc)）（[mayubo2333.github.io/MMLongBench-Doc/](https://mayubo2333.github.io/MMLongBench-Doc/)）

  一个长上下文多模态文档理解基准，旨在评估大型多模态模型在复杂文档理解任务上的性能。包含 1,091 个专家标注的问题，基于 135 份篇幅较长的 PDF 格式文档构建，平均每份文档有 47.5 页和 21,214 个文本标记。这些问题的答案依赖于来自（1）不同来源（文本、图像、图表、表格和布局结构）以及（2）不同位置（即页码）的证据片段。此外，33.0%的问题是跨页问题，需要整合多个页面的证据。22.5%的问题被设计为不可回答，用于检测模型可能产生的幻觉。

- Video-MMMU（[videommmu.github.io/](https://videommmu.github.io/)）

  一个面向多模态、多学科的基准，用于评估 LMMs 从视频中获取和运用知识的能力。Video-MMMU 包含六个专业领域（30个细分学科）中精心挑选的 300 个专家级视频和 900 个人工标注的问题，通过与认知阶段对齐的问题-答案对（感知、理解、适应）来评估知识获取能力。每个视频包含三组问答对，分别对应知识获取的三个阶段：感知（识别与知识相关的关键信息）、理解（掌握底层概念）和适应（将知识应用于新情境）。此外，评估模型“增量准确率”——即在观看视频后性能的提升幅度。

# OCR与嵌入评测

[Supercharge your OCR Pipelines with Open Models](https://huggingface.co/blog/ocr-open-models)

在测试不同的 OCR 模型时，它们在不同文档类型、语言等方面的性能差异很大。

- OmniDocBench（[OmniDocBench/README_zh-CN.md at main · opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench/blob/main/README_zh-CN.md)）

  一个针对真实场景下多样性文档解析评测集，这个广泛使用的基准测试因其多样化的文档类型而脱颖而出，包括书籍、杂志和教科书。其评估标准设计精良，支持 HTML 和 Markdown 格式的表格。

- olmOCR-Bench（[olmocr/olmocr/bench at main · allenai/olmocr](https://github.com/allenai/olmocr/tree/main/olmocr/bench)）

  该基准在评估英语方面非常成功。

- CC-OCR

- Embedding Leaderboard（[MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard)）

# 视频音频生成和角色扮演评测

- DesignArena（[www.designarena.ai/leaderboard](https://www.designarena.ai/leaderboard)）

- Speech-DRAME（[Anuttacon/speech_drame](https://github.com/Anuttacon/speech_drame)）

  用于评估语音角色扮演场景中由人工智能生成的语音

- UNO-Bench（[UNO-Bench](https://meituan-longcat.github.io/UNO-Bench/)）

  一个统一基准，用于探索全模型中单模态与全模态之间的组合规律，UNO-Bench 中几乎 100% 的问题都需要对音频和视觉信息的联合理解。除了传统的多项选择题外，我们还提出了一种创新的多步骤开放式问答格式，以评估复杂推理能力。

  我们的材料具有三个关键特性：a. 多元来源——主要来自众包的真实世界照片和视频，辅以无版权限制的网站和高质量公共数据集。b. 丰富多样的主题——涵盖社会、文化、艺术、生活、文学和科学。c. 实时录制音频——由超过 20 位真人说话者录制的对话，确保音频特征丰富，反映真实世界的声音多样性。

- Vue

  字节推出的视频理解测试。

  - VUE-STG：在实际场景中全面评估 STG（时空定位）能力。1）视频时长覆盖约 10 秒至 30 分钟，支持长上下文推理；2）查询格式多数转换为名词短语，同时保留句子级表达能力；3）标注质量采用人工精准标注所有真实时间范围与边界框；4）评估指标采用优化的 vIoU/tIoU/vIoU-Intersection 方案进行多片段时空评估。
  - VUE-TR-V2：视频问答（Video QA）测试，实现了更均衡的视频时长分布和更贴近用户习惯的查询设计。

- Moral RolePlay（[digitalhuman/RolePlay_Villain at main · Tencent/digitalhuman](https://github.com/Tencent/DigitalHuman/tree/main/RolePlay_Villain)）

‍
