# Summary of Large Model Benchmark Leaderboards 

Tips: Benchmarks constitute the de-facto standard for evaluating model performance, but the tests themselves are not infallible and may even have systemic flaws—for example τ²-bench.(Paper: [arxiv.org/abs/2511.16842](https://arxiv.org/abs/2511.16842))

------

# Comprehensive Benchmarks

- **Artificial Analysis**
   ([AI Model & API Providers Analysis | Artificial Analysis](https://artificialanalysis.ai/))

  - **AA Intelligence**
     ([Artificial Analysis Intelligence Index | Artificial Analysis](https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index))

    A comprehensive benchmark that integrates seven challenging evaluations to holistically measure AI capabilities in math, science, programming, and reasoning. It aggregates performance on: MMLU-Pro, GPQA Diamond, HLE, LCB, SciCode, AIME 2025, IFBench, LCR, Terminal-Bench Hard, and 𝜏²-Bench Telecom.

  - **Artificial Analysis Openness Index**
     ([https://artificialanalysis.ai/evaluations/artificial-analysis-op...](https://artificialanalysis.ai/evaluations/artificial-analysis-openness-index))

    A standardized and independently evaluated metric to measure how open AI models are in terms of usability and transparency. Openness is not just about downloadable model weights—it also concerns licensing, data, and methodology. Models scoring 100 on the Openness Index provide open weights, a permissive license, and fully released training code, pre-training data, and post-training data—allowing users not only to use the model, but also to fully reproduce its training process or draw from some or all of the creators’ methods to build their own models.

- **SEAL LLM Leaderboards** ([scale.com/leaderboard](https://scale.com/leaderboard))

  Evaluating the agent capabilities, frontier performance, safety, 和 public sentiment of the latest LLMs.

- **Epoch AI** ([epoch.ai/benchmarks](https://epoch.ai/benchmarks))

  Features multiple benchmarks.

  The Epoch Capabilities Index (ECI) aggregates scores from diverse AI benchmarks into a single "general capability" scale, enabling model comparisons even over long time spans where individual benchmarks have saturated.

- **Sansa Bench** ([trysansa.com/benchmark](https://trysansa.com/benchmark))

  A specialized benchmark designed to evaluate models on complex, real-world tasks and use cases. It includes detailed leaderboards across multiple domains such as academia, office productivity, content moderation, 和 more.

  Content Moderation Leaderboard ([trysansa.com/benchmark?dimension=censorship](https://trysansa.com/benchmark?dimension=censorship))

- **LMArena**
   ([Overview Leaderboard | LMArena](https://lmarena.ai/leaderboard/))

- **OpenCompass**
   ([OpenCompass 司南 - Leaderboard](https://rank.opencompass.org.cn/home))

- **LiveBench**
   ([LiveBench](https://livebench.ai/#/))

  An LLM benchmark designed to avoid test-set contamination and enable objective evaluation, covering reasoning, coding, math, and data analysis.

- **NeMo Evaluator SDK**
   ([NVIDIA-NeMo/Evaluator: Open-source library for scalable, reproducible evaluation of AI models and benchmarks.](https://github.com/NVIDIA-NeMo/Evaluator))

- **LRM-Eval**
   ([LRM-Eval](https://flageval-baai.github.io/LRM-Eval/))

  Text tasks include the following subtasks:

  - **Problem Solving**
    - University course problems, word puzzles, and decoding
  - **Algorithmic Coding**
    - Recently released programming problems
  - **Task Completion**
    - Instruction following, multi-turn instruction following, long-context understanding
  - **Factuality & Refusal**
    - Long-tail knowledge
  - **Safety**
    - Harmful content generation and jailbreaks

------

# Coding Benchmarks

- **SWE-bench**
   ([SWE-bench Leaderboards](https://www.swebench.com/index.html))

  One of the most popular evaluation suites in software engineering—SWE-bench is a benchmark for assessing the ability of large language models (LLMs) to solve real-world software issues from GitHub. The benchmark requires an agent to take in a code repository and an issue description, and then generate a patch that solves the problem.

  - **Details**

    Each sample in the SWE-bench test set comes from a resolved GitHub issue across 12 open-source Python repositories. Each sample is associated with:

    - A pull request (PR) that contains the solution code
    - Unit tests used to verify correctness

    These tests fail before the solution code in the PR is added, and pass afterwards; they are therefore called **FAIL_TO_PASS** tests. Each sample also has **PASS_TO_PASS** tests that pass both before and after the PR to ensure unrelated functionality is not broken.

    For each SWE-bench sample, the agent receives:

    - The original GitHub issue text (the issue description)
    - Access to the codebase

    Based on this, the agent must edit files in the repository to solve the issue. The test cases are *not* visible to the agent.

    The proposed changes are evaluated by running both FAIL_TO_PASS and PASS_TO_PASS tests:

    - If FAIL_TO_PASS tests pass, the issue is considered solved
    - If PASS_TO_PASS tests pass, the patch is considered not to have broken other parts of the codebase

    Only when *both* sets of tests pass is the original GitHub issue considered fully resolved.

  - **SWE-bench Verified**

    A 500-instance human-curated subset released by OpenAI (to address underestimation of agent capabilities on the original SWE-bench dataset).

    Existing agent frameworks often rely heavily on Python-specific tools, leading to overfitting to SWE-bench Verified.

  - **SWE-Bench Pro (Public Dataset)** ([scale.com/leaderboard/swe_bench_pro_public](https://scale.com/leaderboard/swe_bench_pro_public))

    An upgraded version of SWE-Bench Verified.

    A benchmark specifically designed to provide a rigorous and realistic evaluation of AI agents in the domain of software engineering. It addresses several limitations of existing benchmarks by tackling four key challenges:

    1. Data Contamination: Models may have encountered the evaluation code during training, making it difficult to determine whether they are genuinely solving problems or merely recalling memorized solutions.
    2. Limited Task Diversity: Many benchmarks fail to capture the full spectrum of real-world software engineering challenges, often focusing only on simple utility libraries.
    3. Oversimplified Problems: Ambiguous or underspecified issues are frequently excluded from benchmarks, which does not reflect the actual workflow of real-world developers.
    4. Unreliable and Irreproducible Testing: Inconsistent test environments make it hard to verify whether a proposed solution truly works or if passing tests are merely the result of favorable (or incorrect) environment configurations.
  - **SWE-bench Multilingual**

    Aims to evaluate LLMs’ software engineering capabilities across multiple programming languages. SWE-bench Multilingual contains 300 carefully curated software engineering tasks derived from real-world pull requests across 42 GitHub repositories and 9 languages (C, C++, Go, Java, JavaScript, TypeScript, PHP, Ruby, Rust). These repositories span web frameworks, data storage and processing tools, core utilities, and popular libraries.

  - **Multi-SWE-Bench**
     ([Multi-SWE-bench --- Multi-SWE-bench](https://multi-swe-bench.github.io/#/))

    A modified version released by ByteDance Seed to assess LLMs’ problem-solving abilities across multiple programming languages. The dataset contains 1,632 feature-driven software development tasks covering 7 languages: Java, TypeScript, JavaScript, Go, Rust, C, and C++. Evaluation is performed by verifying the project’s built-in tests, using post-PR behavior as the reference solution.

  - **Limitations**

    Dataset-based evaluation is inherently limited, and SWE-bench is no exception. Since the benchmark is built from scrapes of public GitHub repositories, large foundation models pretrained on internet text are highly likely to be contaminated on these tasks. SWE-bench also only covers a narrow slice of medium-risk autonomy; it must therefore be complemented with other evaluation methods.

- **terminal-bench**
   ([Terminal-Bench --- Terminal-Bench](https://www.tbench.ai/))
   ([Artificial Analysis](https://artificialanalysis.ai/evaluations/terminalbench-hard))

  A suite of tasks and an evaluation framework for assessing AI agents performing complex tasks in a terminal environment. Example tasks include compiling and packaging code repositories, downloading datasets and training classifiers on them, and setting up servers. Each Terminal-Bench task consists of:

  - A natural language description
  - A Docker environment
  - A test script to verify success
  - A reference (“ideal”) solution

- **ArtifactsBench**
   ([ArtifactsBench: Bridging the Visual-Interactive Gap in LLM Code Generation Evaluation](https://artifactsbenchmark.github.io/))

  The first automated multimodal benchmark from Tencent for evaluating LLM-generated visual artifacts. It can render dynamic outputs and, via an MLLM judge guided by fine-grained checklists, evaluate both fidelity and interactivity. It contains 1,825 high-quality, challenging prompts across nine categories: game development, SVG generation, web apps, simulations, data science, management systems, multimedia editing, quick utilities, and others. ArtifactsBench is constructed via an eight-stage pipeline: extraction & filtering, human & LLM rewrites, classification & difficulty filtering, few-shot labeling, checklist generation, model generation, human QA & quality control, and final data integration.

- **SWE-Dev**
   ([DorothyDUUU/SWE-Dev](https://github.com/DorothyDUUU/SWE-Dev))

  The first large-scale benchmark and training corpus for **feature-driven development (FDD)**—the practical task of adding *new features* to an existing codebase.

- **LiveCodeBench**
   ([LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code](https://livecodebench.github.io/index.html))
   ([Artificial Analysis](https://artificialanalysis.ai/evaluations/livecodebench))

  Built from regular contests on LeetCode, AtCoder, and Codeforces to provide a holistic benchmark for continuously evaluating LLM performance across diverse coding scenarios. It focuses not only on code generation but also on broader coding capabilities such as self-debugging, code execution, and test-output prediction.

  - **GSO Benchmark**

    A challenging software optimization benchmark for evaluating software engineering agents. Using 102 difficult optimization tasks in 10 codebases, it measures models’ ability to develop high-performance software by comparing runtime improvements against expert-developer baselines.

- **Aider’s polyglot benchmark**
   ([Aider LLM Leaderboards | aider](https://aider.chat/docs/leaderboards/))

  Evaluates LLMs on 225 difficult Exercism coding exercises across C++, Go, Java, JavaScript, Python, and Rust.

- **SciCode**
   ([SciCode Benchmark](https://scicode-bench.github.io/))
   ([Artificial Analysis](https://artificialanalysis.ai/evaluations/scicode))

  Designed to evaluate language models’ ability to generate code that solves real scientific research problems. SciCode spans 16 sub-fields across six disciplines: physics, mathematics, materials science, biology, and chemistry. It mirrors real workflows in scientific research: identifying key concepts and facts, then translating them into computational and simulation code.

  Focus areas:

  1. Numerical methods
  2. Systems simulation
  3. Scientific computing

- **OJBench**
   ([He-Ren/OJBench](https://github.com/He-Ren/OJBench))

  Evaluates LLMs’ contest-level code reasoning capabilities. The dataset focuses on human competitive programming and contains 232 carefully filtered problems from NOI (China National Olympiad in Informatics) and ICPC (International Collegiate Programming Contest). Each problem is labeled with difficulty (easy/medium/hard) based on contestant votes and real submissions. OJBench supports both Python and C++.

- **Roo Code evals**
   ([Evals | Roo Code](https://roocode.com/evals))

- **CodeClash**
   ([CodeClash](https://codeclash.ai/))

  A benchmark for evaluating goal-driven software engineering in AI systems. Contemporary coding benchmarks are task-oriented: models receive explicit instructions and are checked via unit tests. Real software development, however, is **goal-oriented** (“improve user retention”, “reduce costs”, “increase revenue”), requiring autonomous, iterative, often competitive processes.

  In CodeClash, two or more LLM agents compete in multi-round tournaments in a shared “code arena.” During the tournament, each agent iteratively improves its codebase to achieve a high-level competitive objective (e.g., accumulating resources, surviving the longest).

- **METR: Measuring AI Ability to Complete Long Tasks** ([https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/))

  AI performance is evaluated based on the length of tasks an AI agent can successfully complete. A model's capability is characterized by "the duration of tasks—measured in human time—that the model can complete successfully with x% probability." Current models achieve near 100% success rates on tasks that take humans less than 4 分钟之前, but their success rate drops below 10% on tasks requiring humans more than approximately 4 小时之前.

- **Codeforces**

  One of the largest global platforms for algorithm practice and programming contests.

------

## AI Coding Agents

- **LiveSWEBench**
   ([LiveSWEBench](https://liveswebench.ai/))

  A benchmark for evaluating AI agent applications in software engineering.

  It evaluates each assistant on three task types:

  - **Agent Programming**: The assistant receives a high-level task and must complete it fully autonomously.
  - **Goal-Directed Editing**: The assistant receives more specific instructions and the target files to edit (still runs as an agent).
  - **Autocompletion**: The assistant receives partial code fragments that it must complete.

- **DPAI Arena**
   ([DPAI — Developer Productivity AIrena](https://dpaia.dev/))

  The first open, multi-language, multi-framework, multi-workflow benchmark platform for AI coding agents, aimed at measuring their effectiveness on real-world software engineering tasks. It uses a flexible, track-based architecture to support workflows such as patch generation, bug fixing, PR review, test generation, static analysis, and more.

------

# Agentic Benchmarks

- **GAIA**
   ([GAIA Leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard))

  A general-purpose benchmark that evaluates AI assistants’ capabilities in reasoning, multimodal understanding, web browsing, and tool use through real-world problems.

- **Gaia2 Leaderboard**
   ([Gaia2 Agents Evaluation Leaderboard](https://huggingface.co/spaces/meta-agents-research-environments/leaderboard))

  A benchmark designed to measure general agent intelligence. Unlike traditional search-and-execute tasks, Gaia2 runs asynchronously, requiring agents to handle ambiguity and noise, adapt to dynamic environments, collaborate with other agents, and operate under time constraints. It evaluates dimensions such as:

  - Execution (instruction following, multi-step tool use)
  - Search (information retrieval)
  - Ambiguity (unclear or incomplete instructions)
  - Adaptation (dynamic environment changes)
  - Time (time management and scheduling)
  - Noise (irrelevant information, random tool failures)
  - Multi-agent collaboration

- **BrowseComp**
   ([BrowseComp: a benchmark for browsing agents | OpenAI](https://openai.com/index/browsecomp/))

  A benchmark by OpenAI for evaluating browsing agents’ ability to locate complex, hard-to-find information on the internet. It focuses on questions that have short answers and essentially a single correct solution.

  - **BrowseComp-ZH**
     ([BrowseComp-ZH/README-ZH.md](https://github.com/PALIN2018/BrowseComp-ZH/blob/main/README-ZH.md))

    The first high-difficulty benchmark specifically targeting LLMs’ retrieval and reasoning abilities within the Chinese web ecosystem. Inspired by BrowseComp (Wei et al., 2025), it constructs complex multi-hop retrieval and reasoning tasks tailored to the Chinese information environment, where models must handle platform fragmentation, linguistic characteristics, and content moderation.

- **FACTS Benchmark** ([www.kaggle.com/benchmarks/google/facts/leaderboard](https://www.kaggle.com/benchmarks/google/facts/leaderboard))

  A parameterized benchmark designed to evaluate a model's ability to accurately access and utilize its internal knowledge in factual question-answering scenarios. It includes a public set of 1,052 questions and a private set of 1,052 questions.

  A search benchmark designed to assess a model's capability to use search as a tool for retrieving information and correctly integrating it. It comprises a public dataset of 890 entries and a private dataset of 994 entries.

  A multimodal benchmark designed to test a model’s ability to answer prompts related to input images in a factually accurate manner. It consists of a public dataset of 711 items and a private dataset of 811 items.

  Grounding Benchmark – v2: An expanded benchmark for evaluating a model’s ability to provide factually grounded responses within the context of a given prompt.

- **DeepSearchQA** ([www.kaggle.com/benchmarks/google/dsqa/leaderboard](https://www.kaggle.com/benchmarks/google/dsqa/leaderboard))

   A benchmark introduced by Google comprising 900 prompts designed to evaluate agents' ability to perform difficult, multi-step information retrieval tasks across 17 distinct domains. Unlike traditional benchmarks that focus solely on single-answer retrieval or broad factual accuracy, DeepSearchQA features a set of challenging, hand-crafted tasks aimed at assessing an agent’s capacity to execute complex search strategies and generate comprehensive answer lists.

   Each task is structured as a "causal chain," where the discovery of information in one step depends on the successful completion of the previous step, emphasizing long-term planning and contextual retention.

- **DeepResearch Bench** ([https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leaderboard](https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leaderboard))

  A comprehensive benchmark for deep research agents, consisting of 100 PhD-level research tasks. Each task is carefully designed by 22 domain experts across different fields, with 50 tasks in Chinese and 50 in English.


- **xbench**
   ([xbench --- xbench](https://xbench.org/))

  A benchmark co-developed by domestic organizations. It consists of two complementary tracks to measure both frontier intelligence and practical value of AI systems:

  - **AGI Tracking**: Evaluates core model capabilities such as reasoning, tool use, and memory.
  - **Professional Alignment**: A new evaluation system based on workflows, environments, and business KPIs, co-designed with domain experts.

  AGI-tracking assessments measure frontier capabilities; professional-alignment assessments reflect real-world utility by aligning with business KPIs and operational scale in dynamic, domain-specific tasks.

  xbench is built as a dynamic evaluation system with both tracks continuously updated.

  - **xbench-ScienceQA**

    Focuses on foundational scientific knowledge.

  - **xbench-DeepSearch**

    Focuses on search and information-retrieval tool use, tuned for Chinese scenarios.

  - **xbench-Profession-Recruiting**

    Focuses on realistic recruiting workflows and industry standards, covering job requirement breakdown, candidate profiling, candidate experience enrichment, resume screening, network understanding, and open talent search. It evaluates agents on industry understanding, talent sourcing capabilities, and evaluation skills.

  - **xbench-Profession-Marketing**

    Focuses on realistic marketing workflows and standards, primarily KOL search tasks—covering client requirement discovery, KOL matching, and content distribution monitoring and strategy adjustment.

- **IFBench**
   ([IFBench Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/ifbench))

  Evaluates models’ ability to follow precise instructions under 58 diverse, verifiable out-of-distribution constraints, testing whether models can adhere to strict output requirements.

- **Humanity's Last Exam (HLE)**
   ([Humanity's Last Exam](https://lastexam.ai/))
   ([Artificial Analysis](https://artificialanalysis.ai/evaluations/humanitys-last-exam))

  A frontier-level multimodal benchmark aimed to be the “last closed academic benchmark” across a wide range of disciplines. It contains 2,500 high-difficulty questions spanning over 100 subjects. The questions are public; a hidden test set is reserved to detect model overfitting.

  Achieving high accuracy on HLE demonstrates expert-level performance on closed, verifiable problems and cutting-edge scientific knowledge, but does *not* imply autonomous research capability or general AI. HLE is about structured academic questions rather than open-ended research or creative problem-solving. It specifically measures technical knowledge and reasoning.

  Two settings:

  - **w/ tools**: evaluates agentic abilities
  - **w/o tools**: evaluates model intelligence alone

- **τ²-bench**
   ([τ-bench --- τ-bench](https://taubench.com/#home))
   ([Artificial Analysis](https://artificialanalysis.ai/evaluations/tau2-bench))
   (Found to have serious flaws.)

  A simulation framework for evaluating customer service agents across domains, benchmarking AI agents in collaborative real-world scenarios. τ-bench requires agents to **coordinate, guide, and assist users** in complex enterprise domains to achieve shared goals. Domains include overall, retail, telecom, and airline.

  It introduces a new paradigm for evaluating conversational AI via simultaneous simulation of both agent and user, dynamically modifying a shared global state. The telecom domain, for example, tests an agent’s ability to guide users through technical troubleshooting, probing problem-solving and communication skills.

  **Major issues found:**

  1. **Policy compliance**: Tasks whose expected actions violate domain policies (e.g., providing compensation where policy forbids, canceling a flight that has already departed).
  2. **Database accuracy**: Incorrect item IDs, passenger information, or payment methods inconsistent with the actual database.
  3. **Logical consistency**: Impossible scenarios (e.g., redeeming the same item twice when policy forbids it).
  4. **Evaluation ambiguity**: Vague task descriptions causing inconsistent evaluation outcomes.

- **τ²-Bench-Verified**
   ([github.com/amazon-agi/tau2-bench-verified](https://github.com/amazon-agi/tau2-bench-verified))

  A corrected, human-verified version of the original τ²-bench. It fixes issues where task definitions, expected actions, and evaluation criteria were misaligned with domain policies or database contents.

- **FinSearchComp**
   ([FinSearchComp Benchmark](https://randomtutu.github.io/FinSearchComp/))

  The first benchmark specifically for open-ended financial search. Real-world financial decision-making requires three core abilities:

  1. Discovering the right signals
  2. Verifying and integrating multiple sources
  3. Forming evidence-based judgments under time pressure

  FinSearchComp provides an end-to-end evaluation framework for open financial tasks. It covers three subtasks across two subsets (global and Greater China): timely data acquisition, simple historical queries, and complex historical investigations.

- **GDPval-AA** ([artificialanalysis.ai/evaluations/gdpval-aa](https://artificialanalysis.ai/evaluations/gdpval-aa))

  An evaluation framework developed for OpenAI's GDPval dataset. It assesses AI models on real-world tasks across 44 professions and 9 major industries. The benchmark includes 220 tasks that require models to generate diverse outputs—such as documents, slides, charts, 和 spreadsheets—to simulate authentic work deliverables in fields like finance, healthcare, law, and other professional domains.

- **TheAgentCompany**
   ([The Agent Company](https://the-agent-company.com/))

  Evaluates LLM agents performing real-world professional tasks, acting like digital employees—browsing the web, writing code, running programs, and communicating with colleagues.

- **VitaBench**
   ([VitaBench](https://vitabench.github.io/))

  Benchmarks LLM agents with versatile interactive tasks in real-world applications, using frequent life scenarios such as food delivery, dining out, and travel. It has 66 tools and cross-scenario composite tasks, evaluating agents along three dimensions: deep reasoning, tool use, and user interaction.

- **Toolathlon**
   ([Tool Decathlon - Toolathlon](https://toolathlon.xyz/introduction))

  A benchmark for general tool use in real software environments. It covers 32 software applications and 604 tools, each task requiring long-horizon tool use. There are 108 manually designed or scripted tasks, averaging ~20 interaction turns per task.

- **BFCL-V4** ([gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html))

  Stands for the Berkeley Function-Calling Leaderboard, which evaluates the ability of large language models (LLMs) to accurately invoke functions (i.e., tools).

- **MCP-Universe**
   ([mcp-universe.github.io](https://mcp-universe.github.io/))

  A benchmark suite built around real-world Model Context Protocol (MCP) servers.

- **MCPMark**
   ([mcpmark.ai](https://mcpmark.ai/))

  A comprehensive stress-test benchmark suite for MCP, with diverse verifiable tasks to evaluate models and agents in real MCP application scenarios. It includes MCP servers such as Notion, GitHub, Filesystem, Postgres, and Playwright/Playwright-WebArena.

- **MCP Atlas** ([scale.com/leaderboard/mcp_atlas](https://scale.com/leaderboard/mcp_atlas))

  Evaluates language models' ability to handle real-world tool use through the Model Context Protocol (MCP), measuring performance on multi-step workflows. The benchmark comprises 1,000 human-authored tasks, each requiring multiple tool calls selected from over 40 MCP servers and more than 300 tools. Tasks range from single-domain queries that need only 2–3 tools with straightforward chaining, to complex workflows requiring 5+ tools with conditional branching and error handling.

  Each task includes carefully curated distractor tools—plausible but incorrect options—selected by data annotators from the same category as the required tools. The evaluation framework exposes 12–18 tools per task (3–7 required tools plus 5–10 distractors), compelling agents to reason based on tool descriptions rather than resorting to brute-force invocation.

- **SCONE-bench**
   ([red.anthropic.com/2025/smart-contracts/](https://red.anthropic.com/2025/smart-contracts/))

  Anthropic’s first benchmark that measures agents’ ability to exploit smart contracts via the *total value stolen*. For each target contract, the agent must:

  - Identify vulnerabilities
  - Generate an exploit script that, when executed, increases the attacker’s native token balance by at least a threshold

  SCONE-bench avoids synthetic bugs or hypothetical models and is grounded in on-chain value. It provides:

  1. A benchmark of 405 smart contracts with real exploits between 2020–2025 on Ethereum, BNB Chain, and Base (sourced from DefiHackLabs).
  2. A base agent running in a sandboxed environment via MCP tools with a 60-minute time limit to attack each contract.
  3. A Docker-based evaluation framework that runs a forked local chain at a specified block height for reproducibility.
  4. Plug-and-play support for pre-deployment stress-testing: developers can apply agents to audit their contracts before mainnet deployment.

------

## AI Agents (Web / GUI Agents)

- **Online-Mind2Web**
   ([Online_Mind2Web Leaderboard](https://huggingface.co/spaces/osunlp/Online_Mind2Web_Leaderboard))

  A benchmark for evaluating web agents on real sites. It spans 300 tasks across 136 popular websites in multiple domains, and uses an LLM-as-judge (WebJudge) for automatic evaluation. Tasks are divided into three difficulty levels based on the number of human-annotated steps:

  - Easy: 1–5 steps
  - Medium: 6–10 steps
  - Hard: 11+ steps

------

## LLM Memory & Personalization

- **LoCoMo**
   ([LoCoMo](https://snap-research.github.io/locomo/))

  Evaluates ultra-long-term conversational memory in LLM agents, across tasks such as Q&A, event summarization, and multimodal dialogue generation.

  - **Q&A Task**
     The agent must accurately “recall” past context and integrate relevant information into future responses. The benchmark directly evaluates memory via questions, categorized into five reasoning types: single-step, multi-step, temporal, commonsense & world knowledge, and adversarial.
  - **Event-graph Summarization**
     The agent must identify long-range causal and temporal relations in the dialogue to generate empathetic, context-aware responses. Event-graph summarization measures understanding of causality and timelines by requiring the model to extract events associated with each speaker as a graph.
  - **Multimodal Dialogue Generation**
     Dialogue agents must use contextual information from past conversations to produce responses aligned with ongoing narrative logic. LoCoMo evaluates this via multimodal dialogue generation.

- **PERSONAMEM**
   ([PersonaMem](https://zhuoqunhao.github.io/PersonaMem.github.io/))

  A large-scale benchmark for dynamic user modeling and personalized responses. It includes interactions between more than 180 simulated users and LLMs, with up to 60 multi-turn dialogues per user (~1M tokens), across 15 personalization scenarios and 7 categories of real-time user queries.

  Each instance contains:

  - A user persona with static attributes (e.g., demographics)
  - Dynamic attributes (e.g., evolving preferences)

  Users interact with chatbots in multi-turn sessions on topics such as food, travel planning, and counseling. As preferences evolve, the benchmark provides labeled queries to test whether LLMs can deliver the most appropriate responses to first-person user queries.

- **Personalized Deep Research**
   ([arxiv.org/abs/2509.25106](https://arxiv.org/abs/2509.25106))

  The first benchmark to evaluate personalization in deep-research agents. The platform covers 50 diverse research tasks across 10 domains, paired with 25 real user personas that combine structured attributes and dynamic real-world contexts, yielding 250 real user task queries. It proposes a PQR evaluation framework:

  - **P**ersonalization: alignment with user persona
  - **Q**uality: content quality
  - **R**eliability: factual correctness

------

## Visual Grounding & GUI Agents

- **AndroidDaily**
   ([opengelab.github.io/index_zh.html](https://opengelab.github.io/index_zh.html))

  A multidimensional, dynamic benchmark for real-world scenarios. It focuses on six core dimensions of modern life (food, transportation, shopping, housing, information consumption, entertainment), prioritizing popular apps that dominate these categories to ensure realistic outcomes (payments, bookings, etc.) and tight integration between online and offline behavior.

  - **Static testing**: 3,146 operations with task descriptions and step screenshots; the agent must predict each step’s action type and parameters (e.g., click coordinates, text input), focusing on numerical accuracy.
  - **End-to-end testing**: In a fully functional test environment (real devices or emulators), the agent must autonomously complete tasks from start to finish; overall task success rate is the main metric.

- **AndroidWorld**
   ([android_world](https://google-research.github.io/android_world/))

  A fully functional Android environment providing reward signals for 116 programming tasks across 20 real Android apps. Unlike static test sets, AndroidWorld dynamically constructs parameterized, natural-language tasks for testing at scale on realistic workloads.

- **ScreenSpot-V2**
   ([ScreenSpot-V2](https://gui-agent.github.io/grounding-leaderboard/screenspot.html))

  A GUI grounding benchmark that maps natural language instructions to pixel-level targets on the screen.

- **ScreenSpot-Pro**
   ([GUI Grounding Leaderboard](https://gui-agent.github.io/grounding-leaderboard/))

  Focuses on high-resolution professional computer software, evaluating GUI grounding in professional environments (e.g., complex charts in desktop applications).

- **OSWorld-G**
   ([osworld-grounding.github.io](https://osworld-grounding.github.io/))

  Evaluates fine-grained functional component understanding with 564 carefully annotated samples covering tasks such as text matching, element recognition, layout understanding, and precise manipulation.

- **MMBench-GUI**
   ([MMBench-GUI](https://github.com/open-compass/MMBench-GUI))

  A comprehensive benchmark for GUI agents across Windows, macOS, Linux, iOS, Android, and Web. In addition to task success, it proposes the **Efficiency-Quality Area (EQA)** metric to measure efficiency. It covers four levels:

  - GUI content understanding
  - GUI element grounding
  - GUI task automation
  - GUI task collaboration

------

# Intelligence Benchmarks

- **ARC-AGI-2** ([arcprize.org/leaderboard](https://arcprize.org/leaderboard))

  Focuses on tasks that are relatively simple for humans but difficult or even impossible for current AI systems, thereby revealing capability gaps that do not naturally emerge through scaling alone.

  1. All evaluation sets (public, semi-private, and private) now contain 120 tasks each (increased from 100).
  2. Tasks susceptible to brute-force search have been removed from the evaluation sets—specifically, all tasks solved in the 2020 Kaggle competition.
  3. Controlled human testing has been conducted to calibrate task difficulty, ensure Inter-Human Difficulty Disparity (IDD), and verify that at least two humans can solve each task under a pass@2 criterion (aligned with the rules applied to AI systems).
  4. New tasks have been designed based on research insights—including symbolic interpretation, compositional reasoning, and contextual rule learning—to specifically challenge AI reasoning systems.

- **MMLU-Pro**
   ([MMLU-Pro Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/mmlu-pro))

  A multi-task understanding dataset to rigorously evaluate LLMs. It contains 12k complex questions across many disciplines. Each question has 10 options and emphasizes reasoning-centric problems.

- **Frames**

  A comprehensive dataset for evaluating Retrieval-Augmented Generation (RAG) systems on factuality, retrieval accuracy, and reasoning. It has 824 challenging multi-hop questions requiring information from 2–15 Wikipedia articles across topics like history, sports, science, animals, and health.

- **SealQA**
   ([vtllms/sealqa](https://huggingface.co/datasets/vtllms/sealqa))

  Seal-0 evaluates search-augmented LLMs on factoid questions where web search results are conflicting, noisy, or unhelpful, focusing on the most challenging queries on which chat models like GPT-4.1 have near-zero accuracy.

- **Stanford HELM**
   ([Capabilities - HELM](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard))

- **Zenmux**
   ([Benchmark - ZenMux](https://zenmux.ai/benchmark))

  Provides a cost-performance curve and a score leaderboard. It tests every available provider channel for each model and uses the text version of **Humanity’s Last Exam** (Scale AI) as the primary benchmark.

- **AA-LCR**
   ([Artificial Analysis Long Context Reasoning Benchmark Leaderboard](https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning))

  A benchmark for long-context reasoning across multiple documents. Inputs are up to 100k tokens (cl100k_base), and models must integrate information from multiple positions across documents to infer answers. It covers seven text-only document types: corporate reports, industry reports, government consultations, academic papers, legal documents, marketing materials, and survey reports.

- **Fiction-liveBench**
   ([Fiction.liveBench Sept 29 2025](https://fiction.live/stories/Fiction-liveBench-Sept-06-2025/oQdzQvKHw8JyXbN87/home))

  Evaluates long-context understanding for story writing. It uses a curated set of long, complex storylines and a large set of verified quizzes generated from compressed versions of these stories.

- **Context Arena**
   ([contextarena.ai](https://contextarena.ai/))

- **Context-Bench**
   ([Letta Leaderboard](https://leaderboard.letta.com/))

  Evaluates long-horizon document operations, entity-relation tracking, and multi-step information retrieval.

- **Needle in a Haystack**
   ([LLM-NeedleInAHaystack/README_CN.md](https://github.com/Lianues/LLM-NeedleInAHaystack/blob/main/README_CN.md))

  A recall benchmark using a “needle in a haystack” method to test LLM recall:

  1. Construct a test text of fixed length, randomly inserting multiple 4-digit numbers (1000–9999).
  2. Ask the model to extract all 4-digit numbers and output them in JSON in order of appearance.
  3. Score using an edit-distance (Levenshtein) based metric.

- **Hallucination Leaderboard**
   ([LLM Hallucination Leaderboard](https://huggingface.co/spaces/vectara/leaderboard))

  Evaluates the frequency of hallucinations when models summarize documents. Short documents are given and models must summarize *only* based on the given facts. The metric is factual consistency with the source, *not* absolute real-world correctness.

- **AA-Omniscience**
   ([Artificial Analysis Omniscience Index](https://artificialanalysis.ai/evaluations/omniscience))

  Covers 6 domains—Business; Humanities & Social Sciences; Health; Law; Software Engineering; Science, Engineering & Math—with 6,000 questions across 42 topics. It reports:

  - **Accuracy** (percentage correct)
  - **Hallucination rate** (wrong answers as a percentage of all non-abstentions)
  - **Omniscience Index** (+1 for correct, –1 for incorrect, 0 for abstain)

- **R-HORIZON**
   ([R-HORIZON](https://reasoning-horizon.github.io/))

  The first systematic framework for evaluating and improving long-chain reasoning in LRMs. It introduces **Query Composition**, which constructs dependencies among questions to transform isolated tasks into complex multi-step chains:

  1. **Information extraction**: extract core values and variables from individual questions.
  2. **Dependency construction**: embed earlier answers into later conditions.
  3. **Chain reasoning**: the model must sequentially solve all sub-questions to get the final answer.

- **TRUEBench**
   ([TRUEBench](https://huggingface.co/spaces/SamsungResearch/TRUEBench))

  A benchmark for instruction following in LLMs, evaluating their effectiveness as productivity assistants.

- **SpeechMap**
   ([SpeechMap.AI Explorer](https://speechmap.ai/))

  Explores the boundaries of AI speech. It tests how models respond to sensitive and controversial prompts across providers, countries, and topics. While most benchmarks focus on what models *can* do, SpeechMap measures what they *won’t* do: what they avoid, refuse, or block.

- **WeirdML** ([htihle.github.io/weirdml.html](https://htihle.github.io/weirdml.html))

  Presents large language models (LLMs) with a series of unusual and non-standard machine learning tasks designed to require careful reasoning and genuine understanding to solve. WeirdML aims to evaluate an LLM’s ability to:
  Truly understand the properties of the data and the nature of the problem
  Design an appropriate machine learning architecture and training setup for the problem, 和 generate executable PyTorch code that implements a working solution
  Debug and iteratively improve the solution over five rounds based on terminal output and test-set accuracy
  Make effective use of limited computational resources and time

- **SimpleQA**

  A benchmark released by OpenAI for evaluating LLMs on short factual questions. It contains 4,326 questions, each designed to have a single correct answer and be easy to score.

------

## AI4S (Specialized Domains)

- **AIME25**
   ([AIME 2025 Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/aime-2025))

  A dataset of math answers sourced from the 2025 AIME I exam (American Invitational Mathematics Examination). Suitable for QA tasks; fewer than 1,000 records; English only.

- **HMMT 2025**

  Problems from the February 2025 HMMT (Harvard-MIT Mathematics Tournament) used in the MathArena Leaderboard.

- **AMO-Bench**
   ([AMO-Bench](https://amo-bench.github.io/))

  Released by Meituan LongCat, this benchmark includes 50 original competition-level problems authored by experts, with difficulty on par with or exceeding the IMO.

- **IMO Bench**
   ([imobench.github.io](https://imobench.github.io/))

  A benchmark reviewed by IMO medalists and mathematicians (10 gold and 5 silver medals in total). IMO problems demand rigorous multi-step reasoning and creativity beyond formula application. IMO-Bench targets this difficulty level and consists of:

  - **IMO-AnswerBench**: large-scale correct-answer testing
  - **IMO-ProofBench**: higher-level evaluation targeting proof writing
  - **IMO-GradingBench**: aims to advance automatic evaluation of long mathematical solutions

- **FrontierMath** ([epoch.ai/frontiermath](https://epoch.ai/frontiermath))

  Contains 350 original mathematical problems (including 50 at the highest difficulty level, Tier 4), spanning from challenging undergraduate-level questions to problems that may take expert mathematicians several days to solve. Requirements:

  1. Clear and verifiable answers: Each problem must have a well-defined solution that can be objectively verified.
  2. Guessing resistance: Solutions must be "guess-proof"—i.e., random guessing or simple brute-force approaches should have virtually no chance of success.
  3. Computational feasibility: For computationally intensive problems, a script must be provided demonstrating how the answer can be obtained using only standard knowledge in the relevant field. The cumulative runtime of these scripts on standard hardware must be under one minute.

- **PutnamBench**
   ([PutnamBench Leaderboard](https://trishullab.github.io/PutnamBench/leaderboard.html))

  A benchmark for formalized mathematical reasoning using Putnam competition problems. It includes 1,712 hand-constructed formal problems sourced from the William Lowell Putnam Mathematical Competition, the premier North American undergraduate math contest.

  - 660 problems are formalized in Lean 4
  - 640 in Isabelle
  - 412 in Coq

- **GPQA Diamond**
   ([GPQA Diamond Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/gpqa-diamond))

  The hardest 198 questions from the GPQA benchmark, explicitly “Google-proof,” requiring true scientific expertise instead of search skills. These graduate-level physics, biology, and chemistry questions can only be reliably answered by PhD-level experts, making them ideal for testing deep scientific reasoning.

- **CritPt**
   ([CritPt Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/critpt))

  A benchmark of 71 comprehensive research-level physics reasoning challenges.

- **FrontierScience** ([openai.com/index/frontierscience/](https://openai.com/index/frontierscience/))

  A new benchmark introduced by OpenAI designed to evaluate expert-level scientific capabilities. FrontierScience is written and verified by experts in physics, chemistry, 和 biology, 和 consists of hundreds of questions crafted to be challenging, original,  meaningful. It includes two tracks: an Olympiad track, which measures Olympiad-style scientific reasoning abilities, 和 a Research track, which assesses real-world scientific research capabilities.

- **Frontier-CS** ([frontier-cs.org/leaderboard](https://frontier-cs.org/leaderboard))

  is an unsolved, open-ended, verifiable, diverse benchmark designed to evaluate AI performance on challenging computer science problems. The tasks included are those that even researchers find difficult to solve, lack known optimal solutions, or require deep domain expertise to attempt.

  The benchmark comprises two leaderboards: Algorithmic and Research.
  The Algorithmic track includes optimization tasks, constructive tasks, interactive tasks.
  The Research track spans six major computer science domains: Operating Systems (OS), High-Performance Computing (HPC), Artificial Intelligence (AI research tasks), Databases (DB), Programming Languages (PL), 和 Security (cybersecurity and vulnerability analysis).

------

# Visual Understanding & Reasoning

- **MMMU**
   ([mmmu-benchmark.github.io](https://mmmu-benchmark.github.io/))

  A large-scale, multi-disciplinary multimodal benchmark for expert-level AGI. It contains 115k multimodal questions collected from university exams, quizzes, and textbooks across six core areas: arts & design, business, science, health & medicine, humanities & social sciences, and technology & engineering. It spans 30 disciplines and 183 subfields with 30 highly heterogeneous image types such as charts, diagrams, maps, tables, sheet music, and chemical structures. It focuses on advanced perception and reasoning grounded in domain expertise.

- **MMMU-Pro**
   ([MMMU-Pro Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/mmmu-pro))

  Extends MMMU with 10-choice questions and visual-only input formats (problems embedded in screenshots or photos). The benchmark has 3,460 questions across six core disciplines, requiring models to jointly process visual and textual information in more realistic scenarios.

- **MATH-Vision**
   ([MATH-Vision Leaderboard](https://mathllm.github.io/mathvision/#leaderboard))

  Evaluates multimodal mathematical reasoning, with 3,040 high-quality visual math problems from real competitions. It covers 16 math subfields and five difficulty levels.

- **CharXiv** ([charxiv.github.io/#leaderboard](https://charxiv.github.io/#leaderboard))

  A comprehensive evaluation suite comprising 2,323 natural, challenging, 和 diverse charts sourced from scientific papers. CharXiv includes two types of questions: (1) descriptive questions that assess the ability to identify basic chart elements, and (2) reasoning questions that require synthesizing information across complex visual components within the chart. To ensure high quality, all charts and questions have been carefully selected, curated, and verified by human experts.

- **ROME**
   ([BAAI/ROME](https://huggingface.co/datasets/BAAI/ROME))

  A visual reasoning benchmark with eight sub-tasks and 281 high-quality questions, each verified so that the image is necessary for answering:

  - **Academia** – problems from university courses
  - **Charts** – figures from recent papers, reports, and blog posts
  - **Puzzles & Games** – Raven’s matrices, word puzzles, gameplay
  - **Memes** – reworked internet memes
  - **Geo** – geographic location inference
  - **Recognition** – fine-grained recognition
  - **Multi-image** – spot-the-difference and video frame reordering
  - **Spatial** – relative positions, depth/distance, height, etc.

- **ZeroBench**
   ([zerobench.github.io](https://zerobench.github.io/))

  An extremely challenging visual benchmark for modern multimodal models, with 100 designer-crafted questions and 334 sub-questions representing the reasoning steps required to solve the main questions.

- **VisuLogic**
   ([VisuLogic](https://visulogic-benchmark.github.io/VisuLogic/))

  A domestic benchmark of 1,000 human-verified questions across six categories (e.g., quantitative changes, spatial relations, attribute comparisons) testing visual reasoning from multiple angles.

- **OCRBench v2**
   ([ocrbench_v2](https://99franklin.github.io/ocrbench_v2/))

  A benchmark for evaluating multimodal models on visual text localization and reasoning. It contains 10k human-validated Q&A pairs with a large proportion of hard samples, covering 31 scenarios including street views, receipts, formulas, charts, etc.

- **MMLongBench-Doc**
   ([Hugging Face Space](https://huggingface.co/spaces/OpenIXCLab/mmlongbench-doc))
   ([MMLongBench-Doc](https://mayubo2333.github.io/MMLongBench-Doc/))

  A long-context multimodal document understanding benchmark. It contains 1,091 expert-annotated questions built on 135 long PDF documents (average 47.5 pages and 21,214 tokens). Answers require evidence from:

  - Multiple modalities (text, images, charts, tables, layout)
  - Multiple locations (pages)

  33% of questions are cross-page; 22.5% are unanswerable to test hallucination tendency.

- **Video-MMMU** ([videommmu.github.io/](https://videommmu.github.io/))

  A multimodal, multidisciplinary benchmark designed to evaluate Large Multimodal Models’ (LMMs) ability to acquire and apply knowledge from videos. Video-MMMU features a curated collection of 300 expert-level videos and 900 human-annotated questions spanning six professional disciplines (covering 30 subfields). Knowledge acquisition is assessed through question-answer pairs aligned with three cognitive stages: Perception, Comprehension, 和 Adaptation.

  Each video is accompanied by three question-answer pairs corresponding to the three stages of knowledge acquisition:
  Perception: identifying key information relevant to the knowledge presented,
  Comprehension: understanding the underlying concepts, and
  Adaptation: applying the acquired knowledge to novel scenarios.

  In addition, the benchmark evaluates models’ delta accuracy—the improvement in performance after watching the relevant instructional video—thereby measuring how effectively LMMs learn from video content.

------

# OCR & Embedding Evaluation

[Supercharge your OCR Pipelines with Open Models](https://huggingface.co/blog/ocr-open-models)

In OCR model evaluation, performance can vary widely across document types, languages, etc.

- **OmniDocBench**
   ([OmniDocBench/README_zh-CN](https://github.com/opendatalab/OmniDocBench/blob/main/README_zh-CN.md))

  A benchmark for real-world diverse document parsing. Widely used for its variety of document types including books, magazines, and textbooks, with well-designed evaluation metrics supporting tables in HTML and Markdown formats.

- **olmOCR-Bench**
   ([olmocr/bench](https://github.com/allenai/olmocr/tree/main/olmocr/bench))

  A benchmark that has proven very effective for evaluating English-language OCR.

- **CC-OCR**

- **Embedding Leaderboard**
   ([MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard))

  A widely used benchmark for embeddings based on the MTEB suite.

------

# Video / Audio Generation & Role-play Evaluation

- **DesignArena**
   ([designarena.ai/leaderboard](https://www.designarena.ai/leaderboard))

- **Speech-DRAME**
   ([Anuttacon/speech_drame](https://github.com/Anuttacon/speech_drame))

  A benchmark for evaluating AI-generated speech in voice role-playing scenarios.

- **UNO-Bench**
   ([UNO-Bench](https://meituan-longcat.github.io/UNO-Bench/))

  A unified benchmark for exploring full-model multimodal composition rules. Almost 100% of questions require joint understanding of audio and visual information. Beyond traditional multiple-choice questions, it introduces a multi-step open-ended QA format for complex reasoning.

  Key properties:

  - **Multi-source**: primarily crowdsourced real-world photos and videos, plus copyright-free websites and public datasets.
  - **Topic diversity**: social, cultural, artistic, daily life, literature, science, etc.
  - **Real audio**: dialogues recorded by 20+ human speakers for rich, realistic acoustic features.

- **Vue**

  A video understanding benchmark from ByteDance.

  - **VUE-STG**: Evaluates spatiotemporal grounding (STG) in realistic settings.
    1. Video durations range from ~10 seconds to 30 minutes (long-context reasoning).
    2. Most queries are converted to noun phrases while retaining sentence-level expressivity.
    3. Annotations are human-precise for all true temporal segments and bounding boxes.
    4. Evaluation uses optimized vIoU/tIoU/vIoU-Intersection metrics for multi-segment spatiotemporal grounding.
  - **VUE-TR-V2**: A Video QA benchmark with more balanced video duration distributions and query formats closer to real user behavior.

- **Moral RolePlay**
   ([RolePlay_Villain](https://github.com/Tencent/DigitalHuman/tree/main/RolePlay_Villain))


  A benchmark for moral role-playing, focusing on character behaviors and ethical boundaries in digital humans.










