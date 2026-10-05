# awesome-llm-benchmarks

[中文](./README.md) | [日本語](./README-jp.md)

Summary of Large Model Evaluation Benchmarks: This section compiles various evaluation benchmarks and resources for large models, covering comprehensive evaluations, coding, agents, reasoning, multimodality, and industry-specific benchmarks. The list is continuously updated.

Tips:

Scores from different benchmarks—even those in different domains—are highly correlated. See *[Benchmark scores are well correlated, even across domains](https://epoch.ai/data-insights/benchmark-correlations)*.

Benchmarks serve as the de facto standard for evaluating model performance; however, they are not infallible and may even contain systematic flaws. Notable examples include τ²-bench, MMLU-Pro, GPQA, and HLE. (See the paper: *[Fantastic Bugs and Where to Find Them in AI Benchmarks](https://arxiv.org/abs/2511.16842)*). Detailed descriptions of these cases are provided below.

Challenges and limitations in evaluating LLMs are discussed in epoch.ai’s blog post *[Why benchmarking is hard](https://epoch.ai/gradient-updates/why-benchmarking-is-hard)*. Anthropic has found that infrastructure configurations can cause fluctuations of several percentage points in agentic coding benchmark results (see *[Quantifying infrastructure noise in agentic coding evals](https://www.anthropic.com/engineering/infrastructure-noise)*).

Testing self-deployed models: Refer to Hugging Face’s official tutorials on using inspect-ai and lighteval ([huggingface.co/docs/inference-providers/guides/evaluation-inspect-ai](https://huggingface.co/docs/inference-providers/guides/evaluation-inspect-ai)), ([github.com/huggingface/lighteval](https://github.com/huggingface/lighteval)), and ([huggingface.co/docs/lighteval/main/en/index](https://huggingface.co/docs/lighteval/main/en/index)).

Hugging Face also provides an open benchmark index where tasks can be browsed by language and tag, and searched via their descriptions ([huggingface.co/spaces/OpenEvals/open_benchmark_index](https://huggingface.co/spaces/OpenEvals/open_benchmark_index)).

epoch.ai conducts third-party quality assessments of benchmarks ([epoch.ai/benchmarks/search?reviewed=verified&reviewed=flawed&reviewed=not-enough-info](https://epoch.ai/benchmarks/search?reviewed=verified&reviewed=flawed&reviewed=not-enough-info)).

- Relative Adoption Metric (RAM) ([atomproject.ai/relative-adoption-metric](https://atomproject.ai/relative-adoption-metric))

  The Relative Adoption Metric (RAM). The “RAM score” is a more appropriate metric for evaluating download statistics of newly released open-source models across different sizes.  
  Score = (Model’s download count) / (Median download count of the top 10 models in the same size category). A score of 1 indicates that the model is likely to rank among the top 10 most downloaded models in its size category.

  Data collection (median values are used):  
  Top 10 models per size category ranked by total downloads (from Hugging Face).  
  Cumulative download counts at key milestones: day 7, 14, 30, 60, 90, 180, and 365 after release.  
  Overall cumulative Hugging Face download counts for each model over time.

## Comprehensive Evaluations

- Artificial Analysis ([AI Model & API Providers Analysis | Artificial Analysis](https://artificialanalysis.ai/))

  - AA Intelligence ([Artificial Analysis Intelligence Index | Artificial Analysis](https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index))

    A comprehensive benchmark integrating seven challenging evaluations to fully assess AI capabilities in mathematics, science, programming, and reasoning. It aggregates performance across 10 evaluations: GDPval-AA v2, Terminal-Bench 2.1, τ³-Bench Banking, HLE, AA-Omniscience Accuracy, SciCode, GPQA Diamond, AA-LCR, CritPt, and AA-Omniscience Non-Hallucination (listed in descending order of weight).
  - Artificial Analysis Capability Indices ([artificialanalysis.ai/models/capabilities](https://artificialanalysis.ai/models/capabilities))

    These newly introduced industry-specific indices enable comparison of model performance across key sectors, including programming, finance and accounting, law, healthcare, strategy and operations, engineering, and economics.

    Each index is built based on common job tasks identified in the O*NET occupational classification system. These tasks include financial modeling, legal research and contract review, clinical decision support, and patient medical record documentation. Relevant capability metrics are derived from each task; the benchmarks most representative of each field are selected and assigned weights according to their frequency of occurrence within that sector.
  - Artificial Analysis Openness Index ([artificialanalysis.ai/evaluations/artificial-analysis-openness-index](https://artificialanalysis.ai/evaluations/artificial-analysis-openness-index))

    A standardized, independent metric measuring the openness of AI models in terms of accessibility and transparency. Openness goes beyond merely making model weights downloadable; it also encompasses licensing agreements, data, and methodologies. Models scoring 100 on this index feature open weights, permissive licenses, and full publication of training code, pre-training data, and post-training data—allowing users not only to utilize the models but also to fully reproduce their training process or draw inspiration from the creators’ methods to develop their own models.
- llm-stats ([llm-stats.com/](https://llm-stats.com/))

    A comprehensive evaluation website featuring a benchmark summary ([llm-stats.com/benchmarks](https://llm-stats.com/benchmarks)).
- vals.ai ([www.vals.ai/home](https://www.vals.ai/home))

    A comprehensive evaluation website categorized into sections such as general, legal, finance, healthcare, mathematics, academia, education, coding, and gaming.

    - Vals Index

      Measures AI models’ ability to perform real-world tasks in finance, law, and software engineering. It calculates a weighted average of model performance across key industries, with weights reflecting each sector’s contribution to the U.S. economy.
- SEAL LLM Leaderboards ([scale.com/leaderboard](https://scale.com/leaderboard))

    Evaluates the agent capabilities, cutting-edge performance, safety, and public perception of the latest LLMs.
- Epoch AI ([epoch.ai/benchmarks](https://epoch.ai/benchmarks))

    Offers multiple benchmarks.

    The Epoch Capability Index (ECI) consolidates scores from various AI benchmarks into a single “general capability” scale, enabling comparison of models even over extended periods when individual benchmarks reach saturation.
- LMArena ([arena.ai/leaderboard/](https://arena.ai/leaderboard/))

    A crowdsourced evaluation platform utilizing blind voting by real users and Elo rankings to compare the overall performance of different large models in real-world interactions.
- OpenCompass ([OpenCompass Leaderboards](https://rank.opencompass.org.cn/home))

    An open-source evaluation platform for developers and users of large models, providing unified assessments across multiple capability dimensions, evaluation sets, and leaderboards. It includes closed-source LLM leaderboards, academic benchmarks, and multimodal evaluation lists.
- LiveBench ([LiveBench](https://livebench.ai/#/))

    An LLM benchmark designed specifically to prevent test set contamination and ensure objective evaluation; it covers reasoning, coding, mathematics, and data analysis.
- CAIS AI Dashboard ([dashboard.safe.ai/](https://dashboard.safe.ai/))

    Comprises five indicators: text, vision, risk, automation, and remote labor index (measuring automation levels in remote work).
- NeMo Evaluator SDK ([NVIDIA-NeMo/Evaluator: Open-source library for scalable, reproducible evaluation of AI models and benchmarks.](https://github.com/NVIDIA-NeMo/Evaluator))

    An open-source SDK developed by NVIDIA for evaluating large models; it enables large-scale, reproducible benchmarking of any model compatible with standard APIs within a unified framework.

## Coding Benchmarks

- SWE-Together ([togetherbench.com/](https://togetherbench.com/))

  A multi-turn benchmark set reconstructed from real user and agent programming sessions, comprising 109 repository-level tasks.
- ITBench ([artificialanalysis.ai/evaluations/itbench-aa](https://artificialanalysis.ai/evaluations/itbench-aa))

  Designed to evaluate agents in Site Reliability Engineering (SRE) scenarios, specifically for Kubernetes fault root-cause analysis. This evaluation includes 59 Kubernetes fault tasks: 40 are from IBM’s public release, while the remaining 19 were provided privately by the ITBench team. Each task is tested three times. In each scenario, agents receive an offline snapshot of a Kubernetes fault, containing alert information, system events, call traces, performance metrics, and application topology; they must then output a structured JSON diagnostic report identifying all key objects causing the fault, such as deployments, services, pod groups, namespaces, and network policies.
- CursorBench ([cursor.com/cn/cursorbench](https://cursor.com/cn/cursorbench))

  An evaluation built upon real Cursor sessions from engineering teams; its tasks originate from actual Cursor usage rather than public code repositories. Many tasks are drawn from internal codebases and controlled sources, thereby reducing the likelihood that models have encountered them during training.
- FrontierCode ([cognition.com/blog/frontier-code-1.1](https://cognition.com/blog/frontier-code-1.1)) (human-written tasks)

  A benchmark designed to measure whether models can truly meet the standards of high-quality production codebases. It is the first benchmark to assess code mergeability, evaluating end-to-end code quality—including correctness, test quality, scope compliance, coding style, and adherence to codebase standards. Difficulty levels are categorized as Extended, Main, and Diamond. The Diamond level contains the 50 most difficult tasks; Main includes the top 100 tasks (including those in Diamond); Extended comprises all 150 tasks.
- Kilo Bench ([kilo.ai/leaderboard](https://kilo.ai/leaderboard))

  Built upon Terminal-Bench’s terminal-intensive task data, it includes 89 real-world tasks covering scenarios such as git operations, password analysis, and QEMU automation. Model testing is conducted exclusively via the Kilo testing framework.
- OpenHands Index ([index.openhands.dev/home](https://index.openhands.dev/home))

  A comprehensive benchmark for evaluating AI coding agents’ performance on real-world software engineering tasks; it also reports on models’ performance and cost-effectiveness. Models are assessed across five categories: Problem Solving (bug fixing), New Project Development (building new applications), Frontend Development (UI design), Testing (test case generation), and Information Gathering.
- APEX-SWE ([www.mercor.com/apex/apex-swe-leaderboard/](https://www.mercor.com/apex/apex-swe-leaderboard/))

  Designed to evaluate the actual daily workflows of software engineers. Unlike unit-level or single-repository bug-fixing benchmarks, it comprises 200 cases spanning two complementary scenarios: (1) Integration tasks requiring end-to-end system construction and deployment across heterogeneous services, assessing models’ ability to orchestrate workflows and synchronize data across diverse services; (2) Observability tasks demanding debugging via production-grade telemetry, evaluating models’ capacity to diagnose and resolve real-world software engineering production issues.
- Ramp SWE-Bench ([labs.ramp.com/swebench#explore](https://labs.ramp.com/swebench#explore)) (human-reviewed tasks)

  A private coding benchmark derived from Ramp’s backend engineering work and based on production environments. Inspired by SWE-Bench, it serves as a behavioral tool to study how coding agents handle tasks assigned to them by Ramp engineers. It includes 80 tasks covering areas such as credit card authorization, bill payments, expense reporting, accounting, procurement, fund management, fraud detection, and agent operations.
- BridgeBench ([www.bridgemind.ai/bridgebench](https://www.bridgemind.ai/bridgebench))

  A vibe coding benchmark introduced by BridgeMind; it uses standardized tasks to evaluate models’ performance in programming scenarios such as debugging, algorithms, refactoring, code generation, UI design, and security.
- ALE-Bench ([sakanaai.github.io/ALE-Bench-Leaderboard/](https://sakanaai.github.io/ALE-Bench-Leaderboard/))

  A benchmark for evaluating AI systems’ performance in score-based algorithmic programming competitions. Drawing upon real tasks from the AtCoder Heuristic Contest (AHC), it presents computationally challenging optimization problems lacking known exact solutions—such as routing and scheduling problems.
- Android Bench ([developer.android.com/bench](https://developer.android.com/bench)) (human-curated tasks)

  Evaluates LLMs’ ability to solve real-world Android development problems; it comprises 100 tasks. Android Bench places greater emphasis on libraries (58% of tasks).
- Kotlin Benchmark ([kotlinlang.org/benchmark/](https://kotlinlang.org/benchmark/))

  An official benchmark introduced by JetBrains to assess AI coding agents’ performance on Kotlin software engineering tasks. Following SWE-bench methodology, it focuses on repository-level Kotlin engineering tasks and includes 105 engineering tasks sourced from active open-source repositories. Each task requires the AI agent to interpret authentic problem descriptions, analyze project context, and generate viable patches. All solutions undergo rigorous validation within a containerized environment; only those passing the prescribed tests are marked as successfully resolved.

- Vision2Web ([vision2web-bench.github.io/](https://vision2web-bench.github.io/))

  A benchmark designed to evaluate whether multimodal coding agents can build real websites based on visual prototypes and structured requirements. It comprises 193 tasks aimed at measuring end-to-end web development capabilities in real-world environments. Each task provides multimodal inputs such as UI prototype images, requirement descriptions, and development resources; agents must then generate executable websites that meet both functional and visual fidelity criteria.
- CADBench ([www.seldon.global/blog/cadbench](https://www.seldon.global/blog/cadbench))

  CADBench measures whether cutting-edge AI models can complete native mechanical design tasks requiring long-term planning within Autodesk Fusion, as well as whether the generated geometries, feature histories, and constraints can pass deterministic verification.
- Animation Bench ([www.physera.ai/research/animation](https://www.physera.ai/research/animation))

  This benchmark aims to test whether coding agents can replicate a webpage’s behavioral logic rather than merely its visual appearance. It includes 48 tasks; models receive 12 to 24 static frames sampled from animations, along with timestamp information if available. They also receive an HAR file containing all HTML, CSS, JavaScript, font, and image resources required to load the webpage. Models do not receive the website’s source project files, videos, or any temporal descriptions beyond the frames and their timestamps.

### AI Coding Agents

- Terminal-Bench 4.0 ([www.tbench.ai/](https://www.tbench.ai/)) (Manually designed and validated)

  A dataset containing 89 carefully curated tasks spanning multiple domains such as software engineering, system administration, data science, security, and scientific computing. It is used to evaluate AI agents’ performance in completing complex tasks within terminal environments. Each task in Terminal-Bench includes task instructions, a Docker environment, a test suite, and a reference solution written by humans. Agents must interact with the environment via command-line tools—such as executing Bash commands and editing files—to explore and solve problems. The evaluation is result-driven; only whether a task is ultimately completed (verified via tests) matters, with no restrictions on the specific implementation methods used by agents.
- SWE Atlas ([labs.scale.com/leaderboard/sweatlas-refactoring](https://labs.scale.com/leaderboard/sweatlas-refactoring))

  SWE Atlas is a benchmark suite used to evaluate AI coding agents across various professional software engineering tasks. Rather than measuring just a single skill, it includes three leaderboards targeting different yet complementary capabilities within the software development lifecycle. Tasks are drawn from 11 production code repositories written in four programming languages—Go, Python, C, and TypeScript—selected from SWE-Bench Pro.

  1. [Codebase Question Answering](https://labs.scale.com/leaderboard/sweatlas-qna) – Understanding complex codebases through runtime analysis and multi-file reasoning. This category contains 124 tasks; agents gain access to code repositories inside Docker containers and must answer a series of in-depth technical questions regarding how the system operates. These questions are designed to demand strong agent reasoning capabilities: agents must run software, trace execution flows across multiple files, and synthesize conclusions accordingly.
  2. [Test Writing](https://labs.scale.com/leaderboard/sweatlas-tw) – Writing meaningful production-grade tests for specified functionalities within a code repository. This category includes 90 tasks; agents access code repositories inside Docker containers where certain critical tests for a key workflow are missing. These tasks possess agent-design characteristics: prompts provide only high-level descriptions of the workflow or behavior to be tested. Agents must autonomously explore the codebase to determine exactly what tests need to be written and where, then execute those tests and submit their results. Tests must cover the behavior described in the prompts, be limited strictly to relevant code, and be written in a clear, maintainable manner consistent with the repository’s conventions and best practices.
  3. [Refactoring](https://labs.scale.com/leaderboard/sweatlas-refactoring) – Refactoring code to improve performance and readability while preserving functionality. This category contains 70 tasks; agents access code repositories inside Docker containers where refactoring prompts are provided. These tasks are inherently agent-oriented: prompts describe at a high level the desired refactored structure and specify which code segments should be extracted, integrated, or reorganized. Agents must independently explore the codebase, comprehend its existing architecture, perform refactoring operations across multiple files, and ensure all existing tests continue to pass. The refactored code must correctly reassemble the specified components, eliminate redundant or obsolete code, update documentation to reflect changes, and avoid introducing regressions or disruptive modifications.
- Next.js AI Agent Evaluations ([nextjs.org/evals](https://nextjs.org/evals))

  Performance data for various AI programming agents on Next.js code generation and migration tasks, including metrics such as success rates and execution times.
- Code Review Bench ([codereview.withmartian.com/](https://codereview.withmartian.com/))

  A benchmark used to evaluate AI code review tools.
- Real-SWE ([realswe.withspecific.com/](https://realswe.withspecific.com/))

  A benchmark for assessing whether cutting-edge AI models can handle real-world enterprise codebases. Each test task originates from a private production environment codebase we obtained permission to use from an actual enterprise.

### Long-Horizon Tasks

- DeepSWE ([deepswe.datacurve.ai/blog/deepswe-v1-1](https://deepswe.datacurve.ai/blog/deepswe-v1-1)) (Manually reviewed tasks)

  A long-horizon software engineering benchmark featuring four key characteristics:

  - Uncontaminated: All tasks are newly written rather than adapted from existing commits or pull requests.
  - Highly diverse: Tasks span 91 code repositories written in five languages—TypeScript, JavaScript, Python, Go, and Rust—covering a broad range of scenarios.
  - Real-world complexity: Prompt lengths are only half those of SWE-bench Pro, yet the amount of code required to produce solutions is 5.5 times greater, and output token counts are roughly double.
  - Reliable verification: Verifiers test software behavior by manually writing code, rather than focusing on implementation details.

- SWE-Marathon ([www.swe-marathon.org/](https://www.swe-marathon.org/))

  An AI agent benchmark designed for ultra-long-horizon software engineering tasks, comprising 20 long-duration tasks spanning four categories of software engineering domains. Each task comes with a dedicated executable environment, human-provided reference solutions, and a multi-layered validation suite.
- MirrorCode ([epoch.ai/MirrorCode](https://epoch.ai/MirrorCode))

  Designed to evaluate AI models’ ability to handle long-term coding tasks. In MirrorCode tests, AI models must re-implement an entire program from scratch without access to its original source code. The AI-generated solutions must produce outputs identical to those of the original program across all end-to-end tests, including reserved test sets. The 25 target programs in MirrorCode span multiple computational domains: Unix utilities, data serialization and query tools, bioinformatics, interpreters, static analysis, cryptography, and compression techniques.

  AI models undergo sandbox isolation, preventing them from accessing the Internet or the original codebases, thereby eliminating any possibility of cheating. Additionally, numerous end-to-end tests remain invisible to the models during coding, so they cannot simply create lookup tables to mimic the original program’s outputs.

### Research Scenarios

- FrontierSWE v2 ([www.frontierswe.com/](https://www.frontierswe.com/))

  A programming benchmark featuring ultra-long-term, open-ended technical challenges; it includes 34 tasks such as optimizing compilers or training state-of-the-art models for protein prediction. On average, agents spend 11 hours per task, yet almost none manage to complete them. Among these, the `granite_inf` task evaluates agents’ ability to optimize model implementations end-to-end within an inference engine.
- kernelbench hard ([kernelbench.com/hard](https://kernelbench.com/hard)) (manually designed)

  A focused upgrade of KernelBench v3, testing whether cutting-edge models can efficiently write Triton/CUDA/CUTLASS/CUTE-DSL/PTX code without cheating. Tests are conducted locally on an RTX Pro 6000 Blackwell device, featuring seven manually designed tasks utilizing a real code agent command-line interface as the testing framework. Tasks include: FP8 GEMM, TopK, Sonic MoE forward pass, KimiDeltaAttention, paged attention decoding, Kahan Softmax, and W4A16 GEMM. All require a deep understanding of the SM120 architecture.
- PostTrainBench ([posttrainbench.com/](https://posttrainbench.com/)) (flawed)

  This benchmark measures the degree of automation in AI research by testing whether AI agents can successfully perform post-training optimization on other language models. Each agent is provided with four base models (Qwen 3 1.7B, Qwen 3 4B, SmolLM3-3B, and Gemma 3 4B), one H100 GPU, and a 10-hour time limit to enhance model performance via post-training optimization.

  Flaw: Models exploit shortcuts by using inference traces from stronger models for supervised fine-tuning (i.e., distillation). In reality, top-performing models on the leaderboard—Claude Opus 4.8 and GLM 5.2—indeed employ this strategy; however, distillation fundamentally contradicts the benchmark’s goal of “recursive self-improvement” since post-training optimization must not rely on external, stronger models.
- WeirdML ([htihle.github.io/weirdml.html](https://htihle.github.io/weirdml.html))

  This benchmark presents LLMs with a series of peculiar and unconventional machine learning tasks requiring careful reasoning and genuine comprehension to solve. It aims to test the following capabilities of LLMs:  
  Truly understanding data characteristics and problem essence  
  Designing appropriate machine learning architectures and training configurations, then generating executable PyTorch code to implement solutions  
  Debugging and refining solutions across five iterations based on terminal outputs and test-set accuracy  
  Making optimal use of limited computational resources and time
- InferenceBench ([inferencebench.ai/](https://inferencebench.ai/))

  Designed to evaluate whether cutting-edge coding agents can optimize LLM service workloads within a fixed computational budget. In each test, agents receive a base LLM, an NVIDIA H100 GPU, a time limit, and a specific objective. They must then set up and run an OpenAI-compatible inference server that maximizes core metrics for the given scenario while passing quality and integrity checks. The goal is to achieve acceleration relative to PyTorch baseline models, either in a specific bottleneck scenario or across a balanced set of overall performance metrics. Four scenarios are covered: prefill latency, decoding latency, throughput, and balanced service performance.
- MLS-Bench ([mls-bench.com/](https://mls-bench.com/))

  This benchmark assesses whether AI systems can devise generalizable and scalable machine learning methods. It comprises 140 tasks across 12 domains, including language models, vision and generation, reinforcement learning, robotics, machine learning systems, AI in scientific research, optimization algorithms, time-series analysis, and causal inference. Each task revolves around a clearly defined research problem; agents must propose modular improvements—such as novel loss functions, attention mechanism variants, samplers, or routing rules—and these improvements are then evaluated for effectiveness across different models, datasets, and random seeds.
- Reward Hacking Bench ([www.rewardhacking.io/](https://www.rewardhacking.io/))

  This benchmark examines potential cheating behaviors of models during post-training tasks. Models are assigned a coding task: training a small base model with limited capabilities to improve its scores across various benchmarks. It evaluates whether models take shortcuts or evade oversight mechanisms during this process.

- RSI-Exam ([rsi-exam.ai/](https://rsi-exam.ai/)) (manually designed tasks)

  Designed to evaluate whether AI agents can self-improve over time and generalize their capabilities to unseen data. The testing process involves allowing agents to conduct hours of autonomous experiments regarding problem-solving methods or mechanisms driving model operation, followed by a final test on a hidden test set. There are 6 main domains, comprising a total of 88 tasks. Domain breakdown: Physical Science and Engineering, AI Models and Agents, Optimization, Planning and Control, Systems and Hardware, Life Sciences and Medicine, Finance, Law, and Business.

### Model Usage

- Claude Code Opus Performance Tracker ([marginlab.ai/trackers/claude-code/](https://marginlab.ai/trackers/claude-code/))

  Detects statistically significant performance degradation of Claude Code Opus on SWE tasks; updated daily: conducts daily benchmarking on a curated subset of SWE-Bench-Pro.

- AI STUPID LEVEL ([aistupidlevel.info/about](https://aistupidlevel.info/about))

  An independent AI model performance monitoring platform that continuously tracks AI model performance. It objectively measures model capabilities by having multiple models perform real coding tasks (`including algorithm implementation, debugging, code refactoring, optimization, and error recovery`), thereby detecting potential performance changes (“drift”) that might otherwise go unnoticed. Multiple trials (n=5) are run for each model; confidence intervals are calculated, and statistical tests are employed to distinguish genuine changes from mere noise. A 7-axis scoring system is used: correctness (35%), adherence to specifications (15%), code quality (15%), efficiency (10%), stability (10%), rejection rate (10%), and recovery capability (5%). Each model runs the coding tasks 5 times using different random seeds; the median score is then calculated, and a 95% confidence interval is derived via the t-distribution.

- Platform Coding Plan Evaluation ([coding.15o.cc/](https://coding.15o.cc/))

  Compares time-to-first-token, average TPS, and total response time across various vendors and models.

## Agentic Benchmarks

- Agents’ Last Exam ([agents-last-exam.org/](https://agents-last-exam.org/))

  A benchmark designed specifically to evaluate the performance of AI agents in long-term, economically valuable, and verifiable real-world tasks. Developed by over 250 industry experts, ALE assesses non-physical industries as defined by O*NET/SOC 2018 (the U.S. federal occupational classification system). The test framework is based on a taxonomy of 55 subdomains grouped into 13 industry clusters, encompassing more than 1,500 specific tasks. ALE is designed as a dynamically evolving benchmark: its task library continues to expand as new workflows and industries are added.
- FACTS Benchmark ([www.kaggle.com/benchmarks/google/facts/leaderboard](https://www.kaggle.com/benchmarks/google/facts/leaderboard))

  A parameterized benchmark measuring a model’s ability to accurately retrieve its internal knowledge in factual Q&A scenarios; it includes a public set of 1,052 questions and a private set of 1,052 questions.  
  A search benchmark testing a model’s ability to utilize search as a tool for retrieving and correctly integrating information; it includes a public dataset of 890 entries and a private dataset of 994 entries.  
  A multimodal benchmark assessing a model’s ability to respond factually to prompts related to input images; it includes a public dataset of 711 entries and a private dataset of 811 entries.

  FACTS Grounding: Evaluates LLMs’ ability to generate factually accurate responses based on provided lengthy documents; it checks whether LLM responses are entirely grounded in the given context and whether they correctly integrate information from lengthy context documents.
- MCPMark ([mcpmark.ai/](https://mcpmark.ai/))

  A comprehensive stress-testing benchmark suite for MCP, featuring diverse verifiable tasks aimed at evaluating models and agents in real-world MCP application scenarios. It includes the following MCPs: Notion, Github, Filesystem, Postgres, Playwright, Playwright-WebArena.
- MCP Atlas ([scale.com/leaderboard/mcp_atlas](https://scale.com/leaderboard/mcp_atlas))

  Evaluates language models’ ability to utilize real-world tools via the Model Context Protocol (MCP), measuring their performance in multi-step workflows. It comprises 1,000 human-written tasks, each requiring multiple tool calls to resolve; these tools originate from over 40 MCP servers and more than 300 distinct tools. Tasks range from simple single-domain queries needing just 2–3 tools to complex workflows requiring over 5 tools, conditional branching, and error handling. Each task includes carefully selected distractor tools that appear plausible but are actually incorrect; these distractors are chosen by annotators from the same categories as the required tools. The framework provides 12–18 tools per task (3–7 required tools plus 5–10 distractors), forcing agents to reason based on tool descriptions rather than making blind calls.
- PinchBench ([pinchbench.com/](https://pinchbench.com/))

  Measures how well LLMs perform as the “brains” behind OpenClaw agents. Real tasks are assigned to agents: scheduling meetings, writing code, handling emails, researching topics, and managing files. It includes 23 distinct task categories, each defined in Markdown files with YAML frontmatter.
- Claw-Eval ([claw-eval.github.io/#/](https://claw-eval.github.io/#/)) (Human-reviewed and verified)

  Contains 104 tasks in both Chinese and English (32 in Chinese + 72 in English), along with 19 simulated services featuring error injection. It employs a three-dimensional scoring system: completion, robustness, and safety. Score = Safety × (0.80 × Completion + 0.20 × Robustness). If a task receives a safety score of zero, its total score becomes zero as well.
- WildClawBench ([internlm.github.io/WildClawBench/](https://internlm.github.io/WildClawBench/))

  Each task runs within a real OpenClaw instance; agents gain access to a real bash shell, real file system, real browser, as well as real email and calendar services. It includes 60 original tasks meticulously crafted to test agents’ abilities in instruction following, multimodal reasoning, long-term planning, code generation, and debugging.
- ClawsBench ([clawsbench.benchflow.ai/#results](https://clawsbench.benchflow.ai/#results))

  A high-fidelity simulated workspace for rigorously evaluating agents—featuring Gmail, calendar, documents, cloud storage, and Slack. It includes 44 structured tasks covering single-service, cross-service, and safety-critical scenarios.
- ClawMark ([claw-mark.com/leaderboard](https://claw-mark.com/leaderboard))

  A benchmark for collaborator agents designed to work alongside humans across multiple workdays and various services. It encompasses 100 tasks spanning 13 professional domains, utilizing a fully rule-based scoring system—no LLMs are used as judges.
- PawBench ([agentscope-ai.github.io/PawBench/](https://agentscope-ai.github.io/PawBench/))

  Evaluates the performance of (Model × Harness) combinations on production-like tasks; the same set of 150 tasks is run across multiple (Model × Harness) pairings, allowing independent assessment of both model and harness contributions. It includes 3 Agent Harnesses: Hermes, OpenClaw, and QwenPaw.

  Tasks are labeled across 5 dimensions:

  Application scenarios: e.g., office collaboration, software engineering, automated scripting, multimodal content generation.  
  Atomic capabilities: e.g., tool invocation, Skill usage, planning, logical reasoning, self-verification.  
  Complexity: L1 / L2 / L3, preventing high scores solely through easy tasks.  
  Input modality: Distinguishes purely text-based tasks from multimodal tasks involving images, audio, video, etc.  
  Operating environment: Differentiates offline sandbox tasks from web-search/web-retrieval tasks requiring internet access.

### White-collar Economic Value Tasks

For more information, please refer to *[What do “economic value” benchmarks tell us?](https://epoch.ai/blog/what-do-economic-value-benchmarks-tell-us)*.

- GDPval-AA ([artificialanalysis.ai/evaluations/gdpval-aa](https://artificialanalysis.ai/evaluations/gdpval-aa))

  An evaluation framework developed for OpenAI’s GDPval dataset. It tests AI models’ performance on real-world tasks across 44 occupations and 9 major industries. Comprising 220 tasks, it requires models to generate diverse outputs such as documents, slides, charts, and spreadsheets, thereby simulating actual work outcomes in finance, healthcare, legal services, and other professional fields.
- $OneMillion-Bench ([xbench.org/profession/onemillion](https://xbench.org/profession/onemillion))

  Designed to cover real-world application scenarios in China and abroad that possess high economic value, distinctiveness, and are amenable to automated evaluation. The name “One Million Dollars” originates from estimates based on official salary data: multiplying the estimated time required for each task by the hourly wage of senior experts in respective fields reveals that completing all 200 tasks would incur an expert labor cost of nearly $1 million. This benchmark aims to address a more practical question: whether AI can truly and reliably replace humans in performing high-value tasks without any errors throughout the entire process.
- AA-Briefcase ([artificialanalysis.ai/articles/aa-briefcase](https://artificialanalysis.ai/articles/aa-briefcase)) (Manually designed tasks)

  Designed to assess models’ ability to perform real-world knowledge-based tasks within complex projects. This test requires models to complete multi-week knowledge-work projects, each involving numerous interrelated tasks and thousands of input files. Models are evaluated within a coherent long-term project framework where tasks progress weekly, all relying on shared contextual information and requiring outputs such as financial models, boardroom presentations, and design prototypes that meet real-world requirements. These tasks were meticulously crafted over several months by experts in data science, product management, and corporate strategy from companies including Google, McKinsey & Company, and Boston Consulting Group; they are derived from actual professional experiences.

  Models must process hundreds of input files per task, including Slack chat logs, emails, corporate documents, meeting minutes, and large-scale data exports. In total, AA-Briefcase contains nearly 2,000 original files; the email and Slack export data alone include over 3,500 emails and 25,000 messages. These information sources are fragmented and disorganized, often containing contradictory information typical of real-world scenarios, thereby testing whether models can handle ambiguities and uncertainties inherent in actual knowledge-based work.
- AA-AnalystAgent ([artificialanalysis.ai/evaluations/aa-analyst-agent](https://artificialanalysis.ai/evaluations/aa-analyst-agent))

  Specifically designed to evaluate models’ quantitative analysis capabilities on real-world spreadsheets and documents. It comprises 80 questions spanning 14 business and scientific domains, covering topics such as healthcare expenditure reports, trade and commodity statistics, hydrological and meteorological data, government funding allocations, energy cost modeling, financial modeling, environmental reports, and project scheduling. These questions are drawn from five workflow categories observed in actual analysts’ work: data source identification and diagnosis, data filtering and summarization, ratio/trend/sensitivity analysis, profit-and-loss modeling, and cash flow/balance sheet/valuation modeling.
- Remote Labor Index (RLI) ([scale.com/leaderboard/rli](https://scale.com/leaderboard/rli))

  The Remote Labor Index (RLI) is a benchmark measuring AI agents’ ability to perform economically valuable multimedia remote tasks on professional freelance platforms in the real world. It includes 240 tasks, with questions compiled bottom-up from 358 verified freelancers on Upwork. Agents are required to complete tasks involving visual outputs, such as web design, product art creation, and video editing. They must comprehend and generate complex multi-file deliverables encompassing dozens of distinct file types, including documents, audio files, videos, 3D models, and CAD files.
- APEX-Agents ([www.mercor.com/apex/apex-agents-leaderboard/](https://www.mercor.com/apex/apex-agents-leaderboard/)) ([artificialanalysis.ai/evaluations/apex-agents-aa](https://artificialanalysis.ai/evaluations/apex-agents-aa))

  Measures whether cutting-edge AI agents can perform long-duration, cross-application tasks across three professional service roles: investment banking analysts, management consultants, and corporate lawyers.

  - APEX ([www.mercor.com/apex/apex-v1-leaderboard/](https://www.mercor.com/apex/apex-v1-leaderboard/))

    Evaluates whether advanced models possess the capability to perform economically valuable tasks within four occupational categories: investment banking assistants, management consultants, attorneys at major law firms, and primary care physicians (MDs).
  - The AI Consumer Index (ACE) ([www.mercor.com/apex/ace-leaderboard/](https://www.mercor.com/apex/ace-leaderboard/))

    Assesses cutting-edge AI models’ performance on everyday consumer tasks such as shopping, dining, gaming, and DIY activities.
- AutomationBench ([zapier.com/benchmarks](https://zapier.com/benchmarks))

  A test for automating complex agent workflows simulating SaaS applications. It evaluates AI agents’ ability to execute end-to-end workflows using 47 real-world tools spanning six business functions: sales, marketing, operations, customer service, finance, and human resources. This test framework is built upon actual patterns observed in over 2 billion monthly tasks performed by 3.7 million companies. Each task initiates a miniature simulated company and assigns the agent a request typical of what a real employee would receive—including CRM records, inbox conversations, spreadsheets, and customer support tickets—along with various pitfalls such as outdated data entries, nearly identical names, and hidden rules embedded within the inbox—after which the resulting outputs are scored. The scoring criteria focus not on the agent’s responses but on the data it generates.

  - AutomationBench-AA ([artificialanalysis.ai/evaluations/automationbench-aa](https://artificialanalysis.ai/evaluations/automationbench-aa))

    Unlike the leaderboard provided by Zapier, the primary scores in AutomationBench-AA reflect the percentage of tasks a model successfully completes without violating any safety constraints.

- EnterpriseOps-Gym-AA ([artificialanalysis.ai/evaluations/enterprise-ops-gym-aa](https://artificialanalysis.ai/evaluations/enterprise-ops-gym-aa))

  This benchmark evaluates whether LLM agents can complete multi-step enterprise workflows that depend on specific states. Each test task is confined to a single business domain: everyday collaboration tools such as email, calendars, team collaboration platforms, and cloud storage services; as well as core business systems like customer service, human resources, and IT service management. The eighth domain, “Mixed Domains,” features tasks requiring interaction with multiple systems simultaneously. In all tests, agents operate within a real-time sandbox environment and must utilize available tools to perform actual operational tasks. Scoring is based on the final state of the underlying databases rather than conversation logs, ensuring scores accurately reflect whether tasks were successfully completed. No partial credit is awarded; a task is deemed successful only when all validators confirm the resulting state meets requirements.

### AI Agents

- ClawArena ([github.com/aiming-lab/ClawArena?tab=readme-ov-file](https://github.com/aiming-lab/ClawArena?tab=readme-ov-file))

  A multi-session real-world benchmark for AI programming agents, comprising 64 scenarios across 8 domains: technology/HR, hospitals, NGOs, clinical settings, content creation, finance, human resources, and campuses. It includes 1,879 evaluation rounds integrating multiple-choice reasoning and execution validation. Agents must also handle multi-session contexts, requiring them to reason over workspace files, multi-channel chat logs, and dynamically injected updates throughout the evaluation.

### DeepResearch

- FutureSearch-Deep Research Bench (DRB) ([evals.futuresearch.ai/](https://evals.futuresearch.ai/))

  This benchmark assesses LLM agents’ research capabilities online. It features 169 diverse real-world tasks, each accompanied by 10 to 100,000 offline-stored web pages for agents to search through and reason upon.

### Visual Localization & GUI Agents

- Cua-Bench ([cua.ai/cuabench](https://cua.ai/cuabench)) (manually designed and reviewed tasks)

  Tailored for the electrical engineering field, this benchmark evaluates AI agents’ ability to perform complex tasks on desktop computers primarily via keyboard and mouse, or on mobile devices using touchscreen interactions. It includes 25 KiCad circuit design tasks crafted by experts; KiCad is a comprehensive electronic design automation suite featuring extensive shortcut commands, multiple interconnected editors, and numerous modal dialog boxes. Cua-Bench consists of three components:

  1. **Base Images** — pre-packaged Windows, Linux, macOS, and Android environments containing all applications and dependencies required for running the benchmark.
  2. **Task Dataset** — verifiable, dynamically changing computer usage environments spanning multiple operating systems.
  3. **Evaluation/Training Toolkit** — tools for executing the benchmark, generating data, and testing agents.

- cua-speedrun ([cuaspeedrun.com/](https://cuaspeedrun.com/))

  This platform measures both the time required for models to complete each task within the computer usage domain and the associated operational costs incurred by model invocations.

## Intelligence Benchmarks

- Conceptual Reasoning Index (CRI) ([conceptualreasoning.ai/](https://conceptualreasoning.ai/)) (human-verified tasks)

  A conceptual reasoning benchmark comprising three tests. When empirical evidence is limited and no (practically verifiable) answers exist, models must rely heavily on logical reasoning to draw conclusions. We refer to this as conceptual reasoning. To measure LLMs’ progress in this ability, we developed three benchmark tests: LMCA, ACCoRD, and DTBench capabilities.

  LMCA (Language Model Conceptual Argumentation) is a carefully curated dataset rated by experts, containing conceptual arguments on diverse topics such as decision theory, philosophy, and risks associated with advanced AI. Focusing on the argumentation process helps circumvent the difficulty of verifying final answers to conceptual questions. It includes 560 stance texts and 1,461 arguments corresponding to these texts. A model’s ability to evaluate the quality of arguments is measured by comparing its scores to those assigned by human experts.

  ACCoRD (Assessment of Consistency in Conceptual Reasoning) measures the degree of logical consistency between beliefs and preferences expressed by models regarding conceptual questions. For instance, if we ask a model for the probability P(A) of event A occurring, and then ask for the probability P(A&B) of events A and B occurring simultaneously, do these two values satisfy P(A) ≥ P(A&B)? All consistency constraints in this dataset require models to provide numerical probability estimates or preference rankings.

  DTBench capabilities (Decision Theory Benchmark) is a dataset of 407 manually crafted multiple-choice questions designed to evaluate models’ reasoning abilities in decision theory scenarios involving accurate predictions about their own behavior or interactions with (near) copies.

- ARC-AGI-3 ([arcprize.org/leaderboard](https://arcprize.org/leaderboard))

  Focuses on tasks that are relatively easy for humans but difficult or even impossible for AI, thereby revealing capability gaps that cannot naturally emerge merely through “scaling up.”

- dig.bench ([digbench.ai/](https://digbench.ai/))

  dig.bench is a benchmark for measuring scientific discovery abilities. It comprises 70 games designed to test whether agents can experimentally discover unknown rules governing each game; every game contains hidden transformation rules that agents must uncover through interaction and experimentation. All games are presented in text form, placing them within the natural domain of language models: models can explore without any visual distractions, so dig.bench tests pure discovery capabilities. These games are categorized into 7 difficulty levels.

- SimpleBench ([simple-bench.com/](https://simple-bench.com/))

  A multiple-choice text benchmark for LLMs; individuals possessing non-specialist (high school-level) knowledge perform better on this benchmark than state-of-the-art models. SimpleBench contains over 200 questions covering temporal and spatial reasoning, social intelligence, and what we call linguistic adversarial robustness (or brainteasers).

- AA-Omniscience ([Artificial Analysis Omniscience Index | Artificial Analysis](https://artificialanalysis.ai/evaluations/omniscience)) (contains significant flaws)

  Comprises 6,000 questions across 42 topics in 6 domains (“Business,” “Humanities & Social Sciences,” “Health,” “Law,” “Software Engineering,” and “Science, Engineering & Mathematics”). Three metrics are used: accuracy (percentage of correct answers), hallucination rate (percentage of incorrect answers among all non-abstained responses), and Omniscience Index (+1 for correct answers, -1 for incorrect answers, 0 for abstained responses).

  Significant flaws: Many questions in this benchmark have been identified by the community as problematic.

- Bullshit Benchmark ([petergpt.github.io/bullshit-benchmark/viewer/index.html](https://petergpt.github.io/bullshit-benchmark/viewer/index.html))

  Features 55 completely illogical “nonsensical” questions; it evaluates whether models opt to refute these questions or earnestly attempt to answer them.

- Pencil Puzzle Bench ([ppbench.com/](https://ppbench.com/))

  A framework for evaluating LLMs’ reasoning abilities via pencil puzzles; it includes 300 puzzles belonging to 20 different types.

- AttuneBench ([public.attunebench.com/](https://public.attunebench.com/))

  Designed to measure LLMs’ emotional intelligence; it is built upon 200 real-world multi-turn human-AI dialogues. Models are scored across various capabilities, including emotion perception, comprehension, utilization to guide thought processes, and emotion regulation during interactions. This benchmark adopts the same dimensions psychologists use to assess human emotional intelligence, allowing us to translate this complex concept into concrete, measurable behaviors. We also evaluate models’ response generation abilities as well as dialogue-level metrics such as trajectories of emotional change and participants’ self-reported dialogue objectives.

### Hallucinations and Long-Context Recall

Regarding long-context recall and reasoning issues, refer to this article: *[Evaluating Long Context (Reasoning) Ability](https://nrehiew.github.io/blog/long_context/)*

- HalluHard ([halluhard.com/](https://halluhard.com/))

  A multi-turn hallucination benchmark comprising 950 questions across four domains: legal cases (250), research questions (250), medical guidelines (250), and programming (200). It employs a user LLM to generate engaging follow-up questions, then measures model performance across three dialogue rounds (the initial question plus two follow-ups). HalluHard aims to elicit open-ended responses while requiring models to base factual claims on cited sources. This design ensures the benchmark specifically targets hallucinations (unsubstantiated factual errors) rather than other aspects of responses. For legal, research, and medical domains, five claims per response are extracted and evaluated individually; for programming, each response is judged as a whole. The verification process involves extracting claims, searching the web for evidence, and obtaining full-text sources (including PDF parsing) to confirm whether cited materials support generated content.

- AA-LCR ([Artificial Analysis Long Context Reasoning Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning))

  A benchmark designed specifically to evaluate a language model’s ability to perform reasoning across multiple long documents. The model is required to read an input of 100,000 tokens (measured using the cl100k_base tokenizer), integrate information from multiple locations within the input documents, and derive an answer accordingly. It aims to realistically reflect the reasoning tasks that knowledge workers expect language models to perform. It covers seven types of plain text documents: corporate reports, industry reports, government consultations, academic literature, legal documents, marketing materials, and survey reports.
- Context Arena (MRCR v2) ([contextarena.ai/](https://contextarena.ai/)) (flawed)

  The data source is MRCR released by OpenAI. OpenAI’s MRCR test aims to evaluate large language models’ (LLMs) ability to handle complex conversation histories. Its key aspects include:  
  Core task: Identifying and distinguishing multiple identical pieces of information (“needles”) within a lengthy conversation (“haystack”).  
  Setup: Inspired by Google’s [MRCR evaluation](https://arxiv.org/pdf/2409.12640v2), this version inserts 2, 4, or 8 identical requests (e.g., “Write a poem about tapirs”) along with distracting requests. The key information/distracting information is generated by GPT-4o to ensure natural integration.  
  Challenge: The model must retrieve a specific instance based on sequence order (e.g., the second poem), requiring careful tracking of the conversation. It must also prepend a specific random code (hash value) to its answer.

  Flaw: Relying on just 1,000 samples, Microsoft MAI’s technical report shows they managed to raise scores from 60% to over 90% (in reality, they achieved a 90% score using a smaller model, which actually makes the situation worse).
- LOCA-bench ([github.com/hkust-nlp/LOCA-bench](https://github.com/hkust-nlp/LOCA-bench))

  Designed to evaluate the performance of language agents under extreme yet controllable context growth scenarios. Upon receiving a task prompt, LOCA-bench utilizes automated and scalable environment state control to adjust the agent’s context length. It expands the context length to any desired scale while preserving the original task semantics.

### AI4S (Specialized Domains)

- MathArena ([matharena.ai/?view=problem](https://matharena.ai/?view=problem))

  MathArena is a platform for evaluating LLMs’ performance on the latest mathematics competitions, aiming to rigorously test their reasoning and generalization abilities when faced with new mathematical problems they’ve never encountered during training. To assess performance, each model runs four times on every problem; we then calculate its average score and total cost (in USD) across all runs. The displayed cost represents the average expense for the model to run once on all problems in a single competition.

  It includes benchmarks such as ArXivMath, IMProofBench, MathArenaApex, Visual Math, Final-Answer Comps, Proof-Based Comps, Project Euler, BrokenArXiv, and ArXivLean.

  ArXivMath: Designed to evaluate LLMs’ performance in tackling mathematical research problems drawn from recent arXiv papers. (Manually curated tasks)

  IMProofBench: Aims to evaluate language models’ research-level mathematical reasoning abilities via proof-based problems. Models have access to multiple tools, including web search and code execution capabilities.

  MathArenaApex: A set of 12 highly challenging problems selected from major mathematics competitions held in 2025.

  Visual Math: Problems from the Kangaroo Mathematics Competition. Final-Answer Comps and Proof-Based Comps: Problems from recent major mathematics competitions.

  Project Euler: A problem collection merging mathematics and programming, focusing on solving complex problems through a combination of mathematical insight, algorithmic thinking, and programming skills.

  BrokenArXiv: Tests models’ reliability in mathematical reasoning. Problems are extracted from recent arXiv papers and slightly altered to become seemingly plausible yet provably false statements. Models earn points if they refuse to prove such statements and explicitly identify them as false in their current form. (Manually curated tasks)

  ArXivLean: A benchmark for conducting research-level mathematical theorem proving in Lean; all samples are taken from the latest arXiv papers.
- FrontierMath ([epoch.ai/frontiermath/tiers-1-4](https://epoch.ai/frontiermath/tiers-1-4))

  Contains 350 original mathematical problems (including 50 of the highest difficulty level 4), ranging from challenging university-level problems to ones that might require expert mathematicians several days to solve. Requirements:

  1. Clear and verifiable answers  
  2. Resistance to guessing: Answers must possess “guess-proof” properties, meaning random attempts or simple brute-force methods have almost no chance of success  
  3. Computational feasibility: When solving computation-intensive problems, scripts must be provided demonstrating how to obtain answers using only standard domain knowledge. These scripts must run for less than one minute on standard hardware in total.
- MathScienceBench ([math.science-bench.ai/benchmarks/](https://math.science-bench.ai/benchmarks/))

  An AI benchmark targeting research-level mathematical problems. Active researchers submit PhD-level mathematical problems situated within a research context to test large models’ performance in highly difficult mathematical reasoning tasks.
- MathDuels ([mathduels.ai/](https://mathduels.ai/))

  A self-play mathematics benchmark where each cutting-edge model must both solve problems crafted by other models and generate problems for them; thus, difficulty increases as the overall capability of participating models rises. Based on these two roles, the system calculates two scores: problem-solving ability score and problem-generation ability score.

- PutnamBench ([trishullab.github.io/PutnamBench/leaderboard.html](https://trishullab.github.io/PutnamBench/leaderboard.html))

  A benchmark for evaluating formal mathematical reasoning in the Putnam Mathematical Competition. It comprises 1,712 manually constructed formal problems sourced from the William Lowell Putnam Competition, North America’s premier undergraduate mathematics contest. Of these, 660 problems are formalized using Lean 4, 640 using Isabelle, and 412 using Coq.
- GPQA Diamond ([artificialanalysis.ai/evaluations/gpqa-diamond](https://artificialanalysis.ai/evaluations/gpqa-diamond)) (severely flawed)

  The 198 most difficult problems within the GPQA benchmark, specifically designed to be “Google-proof” by requiring genuine scientific expertise rather than mere search skills. These graduate-level problems in physics, biology, and chemistry can only be consistently solved by domain experts holding doctoral degrees; thus, they serve as excellent indicators of true scientific reasoning capabilities.

  Numerous errors were identified during OCR recognition and data entry; the overall data processing workflow proved disastrous. (Source: [Humanity's Last Hallucination : A Forensic Audit of the Scientific Insolvency in GPQA and HLE](https://zenodo.org/records/18293568))
- Humanity's Last Exam ([lastexam.ai/](https://lastexam.ai/)) ([Artificial Analysis](https://artificialanalysis.ai/evaluations/humanitys-last-exam)) (severely flawed)

  A multimodal benchmark situated at the forefront of human knowledge, intended to serve as the final closed-book academic benchmark covering a vast array of disciplines. It contains 2,500 highly challenging problems spanning over 100 fields. While these problems are publicly released, a separate test set remains undisclosed to evaluate potential model overfitting.

  Achieving high accuracy on HLE would demonstrate that a model performs at an expert level on closed-book, verifiable problems involving cutting-edge scientific knowledge; however, this alone does not imply autonomous research capabilities or “Artificial General Intelligence.” HLE evaluates structured academic problems rather than open-ended scientific inquiry or creative problem-solving, thus serving as a measure of technical knowledge and reasoning ability.

  Two variants exist: (w/ tools) with tools (testing agentic capabilities) and (w/o tools) without tools (testing the model’s intrinsic intelligence).

  Numerous errors were identified during OCR recognition and data entry; the overall data processing workflow proved disastrous. (Source: [Humanity's Last Hallucination : A Forensic Audit of the Scientific Insolvency in GPQA and HLE](https://zenodo.org/records/18293568))

  - HLE-Diamond ([lastexam.ai/blog/hle-diamond](https://lastexam.ai/blog/hle-diamond))

    A carefully curated subset of the HLE dataset, consisting of exactly 1,000 problems.
- CritPt ([CritPt Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/critpt))

  A benchmark designed to assess LLMs’ performance on research-grade physics reasoning tasks; it includes 71 comprehensive research challenges.
- PhyArena (HiPhO) ([phyarena.github.io/](https://phyarena.github.io/))

  A benchmark for evaluating the physical reasoning capabilities of LLMs and MLLMs. HiPhO stands for High School Physics Olympiad benchmark.
- Terminal-Bench-Science ([www.terminal-bench-science.ai/](https://www.terminal-bench-science.ai/)) (human-reviewed tasks)

  This benchmark gauges AI agents’ capabilities via a series of complex workflows drawn from scientific research and meticulously selected by experts. Currently, it comprises 70 tasks spanning life sciences, physical sciences, earth sciences, mathematics, and engineering.
- BioMysteryBench ([www.vals.ai/benchmarks/biomysterybench](https://www.vals.ai/benchmarks/biomysterybench))

  Designed to test whether language models can analyze biological datasets to reconstruct missing information—for instance, determining which tissue a sample originates from. Each task begins with real biological data from which certain metadata has been removed; models must then employ computational analysis to recover this missing data by examining files, running command-line tools, writing code, and querying biological databases. Tasks span sequencing, expression analysis, variant detection, epigenomics, metagenomics, proteomics, and metabolomics.

### Industry-Specific Benchmarks

#### Medicine

- CHI-Bench ([actava.ai/benchmarks](https://actava.ai/benchmarks))

  Evaluates long-term healthcare workflows across three key domains: prior authorizations in medical institutions, payer utilization management, and healthcare management. For each task, agents are presented with clinical cases within a highly accurate simulation environment integrating 20 medical applications and 87 MCP tools. Agents must leverage a managed healthcare operations manual containing over 1,290 documents, utilize these tools, and produce outputs appropriate to their respective roles to ultimately resolve each case.
- Medical Long Context Reasoning (MLCR) benchmark ([www.wisedocs.ai/blogs/medical-long-context-reasoning](https://www.wisedocs.ai/blogs/medical-long-context-reasoning)) (human-designed tasks)

  Assesses how effectively models handle various questions commonly posed by professionals when reviewing medical cases, particularly when dealing with longer documents. Ten synthetic medical cases based on real-world scenarios are provided; each case contains 50–150 cross-disciplinary medical summaries. For each case, 250 questions of varying difficulty levels are posed to the LLMs. Additionally, to gauge the impact of redundant context, medical records containing irrelevant information are simulated. OCR-extracted text from common medical and insurance forms is inserted to fill the models’ context windows, thereby creating “needle-in-a-haystack” style challenges.

- MLCR-AA ([artificialanalysis.ai/evaluations/mlcr-aa](https://artificialanalysis.ai/evaluations/mlcr-aa))

  Designed to evaluate a model’s reasoning capabilities when handling lengthy medical case documents. The AA version selects 60 test questions from the two most difficult categories in the original MLCR dataset to form a private test set: one is the “expert-level” category, which requires the model to perform professional medical reasoning based on the entire case document; the other is the “composite” category, where multiple independent questions are combined into a single query request.

  Answers to each question must be based on a complete medical record spanning 100 to 150 pages. Subsequently, a panel of three models evaluates the responses based on three criteria: completeness, accuracy, and conciseness. The conciseness test aims to prevent models from generating overly lengthy replies; specifically, the response length must not exceed five times that of an expert-provided answer. The accuracy metric verifies whether the model’s response is genuinely grounded in the original document and case context; the completeness metric assesses whether the model includes all key information present in the expert-provided answer. The conciseness test ensures that no redundant content is generated.

- DrugDiscoveryBench ([labs.scale.com/leaderboard/drugdiscoverybench](https://labs.scale.com/leaderboard/drugdiscoverybench))

  Used to evaluate the reliability of cutting-edge coding agents when performing multi-step computational and information retrieval tasks required during the early stages of drug discovery. This benchmark comprises 82 tasks covering the entire early discovery workflow: target identification and validation, screening of potential candidate molecules from patents, databases, and literature, selection of lead compounds from candidate molecules followed by structure-activity relationship analysis, and optimization of those lead compounds. The benchmark concludes prior to candidate compound selection; aspects such as pharmacokinetics (DMPK), toxicology, formulation development, and clinical trials are not included in the evaluation.

- HealthBench Professional ([medicalsphere.ai/benchmarks/healthbench-professional](https://medicalsphere.ai/benchmarks/healthbench-professional))

  A medical benchmark consisting of 525 tasks authored by physicians, designed to assess LLMs’ performance in three real-world clinical scenarios: diagnostic consultation, documentation and record-keeping, and medical research. Each task takes the form of a single- or multi-turn dialogue, derived from actual interactions between physicians while testing “ChatGPT for Clinicians.” These tasks are evaluated by three or more physicians according to predefined scoring criteria.

#### Legal

For legal-domain benchmarks, refer to Appendix 3 in *[LLM Agents in Law: Taxonomy, Applications, and Challenges](https://arxiv.org/pdf/2601.06216)*.

- Harvey LAB-AA ([artificialanalysis.ai/evaluations/harvey-lab-aa](https://artificialanalysis.ai/evaluations/harvey-lab-aa))

  Designed to measure AI agents’ ability to perform actual legal work rather than merely answering isolated legal questions. For each task, the agent receives partner-like instructions along with a set of case documents placed within a sandbox environment. The agent must read these materials, synthesize relevant information, and ultimately produce a legal deliverable. This benchmark includes 120 legal tasks spanning 24 areas of legal practice, ranging from mergers and acquisitions, capital markets, taxation, litigation, to bankruptcy.

#### Business and Finance

- Vending-Bench 2 ([andonlabs.com/evals/vending-bench-2](https://andonlabs.com/evals/vending-bench-2))

  A benchmark used to measure how well AI models operate a business over an extended period. The model must simulate running a vending machine business for one year; its performance is evaluated based on the ending balance of its bank account.

- YC-Bench ([collinear-ai.github.io/yc-bench/#leaderboard](https://collinear-ai.github.io/yc-bench/#leaderboard))

  This benchmark assesses agents’ capabilities by having them manage a startup over a simulated one-year entrepreneurial cycle involving hundreds of decision-making rounds. The agent must manage employees, select task contracts, and maintain profitability within a partially observable environment; adverse clients and rising payroll costs can trigger cascading effects due to poor decisions. Each model is tested using three sets of random seeds, with all models starting with $200,000 in capital.

- CEO-Bench ([ceobench.com/](https://ceobench.com/))

  Agents operate a simulated AI startup for 500 days. They are provided with $1 million in initial cash, and their performance is judged based on the cash balance at the end of the simulation. Agents conduct operations via programmable interfaces, granting them access to business databases, corporate management tools, and social media platforms.

- Finance Agent ([www.vals.ai/benchmarks/fabv2](https://www.vals.ai/benchmarks/fabv2))

  This benchmark tests agents’ ability to perform tasks expected of junior financial analysts. It contains 537 questions covering information retrieval, market research, and predictive analysis.

- TaxCalcBench ([github.com/column-tax/tax-calc-bench?tab=readme-ov-file](https://github.com/column-tax/tax-calc-bench?tab=readme-ov-file))

  An evaluation of cutting-edge models in performing U.S. tax calculation tasks. It includes 51 pairs of user inputs and corresponding expected tax filing outputs; these scenarios involve relatively straightforward tax situations and encompass filing status, sources of income, tax credits, and deductions.

- Diligence Stack Agent Bench ([csbench.com/benchmarks/diligence-stack-agent](https://csbench.com/benchmarks/diligence-stack-agent))

  Designed to measure models’ performance in executing tasks related to financial modeling and assessing financial health and research. The test set consists of two private knowledge bases specifically constructed for the Diligence Stack research workflow.

- Commerce Agent Bench ([github.com/Accio-org/CommerceAgentBench](https://github.com/Accio-org/CommerceAgentBench))

  Designed to evaluate whether agents can complete long-term business workflows, rather than merely answering related questions. It comprises 107 tasks covering browser operations, usage of native-like CLI tools, API/MCP workflow handling, document and spreadsheet creation, public web research, vendor analysis, product launches, logistics, and various other business operations.

### Special Scenarios

- SpeechMap ([SpeechMap.AI Explorer](https://speechmap.ai/))

  Aims to explore the boundaries of AI-generated speech. It tests how language models respond to sensitive and controversial prompts across different providers, countries, and topics. Most AI benchmarks measure what models can do; we focus on what they cannot do—the content they avoid, refuse, or block.
- HiL-Bench ([labs.scale.com/leaderboard/hil](https://labs.scale.com/leaderboard/hil))

  Used to measure an agent’s ability to determine when to seek help: whether it can recognize situations where missing, ambiguous, or conflicting information cannot be resolved through exploration or reasoning alone, and whether it can ask targeted questions at the right moment to obtain accurate information. It covers two domains: software engineering and text-to-SQL. Tasks are drawn from the SWE-Bench Pro and BIRD datasets, with obstacles intentionally introduced. There are 300 tasks evenly distributed across both domains: 200 public tasks and 100 private tasks reserved for unbiased evaluation. In total, the dataset contains 1,131 obstacles, averaging 3.8 per task.
- Voxelbench ([voxelbench.ai/leaderboard](https://voxelbench.ai/leaderboard))

  A benchmark designed to evaluate language models’ performance in generating voxel-based structures.
- DecodingTrust Bench ([decodingtrust-agent.com/leaderboard](https://decodingtrust-agent.com/leaderboard))

  A dynamic red-teaming framework for AI agents across 15+ domains and 50 sandbox environments, covering various indirect injections and direct prompt injections related to environments, tools, and skills.
- Political Manipulation ([political-manipulation.ai/](https://political-manipulation.ai/))

  Evaluates implicit political bias by comparing how models handle paired prompts involving opposing political themes. Such bias is hardly noticeable in individual responses, as it manifests as inconsistencies between different responses rather than explicit stance preferences. The dataset includes paired left-wing and right-wing coded prompts, rated according to a taxonomy of implicit manipulation techniques. We identify different types of political manipulation by measuring emotional consistency and usefulness consistency.
- Last Translation Benchmark ([last-translation-benchmark.vilda.net/leaderboard-results](https://last-translation-benchmark.vilda.net/leaderboard-results))

  A collection of human-written and peer-reviewed examples (text, images, audio, and video) specifically crafted to “stump” state-of-the-art machine translation models. We also propose a novel evaluation method: each example comes with manually crafted verification rules detailing specific failure scenarios for models.
- LibraryDesignBench ([ldbench.com/](https://ldbench.com/))

  An agent is tasked with designing a library; its performance is then scored solely based on how effectively other agents utilize that library to perform tasks.

#### Prediction

## Visual Understanding and Reasoning

Visual evaluation suffers from instability, primarily due to three reasons: first, the limited size of datasets; second, minor variations such as differences in labeling styles can significantly affect model accuracy and rankings; third, changes imperceptible to humans—such as JPEG compression—can alter benchmark rankings. (See [lisadunlap.github.io/vpbench/](https://lisadunlap.github.io/vpbench/).)

- **MMMU-Pro** ([MMMU-Pro Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/mmmu-pro))

  This benchmark features 10 multiple-choice options per question and introduces a visual-only input format where questions are embedded within screenshots or photographs. It comprises 3,460 questions spanning six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Sciences, and Technology & Engineering. Models are required to process both visual and textual information in scenarios closely resembling real-world situations.

- **ZeroBench** ([zerobench.github.io/](https://zerobench.github.io/))

  An extremely challenging visual benchmark designed for contemporary large multimodal models. It includes 100 meticulously crafted and extensively reviewed challenging questions by a team of designers, along with 334 sub-questions representing individual reasoning steps needed to answer each main question.

- **BabyVision** ([xbench.org/agi/babyVision](https://xbench.org/agi/babyVision))

  Part of xbench’s AGI alignment series, this benchmark focuses on evaluating visual understanding capabilities in “unspeakable” challenges.

- **PerceptionBench** ([www.kimi.com/blog/perception-bench](https://www.kimi.com/blog/perception-bench))

  A benchmark specifically designed to test visual perception abilities. It breaks down visual perception into a series of fundamental capabilities inferred from errors made by current models, rather than being predefined. By tracing failure cases of state-of-the-art models across over 40 benchmarks back to their underlying visual causes, 10 distinct perceptual capabilities and 3,000 validated test questions were identified. Answering these questions requires only observation, without any reasoning or external knowledge.

- **Blueprint-Bench 2** ([andonlabs.com/evals/blueprint-bench-2](https://andonlabs.com/evals/blueprint-bench-2))

  This benchmark tests spatial reasoning abilities by requiring agents to convert photographs of apartments into accurate 2D floor plans. Each agent processes 50 apartments sequentially, examining roughly 20 interior photos per apartment to generate a floor plan depicting room layouts, connections, and relative sizes.

- **MazeBench** ([mazebench.com/blog?post=maze-bench-results](https://mazebench.com/blog?post=maze-bench-results)) (human-designed)

  A 3D open-world environment designed to test agents’ long-term planning and visual-spatial reasoning abilities. It contains hundreds of rooms and puzzle levels.

  Models run MazeBench within their native agent environments, such as Codex and Claude Code, enabling cost-effective execution of lengthy tasks. Via MCP servers, these agents can perform eleven actions: four movement actions, four camera control actions, an undo command, level reset, and a teleportation action allowing movement between rooms. A key feature of MazeBench is its camera rotation functionality; models must rotate the camera upward, downward, left, and right to observe the game environment from twenty different angles. Corresponding movement commands must also be adjusted accordingly.

### World Models

## OCR and Embedding Evaluation

[Supercharge your OCR Pipelines with Open Models](https://huggingface.co/blog/ocr-open-models)

When testing various OCR models, their performance varies significantly across different document types and languages.

- OmniDocBench ([OmniDocBench/README_zh-CN.md at main · opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench/blob/main/README_zh-CN.md))

  A benchmark designed to evaluate the parsing of diverse documents in real-world scenarios. This widely used benchmark stands out due to its variety of document types, including books, magazines, and textbooks. Its evaluation criteria are well-designed and support tables in both HTML and Markdown formats.
- olmOCR-Bench ([olmocr/olmocr/bench at main · allenai/olmocr](https://github.com/allenai/olmocr/tree/main/olmocr/bench))

  This benchmark performs exceptionally well in evaluating English text recognition.
- Real5-OmniDocBench ([huggingface.co/datasets/PaddlePaddle/Real5-OmniDocBench](https://huggingface.co/datasets/PaddlePaddle/Real5-OmniDocBench))

  A new benchmark tailored for real-world scenarios, built upon the OmniDocBench v1.5 dataset. It encompasses five distinct scenarios: scanning, distortion, screen captures, lighting variations, and tilting. Except for the scanned documents, all images were manually captured using handheld mobile devices to closely mimic real-world conditions.
- OCRVerse ([github.com/DocTron-hub/OCRVerse](https://github.com/DocTron-hub/OCRVerse))

  The first end-to-end comprehensive OCR framework capable of integrating both text-centric OCR and vision-centric OCR (e.g., for charts, web pages, and scientific diagrams). The text-centric data covers nine document scenarios—natural scenes, books, magazines, papers, reports, slides, exam sheets, notes, and newspapers—representing common text-based scenarios encountered in daily life and meeting basic OCR requirements. The vision-centric data includes six specialized scenarios: charts, web pages, icons, geometric shapes, circuit diagrams, and molecular structures, focusing on structured professional content.
- Chronicles-OCR ([github.com/VirtualLUOUCAS/Chronicles-OCR/blob/main/README_ZH.md](https://github.com/VirtualLUOUCAS/Chronicles-OCR/blob/main/README_ZH.md))

  A comprehensive benchmark designed to evaluate the cross-temporal visual perception capabilities of Vision-Language Models (VLLMs). It covers the complete evolutionary trajectory of Chinese characters—the “Seven Styles of Chinese Calligraphy.” The dataset comprises 2,800 carefully balanced images (400 images per style × 7 styles), featuring highly diverse physical media ranging from oracle bones to paper-based calligraphy.
- PDF Parse Bench ([github.com/phorn1/pdf-parse-bench](https://github.com/phorn1/pdf-parse-bench))

  Evaluates the effectiveness of various PDF parsing solutions in extracting mathematical formulas from documents.
- Embedding Leaderboard ([MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard))

  Also known as the MTEB Leaderboard; it compares the performance of embedding models across tasks such as retrieval, classification, and clustering using a unified set of tasks.

## Evaluation of Image Generation, Video/Audio Generation, and Role-Playing

- DesignArena ([www.designarena.ai/leaderboard](https://www.designarena.ai/leaderboard))

  It includes multiple leaderboards such as Code Categories, Web App, Mobile, Full Stack, Agent, Builder, Image, Image Editing, Graphic Design, Logo, SVG, Video, Video Editing, and Slides.
- GenExam ([github.com/OpenGVLab/GenExam](https://github.com/OpenGVLab/GenExam))

  The first multidisciplinary benchmark for text-to-image generation evaluation, comprising 1,000 samples across 10 disciplines. The exam-style prompts are organized according to a four-level classification system. It tests models’ abilities in comprehension, reasoning, and generation.
- UNO-Bench ([UNO-Bench](https://meituan-longcat.github.io/UNO-Bench/))

  A unified benchmark designed to explore the combination patterns between unimodal and multimodal capabilities in models. Nearly 100% of the questions in UNO-Bench require joint understanding of audio and visual information. Besides traditional multiple-choice questions, we also propose an innovative multi-step open-ended Q&A format to assess complex reasoning abilities.

  Our dataset possesses three key characteristics: a. Diverse sources—primarily real-world photos and videos collected via crowdsourcing, supplemented by royalty-free websites and high-quality public datasets. b. Rich and varied topics—covering social, cultural, artistic, everyday life, literary, and scientific subjects. c. Real-time recorded audio—conversations recorded by over 20 human speakers, ensuring rich audio features that reflect the diversity of real-world sounds.
- WBench ([meituan-longcat.github.io/WBench/#leaderboard](https://meituan-longcat.github.io/WBench/#leaderboard))

  The first systematic multi-turn evaluation benchmark for interactive video world models, containing 289 test cases and 1,058 interaction rounds. Each use case specifies a world scenario and a multi-turn interaction sequence, covering diverse scenes, styles, topics, and both first- and third-person perspectives. It includes four types of interactions: navigation, subject actions, event editing, and perspective switching. For navigation tasks, WBench integrates text, six-degree-of-freedom poses, and discrete action controls, enabling evaluation of models with different native input interfaces.

### Speech Recognition and Interaction

- τ-voice ([taubench.com/#leaderboard?benchmark=voice](https://taubench.com/#leaderboard?benchmark=voice))

  An extension of the text-based benchmark τ²-bench, it comprises 278 tasks spanning three real-world domains: retail, aviation, and telecommunications. It supports multiple accents, background noise, interruptions, filler words, and non-conversational speech. Speech interaction quality is measured across several aspects: response rate, latency, interruption rate, and selectivity (i.e., whether the model correctly ignores filler words/non-conversational speech).

### Omni

## Decision Making

- Jev Decision Index ([huggingface.co/spaces/multimodalart/jev-decision-index](https://huggingface.co/spaces/multimodalart/jev-decision-index))

  The Decision Index generates a numerical value for each model, indicating its ability to make typed decisions—such as selecting options, labels, tools, ranking results, or probability values—after an open-source reproduction of TypeSafe’s Jev is implemented. All competing models are evaluated using the same fixed test set, which comprises 120,340 requests across 36 static benchmarks; this test set is a fixed subset of the request collection used to score Jev version jev-1.13.0.

## Community Evaluations

Below are independent evaluations conducted by experts in the AI community.

- nao’s LLM Benchmark ([llm2014.github.io/llm_benchmark/](https://llm2014.github.io/llm_benchmark/))

  This is a personal project that conducts long-term evaluations using a privately maintained question bank that is updated periodically. It focuses on testing models’ abilities regarding logic, mathematics, programming, and human intuition. The size of the question bank remains relatively small—typically under 30 questions and 240 test cases—and no publicly available internet questions are used. Questions are updated monthly. They are not made public; the aim is to share an evaluation approach and personal insights.

- knowledge-cutoff ([apoorvumang.github.io/knowledge-cutoff/](https://apoorvumang.github.io/knowledge-cutoff/))

  This is a benchmark designed to evaluate the actual knowledge cutoff point of language models—that is, the extent to which a model truly understands real-world information, which is usually earlier than the cutoff date stated by the model itself.  
  The principle is to test the model on various real events occurring in recent months. For each month, the number of events the model answers correctly is tallied. As time approaches the actual knowledge boundary, the model’s accuracy gradually declines; thus, by plotting accuracy curves month by month, one can roughly determine when the model’s world knowledge ends.

- XSCT Bench ([xsct.ai/](https://xsct.ai/))

  It includes tests based on real product scenarios in fields such as text processing, web development, image generation, Openclaw, and Omni.

- LisanBench ([lisanbench.com/](https://lisanbench.com/))

  This is a personal evaluation created by X user Lisan al Gaib (@scaling01). The model is given a starting English word and must continuously generate the next word while satisfying all of the following strict constraints:

  - It must differ from the previous word by exactly one letter (Levenshtein edit distance = 1).
  - It must be a valid English word (using the words_alpha.txt dictionary containing roughly 370,000 words; only the largest connected component of ~108,000 words is actually used).
  - No previously used words may be repeated.
  - Goal: Generate the longest possible valid chain of words.

  Score = sum of the lengths of the longest chains generated from multiple different starting words.

- Kaggle Benchmarks ([www.kaggle.com/benchmarks?type=community](https://www.kaggle.com/benchmarks?type=community))

  There are two main types of benchmarks: 1) Research benchmarks, created by researchers at AI labs; 2) Community benchmarks, created by the Kaggle community, allowing users to design, run, and share their own custom benchmarks for evaluating AI models. Guide: [www.kaggle.com/docs/benchmarks#How%20to%20create%20a%20benchmark](https://www.kaggle.com/docs/benchmarks#How%20to%20create%20a%20benchmark)

- prinzbench ([github.com/prinz-ai/prinzbench/](https://github.com/prinz-ai/prinzbench/))

  This is a private evaluation tool that ranks LLMs based on their ability to conduct U.S. legal research and analysis (“legal reasoning”) as well as their capacity to locate hard-to-find public information online (“finding a needle in a haystack”). It includes 25 legal research questions and 8 search questions.

- Creative Story‑Writing Benchmark, Elimination Game Benchmark, NYT Connections puzzles, Sycophancy Benchmark, Thematic Generalization Benchmark, Persuasion Benchmark ([github.com/lechmazur](https://github.com/lechmazur))

  **Creative Story‑Writing Benchmark**: Evaluates an LLM’s ability to craft engaging stories while adhering to specific creative requirements. Each story must meaningfully incorporate ten required elements: characters, objects, concepts, attributes, actions, methods, settings, time frames, motivations, and tones.

  **Elimination Game Benchmark**: The “Elimination Game” is a multi-player tournament designed to test LLMs’ social reasoning, strategic planning, and deception capabilities. Players engage in public and private conversations, form alliances, and vote to eliminate other players round by round until only two remain. Then, a jury composed of eliminated players casts a final vote to determine the winner.

  **NYT Connections puzzles**: Large language models (LLMs) are evaluated using 940 “Connections” puzzles from The New York Times.

  **Sycophancy Benchmark**: When the same controversial issue is presented from opposite first-person perspectives, does the model maintain the same judgment or tend to favor the speaker? This benchmark directly measures such contradictions.  
  The core metric is deliberately set to be quite strict: a model is considered sycophantic only if it agrees with both sides of the same controversy when each side narrates the story in the first person. Each test case includes five perspectives: one neutral third-person version, two abridged first-person versions, and two emotionally charged first-person versions.

  **Thematic Generalization Benchmark**: Designed to test whether LLMs can infer a specific underlying theme from a few examples, exclude broader but incorrect patterns via counterexamples, and then identify the sole correct match among similar distractors. Each test item provides the model with: 3 positive examples, 3 counterexamples that follow broader or adjacent patterns but do not fully match, and 8 candidate options, exactly one of which is the hidden correct match.

  **Persuasion Benchmark**: Measures the extent to which one language model can alter the stance of another model over multiple conversations. In each run, one model is designated as the persuader and the other as the target; they then discuss the same proposition.

### Games

- AI Poker Leaderboard ([benchmark.gtowizard.com/](https://benchmark.gtowizard.com/))

  Real-time rankings of models competing against GTO Wizard AI (the current state-of-the-art AI poker agent).
- RuneBench ([maxbittker.github.io/runebench/](https://maxbittker.github.io/runebench/))

  Evaluates AI capabilities in playing *RuneScape*, requiring them to complete various tasks within the game world. It measures AI behavior within the “observe, decide, act” cycle.

## Data Quality Evaluation

- OpenDataArena ([opendataarena.github.io/](https://opendataarena.github.io/))

  It ensures that every post-training dataset is measurable, comparable, and verifiable. It evaluates post-training data across multiple domains (general, mathematics, code, science, and long-chain reasoning) and various modalities (text, images). Variables are controlled by using a fixed model size (Llama3 / Qwen2 / Qwen3 / Qwen3-VL 7-8B) and consistent training configurations. Data lineage analysis: Modern datasets often suffer from excessive redundancy and hidden dependencies. ODA introduces the industry’s first data lineage analysis tool designed to visualize the “pedigree” of open-source data. Structural modeling: Maps relationships between datasets, including inheritance, mixing, and distillation.

## AI Infra Performance

- Modded-NanoGPT Optimization Benchmark ([github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization))

  This benchmark aims to identify efficient neural network optimizers through both collaboration and competition. Unlike the main NanoGPT speedrun challenges, which seek to minimize actual runtime at all costs, this benchmark focuses on reducing the number of optimization steps via algorithmic improvements—meaning methods requiring longer runtime are also acceptable.
- InferenceMAX ([inferencemax.semianalysis.com/](https://inferencemax.semianalysis.com/))

  InferenceMAX conducts benchmarking on popular models across mainstream hardware platforms, updating its test criteria whenever new software versions are released. For each model-hardware combination, it evaluates various tensor parallelism levels and maximum concurrent request counts, ultimately generating comprehensive throughput and latency comparison charts.
- MLPerf Training ([mlcommons.org/benchmarks/training/](https://mlcommons.org/benchmarks/training/))

  The MLPerf Training benchmark suite measures how quickly systems can train models to meet predefined quality targets.
- AA-AgentPerf (AI Hardware Benchmarking & Performance Analysis) ([artificialanalysis.ai/benchmarks/hardware](https://artificialanalysis.ai/benchmarks/hardware))

  This benchmark is designed for agent inference: it simulates real-world programming agent workflows and measures how many agents a system can support simultaneously while still meeting production-grade service criteria. Its core metric is “number of agents supported per megawatt”—the maximum number of agents a hardware platform can handle per megawatt of power while satisfying market-defined performance standards. It also accounts for real-world optimizations such as KV cache reuse, speculative decoding, and separation of prefill/decode phases. Test subjects range from individual accelerators to entire server racks.

  This benchmark maintains a fixed service level and then evaluates how far the system can scale while still meeting that level. Its performance metrics are derived from Artificial Analysis’s serverless API benchmark data—representing actual service tiers currently available on the market. Both speed and latency are measured per request: this includes P25 output speed and P95 time to generate the first token, calculated based on all requests during the test period.
- GPU Benchmark ([perf.svcfusion.com/](https://perf.svcfusion.com/))

  - Allows viewing of FP32, FP16, and BF16 performance across different GPU models.
  - Every data point is generated manually via benchmark scripts; raw manufacturer specifications are not used. Anyone can upload their own test results.
  - Test platform names are clearly labeled, enabling easy comparison of GPU performance across different platforms.
