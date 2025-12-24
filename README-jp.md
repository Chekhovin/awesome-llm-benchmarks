# 大規模モデル評価ベンチマークまとめ 

Tips：ベンチマーク（Benchmark）はモデル性能を評価する事実上の標準になっているが、テスト自体が常に正確というわけではなく、τ²-bench のように体系的な欠陥を含むケースもある。（論文：[arxiv.org/abs/2511.16842](https://arxiv.org/abs/2511.16842)）

LLMの評価における現在の課題と限界については、epoch.ai のブログ記事「Why benchmarking is hard」（epoch.ai/gradient-updates/why-benchmarking-is-hard）をご参照ください。

------

# 総合ベンチマーク

- Artificial Analysis
   （[AI Model & API Providers Analysis | Artificial Analysis](https://artificialanalysis.ai/)）

  - **AA Intelligence**
     （[Artificial Analysis Intelligence Index | Artificial Analysis](https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index)）

    数学・科学・プログラミング・推論における AI 能力を総合的に測るため、7 つのチャレンジングな評価を統合した包括ベンチマーク。
     使用している評価：MMLU-Pro、GPQA Diamond、HLE、LCB、SciCode、AIME 2025、IFBench、LCR、Terminal-Bench Hard、𝜏²-Bench Telecom。

  - **Artificial Analysis Openness Index**
     （[https://artificialanalysis.ai/evaluations/artificial-analysis-op...](https://artificialanalysis.ai/evaluations/artificial-analysis-openness-index)）

    AI モデルの「オープンさ」（利用しやすさ・透明性）を標準化して独立評価する指標。
     オープンであることは、単にモデルの重みがダウンロード可能であるだけではなく、ライセンス、データ、方法論も含む。
     Openness Index で 100 点を獲得するモデルは、

    - オープンな重み
    - 寛容なライセンス
    - トレーニングコード、事前学習データ、微調整データをすべて公開
       を満たし、ユーザはモデルを「利用」できるだけでなく、トレーニング過程を完全に再現したり、開発者のノウハウを用いて自分のモデルを構築することもできる。

- **SEAL LLMリーダーボード**（[scale.com/leaderboard](https://scale.com/leaderboard)）

  最新の大規模言語モデル（LLM）のエージェント能力、最先端性能、安全性、および世論への影響を評価します。

- **Epoch AI**（[epoch.ai/benchmarks](https://epoch.ai/benchmarks)）

  複数のベンチマークを提供しています。

  Epoch能力指数（ECI）は、さまざまなAIベンチマークのスコアを統合し、「汎用能力」を示す単一の尺度にまとめることで、個別のベンチマークがすでに飽和しているような長期的な期間においてもモデル間の比較を可能にします。

- **Sansa Bench**（[trysansa.com/benchmark](https://trysansa.com/benchmark)）

  複雑な実世界のタスクやユースケースでモデルを評価するために特別に設計されたベンチマークです。学術、オフィス業務、コンテンツ審査など、複数の分野ごとに細分化されたランキングを提供しています。

  コンテンツ審査ランキング（[trysansa.com/benchmark?dimension=censorship](https://trysansa.com/benchmark?dimension=censorship)）

- **LMArena**
   （[Overview Leaderboard | LMArena](https://lmarena.ai/leaderboard/)）

- **OpenCompass（司南）**
   （[OpenCompass司南 - 评测榜单](https://rank.opencompass.org.cn/home)）

- **LiveBench**
   （[LiveBench](https://livebench.ai/#/)）

  テストセット汚染を避け、公平な評価を行うことを目的とした LLM ベンチマーク。
   推論・コーディング・数学・データ分析などをカバー。

- **NeMo Evaluator SDK**
   （[NVIDIA-NeMo/Evaluator](https://github.com/NVIDIA-NeMo/Evaluator)）

  スケーラブルかつ再現性のあるモデル評価・ベンチマークのための OSS ライブラリ。

- **LRM-Eval**
   （[LRM-Eval](https://flageval-baai.github.io/LRM-Eval/)）

  テキストタスクは以下のサブタスクから構成される：

  - **問題解決**
    - 大学レベルの講義問題、言葉遊び、デコード問題
  - **アルゴリズム・コーディング**
    - 最近公開されたプログラミング問題
  - **タスク完遂**
    - 指示追従、マルチターン指示追従、長文コンテキスト理解
  - **ファクト性 & 拒否**
    - 長尾知識
  - **安全性**
    - 有害コンテンツ生成、越獄（Jailbreak）の評価

------

# Coding Benchmarks

- **SWE-bench**
   （[SWE-bench Leaderboards](https://www.swebench.com/index.html)）

  ソフトウェアエンジニアリング分野で最も人気のある評価スイートの一つ。
   GitHub 上の実際の Issue を元に、LLM がコードリポジトリと Issue 説明を入力として「バグ修正パッチ」を生成できるかを評価する。

  - **詳細**

    SWE-bench の各サンプルは、12 個の OSS Python リポジトリから取得した「既に解決済みの GitHub Issue」で構成される。
     各サンプルには以下が紐づく：

    - 解決コードを含む Pull Request（PR）
    - コードの正しさを検証する単体テスト

    これらのテストは PR 適用前は失敗し、適用後に成功するため **FAIL_TO_PASS テスト** と呼ばれる。
     さらに、PR 適用前後のどちらでも成功する **PASS_TO_PASS テスト** があり、既存機能を壊していないか確認する。

    各サンプルでエージェントは：

    - GitHub Issue の元のテキスト（Issue 説明）
    - 対応するコードベースへのアクセス

    を与えられ、Issue を解決するようリポジトリ内のファイルを編集する。テストケースはエージェントからは見えない。

    提案された変更は FAIL_TO_PASS と PASS_TO_PASS の両テストで評価される：

    - FAIL_TO_PASS が通れば、Issue は解決されたとみなす
    - PASS_TO_PASS も通れば、他の機能を壊していないとみなす

    2 つのテスト群がすべて成功した場合にのみ、元の GitHub Issue が完全解決と判断される。

  - **SWE-bench Verified**

    OpenAI が公開した、500 サンプルからなる人手精選サブセット。
     元の SWE-bench がエージェントの能力を過小評価している、という問題を補正するために作成。

    既存のエージェントフレームワークは Python 固有ツールに強く依存することが多く、SWE-bench Verified に過度適合しがちである。

  - **SWE-Bench Pro（公開データセット）**（[scale.com/leaderboard/swe_bench_pro_public](https://scale.com/leaderboard/swe_bench_pro_public)）

    SWE-Bench Verified のアップグレード版です。

    これは、ソフトウェアエンジニアリング分野におけるAIエージェントを厳密かつ現実的に評価するために設計されたベンチマークです。既存のベンチマークに見られるいくつかの限界を、以下の4つの主要課題に取り組むことで解決することを目指しています。

    1. データ汚染：モデルが学習中に評価用コードにすでに触れている可能性があり、そのためモデルが実際に問題を解いているのか、それとも記憶に頼って解答を再現しているのかを判断することが困難になります。
    2. タスクの多様性不足：多くのベンチマークは、現実世界のソフトウェア開発が抱える幅広い課題を十分にカバーできておらず、単純なユーティリティライブラリに偏りがちです。
    3. 問題の過剰な単純化：曖昧さや仕様の不明確さを含む課題はベンチマークから除外されがちですが、これは実際の開発者の作業フローとはかけ離れたものです。
    4. 信頼性の低さと再現性の欠如：環境設定が一貫していないため、提出された解決策が本当に有効なのか、あるいは単に環境の設定ミスによって偶然通っただけなのかを判断することが困難になります。

   - **SWE-bench Multilingual**

     複数のプログラミング言語にまたがるソフトウェアエンジニアリング能力を評価する。
     42 の GitHub リポジトリ・9 言語（C、C++、Go、Java、JavaScript、TypeScript、PHP、Ruby、Rust）の実際の PR から 300 のタスクを厳選。
     Web フレームワーク、データストレージ/処理ツール、コアユーティリティ、人気ライブラリなど広い領域をカバーする。

  - **Multi-SWE-Bench**
     （[Multi-SWE-bench](https://multi-swe-bench.github.io/#/)）

    ByteDance Seed による改変版。7 言語（Java、TypeScript、JavaScript、Go、Rust、C、C++）にまたがる 1,632 個の機能開発タスクを収録し、LLM の問題解決能力を評価する。
     評価はプロジェクトに内蔵されたテストを実行し、PR 適用後の挙動を正解として検証する。

  - **制約・限界**

    静的データセットに基づく評価には本質的な限界があり、SWE-bench も例外ではない。
     公開 GitHub リポジトリからスクレイピングしたデータであるため、Web テキストで事前学習した巨大モデルは高確率でテストセット汚染を起こしうる。
     また SWE-bench がカバーするのは「中リスクレベルの自律性」に限られるため、他の評価と組み合わせる必要がある。

- **terminal-bench**
   （[Terminal-Bench](https://www.tbench.ai/) / [Artificial Analysis](https://artificialanalysis.ai/evaluations/terminalbench-hard)）

  ターミナル環境で複雑なタスクをこなす AI エージェントの能力を測るタスク群と評価フレームワーク。
   例：リポジトリのビルド・パッケージング、データセットのダウンロードと分類器の学習、サーバセットアップなど。

  各タスクは

  - 自然言語によるタスク説明
  - Docker 環境
  - 成功判定のためのテストスクリプト
  - 参考となる「理想解」
     から構成される。

- **ArtifactsBench**
   （[ArtifactsBench](https://artifactsbenchmark.github.io/)）

  Tencent による、LLM が生成した「視覚的アーティファクト」を自動・多モーダルに評価する最初のベンチマーク。
   動的な出力をレンダリングし、MLLM（マルチモーダル LLM）ジャッジが詳細なチェックリストに基づいて忠実度とインタラクティビティを評価する。

  1,825 個の高品質で難度の高いプロンプトを収録し、

  - ゲーム開発
  - SVG 生成
  - Web アプリ
  - シミュレーション
  - データサイエンス
  - 管理システム
  - マルチメディア編集
  - 小さなユーティリティ
  - その他
     の 9 カテゴリーをカバー。

- **SWE-Dev**
   （[DorothyDUUU/SWE-Dev](https://github.com/DorothyDUUU/SWE-Dev)）

  「Feature-Driven Development（FDD）」＝既存コードベースに新機能を追加するタスクに特化した、初の大規模ベンチマーク兼トレーニングコーパス。

- **LiveCodeBench**
   （[LiveCodeBench](https://livecodebench.github.io/index.html) / [Artificial Analysis](https://artificialanalysis.ai/evaluations/livecodebench)）

  LeetCode、AtCoder、Codeforces の定期コンテスト問題を継続的に収集して構築したベンチマーク。
   コード生成だけでなく、自己デバッグ・コード実行・テスト出力の予測など、より広いコーディング能力を評価する。

  - **GSO Benchmark**

    ソフトウェアエンジニアリングエージェント向けの難度の高い「ソフトウェア最適化」ベンチマーク。
     10 個のコードベースにまたがる 102 個の最適化タスクに対して、専門開発者の最適化と比較してどれだけ実行速度を改善できるかを測る。

- **Aider’s polyglot benchmark**
   （[Aider LLM Leaderboards | aider](https://aider.chat/docs/leaderboards/)）

  C++、Go、Java、JavaScript、Python、Rust の 6 言語で 225 問の難しい Exercism 問題を解かせて LLM を評価。

- **SciCode**
   （[SciCode Benchmark](https://scicode-bench.github.io/) / [Artificial Analysis](https://artificialanalysis.ai/evaluations/scicode)）

  現実の科学研究で出てくる問題を解くコードを生成できるかを評価するベンチマーク。
   物理・数学・材料科学・生物学・化学など 6 分野・16 サブフィールドをカバーし、科学者の典型的なワークフロー：

  1. 重要な科学概念・事実を把握
  2. それを計算・シミュレーションコードに落とし込む

  を再現する。

  主な焦点は

  1. 数値計算手法
  2. システムシミュレーション
  3. サイエンティフィックコンピューティング

- **OJBench**
   （[He-Ren/OJBench](https://github.com/He-Ren/OJBench)）

  LLM の「コンテストレベルのコード推論能力」を評価するベンチマーク。
   中国 NOI（全国情報学オリンピック）、ICPC（国際大学対抗プログラミングコンテスト）から 232 問を厳選し、実際の提出・投票データに基づいて「易・中・難」の 3 ランクに分類。Python / C++ の 2 言語をサポート。

- **Roo Code evals**
   （[Evals | Roo Code](https://roocode.com/evals)）

- **CodeClash**
   （[CodeClash](https://codeclash.ai/)）

  **目標駆動型ソフトウェアエンジニアリング** における AI システムの能力を評価するベンチマーク。
   従来のコーディングベンチマークは「タスク駆動」（明示的な仕様 + ユニットテスト）だが、現実の開発は

  - 「ユーザー維持率を上げる」
  - 「コストを下げる」
  - 「売上を増やす」
     などの抽象的な目標が先にあり、それをコードで実現するのは自律的・反復的・ときに競争的なプロセスである。

  CodeClash では、2 体以上の LLM エージェントが「コードアリーナ」で複数ラウンドのトーナメント形式で競い合う。
   それぞれのエージェントは、自分のコードベースを繰り返し改善しながら、高レベルな競争目的（資源の獲得、最長生存など）を達成しようとする。

- **METR：長時間かかるタスクをAIが完遂できる能力の測定**（[https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)）

  AIの性能を、AIエージェントが完遂可能なタスクの長さ（所要時間）で評価する。具体的には、「そのモデルがx％の確率で成功裏に完了できる、人間が実行するのに必要な時間の長さ」によってモデルの能力を定量化する。（現状のモデルは、人間が4分未満で完了できるタスクでは成功率がほぼ100％に達するが、人間が約4時間以上かかるタスクでは成功率が10％を下回る。）

- **Codeforces**

  世界最大級のアルゴリズム練習・プログラミングコンテストプラットフォーム。

## AI コーディングエージェント

- **LiveSWEBench**
   （[LiveSWEBench](https://liveswebench.ai/)）

  AI コーディングエージェントのソフトウェアエンジニアリング能力を評価するベンチマーク。
   各アシスタントを以下 3 種のタスクで評価する：

  - **エージェント・プログラミング**：高レベルなタスクを与え、完全自律で完遂させる
  - **目的指向の編集**：より具体的な指示と編集対象ファイルを与え、エージェントとして編集させる
  - **オートコンプリート**：部分的なコードスニペットの続きを完成させる

- **DPAI Arena**
   （[DPAI — Developer Productivity AIrena](https://dpaia.dev/)）

  世界初の「オープン・多言語・多フレームワーク・多ワークフロー」AI コーディングエージェントベンチマーク。
   パッチ生成、バグ修正、PR レビュー、テスト生成、静的解析など、複数のワークフローに対して公平かつ再現性のある比較を行うため、柔軟なトラック制アーキテクチャを採用している。

------

# Agentic Benchmarks

- **GAIA**
   （[GAIA Leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard)）

  現実世界に即した問題を通じて、汎用 AI アシスタントの

  - 推論
  - マルチモーダル理解
  - Web ブラウジング
  - ツール使用
     能力を評価する総合ベンチマーク。

- **Gaia2 Leaderboard**
   （[Gaia2 Agents Evaluation Leaderboard](https://huggingface.co/spaces/meta-agents-research-environments/leaderboard)）

  一般的な知能エージェント能力を測るベンチマーク。
   従来の「検索してすぐ実行」型タスクと異なり、非同期に実行されるため、エージェントは

  - あいまいさとノイズ
  - 変化する環境
  - 他エージェントとの協調
  - 時間制約

  に対処しなければならない。

  評価軸：

  - 実行（指示追従・多段ツール利用）
  - 検索（情報検索）
  - 曖昧性（不完全指示への対処）
  - 適応性（環境変化への対応）
  - 時間（時間管理・スケジューリング）
  - ノイズ（関係のない情報・ランダムなツール故障）
  - マルチエージェント協調

- **BrowseComp**
   （[BrowseComp: a benchmark for browsing agents | OpenAI](https://openai.com/index/browsecomp/)）

  OpenAI によるブラウジングエージェント向けベンチマーク。
   Web 上の「複雑かつ見つけにくい情報」を、短い一意解の形でどれだけ正確に探し出せるかを評価する。

  - **BrowseComp-ZH**
     （[BrowseComp-ZH/README-ZH.md](https://github.com/PALIN2018/BrowseComp-ZH/blob/main/README-ZH.md)）

    中国語 Web エコシステムにおける LLM の検索・推論能力を評価する初の高難度ベンチマーク。
     BrowseComp（Wei ら, 2025）に着想を得て、中国語圏特有の

    - プラットフォームの断片化
    - 言語特性
    - コンテンツ検閲

    といった課題を考慮した複雑な多段検索・推論タスクを構築している。

- **FACTS Benchmark**（[www.kaggle.com/benchmarks/google/facts/leaderboard](https://www.kaggle.com/benchmarks/google/facts/leaderboard)）

  事実に基づく質問応答のシナリオにおいて、モデルが内部知識を正確に活用できる能力を測定するためのパラメータ化されたベンチマーク。公開セット（1,052問）および非公開セット（1,052問）で構成されています。

  検索をツールとして用いて情報を取得し、それを正しく統合する能力を評価するための検索ベンチマーク。公開データセット（890件）および非公開データセット（994件）で構成されています。

  入力画像に関連するプロンプトに対して、事実に忠実な形で回答できる能力を評価するためのマルチモーダルベンチマーク。公開データセット（711件）および非公開データセット（811件）で構成されています。

  Grounding Benchmark - v2：与えられたプロンプトの文脈において、事実に基づいた回答を提供できる能力を評価するための拡張版ベンチマークです。

- **DeepSearchQA**（[www.kaggle.com/benchmarks/google/dsqa/leaderboard](https://www.kaggle.com/benchmarks/google/dsqa/leaderboard)）

  Googleが公開した、900のプロンプトからなるベンチマークで、エージェントが17の異なる分野において困難な多段階情報検索タスクを遂行する能力を評価することを目的としています。従来の単一回答の検索や広範な事実の正確性に焦点を当てたベンチマークとは異なり、DeepSearchQAは、エージェントが複雑な検索計画を実行し、包括的な回答リストを生成できるかを評価するための、挑戦的な手作業によるタスク群で構成されています。

  各タスクは「因果連鎖（causal chain）」として設計されており、あるステップでの情報発見が、その前のステップが成功裏に完了していることに依存するようになっています。これにより、長期的な計画立案とコンテキストの維持が重視されます。

- **DeepResearch Bench**（[https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leaderboard](https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leaderboard)）

  深層リサーチエージェント向けの総合ベンチマークで、100件の博士レベルのリサーチタスクで構成されています。各タスクは、22名の異なる分野のドメイン専門家によって綿密に設計されており、そのうち50件は中国語タスク、残り50件は英語タスクです。


- **xbench**
   （[xbench](https://xbench.org/)）

  国内機関が共同開発したベンチマーク。
   AI システムの「最先端知能」と「実用価値」の双方を測るため、2 つの補完的トラックから成る：

  - **AGI Tracking**：推論・ツール使用・記憶などのコア能力を評価
  - **Professional Alignment**：ワークフロー・環境・ビジネス KPI に基づく評価体系で、実務上の有用性を評価

  xbench は動的な評価システムとして設計されており、両トラックとも継続的に更新される。

  - **xbench-ScienceQA**
     科学分野の基礎知識を評価。
  - **xbench-DeepSearch**
     検索・情報検索シナリオにおけるツール利用能力を評価。中国語の現場ニーズに合わせて設計。
  - **xbench-Profession-Recruiting**
     現実の採用ワークフローと業界標準に基づき、
    - 職務要件分解
    - 人材プロファイリング
    - 候補者情報の補完
    - 履歴書スクリーニング
    - ネットワーク関係理解
    - オープンタレント検索
       などを通じて、エージェントの業界理解・人材サーチ能力・評価スキルを測定。
  - **xbench-Profession-Marketing**
     実際のマーケティングワークフローに基づき、主に KOL 検索を対象とする。
     クライアント要件のヒアリング・KOL マッチング、コンテンツ配信のモニタリングと戦略調整などを評価。

- **IFBench**
   （[IFBench Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/ifbench)）

  58 種類の多様で検証可能な「分布外制約」のもとで、モデルがどれだけ厳密に指示を守れるかを評価する。
   出力形式・スタイルなど細かい制約への追従能力をテスト。

- **Humanity's Last Exam（HLE）**
   （[Humanity's Last Exam](https://lastexam.ai/) / [Artificial Analysis](https://artificialanalysis.ai/evaluations/humanitys-last-exam)）

  専門家レベルの最先端知識を測る、多分野マルチモーダルベンチマーク。
   100 以上の科目・2,500 問から構成される。問題は公開される一方で、「過学習検出用の非公開テストセット」も別途保持されている。

  HLE で高精度を達成することは、

  - 閉形式で検証可能な問題
  - 最先端の科学知識
     に対して専門家レベルの性能を持つことを示すが、
     それだけで自律研究能力や汎用 AI（AGI）であることを意味するわけではない。
     あくまで「構造化された学術問題」に対する技術知識と推論力を測る。

  評価設定は 2 つ：

  - **w/ tools**：ツール利用を含むエージェント能力を評価
  - **w/o tools**：ツールなしでモデル本体の知能を評価

- **τ²-bench**
   （[τ-bench](https://taubench.com/#home) / [Artificial Analysis](https://artificialanalysis.ai/evaluations/tau2-bench)）
   ※重大な欠陥が指摘されている。

  複数ドメインにわたるカスタマーサポートエージェントを評価するシミュレーションフレームワークで、
   小売・通信・航空などの現実的なコラボレーションシナリオを模擬し、エージェントの対話能力を測定する。

  エージェントには「複雑な企業ドメインでユーザーと協調し、ガイドし、支援して共通の目標を達成する」ことが求められる。
   エージェントとユーザーを同時にシミュレートし、共有グローバルステートを動的に更新することで、新しい評価パラダイムを提案している。

  しかし検証の結果、以下のような問題が見つかった：

  1. **ポリシー違反**：
      想定される行動がドメインポリシーに反している（例：補償が禁止されている状況で補償を行う、すでに出発済みのフライトをキャンセルする等）。
  2. **データベースの不整合**：
      アイテム ID、乗客情報、支払方法などが実データベースと矛盾。
  3. **論理矛盾**：
      ポリシー的に禁止されているにもかかわらず同じアイテムを二度引き換える等、実現不可能なシナリオ。
  4. **評価の曖昧さ**：
      タスク記述が不明瞭で、評価結果が一貫しない。

- **τ²-Bench-Verified**
   （[github.com/amazon-agi/tau2-bench-verified](https://github.com/amazon-agi/tau2-bench-verified)）

  元の τ²-bench に対して、

  - タスク定義
  - 期待行動
  - 評価基準
     とドメインポリシー／データベースの整合性を人手で検証・修正したバージョン。

- **FinSearchComp**
   （[FinSearchComp Benchmark](https://randomtutu.github.io/FinSearchComp/)）

  オープンエンドな金融検索に特化した初のベンチマーク。
   現実の金融意思決定タスクに必要な 3 つのコア能力：

  1. 適切なシグナルを発見する
  2. 複数ソースを検証・統合する
  3. 時間制約下で、証拠に基づく判断を下す

  を評価するため、グローバルと大中華圏の 2 サブセットで

  - 時系列データ取得
  - 単純な過去問合せ
  - 複雑な歴史調査
     の 3 サブタスクを設計している。

- **GDPval-AA**（[artificialanalysis.ai/evaluations/gdpval-aa](https://artificialanalysis.ai/evaluations/gdpval-aa)）

  OpenAIのGDPvalデータセット向けに開発された評価フレームワークです。このフレームワークは、44の職種および9つの主要業界にわたって、AIモデルが実際の業務タスクにおいてどのように性能を発揮するかを評価します。金融、医療、法律その他の専門分野における実務成果物を模擬するために、文書、スライド、チャート、スプレッドシートなど多様な出力を生成することを要求する220のタスクで構成されています。

- **TheAgentCompany**
   （[The Agent Company](https://the-agent-company.com/)）

  現実の専門タスクを行う LLM エージェントを評価する枠組み。
   AI エージェントが「デジタル社員」として、Web ブラウズ・コード記述・プログラム実行・同僚とのコミュニケーションなどを行えるかを見る。

- **VitaBench**
   （[VitaBench](https://vitabench.github.io/)）

  フードデリバリー、外食、旅行など、高頻度の日常シナリオを用いて設計された「対話的ベンチマーク」。
   66 個のツールと複数シナリオをまたぐ複合タスクを用意し、

  - 深い推論
  - ツール使用
  - ユーザーとの対話
     の 3 軸からエージェントを評価する。

- **Toolathlon**
   （[Toolathlon](https://toolathlon.xyz/introduction)）

  現実世界のソフトウェア環境を前提にした「一般的なツール使用能力」を測るベンチマーク。
   32 のソフトウェア・604 のツールをカバーし、108 個のタスクを人手設計／スクリプト生成している。
   各タスクは平均約 20 ターンのインタラクションが必要な長期ツール利用を伴う。

- **BFCL-V4**（[gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html)）

  正式名称は「Berkeley Function-Calling Leaderboard（バークレー関数呼び出しリーダーボード）」で、大規模言語モデル（LLM）が関数（すなわちツール）を正確に呼び出せる能力を評価します。

- **MCP-Universe**
   （[mcp-universe.github.io](https://mcp-universe.github.io/)）

  実際の Model Context Protocol（MCP）サーバをベースにした LLM ベンチマークスイート。

- **MCPMark**
   （[mcpmark.ai](https://mcpmark.ai/)）

  MCP アプリケーションにおける LLM・エージェント能力を測る総合ストレステストベンチマーク。
   Notion・GitHub・Filesystem・Postgres・Playwright・Playwright-WebArena など、多様な MCP サーバを含む。

- **MCP Atlas**（[scale.com/leaderboard/mcp_atlas](https://scale.com/leaderboard/mcp_atlas)）

  モデルコンテキストプロトコル（MCP）を通じて、言語モデルが現実世界のツールを適切に活用できるかを評価するベンチマークであり、特に複数ステップからなるワークフローにおけるパフォーマンスを測定します。このベンチマークには、1,000件の人手で作成されたタスクが含まれており、各タスクは40以上のMCPサーバーおよび300以上のツールの中から複数のツールを呼び出して解決する必要があります。タスクの難易度は、2～3つのツールと単純なチェーンで完結する単一ドメインのクエリから、5つ以上のツールを必要とし、条件分岐やエラー処理を含む複雑なワークフローまで幅広くカバーしています。

  各タスクには、一見妥当だが実際には誤った選択となる「妨害ツール（distractor tools）」が慎重に選定されています。これらの妨害ツールは、データアノテーターが必須ツールと同じカテゴリから選びます。評価フレームワークでは、各タスクごとに12～18個のツール（うち3～7個が必須ツール、5～10個が妨害ツール）を提示し、エージェントがツールの説明に基づいて推論を行うことを強制し、無作為な試行を防いでいます。
- **SCONE-bench**
   （[red.anthropic.com/2025/smart-contracts/](https://red.anthropic.com/2025/smart-contracts/)）

  Anthropic による、スマートコントラクトの「資金搾取能力」でエージェントを評価する初のベンチマーク。
   各ターゲットコントラクトについて、エージェントは：

  - 脆弱性を特定し
  - 搾取スクリプトを生成し
  - 実行者のネイティブトークン残高を所定閾値以上増やす

  必要がある。

  SCONE-bench は架空のバグではなく、2020〜2025 年に Ethereum / BNB Chain / Base 上で実際に悪用された 405 件のスマートコントラクトを対象とする。

  さらに、

  1. MCP ツールを用いて 60 分以内に攻撃を試みるベースエージェント
  2. 指定ブロック高でフォークしたローカルチェーンを Docker コンテナ内で実行する評価フレームワーク
  3. メインネットデプロイ前にスマートコントラクトをストレステストできる「事前監査」機能

  を提供する。

------

## AI エージェント（Web / GUI エージェント）

- **Online-Mind2Web**
   （[Online_Mind2Web Leaderboard](https://huggingface.co/spaces/osunlp/Online_Mind2Web_Leaderboard)）

  実在サイト上での Web エージェント性能を評価するベンチマーク。
   136 の人気サイトにまたがる 300 タスクを収録し、LLM ジャッジ（WebJudge）による自動評価を採用。

  人手アノテーションのステップ数に応じて、タスクは

  - 易：1–5 ステップ
  - 中：6–10 ステップ
  - 難：11 ステップ以上

  に分類される。

------

## LLM の記憶とパーソナライゼーション

- **LoCoMo**
   （[LoCoMo](https://snap-research.github.io/locomo/)）

  LLM エージェントの「超長期対話メモリ」を評価するベンチマーク。
   QA、イベント要約、多モーダル対話生成などのタスクを通じて、長期記憶能力を測る。

  - **QA タスク**
     エージェントは過去コンテキストを正確に「想起」し、将来の応答に適切に統合する必要がある。
     質問は以下 5 種類の推論タイプに分類される：
    1. 単一ステップ推論
    2. マルチステップ推論
    3. 時系列推論
    4. 常識・世界知識推論
    5. 敵対的推論
  - **イベントグラフ要約**
     対話の中の長距離因果関係・時間関係を捉え、共感的で文脈に沿った応答を生成できるかを評価。
     各話者に関連するイベントグラフを「正解」とし、対話履歴からそれを抽出させる。
  - **マルチモーダル対話生成**
     過去対話から得られるコンテキストを活かして、ストーリー展開と整合的な応答を生成できるかを評価。

- **PERSONAMEM**
   （[PersonaMem](https://zhuoqunhao.github.io/PersonaMem.github.io/)）

  動的ユーザモデリングとパーソナライズ応答に関する大規模ベンチマーク。
   180 以上のシミュレートユーザと LLM の対話を含み、ユーザあたり最大 60 ターン、合計約 100 万トークン。
   15 種類のパーソナライゼーションシナリオと 7 カテゴリのリアルタイムクエリをカバーする。

  各インスタンスには、

  - 静的属性（人口統計など）
  - 動的属性（変化する嗜好など）
     を持つ「ユーザ・ペルソナ」が含まれる。

  ユーザは食事・旅行・カウンセリング等のテーマでチャットボットとマルチターン対話を行い、嗜好が時間とともに変化していく。
   ベンチマークは、ユーザの「一人称クエリ」に対して、LLM がどれだけ適切なパーソナライズ応答を返せるかを評価する。

- **Personalized Deep Research**
   （[arxiv.org/abs/2509.25106](https://arxiv.org/abs/2509.25106)）

  深い調査・研究タスクを行うエージェントの「パーソナライズ能力」を評価する最初のベンチマーク。
   10 領域・50 個の多様な研究タスクと、25 の実ユーザペルソナ（構造化属性 + 動的現実コンテキスト）を組み合わせ、合計 250 個の実ユーザクエリを構成する。

  評価には PQR フレームワークを用いる：

  - **P（Personalization）**：ペルソナへの適合度
  - **Q（Quality）**：内容の質
  - **R（Reliability）**：事実の正確性

------

## 視覚グラウンディング & GUI エージェント

- **AndroidDaily**
   （[opengelab.github.io](https://opengelab.github.io/index_zh.html)）

  実世界シナリオに対する多次元・動的ベンチマーク。
   現代生活の 6 つのコア領域（食・移動・ショッピング・住・情報消費・エンタメ）を対象に、各カテゴリで主要なアプリを優先的に採用している。
   これにより、支払い・予約など「現実のアウトカム」を伴うタスクを再現し、オンラインとオフラインの統合を重視している。

  - **静的テスト**：
     3,146 の操作サンプル。
     タスク説明とステップごとのスクリーンショットに基づき、エージェントは
    - アクション種別（タップ・入力など）
    - パラメータ（座標・テキストなど）
       を予測する。主な指標は数値的正確さ。
  - **エンドツーエンドテスト**：
     実機・エミュレータ上の完全なテスト環境で、エージェントがタスクを自律的に完遂できるかを評価。
     評価指標の中心は「タスク成功率」で、複雑環境における総合能力をよりリアルに反映する。

- **AndroidWorld**
   （[android_world](https://google-research.github.io/android_world/)）

  20 の実 Android アプリで 116 個のプログラミングタスクを提供する、完全機能の Android 環境。
   静的テストセットではなく、自然言語でパラメータ化されたタスクを動的に生成できるため、スケールと現実性の高い評価が可能。

- **ScreenSpot-V2**
   （[ScreenSpot-V2](https://gui-agent.github.io/grounding-leaderboard/screenspot.html)）

  GUI グラウンディングベンチマーク。
   自然言語指示を画面上のピクセルレベルのターゲットにマッピングする能力を測る。

- **ScreenSpot-Pro**
   （[GUI Grounding Leaderboard](https://gui-agent.github.io/grounding-leaderboard/)）

  高解像度のプロフェッショナルソフトウェアに特化した GUI グラウンディングベンチマーク。
   複雑なチャートを含むデスクトップアプリ等を対象とする。

- **OSWorld-G**
   （[osworld-grounding.github.io](https://osworld-grounding.github.io/)）

  GUI 上の細粒度な機能コンポーネント理解を評価するベンチマーク。
   テキストマッチング・要素認識・レイアウト理解・精密操作など、564 サンプルを収録。

- **MMBench-GUI**
   （[MMBench-GUI](https://github.com/open-compass/MMBench-GUI)）

  Windows / macOS / Linux / iOS / Android / Web の 6 大プラットフォームにわたる GUI Agent の総合能力を評価。
   単に「成功したか」だけでなく、「効率–品質面積（EQA：Efficiency-Quality Area）」指標を導入してタスク効率を測る。

  評価レイヤーは 4 つ：

  1. GUI コンテンツ理解
  2. GUI 要素グラウンディング
  3. GUI タスク自動化
  4. GUI タスク協調

------

# Intelligence Benchmarks

- **ARC-AGI-2**（[arcprize.org/leaderboard](https://arcprize.org/leaderboard)）

  人間にとっては比較的簡単だが、AIにとっては困難、あるいは不可能に近いタスクに焦点を当てることで、「スケーリング」だけでは自然には獲得できない能力のギャップを明らかにすることを目指しています。

  1. 全ての評価セット（公開、準非公開、非公開）は、従来の100タスクから拡張され、現在はそれぞれ120タスクを含んでいます。
  2. ブルートフォース検索（総当たり探索）による影響を受けやすいタスク（すなわち、2020年のKaggleコンペティションで既に解かれたすべてのタスク）は、評価セットから除外されました。
  3. 評価セットの難易度を適切に調整し、人間間の難易度差異（IDD: Inter-Human Difficulty Disparity）を確保するとともに、少なくとも2人の人間がAIと同等の条件（pass@2）でタスクを解けることを確認するため、制御された人間テストが実施されました。
  4. 記号的解釈、構成的推論、文脈依存ルールなどの研究知見に基づき、AIの推論システムに挑戦する新しいタスクが設計されました。

- **MMLU-Pro**
   （[MMLU-Pro Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/mmlu-pro)）

  多タスク理解ベンチマーク。
   多数の学問領域から 12,000 の複雑な問題を収録し、各問題は 10 択。
   より「推論に重きを置いた問題構成」になっている。

- **Frames**

  検索拡張生成（RAG）システムの

  - ファクト性
  - 検索精度
  - 推論能力
     を総合的に評価するデータセット。

  824 の難しいマルチホップ質問を収録し、2〜15 本の Wikipedia ページから情報を統合する必要がある。
   歴史・スポーツ・科学・動物・健康など幅広いトピックを含む。

- **SealQA（Seal-0）**
   （[vtllms/sealqa](https://huggingface.co/datasets/vtllms/sealqa)）

  Web 検索結果が

  - 矛盾している
  - ノイズが多い
  - 役に立たない
     場合でも、検索拡張 LLM が Factoid QA をどれだけこなせるかを評価する。

  GPT-4.1 など既存チャットモデルの精度がほぼゼロに近いような「特に難しい質問」に焦点を当てている。

- **Stanford HELM**
   （[HELM Capabilities Leaderboard](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard)）

  Stanford CRFM による包括的 LLM 評価フレームワーク。
   多数のタスク・指標を横並びで比較できる。

- **Zenmux**
   （[Benchmark - ZenMux](https://zenmux.ai/benchmark)）

  各モデルをあらゆる提供チャネルでテストし、

  - コスト vs 性能のトレードオフ曲線
  - 総合スコアランキング
     を示す。

  主な評価ベンチマークとして、Scale AI の「Humanity’s Last Exam（テキスト版）」を使用。

- **AA-LCR**
   （[Artificial Analysis Long Context Reasoning Benchmark Leaderboard](https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning)）

  長文・複数ドキュメント間での推論能力を評価するベンチマーク。
   cl100k_base トークナイザで最大 10 万トークンの入力を想定し、

  - 企業レポート
  - 業界レポート
  - 政府コンサルテーション
  - 学術論文
  - 法律文書
  - マーケ資料
  - 調査レポート

  など 7 種のテキスト文書から、多地点の情報を統合して答えを導く。

- **Fiction-liveBench**
   （[Fiction.liveBench](https://fiction.live/stories/Fiction-liveBench-Sept-06-2025/oQdzQvKHw8JyXbN87/home)）

  物語生成タスクにおける長文コンテキスト理解を評価するベンチマーク。
   とても長く複雑なストーリーを十数本厳選し、それらの圧縮版から作成した多数のクイズでモデルをテストする。

- **Context Arena**
   （[contextarena.ai](https://contextarena.ai/)）

- **Context-Bench**
   （[Letta Leaderboard](https://leaderboard.letta.com/)）

  ドキュメントチェーン操作・エンティティ関係追跡・多段情報検索など、長期的なコンテキストマネジメント能力を評価する。

- **Needle in a Haystack**
   （[LLM-NeedleInAHaystack](https://github.com/Lianues/LLM-NeedleInAHaystack/blob/main/README_CN.md)）

  「干し草の中の針」方式で、LLM のリコール能力を評価するベンチマーク。

  手順：

  1. **テストテキスト構築**：
      固定長コンテキスト内に、ランダムな 4 桁数（1000–9999）を複数挿入。
  2. **モデルタスク**：
      テキストからすべての 4 桁数を抽出し、出現順に JSON 形式で出力させる。
  3. **スコアリング**：
      Levenshtein 距離ベースの編集距離アルゴリズムでモデル出力を採点。

- **Hallucination Leaderboard**
   （[LLM Hallucination Leaderboard](https://huggingface.co/spaces/vectara/leaderboard)）

  短い文書を渡し、「与えられた事実にのみ基づいて要約を出す」ようモデルに指示し、
   元文書との事実整合性（realism ではなく faithfulness）を測ることで、幻覚頻度を評価する。

- **AA-Omniscience**
   （[Artificial Analysis Omniscience Index](https://artificialanalysis.ai/evaluations/omniscience)）

  「ビジネス」「人文・社会科学」「ヘルス」「法律」「ソフトウェアエンジニアリング」「科学・工学・数学」の 6 分野・42 トピックから 6,000 問を収録し、

  - 正答率
  - 幻覚率（誤答 / 非回避回答）
  - 全知指数（正答 +1 / 誤答 −1 / 回避 0）

  の 3 指標で評価する。

- **R-HORIZON**
   （[R-HORIZON](https://reasoning-horizon.github.io/)）

  LRMs（Large Reasoning Models）の「長鎖推論能力」を系統的に評価・向上させるフレームワーク。
   Query Composition という手法で問題同士の依存関係を構成し、孤立したタスクを複雑なマルチステップ推論チェーンに変換する。

  手順：

  1. **情報抽出**：
      各問題から核心となる数値・変数を抽出。
  2. **依存構築**：
      先行問題の答えを後続問題の条件に埋め込む。
  3. **チェーン推論**：
      すべてのサブ問題を順序通りに解かなければ最終回答が得られない構造にする。

- **TRUEBench**
   （[TRUEBench](https://huggingface.co/spaces/SamsungResearch/TRUEBench)）

  指示追従能力を評価するベンチマーク。
   LLM が「生産性アシスタント」として、人間の指示をどれだけ正確にこなせるかを見る。

- **SpeechMap**
   （[SpeechMap.AI Explorer](https://speechmap.ai/)）

  AI の発話境界を探るベンチマーク。
   複数国・複数トピック・複数プロバイダにまたがって、

  - モデルが何に答えるのか
  - 何を拒否・回避するのか
     を可視化する。

  多くのベンチマークが「モデルが何をできるか」に焦点を当てるのに対し、SpeechMap は「何をしないか」に注目する。

- **WeirdML**（[htihle.github.io/weirdml.html](https://htihle.github.io/weirdml.html)）

  大規模言語モデル（LLM）に対して、細やかな思考と真の理解を必要とする一連の奇妙で非伝統的な機械学習タスクを提示し、以下の能力を評価することを目的としています：
  データの特性および問題の本質を真に理解する力
  問題に適した機械学習アーキテクチャおよび学習設定を設計し、実行可能なPyTorchコードを生成して解決策を実装する力
  ターミナル出力およびテストセット上の精度に基づき、5回の反復を通じて解決策をデバッグ・改善する力
  限られた計算リソースと時間を使い切る力

- **SimpleQA**

  OpenAI による「短く事実を問う質問」に特化したベンチマーク。
   4,326 問から成り、各質問には単一の正解があり、自動採点しやすいよう設計されている。

------

## AI4S（専門ドメイン向け）

- **AIME25**
   （[AIME 2025 Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/aime-2025)）

  2025 年 AIME I（アメリカ数学招待試験）から抽出した数学問題の解答データセット。
   QA タスクに適しており、レコード数は 1,000 未満、言語は英語。

- **HMMT 2025**

  2025 年 2 月の HMMT（Harvard-MIT Mathematics Tournament）の問題。
   MathArena Leaderboard で使用される。

- **AMO-Bench**
   （[AMO-Bench](https://amo-bench.github.io/)）

  Meituan LongCat チームによるベンチマーク。
   競技数学の専門家が作成した 50 問からなり、IMO と同等以上の難度を持つ。

- **IMO Bench**
   （[imobench.github.io](https://imobench.github.io/)）

  合計 10 個の金メダル・5 個の銀メダルを持つ IMO メダリスト・数学者チームによって精査されたベンチマーク。
   IMO 問題は多段推論と創造性を必要とするため、IMO レベルの難題に特化している。

  - **IMO-AnswerBench**：正答率評価
  - **IMO-ProofBench**：証明生成の質を評価
  - **IMO-GradingBench**：長い数学解答の自動採点に挑戦するベンチマーク

- **FrontierMath**（[epoch.ai/frontiermath](https://epoch.ai/frontiermath)）

  原創的な数学問題350問（うち50問は最高難易度のレベル4）を含み、挑戦的な大学レベルの問題から、専門の数学者が数日かけて解くような難問までをカバーしています。このベンチマークでは以下の要件を満たす必要があります。

  1. 明確かつ検証可能な解答
  2. 推測への耐性：解答は「推測不可能」でなければならず、ランダムな試行や単純なブルートフォース手法ではほぼ成功しないように設計されていること
  3. 計算的実現可能性：計算集約型の問題については、その分野の標準的な知識のみに基づいて解答に至る方法を示すスクリプトを含めること。これらのスクリプトの標準ハードウェア上での累積実行時間は1分未満でなければならない

- **PutnamBench**
   （[PutnamBench Leaderboard](https://trishullab.github.io/PutnamBench/leaderboard.html)）

  William Lowell Putnam Mathematical Competition（北米トップレベルの学部数学コンテスト）の問題を形式化し、
   Lean 4・Isabelle・Coq で 1,712 問を定式化したベンチマーク。

  - Lean 4：660 問
  - Isabelle：640 問
  - Coq：412 問

- **GPQA Diamond**
   （[GPQA Diamond Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/gpqa-diamond)）

  GPQA ベンチマーク中、最も難しい 198 問からなるサブセット。
   「Google 検索で答えが見つからない」よう設計されており、物理・生物・化学の大学院レベル問題で構成。
   安定して正解できるのは PhD レベルの専門家に限られるとされ、深い科学的推論力の評価に適している。

- **CritPt**
   （[CritPt Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/critpt)）

  71 個の研究レベル物理チャレンジから成るベンチマーク。
   LLM の高度な物理推論能力を評価する。

- **FrontierScience**（[openai.com/index/frontierscience/](https://openai.com/index/frontierscience/)）

  OpenAIが開発した、専門家レベルの科学的能力を評価することを目的とした新しいベンチマークです。FrontierScienceは、物理学・化学・生物学の各分野の専門家によって作成・検証されており、挑戦的で独創的かつ意味のある数百問の問題で構成されています。FrontierScienceには2つのトラックがあります：「オリンピアドトラック」は国際科学オリンピアドに類似したスタイルの科学的推論能力を測定し、「リサーチトラック」は実際の科学研究能力を評価します。

- **Frontier-CS**（[frontier-cs.org/leaderboard](https://frontier-cs.org/leaderboard)）

  は、AIが難解な計算機科学の課題にどの程度対応できるかを評価するための、未解決・オープンエンド・検証可能かつ多様性に富んだベンチマークです。このベンチマークに含まれる問題は、研究者にとっても解決が困難なもの、最適解がまだ知られていないもの、または高度な専門知識がなければ取り組むことすら難しいものばかりです。

  ベンチマークは「アルゴリズム部門」と「研究部門」の2つのリーダーボードに分かれています。
  アルゴリズム部門には、最適化タスク、構築タスク、インタラクティブタスクが含まれます。
  研究部門は、オペレーティングシステム（OS）、高性能計算（HPC）、人工知能（AI研究タスク）、データベース（DB）、プログラミング言語（PL）、セキュリティ（サイバーセキュリティおよび脆弱性分析）という6つの主要な計算機科学分野をカバーしています。

------

# 視覚理解と推論

- **MMMU**
   （[mmmu-benchmark.github.io](https://mmmu-benchmark.github.io/)）

  専門家レベルの AGI を目指す大規模・多分野・マルチモーダル理解＆推論ベンチマーク。
   大学試験・小テスト・教科書から精選した 11.5 万問のマルチモーダル問題を収録し、

  - アート＆デザイン
  - ビジネス
  - サイエンス
  - ヘルス＆メディシン
  - 人文＆社会科学
  - テクノロジー＆エンジニアリング

  の 6 分野・30 学科・183 サブフィールドをカバーする。

  図表・ダイアグラム・地図・表・楽譜・化学構造など、多様な画像タイプに対する高度な認識・推論能力が求められる。

- **MMMU-Pro**
   （[MMMU-Pro Benchmark Leaderboard | Artificial Analysis](https://artificialanalysis.ai/evaluations/mmmu-pro)）

  MMMU を拡張し、10 択問題と「スクリーンショット・写真のみ」に埋め込まれた問題形式を追加。
   3,460 問を収録し、現実的なシナリオでの視覚 + テキスト統合能力を評価する。

- **MATH-Vision**
   （[MATH-Vision Leaderboard](https://mathllm.github.io/mathvision/#leaderboard)）

  視覚的な数学コンテキストを含む 3,040 問のコンテスト問題から構成されるベンチマーク。
   16 の数学分野・5 段階の難度をカバーし、多モーダル数学推論能力を測る。

- **CharXiv**（[charxiv.github.io/#leaderboard](https://charxiv.github.io/#leaderboard)）

  科学論文から抽出された2,323点の自然で挑戦的かつ多様なチャート（図表）を含む包括的な評価スイートです。CharXivには以下の2種類の問題が含まれています：
  （1）チャートの基本要素を識別する記述的問題、
  （2）チャート内の複雑な視覚要素の情報を統合して解答する推論問題。

  品質を確保するため、すべてのチャートおよび問題は人間の専門家によって慎重に選定・整理・検証されています。

- **ROME**
   （[BAAI/ROME](https://huggingface.co/datasets/BAAI/ROME)）

  281 問・8 サブタスクから成る視覚推論ベンチマーク。
   各サンプルは「画像がなければ答えられない」ように設計されている。

  サブタスク例：

  - Academia：大学講義の問題
  - Charts：論文・レポート・ブログ等からの図表
  - Puzzles & Games：ラベン行列、パズル、ゲームプレイ
  - Memes：改変ミーム画像
  - Geo：地理的な位置推定
  - Recognition：細粒度識別
  - Multi-image：間違い探し、動画フレームの並べ替え
  - Spatial：位置関係・距離・高さなど

- **ZeroBench**
   （[zerobench.github.io](https://zerobench.github.io/)）

  現代のマルチモーダルモデルにとって非常に難しい視覚ベンチマーク。
   デザイナーチームが作成・レビューした 100 問の難問と、そこから導出される 334 個のサブ問題（中間推論ステップ）で構成される。

- **VisuLogic**
   （[VisuLogic](https://visulogic-benchmark.github.io/VisuLogic/)）

  国内機関による 1,000 問の視覚推論ベンチマーク。
   量的変化・空間関係・属性比較など 6 カテゴリから多面的に多モーダルモデルの視覚推論能力を評価する。

- **OCRBench v2**
   （[ocrbench_v2](https://99franklin.github.io/ocrbench_v2/)）

  マルチモーダルモデルの「視覚テキストローカライゼーション＆推論」を評価するベンチマーク。
   10,000 の人手検証済み QA ペアを含み、高難度サンプルが大きな割合を占める。
   街景・レシート・数式・チャートなど 31 シナリオをカバー。

- **MMLongBench-Doc**
   （[Hugging Face Space](https://huggingface.co/spaces/OpenIXCLab/mmlongbench-doc) / [MMLongBench-Doc](https://mayubo2333.github.io/MMLongBench-Doc/)）

  長文マルチモーダルドキュメント理解ベンチマーク。
   135 本の長い PDF 文書（平均 47.5 ページ・21,214 トークン）から構成される。
   1,091 の専門家アノテーション QA を収録し、回答は

  - テキスト
  - 画像
  - 図表
  - テーブル
  - レイアウト構造

  といった複数モダリティ・複数ページにまたがる証拠に依存する。

  33% の質問は複数ページに跨り、22.5% は「答えが存在しない」よう設計され、幻覚傾向を測る。
- **Video-MMMU**（[videommmu.github.io/](https://videommmu.github.io/)）

  多モーダルかつ多分野にわたるベンチマークで、大規模マルチモーダルモデル（LMM）が動画から知識を獲得し活用する能力を評価することを目的としています。Video-MMMUは、6つの専門分野（30の細分化されたサブ分野）から厳選された300本の専門家レベルの動画と、それに対応する900問の人手によるアノテーション付き質問で構成されています。このベンチマークでは、認知プロセスの各段階（知覚・理解・適応）に沿った質問－回答ペアを通じて、知識獲得能力を評価します。

  各動画には、知識獲得の3段階に対応した3組の質問－回答ペアが含まれています：
  知覚（Perception）：知識に関連する重要な情報を識別する
  理解（Comprehension）：その背後にある概念を把握する
  適応（Adaptation）：新しい状況にその知識を応用する

  さらに、モデルの「増分精度（delta accuracy）」——すなわち動画視聴後の性能向上度合い——も評価対象としています。

------

# OCR と埋め込み（Embedding）評価

[Supercharge your OCR Pipelines with Open Models](https://huggingface.co/blog/ocr-open-models)

OCR モデルを評価する際、文書種別・言語などによって性能差が非常に大きくなる。

- **OmniDocBench**
   （[OmniDocBench/README_zh-CN](https://github.com/opendatalab/OmniDocBench/blob/main/README_zh-CN.md)）

  現実世界の多様なドキュメント解析を対象とするベンチマーク。
   書籍・雑誌・教科書など、多様な文書タイプを含み、HTML / Markdown 形式のテーブル評価指標が整備されている。

- **olmOCR-Bench**
   （[olmocr/bench](https://github.com/allenai/olmocr/tree/main/olmocr/bench)）

  英語 OCR モデル評価において非常に有効だとされるベンチマーク。

- **CC-OCR**

  （※詳細略）

- **Embedding Leaderboard**
   （[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)）

  MTEB（Massive Text Embedding Benchmark）に基づく、埋め込みモデルの代表的なリーダーボード。

------

# 動画・音声生成 & ロールプレイ評価

- **DesignArena**
   （[designarena.ai/leaderboard](https://www.designarena.ai/leaderboard)）

- **Speech-DRAME**
   （[speech_drame](https://github.com/Anuttacon/speech_drame)）

  ボイスロールプレイシナリオにおける、AI 生成音声の評価ベンチマーク。

- **UNO-Bench**
   （[UNO-Bench](https://meituan-longcat.github.io/UNO-Bench/)）

  音声・画像情報の「マルチモーダル統合能力」を評価する統一ベンチマーク。
   ほぼすべての問題が音声 + 視覚情報の両方を必要とし、従来の多択だけでなく、多段オープン QA 形式も導入して複雑な推論能力を測る。

  特徴：

  - **多様なソース**：クラウドソーシングによる写真・動画、フリー素材サイト、公的データセットなど。
  - **豊富なトピック**：社会・文化・アート・日常生活・文学・科学など。
  - **リアル音声**：20 人以上の話者による録音で、現実世界の音声多様性を再現。

- **Vue**（VUE）

  ByteDance による動画理解ベンチマーク。

  - **VUE-STG**：
     現実的シナリオでの時空間グラウンディング（STG）能力を評価する。
     10 秒〜30 分までの動画を対象とし、
    - 長文コンテキスト推論
    - 名詞句ベースのクエリ
    - すべての真の時間区間・バウンディングボックスの人手アノテーション
       を提供。評価には最適化された vIoU / tIoU / vIoU-Intersection を使用する。
  - **VUE-TR-V2**：
     Video QA ベンチマーク。動画長分布をより均等にし、ユーザの問い合わせスタイルに近いクエリ設計を行っている。

- **Moral RolePlay**
   （[RolePlay_Villain](https://github.com/Tencent/DigitalHuman/tree/main/RolePlay_Villain)）

  デジタルヒューマンの「道徳的ロールプレイ」を評価するベンチマーク。

   キャラクター行動と倫理的制約のバランスを測る。

- **OpenGameEval**（[https://github.com/Roblox/open-game-eval/blob/main/LLM_LEADERBOARD.md](https://github.com/Roblox/open-game-eval/blob/main/LLM_LEADERBOARD.md)）

  Robloxのゲーム開発タスクにおいて大規模言語モデル（LLM）を評価するためのフレームワーク。










