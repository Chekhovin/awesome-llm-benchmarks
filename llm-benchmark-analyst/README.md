# llm-benchmark-analyst

在用户指定的评测范围内，完成模型比较、领域选型、基准解释和榜单覆盖核验。benchmark 内容完全依据《大模型评测榜单汇总.md》重建，没有旧清单兼容层。

## 来源

- 导入文件：`D:\Desktop\大模型评测榜单汇总.md`
- 导入日期：2026-10-05
- 包内完整快照：[references/benchmark-source.md](references/benchmark-source.md)
- 原始文件 SHA-256：`a36c85a76c37b77356251445edfae2d48b376e579c8bd10c9a35221182e2ad70`

包内快照与本次导入文件字节一致。该日期表示清单导入时间，不代表榜单已在线复核。运行时使用包内相对路径，不依赖原电脑桌面目录。

## 入口与参考

| 文件 | 用途 |
| --- | --- |
| [SKILL.md](SKILL.md) | 主入口、范围边界与任务流程 |
| [AGENTS.md](AGENTS.md)、[CLAUDE.md](CLAUDE.md) | 相应宿主的精简入口 |
| [PORTABLE_PROMPT.md](PORTABLE_PROMPT.md) | 跨工具指令；需连同源快照提供 |
| [references/core-dimensions.md](references/core-dimensions.md) | 按任务定位新清单，保留子榜和跨栏目关系 |
| [references/search-playbook.md](references/search-playbook.md) | 身份、网页结果归属、可比性及覆盖账本 |
| [references/data-defect-warnings.md](references/data-defect-warnings.md) | 来源内的风险主张及解释边界 |
| [references/report-template.md](references/report-template.md) | 按请求规模选用的输出字段 |

复制整个目录即可使用，不需要专属 MCP 或固定检索工具。`agents/openai.yaml` 保留原有显示设置；本包不自动安装或改变任何宿主配置。

更新来源时，用用户明确指定的新文档完整替换快照，并同步重建所有路由、风险与入口说明；更新哈希和定位信息，检查链接与源文档覆盖，排除已移出清单的内容。备份保存在技能目录之外，不能作为运行时回退来源。
