# 数据清单与审计

环境录音并不自动构成可循环训练真值。优先选择稳定的雨、风、水流、机械底噪等连续声景，排除人声、旋律音乐、爆炸等强起止事件。候选可来自 [WavCaps Freesound 子集](https://github.com/XinhaoMei/WavCaps)；[freefield1010](https://arxiv.org/abs/1309.5275)适合声景筛查，但其标准片段为 10 秒，通常不足以满足本项目默认 10 秒加 0.5 秒重叠区的构造要求。

`loopscape.curate_wavcaps` 读取 WavCaps 的 `fsd_final.json`，按描述和标签中的环境声音关键词、元数据时长筛选，并查找本地以 Freesound 数字 ID 命名的音频。默认关键词只是候选召回，不能替代实际听音与挑选。输出 `review_template.csv`、`approved_manifest.jsonl` 和 `review_rejected.jsonl`。第一次运行时批准清单为空；将模板复制为独立的 `review.csv` 后逐条核查并填写。后续运行通过 `--review-csv` 导入已核查条目。模板文件每次都会重建，因此不要直接在模板里保存审核结论。若音频命名不符合 ID 规则，可在审核 CSV 的 `audio_path` 填写实际路径。

每条原始清单必须包含 `source_id`、`source_url`、`author`、`license`、`ai_training_allowed`、`permission_checked_at`、`path`、`caption`；`group_id` 推荐用上传者或原始录音标识。`ai_training_allowed` 必须由人工核实后写入布尔值 `true`。目前预处理仅接受 `cc0`、`cc-by-4.0`、`cc-by-nc-4.0`；这只是项目数据过滤规则，不能替代对具体条款和上传者偏好的判断。[Freesound 官方 FAQ](https://freesound.org/help/faq/#can-my-sounds-be-used-to-train-generative-artificial-intelligence-gen-ai-models)说明了额外的生成式 AI 偏好。

`loopscape.prepare` 将长源录音裁取 10.5 秒连续区间，用最后 0.5 秒平滑修正开头，得到 10 秒片段。这样在接缝处使用原录音里相邻的两段声音。处理后的清单保留来源和处理参数；拒绝原因写入 `data/processed/rejected.jsonl`。数据按 `group_id` 的 SHA-256 确定性分配到训练、验证和测试集。

预处理后需要人工抽查：

1. 在循环播放两次时有无明显重复、静音或淡化。
2. 文字描述与实际声音是否匹配。
3. 上传者和许可证信息是否在检查日期仍允许该学术用途。
4. 同一原始录音、改名副本和同一作者高度相似声音是否跨分组泄漏。

训练数据、生成音频、权重和缓存默认被 `.gitignore` 排除。若需要交付实验数据，应由用户另行通过符合许可证的方式共享数据清单及获取步骤。
