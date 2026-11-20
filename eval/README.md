# 翻译回归评测

> 评测数据（`data/`）和转写缓存来自开发时使用的音频，**未随仓库公开**；
> 这里只有评测脚本，要运行需要自备同样格式的数据。

每次改动断句或翻译相关的代码后跑一遍，确认没有引入条数错位、时间轴重叠，翻译质量没有退步。

```powershell
conda activate subtitle-whisper
$env:PYTHONUTF8=1
python eval/run.py                                   # 默认后端（config.yaml），开发集
python eval/run.py --mt qwen3-8b --tag q3            # 换后端
python eval/run.py --set translation.granularity=cue --mt qwen3b --tag old   # 旧逐条方式对照
python eval/run.py --split test                      # 测试集：只在最终验证时看，不要拿来调参
```

- 不需要原始音频：读取项目 `.cache/` 里的转写缓存（清单在 `data/transcripts.json`），
  走与命令行**同一条**代码路径 `subtitle.pipeline.segment_and_translate`。
- 输出写到 `eval/out/<tag>/`：三种 VTT + `summary.json`（每条评测条目的译文与得分）+ `units_<文件>.json`
  （每个翻译单元：送给模型的原文、译文、处理方式、占几条原文条、排成几条单语条——查“合并/排版”问题时看这个）。
- 打印：格式校验（`verify_vtt.py`）、每个文件的结构指标（原文条数、单语条数、翻译单元数、碎片条、假名、
  相邻重复、超过 20 字的单语条、<0.5 秒的条、翻译耗时）、评测条目的“意思正确”数和失败条目。
- **主指标按单语中文文件（`_cn`）打分**；双语文件的中文行同样打分并列出。

## 数据格式（需自备）
- `data/evalset.json`：评测条目；每条有参考译文、“关键意思”正则（同义词放宽）、已知错误译法，按时间与输出字幕对齐。
- `data/split.json`：开发集 / 测试集划分（应在看任何结果之前划分）。
- `data/cues_all.json`、`data/transcripts.json`：原文字幕条和转写缓存清单。
- 局限：正则打分是有噪声的代理指标，会有漏判；看失败条目时请人工确认。
