# Autoscripter — 本地离线语音识别 + 翻译 + 字幕生成

输入音频/视频，本地完成 **语音识别 (ASR) → 翻译 (MT) → 生成 .vtt 字幕**。
所有模型在本地运行，**不调用任何在线 API**。主要面向日语，也支持中/英/韩/粤等。

> **AI 生成声明**：本项目的代码、测试和文档主要由 AI（Anthropic 的 Claude，通过 Claude Code）编写；
> 作者负责提出需求、做取舍决定、核对实际效果。代码按现状提供，使用前请自行评估。
>
> 评测数据（`eval/data/`）包含开发时测试音频的内容，未随仓库公开。

---

## 1. 环境搭建

### 1.1 测试机环境（见"复现信息"一节获取精确版本）
- OS: Windows 11
- GPU: NVIDIA RTX 4060 Laptop, **8 GB 显存**
- 驱动支持 CUDA 13.x；本项目 PyTorch 使用 **cu124** wheel（wheel 自带 CUDA 运行时，无需系统 CUDA Toolkit 匹配）
- Python **3.11**
- conda（Anaconda/Miniconda）

### 1.2 创建环境

**推荐：一条命令**（Windows；含 ffmpeg、CUDA 12.4 版 torch/torchaudio 和全部固定版本的依赖，约 3GB 下载）：

```powershell
conda env create -f environment.yml
conda activate subtitle-whisper
```

**或者手动装**（效果相同）：

```powershell
conda create -n subtitle-whisper python=3.11 -y
conda activate subtitle-whisper

# ffmpeg（用于解码任意音/视频）
conda install -c conda-forge ffmpeg -y

# 全部 Python 依赖（requirements.txt 里已经写了 CUDA 12.4 版 torch/torchaudio 的下载源）
pip install -r requirements.txt
```

> **人声分离 (`--separate-vocals`)**：依赖 `demucs` + `torchaudio`。Demucs(torch) 与
> faster-whisper(ctranslate2) 的 cuDNN 在同进程会符号冲突并崩溃，因此本项目在**独立
> 子进程**里运行 Demucs CLI 做隔离。首次使用会联网下载 htdemucs 模型（约 80MB，到
> `模型目录/.hf_cache`），之后离线可用；GPU 上分离 14min 音频约 30–60 秒。

> **CPU-only**：把 torch 换成 `pip install torch`（CPU 版）。ASR 会用 int8，翻译较慢。

> **依赖冲突**：`funasr` / `faster-whisper` / `transformers` / `bitsandbytes` 对
> `numpy`、`torch` 版本较敏感。本项目逐个安装并验证通过（版本见 `requirements.txt`）。
> 若你的机器上冲突无法解决，可只装 whisper + Qwen/NLLB 路径，暂不装 `funasr`
> （SenseVoice 后端将不可用，其余功能正常）。

### 1.3 bitsandbytes（8GB 显存跑默认翻译模型时需要）
默认翻译模型是 **Hy-MT2-7B**（腾讯翻译专用模型）。显存 < 16GB 时用 bitsandbytes 现场量化成
4-bit（8GB 卡上模型占约 4.65GB）。Windows 上 `bitsandbytes` 需 0.43+ 才有官方 wheel（本项目用
0.50.2）；装不上或加载失败时程序会自动回退到 **NLLB-200-600M** 并在屏幕上提示。

### 1.4 日语断句分词（fugashi + unidic-lite）
日语断句用 `fugashi` + `unidic-lite`（MeCab 形态素分析 + 词典，约 48MB，已写入 requirements.txt）
判断词和文节边界，避免把词从中间切开。没装时自动退回旧的启发式规则（日志会告警）。

---

## 2. 模型

所有模型统一下载到项目里的 **`model/`** 目录（`config.yaml` 的 `model_dir` 或 `--model-dir` 可改）。
下载用 `huggingface_hub.snapshot_download(local_dir=...)`，每个模型一个子目录，
已存在则跳过。`HF_HOME` / `HF_HUB_CACHE` 也指向该目录，避免缓存写 C 盘。
支持镜像：`setx HF_ENDPOINT https://hf-mirror.com`（或当前会话 `$env:HF_ENDPOINT=...`），
支持断点续传。

### 2.1 使用的仓库名（已在 HuggingFace 确认，v0 时下载）

| 角色 | HuggingFace 仓库 | 本地子目录 |
|---|---|---|
| Whisper 默认 (large-v3) | `Systran/faster-whisper-large-v3` | `faster-whisper-large-v3` |
| Whisper large-v3-turbo | `deepdml/faster-whisper-large-v3-turbo-ct2` | `faster-whisper-large-v3-turbo` |
| Whisper medium / small | `Systran/faster-whisper-medium` / `-small` | 同名 |
| 日语 kotoba-whisper | `kotoba-tech/kotoba-whisper-v2.0-faster` | `kotoba-whisper-v2.0-faster` |
| SenseVoice | `FunAudioLLM/SenseVoiceSmall` | `SenseVoiceSmall` |
| funasr VAD | `funasr/fsmn-vad` | `fsmn-vad` |
| **翻译默认** `hymt2` | `tencent/Hy-MT2-7B`（commit 9b0eb4e8，Apache-2.0，v1 时下载） | `Hy-MT2-7B` |
| 翻译快速档 `hymt2-1.8b` | `tencent/Hy-MT2-1.8B`（Apache-2.0，v1 时下载） | `Hy-MT2-1.8B` |
| 翻译备用 `qwen3-8b` | `unsloth/Qwen3-8B-bnb-4bit`（Qwen/Qwen3-8B 的 NF4 预量化版，Apache-2.0，v1 时下载） | `Qwen3-8B-bnb-4bit` |
| 翻译 `qwen3b`（旧默认） | `Qwen/Qwen2.5-3B-Instruct` | `Qwen2.5-3B-Instruct` |
| 翻译 `qwen7b` | `Qwen/Qwen2.5-7B-Instruct`（按需下载） | `Qwen2.5-7B-Instruct` |
| 翻译回退 `nllb` | `facebook/nllb-200-distilled-600M` | `nllb-200-distilled-600M` |
| 旧版 `hunyuan` | `tencent/Hunyuan-MT-7B`（腾讯混元社区许可，实测不如 Hy-MT2，按需下载） | `Hunyuan-MT-7B` |

> 注：`Systran/faster-whisper-large-v3-turbo` 不存在，turbo 使用社区转换版
> `deepdml/faster-whisper-large-v3-turbo-ct2`（下载量最高、维护活跃）。
> 为什么默认换成 Hy-MT2-7B、为什么 Qwen3-8B 用预量化仓库：依据是开发时在 8GB 卡上实测的
> 显存/速度/质量对比（审查记录未公开）。

### 2.2 预下载

```powershell
# 推荐组合：whisper + kotoba + sensevoice(+vad) + hymt2 + nllb
python main.py download --all-recommended

# 单独下载
python main.py download --asr whisper --mt hymt2
python main.py download --asr kotoba-whisper
python main.py download --asr sensevoice      # 同时会拉 fsmn-vad
```

运行时找不到本地模型会自动触发下载。

**模型版本是固定的**：常用模型（whisper large-v3、kotoba-whisper、Hy-MT2-7B / 1.8B、Qwen3-8B、Qwen2.5-3B、NLLB）
下载的是开发和测试时用的那个 commit（见 `subtitle/models.py` 的 `MODEL_REVISIONS`），仓库作者以后更新了也不受影响；
其余模型（SenseVoice、fsmn-vad、whisper turbo/medium/small、Qwen2.5-7B、Hunyuan-MT）下载最新版。

---

## 3. 使用

### 3.0 图形界面（v3 新增）
```powershell
python main.py gui                 # 自动打开浏览器；地址 http://127.0.0.1:7860（只在本机可访问）
python main.py gui --port 7861 --no-browser
```
- 把音频/视频**拖进网页**（可以多个），或者填本机路径（文件或文件夹，可多行）；
- “基本设置”就是命令行参数（输出哪些字幕、源/目标语种、模型、精度、VAD、人声分离、术语表……）；
- “高级参数”是 `config.yaml` 里的超参数（断句字数/时长、翻译单元合并、Whisper 的各项阈值），**只对这一次运行
  生效，不会改 `config.yaml`**；改了语音识别参数会重新转写（不会复用默认参数的缓存）；
- 运行时显示日志、红字警告和最后的汇总；完成后可以下载字幕、在表格里查看（可搜索）、边听边看（原文件 ≤300MB 时）；
  网页播放器里每条字幕会一直显示到下一条开始（长段没人声时上一句会一直挂着），**只影响网页里的显示，下载的字幕文件不变**；
- “查看已有字幕”页：打开任意 .vtt，表格查看 + 格式校验 + 质量警告（同 `verify_vtt.py`）。

和命令行走同一套代码（`cli.run_pipeline`），命令行的功能都还在。不开公网分享、不发统计数据、界面用系统字体。

**字幕自动保存**（每个文件完成就写出，不用点下载；汇总里会写“字幕已自动保存到：…”）：
拖进来的文件 → 项目的 `output/`；填路径的文件 → 原文件旁边；填了“字幕输出目录” → 那里。
网页上的“点文件名”只是通过浏览器另存一份。

**界面的缓存**都在项目的 `.cache/gradio/s<进程号>/` 里：拖进来的文件的副本（和原文件一样大）、下载用和播放用的副本。
按 **Ctrl+C** 退出时 Gradio 会删掉；直接关窗口或程序崩溃时来不及删，**下次启动界面时自动删**。不需要手动清理。
`.cache` 里另外长期保留的只有转写缓存（`*.json`，每个 0.1~0.2MB，同一个文件再处理时跳过识别）和分词词典 `jieba.cache`（约 9MB），
都可以随时删（删了只是下次重新生成）。

### 3.1 命令行

```powershell
# 默认：日语→中文，双语字幕，whisper + Hy-MT2-7B（整句翻译）
python main.py run "D:\media\sample.mp3"

# 知道是什么语言就直接指定（比自动识别可靠；识别结果和把握每次都会用黄字/红字显示）
python main.py run "D:\media\sample.mp3" --src-lang ja

# 一次给多个文件/文件夹
python main.py run a.mp3 b.wav "D:\media\作品C"

# 只要中文字幕
python main.py run "D:\media\sample.mp3" --mode mono-tgt

# 指定源/目标语言、模式
python main.py run input.mp4 --src-lang ja --tgt-lang zh --mode bilingual

# 开启语种适配（自动检测语种并按表选后端）
python main.py run input.mp3 --lang-adapt

# 适配 + 日语改用 sensevoice
python main.py run input.mp3 --lang-adapt --ja-asr sensevoice

# 批量处理文件夹（递归）
python main.py run "D:\media" --recursive --output-dir "D:\subs"

# 换翻译后端（转写缓存会复用，不重复转写）
python main.py run input.mp3 --mt hymt2-1.8b   # 更快，质量约等于旧默认 3B 的整句翻译
python main.py run input.mp3 --mt qwen3-8b     # 通用 LLM 备用
python main.py run input.mp3 --mt nllb

# 旧的逐条翻译方式（对照用）
python main.py run input.mp3 --mt qwen3b --mt-granularity cue
```

#### 运行时的提示和汇总（v3 起）
- **红字**：结果可能不是你以为的那样——翻译模型换成了低质量的 NLLB（显存不够或加载失败）、语音识别退到 CPU、
  语种识别把握低于 50%、识别出的语种就是目标语种（这个文件不会被翻译）、有句子没翻译出来（保留了原文）、
  翻译出错、字幕“延续显示”或超字数、某个文件处理失败。
- **黄字**：关键信息——每个文件识别出的语种和把握、片段数和被质量过滤丢掉的段数、实际用的翻译模型和精度。
- 每个文件翻译完**立刻写出**；一个文件出错不影响其他文件（以前要等全部翻译完才写，一个出错全部白做）。
- 最后打印汇总：每个文件一行，需要注意的项目逐条列出。**有文件失败时返回码为 1**（以前总是 0 并打印“全部完成”）。
- 翻译模型连续 3 句都出错（如显存不足）时停止翻译这个文件并记为失败，而不是写出一份几乎全是原文的“译文”
  （`translation.max_consecutive_errors`）。
- 输出被重定向到文件时不加颜色；设置环境变量 `NO_COLOR=1` 也可以关掉颜色。

### 3.2 主要参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `input` | 文件或文件夹（可以给多个） | — |
| `--recursive` | 文件夹递归 | 关 |
| `--mode` | `bilingual` / `mono-src` / `mono-tgt` | bilingual |
| `--src-lang` | 源语言（`auto` 自动检测） | auto |
| `--tgt-lang` | 目标语言 | zh |
| `--asr` | `whisper` / `kotoba-whisper` / `sensevoice` | whisper |
| `--mt` | `hymt2` / `hymt2-1.8b` / `qwen3-8b` / `qwen3b` / `qwen7b` / `nllb` / `hunyuan` | **hymt2** |
| `--mt-precision` | 翻译模型精度：`auto`（按显存自动选，见第 5 节）/ `full` / `int8` / `nf4`（强制） | auto |
| `--mt-granularity` | `sentence`（整句翻译再分配回各条）/ `cue`（旧：逐条翻译） | sentence |
| `--mono-display` | 单语译文：`sentence`（译文自己排版，一句话尽量一条，条数可以比原文少）/ `repeat`（旧：条数同原文，短句在其每一条都显示整句）/ `split`（旧：条数同原文，按比例切分） | **sentence** |
| `--lang-adapt` / `--no-lang-adapt` | 语种自适应选后端 | 关 |
| `--ja-asr` / `--zh-asr` / `--ko-asr` | 适配时覆盖对应语种后端 | — |
| `--output-dir` | 输出目录 | 输入文件所在目录 |
| `--swap-lines` | 双语调换上下行（默认译文在上） | 关 |
| `--vad` | 打开 VAD 预过滤（**默认关闭**，v2 起）。人声音量很小的音频不要开 | 关 |
| `--no-vad` | 关闭 VAD（现在已是默认；保留是为了兼容旧命令） | — |
| `--vad-threshold` | VAD 阈值 0~1，越低越灵敏；给了这个参数等于同时打开 `--vad` | 0.2 |
| `--separate-vocals` | 先用 Demucs 分离人声再识别（背景音乐/音效很响时用；**人声音量很小的音频不要用**） | 关 |
| `--glossary <file>` | 术语表（每行 `日文=中文` 或 JSON），强制专名译法 | 无 |
| `--model-dir` | 模型目录 | 项目里的 `model/` |

`config.yaml` 里 `asr:` 下的识别参数（`no_speech_threshold`、`drop_*` 等段级过滤阈值）也可以按音频调整；
v3 起转写缓存的文件名里带这些参数的哈希（等于默认值时不带，所以旧缓存仍然有效），**改了参数会重新转写**，
不会悄悄复用旧参数的结果。

### 3.3 VAD、人声分离该不该开（轻声细语、音效较多的音频）
**默认：不开 VAD、不分离人声，直接对原始音频全量识别**（v2 起；此前默认开 VAD）。
实测依据（开发时的对照实验）：
- **VAD 听不见音量很小的人声**：一个 24 分钟、全程轻声说话的音频，Silero VAD 在阈值 0.5 / 0.2 / 0.1 下都只在前 1 分半检测到语音
  （合计 35~54 秒），后面 20 多分钟的台词在转写前就被整段丢掉；另外两个同类音频（21 分钟、34 分钟）几乎整段
  检测不到，输出是空文件。不开 VAD 时 Whisper 能一直识别到结尾。正常音量的对话不受影响（检测到 38%~64% 是语音）。
  开着 VAD 还会让词级时间戳出现数秒的空洞（一条字幕被拉到 7 秒上限、出现单字条）。
- **人声分离（Demucs）会把小音量人声分坏**：同一个 24 分钟的音频，原始音频识别出的 62 段里，分离后有 20 段没了、
  12 段变成读音相近但意思不通的句子，还多出 6 条「ご視聴ありがとうございました」这类幻听（不分离时 0 条）。
  正常音量的对白没有单独对照过。
- 不开 VAD 的代价：静音/音效段可能产生幻听。已有两层兜底：Whisper 的静音幻听跳过
  （`hallucination_silence_threshold`）+ 段级质量过滤（按 `no_speech_prob` / `avg_logprob` / 压缩率丢弃，
  阈值见 `config.yaml`）。仍可能漏进少量拟声词。

```powershell
# 默认（推荐）：原始音频全量识别；语种自动检测不准时（开头是静音/音效）加 --src-lang ja
python main.py run input.wav --src-lang ja

# 背景音乐/音效很响、正常音量说话：可以试分离人声
python main.py run input.wav --separate-vocals

# 长时间没人说话、幻听很多：可以试打开 VAD（小音量的语音会被丢，慎用）
python main.py run input.wav --vad
```

**术语表**（可选，提升专名一致性）：
```
# glossary.txt，每行一条
絶対ゼロ=绝对零
現実世界=现实世界
```
`python main.py run input.wav --glossary glossary.txt`

### 3.4 语种适配表
在 `config.yaml` 的 `lang_adapt_map` 里修改映射，无需改代码：

```yaml
lang_adapt_map:
  ja: whisper          # 默认 whisper large-v3（带标点+词级时间戳，断句/翻译更稳）
  zh: sensevoice
  yue: sensevoice
  ko: sensevoice
  fallback: whisper
```

> 日语默认后端说明：规格原建议 kotoba-whisper，但 kotoba v2.0（蒸馏模型）无标点、
> 无词级时间戳，碎片化导致逐条翻译质量下降；whisper large-v3 带标点与词级时间戳，
> 端到端字幕质量更好，故默认用 whisper。如需 kotoba 更低的日语 WER：
> `--ja-asr kotoba-whisper`（或改回 `config.yaml` 的 `lang_adapt_map.ja`）。

开启 `--lang-adapt` 时，会先用 Whisper 对前 30 秒检测语种（若已用 `--src-lang`
指定则跳过检测），日志打印「检测到语种 X，选用后端 Y」。批量处理时同一后端的
文件会分组、顺序处理，避免反复装卸模型。语种检测的 VAD 跟随 `asr.vad_filter`（默认关；v3 以前固定开，
小音量人声的音频会被当成静音）。

---

## 4. 输出规则

- 输出 `.vtt`，默认写在输入文件所在目录（`--output-dir` 可改）。
- 文件名：`<原名>_<语言代码><原扩展>.vtt`，如 `xxx_cn.mp3.vtt`。
  中文代码用 `cn`，其余用 ISO 639-1（`ja`/`en`/`ko`…）。
- 双语模式一次生成三个文件：
  1. 原文 `xxx_ja.mp3.vtt`
  2. 译文 `xxx_cn.mp3.vtt`
  3. 双语 `xxx_ja-cn.mp3.vtt`（每条两行，默认译文在上、原文在下，`--swap-lines` 调换）
- 单语模式只生成对应一个文件。
- 断句：按标点/最大字符数拆分（中/日 20、英 42），每条 ≤ 7 秒，时间戳用词级
  时间戳校准（SenseVoice 用 VAD 分段时间戳）。日语按形态素/文节边界切分、同一句切出的各条长度均衡
  （不会再出现 `大丈夫で` / `す` 这种把词切开的条）；显示不足 0.6 秒的条向后面的空隙延长。
  句内有长停顿时在停顿处切（Whisper 常把停顿算进后一个词里，使一个词“长达数秒”；v2 起按
  每个字最多 `segmentation.max_char_sec` 秒估计真实发音时间），ASR 文本里的空格也是切点；一条超过 7 秒时
  保留发音最多的 7 秒窗口，而不是截掉结尾。
- 标点：中/日字幕去掉逗号、句号（停顿用空格），**保留问号、感叹号**（v1 起；此前也会删掉 ？！）。
- 译文怎么放进各条：翻译以“句子”为单位（一句话可能跨 2~3 条字幕）。
  - 哪些片段算“一句”：相邻、几乎没有间隔、前一段没有句末标点，**并且前一段话没说完**（日语用形态素判断：
    以终助词、感叹词、用言/助动词的终止形等结尾的算说完了，不再和后面合并；v2 起）。
    这个判断宁可少合并：一句话被识别成两段、前一段恰好以动词或「か」结尾时，会被当成两句分开翻译。
  - 单语中文文件（`_cn`，**v2 起的默认行为**）：译文自己排版，**不再要求与日文条一一对应**。
    一句译文放得下（≤20 字、≤7 秒）就只出一条，时间从这句的第一条日文条到最后一条；放不下就在中文的
    标点/词边界切成尽量少的几条。所以 `_cn` 的条数通常比 `_ja` 少，但每一条都正好由连续的几条日文条
    合并而成，不丢内容、不错位。一句话由几个 ASR 片段合并而成时，不会把上一段的尾巴和下一段的前半拼成一条。
    **每句译文在 `_cn` 里只出现一次**（v3 起由代码保证并检查）：按规则排不出来时（译文比这几条日文条
    容得下的长，或者短到切不开），宁可某条超过 20 字，也不把同一段译文放到两条上；唯一躲不开的情况是
    “译文能切出的段数比 7 秒上限逼出来的条数还少”（如 10 秒的一句只译出「是的」），这时后一条“延续显示”
    同一段文字。两种情况都会计数并红字提示。说话人自己把一句话说两遍不受影响：那是两句、各翻译一次、各显示一次。
    旧行为：`--mono-display repeat`（每条重复显示整句）/ `split`（按比例切分）。
  - 双语文件（`_ja-cn`）和原文文件（`_ja`）：条数和时间轴一致；整句译文按各条日文的字数比例切分，
    同一句话按时间铺开——中日语序不同，**不追求每条中文与下面那段日文逐字对应**。
- 编码 UTF-8 无 BOM，换行 `\n`。

### 4.1 校验与测试
```powershell
python verify_vtt.py "D:\subs"          # 校验目录下全部 vtt
python verify_vtt.py xxx_ja-cn.mp3.vtt  # 校验单个文件
python -m unittest discover -s tests -v # 单元测试（不需要 GPU）
python eval/run.py                      # 翻译回归评测（需要本机的评测数据和转写缓存，未公开，见 eval/README.md）
```
`verify_vtt.py` 检查 WEBVTT 头、序号连续、时间戳格式、时间轴递增不重叠、文件名规则、同源文件一致性
（原文与双语的条数和时间轴相同；单语译文的每一条正好由连续的原文条合并而成并覆盖全部原文条）
（任何一项不通过返回码非 0）；另外打印**质量警告**（不影响返回码）：显示不足 0.3 秒的条、中文轨里的日文假名、
只有 0~1 个汉字的条、`_cn` 文件里相邻两条文字相同。最后一项只看文件分不清“说话人自己说了两遍”和
“同一句译文被放到了两条上”，所以同目录里有原文文件时分两类报：**原文也相同**（多半是说话人重复，正常）/
**原文不同**（可疑）。“可疑”也可能是正常的（如说话人省略着重复了半句，两句译文碰巧一样）。
能确定是不是重复显示的检查在程序内部：每句译文排出的各条拼起来必须正好等于这句译文（`eval/run.py` 的
“重复显示检查”，不通过时返回非 0；单元测试也查这一条）。

---

## 5. 显存与性能（RTX 4060 Laptop 8GB 实测，v0）

显存为 `nvidia-smi` 全局占用增量（WDDM 下无法按进程统计），约数：

| 阶段 | 模型 | 精度 | 显存(约) | 备注 |
|---|---|---|---|---|
| ASR | faster-whisper large-v3 | float16 | ~3.9 GB | 默认；也用于语种检测 |
| ASR | kotoba-whisper v2.0 | float16 | ~2.1 GB | 日语；无词级时间戳 |
| ASR | SenseVoiceSmall + fsmn-vad | — | ~1–1.5 GB | 未单独精确基准 |
| MT | **Hy-MT2-7B**（默认） | 4-bit（现场量化） | 模型 4.65 GB / 整卡 ~7.6 GB | 加载 ~40~45s；20 分钟级文件翻译 ~2~2.7 分钟 |
| MT | Hy-MT2-1.8B | bf16 | 模型 3.58 GB / 整卡 ~5.6 GB | 加载 ~13s；约为 7B 耗时的 1/3 |
| MT | Qwen3-8B | 4-bit（预量化仓库） | 模型 6.07 GB / 整卡 ~7.4 GB | 加载 ~17~24s；整句模式 ~1.4~1.7 分钟/文件 |
| MT | Qwen2.5-3B-Instruct | fp16 | 模型 6.17 GB / 整卡 ~6.9 GB | 旧默认 |
| MT | NLLB-200-600M | fp16 | ~2.9 GB | 回退 |

（v1 实测；“整卡”含桌面常驻的 0.8~2GB，会随你开的程序变化。同一份代码的耗时在不同时刻可相差
一倍左右，只能当量级参考。）

ASR 与 MT **分阶段加载**：先转写完所有文件并卸载 ASR、清空显存，再加载翻译模型，
因此 8GB 足够（峰值不会同时叠加）。

**显存检测（v3 修订）**：按总显存给翻译模型选精度，实际选择每次用黄字显示，`--mt-precision` 可强制：

| 总显存 | Hy-MT2-7B / Qwen-7B / Hunyuan-MT-7B | Qwen2.5-3B | Hy-MT2-1.8B |
|---|---|---|---|
| ≥ 18GB | 全精度（bf16，约 15GB） | fp16 | bf16 |
| 10 ~ 18GB | bitsandbytes 8-bit（约 7.5GB） | fp16 | bf16 |
| 6 ~ 10GB（测试机 8GB） | bitsandbytes 4-bit（约 4.7GB） | ≥7GB fp16，否则 4-bit | bf16 |
| < 6GB | 回退 NLLB-600M（**红字提示**；可以改用 `--mt hymt2-1.8b`） | 4-bit | ≥5GB bf16，否则 4-bit |

7B 级全精度的门槛以前是 16GB——bf16 权重就有 15GB，16GB 的卡加上桌面占用放不下，所以提到 18GB，并加了 8-bit 一档。
**只有 4-bit 这一档在测试机（8GB）上实测过**；8-bit 只在 Hy-MT2-1.8B 上验证过能加载和翻译（8GB 放不下 7B 的 8-bit），
全精度路径没有验证。**加载时真的失败（显存不足、bitsandbytes 不可用）也会自动回退 NLLB**，同样红字提示，
汇总里会写“要求 X，实际用的是 nllb”。

“显存更大就换更好的模型”：现有的模型里 Hy-MT2-7B 已经是评测最好的，
所以显存更大时是让它用更高的精度（不量化），没有自动换成更大的模型——更大的模型要先下载、而且在 8GB 的卡上没法验证。

质量参考（同一评测集的封存测试集 24 条，盲评 0~2 分）：旧默认（3B 逐条）0.96 → 现默认（Hy-MT2-7B 整句）1.62。

---

## 6. 架构

```
main.py                 入口（run / download / gui）
config.yaml             适配表 / 默认值 / 断句、翻译、识别参数
prompts/translate.txt   翻译提示词（可改）
subtitle/
  cli.py                参数解析 + 编排（run_pipeline：ASR 分组 → 翻译 → 逐文件写出 → 汇总）
  gui.py                Gradio 图形界面（调用 cli.run_pipeline）
  console.py            红字/黄字提示，并记下来给汇总和图形界面
  models.py             HF 下载 + 模型注册表 + HF_HOME 设置
  media.py              ffmpeg 解码为 16k 单声道 wav（中间文件在 .cache/wav、.cache/vocals，用完即删）
  segments.py           后端无关的内部数据结构
  vram.py               CUDA/显存检测，精度与量化决策
  cache.py              转写缓存（键含文件哈希 + 后端名 + 语种 + VAD/人声分离 + 非默认的识别参数）
  segmenter.py          断句（日语形态素/文节边界 + 均衡切分；最短显示时长）
  pipeline.py           断句 + 翻译编排（CLI、图形界面与评测共用；按翻译单元检查重复显示）
  units.py              翻译单元：把跨条、跨 ASR 片段的一句话合并
  distribute.py         整句译文排成字幕条（单语自己排版 + 退路 / 双语按比例）
  vtt.py                WEBVTT 写出 + 文件名规则
  asr/                  whisper / kotoba / sensevoice + 语种检测 + 工厂
  mt/                   hymt（Hy-MT2/Hunyuan）/ qwen / nllb + 句子单元校验与回退 + 工厂
verify_vtt.py           格式校验 + 质量警告
tests/                  单元测试（python -m unittest discover -s tests）
eval/                   翻译回归评测脚本（评测数据 data/ 未公开）
```

---

## 7. 翻译策略与已知限制

**翻译策略（v1 起默认，`translation.granularity: sentence`）**：
1. 断句器切出字幕条，同时记住每一句（保留 ASR 原始标点）；
2. 相邻两句间隔 ≤0.3 秒且前一句没有句末标点时合并成一个“翻译单元”（ASR 常把一句话切成零间隔的两段，
   如「私がただれた責任」|「取ってください」）；最多 60 字、3 个 ASR 片段，合并处加「、」；
3. 每个单元单独翻译一次（Hy-MT2 只给前 3 句作背景；Qwen 给前 3 句和后 2 句），输出纯文本；
4. 确定性校验：空、含日文假名、整句不是目标语言、长度异常 → 去掉上下文重试一次 → 仍不合格只对这一句退回
   逐条翻译 → 再不行保留原文并**红字**告警（不丢条）；翻译模型连续 3 句出错时停止这个文件并记为失败；
5. 把整句译文分配回该单元的各条（见第 4 节），条数和时间轴不变。

旧方式（每条单独翻、12 条一批返回 JSON、条数不符二分重试）仍可用 `--mt-granularity cue` 选择，
但它在实测中会把模型合并出的正确整句译文当作“条数错误”丢掉、对碎片单独重译，并产生“合并后补位”的错位
（Subtitle Edit 里专门有 `TranslationShiftGuard` 防这种情况）。

**已知限制（如实说明）**：
- 双语文件里每条中文**不与**下面那段日文逐字对应（同一句话按时间铺开）；单语中文读起来连贯。
- 模型知识的上限：少数引申义/成语在 8GB 能跑的模型里都译不对（例：本项目样本里「私がただれた責任
  取ってください」的「ただれた」是“堕落、不检点”的意思，模型译成“溃烂”）。只能用术语表或人工校对缓解。
- 翻译单元偶尔会把两句独立的话并在一起（零间隔、无标点时），这时前一条可能只显示半句、后半句落到下一条。
- 翻译模型是否会拒译/弱化敏感内容**没有测试过**；校验能发现空或过短的译文，但发现不了用中文写的拒绝语。
- kotoba-whisper v2.0 是蒸馏模型，**不支持词级时间戳**（开启会原生崩溃），已自动
  退回片段级时间戳；SenseVoice 同样用 VAD 分段时间戳。日语时间精度以片段为准。
- Whisper 的 `translate` 任务只能翻英文；本项目统一「先转写原文，再本地翻译」，
  目标语言为英文时也走 MT（一致性更可控）。
- 8GB 显存上 Hy-MT2-7B 走 bitsandbytes 4-bit，生成较慢（每秒约 3~7 个 token）；每次加载要现场量化约 40 秒。
  想更快可用 `--mt hymt2-1.8b`。
- 翻译质量受 ASR 识别质量影响：识别出错会连带翻译出错（如同音错字「刑事様」实为「剣士様」，可用
  `--glossary` 写 `刑事様=剑士大人` 纠正；术语表只注入当前句子里出现的词条）。

## 8. 复现信息
精确版本见 `environment.yml` 与 `requirements.txt`。测试机上的关键版本：

| 组件 | 版本 |
|---|---|
| Python | 3.11 |
| CUDA (torch wheel) | 12.4 (cu124) |
| torch | 2.6.0+cu124 |
| transformers | 5.22.1 |
| faster-whisper | 1.2.1 |
| ctranslate2 | 4.8.5 |
| funasr | 1.4.22 |
| torchaudio | 2.6.0+cu124 |
| demucs | 4.1.0 |
| bitsandbytes | 0.50.2 |
| numpy | 2.4.6 |
| huggingface_hub | 2.4.0 |
| fugashi / unidic-lite | 1.5.2 / 1.0.8（v1 新增，日语断句） |
| jieba | 0.42.1（译文切分用；原本就在依赖里） |
| gradio | 6.32.0（v3 新增，图形界面） |

模型版本：常用的 7 个模型固定在开发时下载并测试过的 commit（完整编号见 `subtitle/models.py` 的
`MODEL_REVISIONS`，v4 起；这些编号在 HuggingFace 上都核对过存在）。

v1 的依赖变更只新增了 `fugashi==1.5.2`、`unidic-lite==1.0.8`（Windows cp311 有现成 wheel /
纯数据包，无需编译器），已同步写入 `requirements.txt` 和 `environment.yml`。

**全新环境验证（v4）**：在 Windows 11 + RTX 4060 8GB 上用 `conda env create -f environment.yml` 新建环境，
装出来的 129 个 Python 包与开发环境逐个版本一致，`pip check` 无冲突，torch 能用 CUDA；
在这个环境里 `pip install --dry-run -r requirements.txt` 不需要再装任何包（两个依赖文件一致）。
从 git 仓库干净克隆的项目（没有模型、转写缓存和样本）里，单元测试 56 个通过、5 个因缺少样本/缓存跳过，
图形界面能正常启动。没有验证的：模型下载（体积太大，只核对了版本编号）、其他操作系统、其他 CUDA 版本。
（v4 以前的 `requirements.txt` 有个问题：一行带 BOM 的注释被排到了文件末尾，pip 会报 `Invalid requirement`，
装不上；`environment.yml` 缺 PyTorch 的下载源。v4 已修好。）

v3 的依赖变更：为图形界面新增 `gradio==6.32.0` 及它带进来的 25 个包（annotated-types、brotli、fastapi、
gradio_client、groovy、hf-gradio、httpcore、httpx、opentelemetry-api、orjson、pandas、pillow、pydantic、pydantic_core、
pydub、python-dateutil、python-multipart、pytz、safehttpx、semantic-version、starlette、tomlkit、typing-inspection、
tzdata、uvicorn），版本都按现有环境里实际装上的写入 `requirements.txt` 和 `environment.yml`。安装前用
`pip install --dry-run` 确认过：只新增这些包，**没有改动任何已有包的版本**（环境里原有的 `httpx2`/`httpcore2`
是另外的包，保留）。只用命令行的话可以不装它们（`gui` 子命令才会 import gradio）。

**从零复现**：
```powershell
conda env create -f environment.yml
conda activate subtitle-whisper
python -m unittest discover -s tests       # 可选：确认环境没问题（不需要 GPU）
python main.py download --all-recommended  # 下载模型（固定版本）
```

## 9. 版本记录
| 版本 | 主要变化 |
|---|---|
| v0 | 初版：Whisper 识别 + 逐条翻译（Qwen2.5-3B），双语/单语 VTT |
| v1 | 整句翻译（翻译单元）+ 默认模型改为 Hy-MT2-7B；日语按形态素/文节断句；保留 ？！ |
| v2 | 单语译文自己排版（条数可以比原文少）；默认不开 VAD、不分离人声；句内停顿切分；“话没说完”才合并 |
| v3 | 图形界面；红字/黄字提示和运行汇总；按显存选翻译精度（新增 8-bit 档）；转写缓存带识别参数哈希 |
| v4 | 默认模型目录改为项目里的 `model/`；`download` 也读 `config.yaml` 的模型目录；粤语每条 20 字；Qwen 后端读取 `translation.generation` 解码参数；修好 `requirements.txt`（以前 pip 会报 Invalid requirement）和 `environment.yml`（可以一条命令建环境）；常用模型固定下载版本 |

## 10. 许可证
本项目代码以 MIT 许可证发布，见 [LICENSE](LICENSE)。用到的模型各有自己的许可证（见第 2 节的表格），下载和使用前请分别确认。
