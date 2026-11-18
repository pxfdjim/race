# B04 正式数据新版流程：读取、分组、定位、匹配、类别与状态识别

本目录是独立的新实现，当前完成**读取与时间轴适配、稳定/抖动 PRI 提取、平台内分组、A-only 与 B 辅助保护版跨平台关联、二维交叉定位、A—B 航迹匹配、R01～R08 类别识别及 target_state 状态模型训练与预测**。状态模型的粗跟识别仍明显偏弱。旧的 `competition/04/29/` 程序和结果，以及 `new_04` 内其他目录的程序、模型、提交文件均未改写。

数值程序和 SVG 绘图使用 Python 3.10+ 标准库；安装 ImageMagick 时可导出 PNG，本次结果已包含图片。`sequence_id` 是短时脉冲序列，`local_cluster_id` 是平台内关联结果，`object_id` 是跨平台对象假设，`target_class` 是型号预测，`target_state` 是0搜索/1粗跟/2精跟的时变预测。**已导出官方字段的两个测试CSV；平台是否接受空编号尚未验证，没有上传。**

## 第九步：测试成绩CSV导出与核验

两个文件位于 `提交结果/P1A_test_pulse_predictions.csv` 和 `提交结果/P1A_test_object_track_links.csv`。全部8,600,592条脉冲与8个已有对象保留，类别和状态来自已有预测，7个对象为 `matched`、1个为 `no_match`，B编号不存在重复占用。

```bash
python export_submission.py --split test --output-dir 提交结果_new
python verify_submission.py --directory 提交结果_new
```

本地对文档明确要求的字段顺序、脉冲唯一与完整、对象保留、合法类别/状态、B编号存在与唯一进行了全量核验。仍有109,645条局部分组号为空、517,760条对象号为空；PDF未说明平台是否接受这些空值。**本地核验不代表平台导入或成绩已验证。** 粗跟模型效果也仍较弱。导出没有删除未解决记录，也没有伪造新对象填满列。见 [提交文件说明.md](提交文件说明.md)、[第九步导出记录.md](第九步导出记录.md) 和 `提交结果/submission_verification.json`。

## 第八步：target_state 训练与预测

统一模型参数为 **`models/target_state/target_state_model.json`**，包含短序列分类器与无PRI窗口回退分类器。训练和测试的所有脉冲都在 `output/state_predicted/{train,test}/pulse_state_predictions.csv` 中保留状态及证据；对象时间表是 `object_state_timeline.csv`，图片是 `target_state_timeline.png`。

```bash
python predict_target_state.py --split both --output-dir output/state_predicted_new
python verify_target_state.py --directory output/state_predicted_new/test
```

完整重新训练的入口是 `train_state_pipeline.py`，默认使用已有模型推断时无需重训。训练基于实际照射时序、幅度变化、短序列形态及观测校准量，真值编号与B编号不作为预测器特征。状态按时间变化，不用型号硬填，也不填补无观测时段。

场景内420–500秒诊断中，默认序列模型整体准确率94.85%，但粗跟召回仅3.83%，宏平均F1为0.6331；**不能把总体数字当作三种状态均已可靠识别**。测试无真值，不报告测试准确率；场景内检查被用于多轮开发，不是未触碰的外部测试。详见 [目标状态识别说明.md](目标状态识别说明.md)、[第八步完成记录.md](第八步完成记录.md) 和模型目录中的详细报告。

类别/状态与关联表已由第九步合并为官方列结构，空编号的平台接受规则仍需确认。

## 第七步：目标类别识别

程序由训练集类别标签学习 RF/PW/PRI 及稳定/抖动模式，不以目标编号、B 编号、平台位置作为类别特征；再汇总对象内多个时间段的波形证据。所有未关联脉冲也保留类别预测。

训练参数保存在 **`models/target_class/target_class_model.json`**，是可直接加载的统计密度模型 JSON。完整结果位于 `output/classified/{train,test}/`，包括 `object_class_predictions.csv`、`sequence_class_predictions.csv`、`pulse_class_predictions.csv` 和按类别着色的 `target_classes.png`。

```bash
python classify_targets.py --split both --output-dir output/classified_new
python verify_target_class.py --directory output/classified_new/test
```

无需每次重新训练。重新拟合的入口是 `train_target_class.py`。非保留实例420–500秒的有PRI序列，按脉冲加权分类准确率为96.91%；三个保留实例约97.61%。这不是测试集全脉冲精度，测试无类别真值，证据权重也不是校准概率。详见 [目标类别识别说明.md](目标类别识别说明.md)、[第七步完成记录.md](第七步完成记录.md) 和 `models/target_class/training_report.json`。

当前分类表是中间结果；状态由第八步补充，官方列结构的合并输出见第九步。

## 第六步：A—B 二维航迹匹配

完整结果位于 `output/ab_matched_2d/{train,test}/`。优先查看测试集的 `matching_overlay.png`、`object_track_links.csv` 和 `object_match_details.csv`。测试集8个 A 对象中7个匹配，`OA00004` 保留未匹配；训练集17个对象中9个匹配。

```bash
python match_tracks.py --split both --output-dir output/ab_matched_2d_new
python verify_track_matching.py --directory output/ab_matched_2d_new/test
```

实际时间对齐后比较整条轨迹的位置、有效重叠及充分支持的运动趋势，允许多个 A 碎对象匹配同一 B，不强制匹配。A 的前序身份已经使用 B 辅助，本步不是独立 A-only 准确率验证。算法与字段见 [AB航迹匹配说明.md](AB航迹匹配说明.md)，全量记录见 [第六步完成记录.md](第六步完成记录.md)。

上述多对一是内部诊断策略；正式提交要求同一B编号只关联一个对象，第九步导出会强制检查，发生冲突时拒绝导出，需回到匹配阶段解决。当前测试的7条匹配无该冲突。

## 第五步：二维交叉定位与航迹

完整结果位于 `output/localized_2d/{train,test}/`。默认使用下述 B 辅助保护版归属，位置只由 A 测向及相应时刻的平台经纬度求解，忽略全部高度。查看 `track_preview.html` 或 `cross_localization.png`、`tracks_2d.png`。

```bash
python localize_objects.py --split both --output-dir output/localized_2d_new
python verify_localization.py --directory output/localized_2d_new/test
```

两平台交叉，三个及以上平台联合加权估计。滤波保留观测缺口，不用 B 坐标补全 A。见 [二维交叉定位说明.md](二维交叉定位说明.md) 与 [第五步完成记录.md](第五步完成记录.md)。

## 第四步补充：B 辅助保护版关联

后续定位采用 `output/fused_b_assisted_guarded/`。原 A-only 结果继续保留在 `output/fused/`，二者不要混用。代码入口为 `assist_fusion.py`、`guard_b_assistance.py`，说明见 [B辅助跨平台关联说明.md](B辅助跨平台关联说明.md) 与 [B辅助改进完成记录.md](B辅助改进完成记录.md)。

## 第四步：A-only 跨平台关联

复用已有 PRI 和平台内结果：

```bash
python fuse_platforms.py \
  --data-root .. --split both \
  --pri-root output/pri \
  --group-root output/grouped \
  --position-root output/group_inputs \
  --output-dir output/fused_new
```

本次完整结果在 `output/fused/train/` 和 `output/fused/test/`。已有目录拒绝覆盖，查看结果无需重跑。参数为 `config/fusion_config.json`，没有模型权重文件。本步不读取 B，不将 A 的 target_id/class/state 作为关联输入，也不预设对象数。

| 文件 | 内容 |
|---|---|
| `fused_objects.csv` | 融合对象摘要：平台、成员数、时间范围和证据量标记 |
| `sequence_object_links.csv` | 每条短序列唯一的 object_id 或未匹配状态 |
| `local_object_memberships.csv` | 本地组与融合对象的多对多成员统计，不是整组硬映射 |
| `cross_platform_events.csv` | 2～4平台短时组合、参与序列、粗交会位置及几何检查证据 |
| `pulse_object_predictions.csv` | 所有原始脉冲的对象归属或未匹配状态，保持输入顺序 |
| `fusion_run.json` | 参数、来源与程序/输出哈希、覆盖率、组合数、对象数 |
| `output_verification.json` | 独立文件与成员一致性核验 |

对象完整键为 **`(scenario_id, data_version, object_id)`**。一个短时组合中每个平台至多一条序列；同一持续对象可在不同时间接纳同一平台的多个片段。每条脉冲最终至多一个对象，证据不足时保留空编号。

算法先以时间重叠及 RF/PW/PRI 筛候选，检查前向测向交会、交会角、多平台残差和距离差能否解释 TOA 相位差，再进行跨时间对象关联。模糊组合暂缓回看；只有单平台观测时可借助邻近融合对象的几何与时间证据补关联。

这里的交会坐标只用于**粗几何关联**，尚不是正式定位航迹。算法假设提供的 TOA 可在平台之间比较；没有估计未知时钟偏差。高度和斜距未知，距离差检查使用水平近似。覆盖率不足、误合并和碎对象均需同时看，不能把高配对同源率当作全量准确率。

运行与评价记录见 [第四步完成记录.md](第四步完成记录.md)，细节见 [跨平台关联说明.md](跨平台关联说明.md)。

训练集独立评价：

```bash
python evaluate_fusion.py \
  --sequence-truth output/group_validation/train_sequence_truth.csv \
  --links output/fused/train/sequence_object_links.csv \
  --output output/fused/train/sequence_evaluation.json

python evaluate_fusion.py \
  --truth-file ../B04赛题初赛A榜训练数据/初赛数据_A榜_A类传感器训练数据.csv \
  --pulse-predictions output/fused/train/pulse_object_predictions.csv \
  --output output/fused/train/pulse_evaluation.json

python verify_fusion.py --fusion-dir output/fused/test
```

标签只由评价程序读取。身份指标按场景/版本跨平台统一计算，不能直接与第三步“每个平台分别计算”的指标相比；覆盖率另列。测试无标签，不报告测试准确率。

复现开发比较与时间留出检查：

```bash
python diagnostics/validate_fusion.py \
  --pipeline-root . --output-dir output/fusion_validation_new
```

本次结果在 `output/fusion_validation/validation_report.json`。开发候选均未达到预设覆盖要求，程序按已声明回退规则选择；该情况已记录，不能当作验证全部达标。

## 第三步：平台内分组

复用已有 PRI 结果，不需要重算 PRI：

```bash
python group_sequences.py \
  --data-root .. --split both \
  --pri-root output/pri \
  --output-dir output/grouped_new
```

默认读取正式 A 原始文件、`output/pri/{train,test}/` 和 `config/group_config.json`。节点位置缓存位于 `output/group_inputs/`；缺失时自动从 A 按0.1秒间隔抽取，保留各平台首尾观测。输入和 PRI 的 SHA256、单位、逐行键均会核对。

本次完整结果已经保存在 `output/grouped/train/`、`output/grouped/test/`，无需为查看结果重新运行。已有目录拒绝覆盖，重跑时选择新目录。

| 文件 | 内容 |
|---|---|
| `local_groups.csv` | 每个平台内的组摘要、confirmed/tentative 证据量标记、成员数及参数范围 |
| `sequence_group_links.csv` | 每条 PRI 短序列的 local_cluster_id、关联代价和依据；待定序列的组号为空 |
| `local_predictions.csv` | 每个原始脉冲一行，保留 pulse_id、sequence_id、PRI 模型和平台内组号/状态 |
| `bearing_observations.csv` | 组内短序列的时刻、测向及当时节点位置，可供后续跨平台融合使用 |
| `grouping_run.json` | 固定参数、来源/程序哈希、组数、脉冲覆盖与分配状态统计 |

完整组键是 **`(scenario_id, data_version, platform_id, local_cluster_id)`**，不能只按 `L00001` 合并不同平台。`local_cluster_id` 尚不是跨平台 object_id。

分组综合方位趋势、时间、RF/PW、PRI 与较弱的 PA 证据；同一目标可以在不同 CPI 改变参数。密集重叠序列的波形/稳定周期相位冲突会阻止合并。模糊序列先暂缓，再用前后观测补判；未知 PRI 脉冲仅在邻近序列证据充分且候选唯一时补分配，其余保留待定。

训练真值只用于独立评价和开发时间窗参数选择，不作为关联输入；本步不使用 B，也不固定目标个数。组号是算法结果，仍可能误合并或将同一目标拆成多个组；confirmed 也不是正确性保证。详见 [平台内分组说明.md](平台内分组说明.md) 和 [第三步完成记录.md](第三步完成记录.md)。

全量训练评价（包含未知 PRI 的补分配）：

```bash
python evaluate_local_groups.py \
  --truth-file ../B04赛题初赛A榜训练数据/初赛数据_A榜_A类传感器训练数据.csv \
  --pulse-predictions output/grouped/train/local_predictions.csv \
  --output output/grouped/train/pulse_evaluation.json
```

指标同时报告覆盖率、组纯度和 B-cubed 精确率/召回率/F1；后者分别反映误合并和误拆分，不能只看组数或纯度。测试集没有真值，只报告分配统计及输出一致性。

## 第二步：PRI 提取

在本目录运行：

```bash
python estimate_pri.py --data-root .. --split both --output-dir output/pri_new
```

分别处理正式训练和测试 A 数据；本步不读取 B，也不使用 A 的 `target_id`、`target_class`、`target_state` 作为输入。默认参数位于 `config/pri_config.json`。不需要训练权重文件。

运行已经完成的结果在 `output/pri/train/`、`output/pri/test/`。每个目录包含：

| 文件 | 内容 |
|---|---|
| `pulse_pri.csv` | 每个输入脉冲一行，含序列、模型、PRI 中心与分位范围、上一脉冲及观测间隔 |
| `pri_sequences.csv` | 序列摘要：脉冲数、起止时间、RF/PW/PA/DOA 统计与验证方式 |
| `pri_run.json` | 参数、输入/输出计数、覆盖率、源文件及程序 SHA256、运行时间 |

输出保持输入顺序，未获得可靠序列的脉冲也保留，标为 `unknown`，PRI 字段为空。结果目录已经存在时程序会拒绝覆盖，请选一个新目录。失败时保留 `.目录名.partial-进程号` 供诊断，只有完整成功才发布正式目录。

核心流程为：**按平台、短时窗构建 RF/PW/DOA 邻域 → 验证稳定周期序列 → 验证有界抖动序列 → 多阶 TOA 差分与图路径补提取 → 输出序列或 unknown**。PA 只在候选边排序时作弱约束。0.5 秒核心窗带 0.08 秒前视；读取批次不是序列边界，也不被解释为 CPI。

算法细节、参数、评价含义和局限见 [PRI算法说明.md](PRI算法说明.md)，运行结果见 [第二步完成记录.md](第二步完成记录.md)。

训练标签独立评价命令：

```bash
python evaluate_pri.py \
  --truth-file ../B04赛题初赛A榜训练数据/初赛数据_A榜_A类传感器训练数据.csv \
  --predictions output/pri/train/pulse_pri.csv \
  --output output/pri/train/evaluation.json
```

这里评价序列同源性和覆盖率。训练集没有真实 PRI 和 CPI 标签，不能把同源纯度当作 PRI 数值准确率。测试集没有真值，不能运行同样的准确性评价。

复现固定训练时间窗的参数比较与留出评价：

```bash
python diagnostics/validate_pri.py \
  --source ../B04赛题初赛A榜训练数据/初赛数据_A榜_A类传感器训练数据.csv \
  --output-dir output/pri_validation_new
```

观测缓存会删除全部 A 标签列，真值另存；RF 门限只在 development 时间窗比较，随后固定参数，在 validation 时间窗评价，不根据测试标签调参。

## 第一步：读取与单位校验

在本目录中执行：

```bash
python prepare_inputs.py \
  --data-root /home/zhougy/competition/new_04 \
  --output-dir output/prepared

python -m unittest discover -s tests -v
```

默认扫描训练与测试的完整 A/B 数据。A 按 20,000 行分块，仅输出前 1,000 行的标准化样例，不额外复制数 GB 的完整 A；后续算法可直接消费同一个流式接口。B 数据较小，完整导出。

需要完整标准化 A CSV 时添加 `--write-normalized`。此时输出会明显增大，但仍按块写入，不将全部脉冲存入内存。只验证一组可用 `--split train` 或 `--split test`；自定义文件可同时指定 `--a-file`、`--b-file`（只对单个 split 使用）。

默认 `--id-check ordered` 利用本次正式数据的顺序数字 pulse_id 检查唯一性，内存与脉冲数无关。输入为无序或字符串 ID 时，使用 `--id-check sqlite`，临时磁盘索引精确检查唯一性，不要求重编号。

## 已实现的处理

| 内容 | 规则 |
|---|---|
| TOA | 源单位由配置明确指定，正式集为 μs；精确转换为整数 ns，保留原始字符串 |
| 小数精度 | 31.927 μs → 31927 ns；不做 int() 截断，不静默丢掉亚纳秒精度 |
| RF/PW/PA | 分别保留 MHz、μs、dBm，不混入 TOA 的时间倍率 |
| A 字段 | 兼容 sensor_lontitude / sensor_longitude，及 param1/2/3 与 RF/PW/PA 等别名 |
| B 字段 | 兼容 target_lontitude / target_longitude / lontitude 等实际拼写 |
| 观测平台 | 根据 platform_id 区分，平台数不固定；不拆掉混合文件、不丢脉冲 |
| 平台位置 | 每条脉冲保留自己的经纬度、高度，不使用平台平均位置 |
| A 时间 | 由公开场景起点＋TOA 建立时间轴，并逐行核对记录 datetime 的显示精度 |
| B 时间 | 使用实际带日期、秒的时间戳，保留各航迹起始偏移、1/2 秒采样和长缺口 |
| B 插值 | 仅在实际相邻观测时刻间插值；默认不跨超过 3 秒的缺口，也不外推 |
| 标签隔离 | A 的 target_id/class/state 不读入观测对象，B 的 target_id 不读入；公开 B 类别可保留 |
| 校验 | 字段数量、非有限值、位置范围、重复脉冲、同航迹重复时刻、单位与日期一致性 |

旧 m0 示例只有分钟级 B 时间，不应由新读取层自动猜成“每行一秒”；此类时间将明确拒绝，原实验仍可用原程序查看。

## 时间原点与限制

`config/input_units.json` 使用公开 D1 的场景起点：

- P1A/train：2026-11-10 09:00:00。
- P1A/test：2026-11-11 09:00:00。

新场景应在配置中补充其已知时间原点。没有时区后缀的日期按同一场景的本地墙钟比较，不假称 UTC；有明确时区后缀时才换算为 UTC。A 记录 datetime 为毫秒显示，逐行允许其显示精度范围内的差异。

这里做的是**按提供时间建立一致坐标轴**，没有估计或校正未知的 A/B 时钟偏差、B 上报延迟，也没有将某条 B 航迹的第一个点强制当作 0 秒。

## 模块与后续接入

```python
import json
from src.dataio import InputConfig, iter_a_batches, read_b_tracks

cfg = InputConfig.from_dict(json.load(open("config/input_units.json")))

for batch in iter_a_batches(a_path, cfg, batch_size=20_000):
    for pulse in batch:
        # pulse.toa 为整数 ns；scope 区分场景、版本、平台。
        # 后续序列算法要跨块维护平台状态，读取块不是 CPI 或目标边界。
        pass

tracks = read_b_tracks(b_path, cfg)
for track in tracks.values():
    point = track.position(time_ns=3_000_000_000,
                           maximum_gap_ns=3_000_000_000)
```

记录保持源文件顺序，读取层不重新排序、不分组、不改变身份编号。PRI 阶段检查场景内 TOA 顺序，并在平台内构建序列；如果自定义数据未排序，应先按场景、版本、TOA 整理，保留原始 pulse_id，并按需要使用 `--id-check sqlite`。

输出 A 样例的 `toa` 是**整数纳秒**，`toa_raw` 和 `toa_input_unit` 描述原始值及原始单位。RF/PW/PA 仍使用原物理单位。`timestamp_ns` 是 TOA 与配置原点组成的时刻，`recorded_timestamp_ns` 是原 CSV 中较粗的 datetime；不可把标准化后的 toa 再按微秒乘一遍。

## PRI 参数单位复核

```bash
python review_pri_parameters.py \
  --baseline-config config/legacy_pri_reference.json \
  --training-audit diagnostics/output/training_pri_audit.json
```

该复核将旧搜索范围、分桶和误差容差换成物理单位，并检查此前训练诊断的主要观测间隔是否落在范围内。候选门限网格另存，不会自动写入现有模型。第二步只比较了其中的 RF 门限，不能宣称所有参数均已最优。

## 输出

- `output/prepared/input_validation.json`：全量读取计数、平台分布、时间范围、源文件 SHA256 和时间一致性。
- `train_A_sample.csv`、`test_A_sample.csv`：标准化 A 小样例，不含真实标签。
- `train_B_normalized.csv`、`test_B_normalized.csv`：完整 B 公开观测及真实时间。
- `output/pri_parameter_review.json`：PRI 单位复核、训练诊断及待验证参数网格。

`prepare_inputs.py` 校验输入，`estimate_pri.py` 提取 PRI，`group_sequences.py` 产生 local_cluster_id，`fuse_platforms.py` 产生跨平台 object_id。当前仍不生成正式提交 CSV。

## 目录位置

新版代码统一保存在 `competition/new_04/zgy/`。`diagnostics/` 为训练诊断与评价，`src/`、`config/` 和 `tests/` 为各步骤实现。原 home 目录下的工作副本保留作备份，后续以本目录为准。
