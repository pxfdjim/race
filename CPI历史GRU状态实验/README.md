# CPI 历史 GRU 状态实验

每个样本以一个有 PRI 的短序列为中心，最多取同一平台、同一对象或已确认局部分组的三个历史 CPI。窗口按 TOA 排序，在时间重叠、间隔超过 250 ms 或切分边界处分段。重叠和间隔用前一 CPI 的结束时刻与下一 CPI 的开始时刻计算。

每个时间步输入 13 维观测。11 维来自当前 CPI 的 RF、PW、PA、脉冲数、持续时间、PRI 数值及其分布，另外 2 维记录与上一个 CPI 的实际间隔和历史存在标记。GRU 隐层为 16 维，读完窗口后预测中心 CPI 的搜索、粗跟或精跟。实际窗口长度控制递推步数。

对照组使用相同网络、初始化、损失和切分，只读取中心 CPI。两组均在 0–300 秒拟合，在 320–400 秒选择轮次和粗跟得分倍率。用户已有的统计 `class` 结果随每条状态预测一并导出。报告按原始脉冲真值计分，含 102,140 条验证短序列和三个完整留出目标。

## 结果

| 留出范围 | 模型 | 三状态准确率 | 粗跟精确率 | 粗跟召回率 | 粗跟 F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 420–500 秒 | 原状态模型 | 94.85% | 35.96% | 3.83% | 6.92% |
| 420–500 秒 | 只看当前 CPI | 90.37% | 0.00% | 0.00% | 0.00% |
| 420–500 秒 | 历史 GRU | 95.22% | 0.00% | 0.00% | 0.00% |
| 三个整目标 | 原状态模型 | 94.00% | 33.54% | 28.97% | 31.09% |
| 三个整目标 | 只看当前 CPI | 87.07% | 0.00% | 0.00% | 0.00% |
| 三个整目标 | 历史 GRU | 93.90% | 21.44% | 7.60% | 11.22% |

历史 GRU 在两个主要留出集上的总体准确率均高于单步对照。在 420–500 秒的真实跟踪脉冲上，粗跟排序 AP 从 9.44% 提升至 12.22%。当前选定的决策倍率在该时间段给出的粗跟预测数为 0。该段 12,542 条真实粗跟脉冲中，4,670 条有历史 CPI，比例为 37.23%。此次历史模型的粗跟指标未达到原状态模型的采用条件。

## 复现

在 `zgy` 下运行。需要 Python 3.10+ 和支持 OpenMP 的 `g++`。

```bash
work=$(mktemp -d /tmp/zgy-gru-state.XXXXXX)
python -B CPI历史GRU状态实验/prepare_gru_state.py --output-dir "$work/prepared"
python -B CPI历史GRU状态实验/fit_gru_state.py --prepared-dir "$work/prepared" --output-dir "$work/fit"
python -B CPI历史GRU状态实验/evaluate_gru_state.py --prepared-dir "$work/prepared" \
  --fit-dir "$work/fit" --output-dir "$work/results"
```

- `prepare_gru_state.py` 构造时间历史、验证 CPI 和 PRI 的来源并保存归一化参数。
- `gru_state.cpp` 实现 GRU 前向、训练和预测。训练脚本先用有限差分核对梯度。
- `results/history_gru.model` 和 `results/current_cpi.model` 保存两组网络的权重。
- `results/comparison.json` 保存指标、决策倍率和输入来源。
- `results/joint_class_state_validation.csv` 保存每条序列的型号、历史长度、原状态、两版 GRU 状态及得分。
