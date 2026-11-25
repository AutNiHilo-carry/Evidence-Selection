# HotpotQA Evidence Selection

Python 3.10+，仅依赖标准库，无需 GPU 或 API。

## 复现实验

```bash
python evidence_selection.py --self-test
python evidence_selection.py hotpot_first100.json --allow-first100 --out results
```

如有官方完整原始文件，可用以下命令；程序将校验完整原始文件的 SHA-256 后取前 100 题。

```bash
python evidence_selection.py hotpot_dev_distractor_v1.json --out results
```

原始数据：https://hotpotqa.github.io/ （Dev distractor）。本次输入为用户按原文件顺序提取的 100 题 JSON，含 100 个唯一 ID，未能独立核验完整原文件哈希；截取文件的 SHA-256 记录在 results/report_data.json。99 题有 10 个 context 段落，1 题有 2 个，保留实际候选集合。

## 输出文件

实验报告单独通过课程平台提交；下列文件由运行脚本生成。
- results/sample_ids.txt：100 道题的 _id（原始顺序）。
- results/summary.csv：三种方法 × 三档预算共 9 组统计。
- results/per_question.csv：900 条逐题评测结果。
- results/details.json：逐题选中证据、标注及桥接扩展轨迹。
- results/report_data.json：汇总、穷举验证、运行环境、输入哈希及案例。

算法计时为每题 5 次运行的中位数，再取 100 题的平均；不含读取、预处理及评测。时间会随运行环境变化。所有选择完成后才读取 supporting_facts；answer 不参与选择和评测。改进方案的规则在运行前固定。
