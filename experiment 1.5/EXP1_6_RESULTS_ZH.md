# Experiment 1.6 结果：3e-6 endpoint-first 剂量探针

日期：2026-07-22 至 2026-07-24  
Run dir：`eaaj-pilot/outputs/exp15_cuda_grpo_gsm8k_caebbcc73461`

## 结论

Exp1.6 按预注册规则在 endpoint expansion gate 处得到 `STOP`。这不是运行失败：
Stage A、float32 Q 测量和六个 endpoint adaptation cells 均完整完成，但 3e-6
没有达到“持久 Q 位移”或“固定预算适应性下降”的扩格门槛。因此没有运行 full-grid
adaptation，也没有运行只对 full-grid 定义的 Phase 4。

## 完整性

- Phase 1：500/500 updates，8 个 checkpoint（0/25/50/100/200/300/400/500），无 safety stop。
- Phase 2：8 个 checkpoint 均以 float32 测量；ckpt-0 identity gate 完全复现 pilot。
- Phase 3 endpoint：ckpt {0,500} × seed {42,43,44}，6/6 cells 均完成 50/50 updates。
- 总本地计算时间：约 19.57 小时（Phase 1 11.76 h、Phase 2 4.6 min、Phase 3 7.74 h）。

## Expansion gate

| Gate | 观察值 | 阈值 | 结论 |
|---|---:|---:|---|
| G-A：late-window mean erank_L12 相对 ckpt-0 位移 | +4.06% | 绝对值 ≥ 7.5% | FAIL |
| G-B：mean3 delta(ckpt-0) − mean3 delta(ckpt-500) | -0.0089 | ≥ +0.05 | FAIL |

`exp16_gate_eval.json` 的最终 verdict 为 `STOP`。

## Endpoint adaptation

| Checkpoint | Seed | SVAMP-300 before | SVAMP-300 after | Delta |
|---:|---:|---:|---:|---:|
| 0 | 42 | 0.5300 | 0.5500 | +0.0200 |
| 0 | 43 | 0.5300 | 0.5567 | +0.0267 |
| 0 | 44 | 0.5300 | 0.5433 | +0.0133 |
| 500 | 42 | 0.6400 | 0.6700 | +0.0300 |
| 500 | 43 | 0.6400 | 0.6733 | +0.0333 |
| 500 | 44 | 0.6400 | 0.6633 | +0.0233 |

ckpt-0 的 mean3 delta 为 +0.0200；ckpt-500 为 +0.0289。后者高 0.0089，
方向与“适应性下降”相反。由于两个 endpoint 的 pre-adaptation accuracy 不同，
任何跨 checkpoint 解释都必须同时报告 `svamp_before`；本结果不建立
erank_L12 对未来固定预算适应性的可靠预测关系。

## 对下一轮的约束

1. 3e-6 × 500 是未达到 G-A/G-B 门槛的剂量点。
2. 1e-5 曾在单条 v2 轨迹约 step 55 出现 collapse，但 Exp1.5.1 三条 seed
   轨迹在 step 80 前均未复现，说明 collapse hazard 对轨迹随机性敏感。
3. 下一轮不能只换一个 seed 得出剂量阈值；需要把 Stage-A seed 作为设计维度。
4. 下一剂量应位于 3e-6 与 1e-5 之间，并继续使用 endpoint-first、早期 checkpoint
   与严格 safety stop，先定位“Q 有持久位移但尚未 collapse”的窗口，再决定是否扩格。
