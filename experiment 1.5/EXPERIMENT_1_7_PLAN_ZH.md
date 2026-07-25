# Experiment 1.7：复制化剂量窗口实验

状态：**设计冻结，尚未正式启动**
冻结日期：2026-07-25

## 1. 为什么 1.7 不能只是“把学习率再调高一点”

已有证据给出三个剂量点：

- exp1.5 v3：`lr=1e-6 × 500`，训练稳定，但 L12 Q 的晚期均值只变化约
  `-1.2%`，没有形成持续位移。
- exp1.6：`lr=3e-6 × 500`，训练稳定，L12 Q 晚期均值变化
  `+4.06%`；端点适应性反而略高，两个 expansion gate 均失败。
- exp1.5 v2 / exp1.5.1：`lr=1e-5` 曾在约 step 55 出现一次坍塌，但三次
  受控复跑均未复制，说明坍塌风险明显依赖训练轨迹，而不是一个确定的
  “学习率阈值”。

因此，下一步最重要的不是再跑一条单 seed 曲线，而是回答：

> 在 3e-6 与 1e-5 之间，是否存在一个能稳定移动 Q、但尚未稳定坍塌的
> 过渡剂量？如果存在，这个效应能否跨 Stage-A 轨迹复制？

## 2. 冻结设计

### 2.1 唯一处理变量

- Stage-A 学习率：`5.5e-6`
- 选择理由：接近 `sqrt(3e-6 × 1e-5) = 5.477e-6`，即已测稳定点和曾观察
  到坍塌点的对数中点。
- 其余模型、数据、GRPO recipe、500-step schedule、测量 dtype 和
  Stage-B recipe 与 exp1.6 保持一致。

### 2.2 Stage-A 复制

独立训练 seed：`42 / 43 / 44`，每条均跑到 500 steps。

检查点：

`0 / 25 / 50 / 75 / 100 / 200 / 300 / 400 / 500`

新增 step 75 是为了提高对 exp1.5 v2 中 step 55 附近转折的分辨率；晚期
位移仍固定使用 `300 / 400 / 500`，保持与 exp1.6 的 gate 可比。

### 2.3 两级 Stage-B

第一级只对每条完整 Stage-A 轨迹的 `ckpt-0` 和 `ckpt-500` 跑
Stage-B seed 42。这样总计 6 个适应性 cell，用于快速判断端点效应是否
跨 Stage-A 轨迹一致。

只有 expansion gate 通过，才补跑 Stage-B seed 43/44。补跑后每条
Stage-A 轨迹拥有 `2 endpoints × 3 Stage-B seeds`，总计 18 个 cell。

## 3. 预注册 gate

对每条完整 Stage-A 轨迹定义：

`q_shift = mean(erank_L12[300,400,500]) / erank_L12[0] - 1`

`endpoint_drop = delta_acc(ckpt-0, Stage-B seed42)
                 - delta_acc(ckpt-500, Stage-B seed42)`

### 3.1 坍塌 gate

- 若至少 2/3 轨迹因 safety stop 未到 500：`STOP_COLLAPSE`。
- 若只有一条坍塌：保留该事件作为过渡窗口证据；对另外两条完整轨迹继续
  Q 测量与端点探针，后续一致性条件要求两条都满足。
- 若不足两条轨迹已完整结束、且也未达到两条 safety stop：`INVESTIGATE`，
  表示实验尚未完成。

### 3.2 Q gate

在完整轨迹中同时满足：

- `abs(median(q_shift)) >= 0.075`
- 至少两条轨迹的 `q_shift` 与中位数同号。

当仅两条轨迹完整时，两条必须同号。

### 3.3 结果 gate

在完整轨迹中同时满足：

- `median(endpoint_drop) >= 0.05`
- 至少两条轨迹 `endpoint_drop > 0`。

当仅两条轨迹完整时，两条都必须为正。

### 3.4 决策

- `EXPAND`：Q gate 或结果 gate 至少一个通过，补跑 Stage-B seed 43/44。
- `STOP`：两 gate 均失败，不扩展。
- `STOP_COLLAPSE`：至少两条 Stage-A 轨迹坍塌，不做 ckpt-500 端点推断。

gate 使用 OR 规则延续 exp1.6：稳健的表征位移本身足以购买额外 outcome
replication；反之，即使 Q 指标未移动，若端点 outcome 跨轨迹一致，也值得
扩展验证。

## 4. 运行顺序

统一入口：

```powershell
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp1_7.ps1" -Action stagea
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp1_7.ps1" -Action measure
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp1_7.ps1" -Action probe
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp1_7.ps1" -Action status
```

只有 `status` 明确打印 `VERDICT: EXPAND` 后才允许：

```powershell
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp1_7.ps1" -Action expand
```

所有 phase 均沿用 `run_exp15.ps1` 的 keep-awake、GPU telemetry、日志和
可恢复执行。

## 5. 预计算力

按 exp1.6 实测外推：

- 三条 Stage-A：约 35 小时；
- 三条 Q 测量：约 15 分钟；
- seed42 端点探针：约 8 小时；
- 若 EXPAND，再增加约 16 小时。

因此 STOP 路径约 43 小时，EXPAND 路径约 59 小时。正式结果必须同时报告
wall time、safety marker、每条轨迹的 Q shift，以及所有已运行 Stage-B
cell；不得只报告跨轨迹平均值。

## 6. 解释边界

- 本实验检验的是小模型、当前 GRPO recipe 下的“剂量窗口”，不是对所有
  RLVR 或所有模型的普遍结论。
- Stage-B seed42 只是 expansion screen；通过后必须以 42/43/44 的完整
  结果为正式端点证据。
- 一条坍塌不是可复制的 hazard 结论，两条或以上才触发
  `STOP_COLLAPSE`。
- 如果两个 gate 再次失败，下一步不应继续在同一 0.5B / beta=0 配置上
  做更细的学习率搜索；应转向 KL 正则臂或更大模型。
