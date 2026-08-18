# Task 3: Source-Grounded Review of Four Plasticity Papers and Search for the Closest Reusable RLVR Pipeline

## 0. 中文摘要（先读这一页）

这四篇是 Tommy 指定的必读材料。我把 PDF 全部下载到 `lit review/task3_core_papers/`，逐页读完正文与附录，并核对了每篇的官方代码仓库。核心结论如下。

**关于「可塑性」的定义，四篇各说各的，而且没有一篇和我们的定义完全一致。** 只有第 4 篇（Zyphra 的 Can Scale Save Us）在操作层面上和我们一致：**固定训练预算下，模型在新目标分布上的进步能力**。第 3 篇（When RL Fails after SFT）在实验结构上最接近我们——它确实从 7 个不同 SFT checkpoint 各自启动 RL，并画出"每个 checkpoint 的 RL 增益"曲线——但它没有报告任何相关系数、没有统计检验、没有多种子重复，而且**数学 RL 到底跑了多少步论文里根本没写**。第 2 篇（PCR）把 plasticity 定义成 GRPO 损失里的 surrogate-gain 那一项的梯度，跟激活/表征完全无关；第 1 篇（Plasticine）是传统 deep RL 的 benchmark，指标定义最规范，但没有 LLM、没有 RLVR、没有"从中间 checkpoint 预测未来学习"这件事。

**没有任何一篇做过我们要做的事**：用早期 RLVR 阶段测到的激活指标，去**预测**后续固定预算适应性。第 4 篇拟合了一条"可塑性衰退起点 vs 模型规模"的幂律（8 个点，留一交叉验证），但那是按模型大小预测起点，不是用激活指标预测某一次训练的未来学习。**这意味着我们的 RQ1 在这四篇里是空白的——这是好消息（有新意），也是坏消息（没有现成的方法论可抄）。**

**关于代码复用：只有 Plasticine 有官方开源仓库（MIT，已核实）。** 第 2、3、4 篇都没有放代码。所以"最接近的现成 RLVR pipeline"不在这四篇里面。我又查了 TRL、verl、OpenRLHF、Open-R1、EasyR1。结论是：**继续用我们现有的 `eaaj-pilot`（TRL 1.6 GRPOTrainer），只从 Plasticine 借指标实现细节、从第 4 篇借 probe 协议**。理由在第 7、8 节，简单说：verl 的 GRPO 配方对 0.5B + Colab L4 来说太重（需要 Ray + FSDP，官方 GSM8K GRPO 脚本最小是 0.6B 且面向多卡），而我们真正难的部分（checkpoint 分叉、Stage-B 独立启动、激活 hook、frozen probe）在任何框架里都得自己写——我们已经写好了。

**我在核对过程中发现了我们自己 pipeline 的两个具体问题，建议优先处理：**

1. **有效秩的探针数量不足。** Qwen2.5-0.5B 的隐藏维度 d = 896，而我们主测量用 `probe_questions = 512` 条提示，得到的激活矩阵是 512×896。行数少于列数，秩最多只能到 512，有效秩因此被采样量截断、而且随 n 有系统性偏差。Plasticine 的代码里专门写了 `assert features.size(0) >= features.size(1)` 就是为了防这个。建议把主测量的探针数提到 ≥ 1024（配置里已有的 `sensitivity_probe_questions = 2048` 正好可用），或者改成按 token 池化取样本。
2. **休眠神经元比例恒为 0 不是"可塑性完好"的证据。** 我查了实际产物 `metrics_ckpt0.json`：三层的 `dormant_frac` 在 τ=0.025 和 τ=0.1 下全是 0.0，而最小归一化分数是 0.13–0.44。原因是结构性的：Qwen2 的 MLP 是 SiLU 门控，`act_fn(gate)*up` 几乎不会精确归零，分数分布天然远离阈值。这个指标在这个模型上**没有分辨力**，不是"没有退化"。第 4 篇用 ε=0.01 且只在 MLP 层测，得到的是 0–95% 的大范围变化，但他们的模型用 GeLU、非门控、且训练了几千亿 token。建议明确记录为"该指标在本设置下无判别力"，并把重点放到有效秩、参与率、各向异性上。

以下是完整的英文技术评审，结构按 Tommy 要求的 Part A–F 组织。

---

## 1. Executive summary

**What was verified.** All four papers were downloaded from arXiv and read in full, including every appendix that exists (`lit review/task3_core_papers/`). Identities, version dates, author lists and code availability were checked against the arXiv abstract pages and, where a repository was claimed, against the GitHub API on 2026-07-26.

**Finding 1 — the four papers do not share a definition of plasticity, and only one shares ours.** Plasticine defines plasticity implicitly, through a battery of six network-level metrics (Appendix D, pp. 10–11) rather than through any learning outcome. Paper 2 (PCR) defines plasticity as the gradient of the GRPO surrogate-gain term (Eq. 5, Eq. 8), i.e. as a *loss component*, not as a capacity. Paper 3 defines it as "the ability of SFT models to undergo reward-driven improvement in subsequent RL" (footnote 1, p. 2) — conceptually identical to ours. Paper 4 gives the cleanest operational definition, and it is the one we use: "degradation in a model's ability to improve on a target distribution under a fixed training budget" (§I, p. 1).

**Finding 2 — only one paper actually launches future training from a checkpoint ladder, and it is not statistically analysed.** Paper 3 trains GRPO independently from seven SFT checkpoints (epochs 0.5/1/2/4/8/16/32) and plots the resulting RL gain per checkpoint (Fig. 5, p. 6). That is structurally our experiment. But: the RL step budget for the math setting is never stated; a single shared seed is used ("seeds are kept identical across initializations", Appendix B, p. 21); no error bars appear in any table; and no correlation coefficient, regression or held-out test relates the pre-RL measurements to the post-RL gain. Paper 4 also probes future learning from checkpoints, with a genuinely fixed budget and discarded probe updates — but its probe is next-token prediction on Vietnamese, not RL, and the checkpoints come from a pretraining stream, not an RL stage.

**Finding 3 — no paper demonstrates that an activation metric predicts future fixed-budget adaptability.** Paper 4 comes closest and reports the honest negative: average parameter magnitude, dormant-unit fraction and attention-head entropy all fail to track the onset of plasticity loss consistently, and the authors state plainly that they "do not yet manage to find a 'smoking gun'" (§I, p. 2). Their one predictive result (Eq. 2, p. 5) predicts *onset as a function of model size*, fit on 8 points with leave-one-out cross-validation — a scaling law, not a within-run early-warning signal. **This is the gap our RQ1 sits in.** It is genuinely open; it is also unsupported by prior positive results, which raises our prior on a null outcome.

**Finding 4 — three of four papers released no code.** Only Plasticine has an official repository (`github.com/RLE-Foundation/Plasticine`, MIT, verified). Papers 2, 3 and 4 contain no repository link anywhere in the PDF or HTML. Paper 2 is worse than that: it cross-references Appendices B, C, D, E, F, G and J–L a dozen times, and **none of those appendices exist in the arXiv v1 PDF or HTML**. Its experimental setup, its algorithm listing and both of its proofs are therefore unverifiable.

**Finding 5 — the closest reusable RLVR pipeline is not in these four papers.** Paper 3's RL stage is built on `verl` + GRPO with published hyperparameters (Appendix B, pp. 20–21), which makes verl the closest *framework* endorsed by a reviewed paper. But verl's own GSM8K GRPO shell scripts start at 0.6B on multi-GPU FSDP2, and the paper's recipe assumes 8×H800. Our own `eaaj-pilot` (TRL 1.6.0 `GRPOTrainer`) already implements the parts nobody's framework gives you for free: frozen splits, checkpoint branching, a frozen probe set, eval-mode activation hooks, effective rank in float64, and a per-checkpoint measurement contract.

**Decision: continue using our current pipeline and only borrow selected components.** Specifics and evidence in §8.

**Finding 6 — two concrete defects in our own measurement contract, found while writing §5.** (a) The primary probe set (512 prompts) is smaller than the hidden dimension (896), so the effective rank is sample-truncated; Plasticine's code guards against exactly this with an assertion. (b) The dormant fraction is identically 0.0 at every layer, checkpoint and threshold in `outputs/local_cuda_grpo_gsm8k_6a075c15808e/measurements/metrics_ckpt0.json`, with a minimum normalised score of 0.13 — the metric has no resolution on a SiLU-gated Qwen2 MLP and must not be reported as evidence that plasticity is preserved. Both are actionable now.

---

## 2. Verified paper identities and sources

All four PDFs were retrieved on 2026-07-26 from `arxiv.org/pdf/<id>` and are stored in `lit review/task3_core_papers/`. Page counts below are from `pypdf`, not from the `file` header (which reports wrong counts for these linearised PDFs).

| # | arXiv | Title (verified) | Version used | Pages | Official code |
| --- | --- | --- | --- | --- | --- |
| 1 | 2504.17490 | Plasticine: Accelerating Research in Plasticity-Motivated Deep Reinforcement Learning | v2, 10 Feb 2026 (v1: 24 Apr 2025) | 21 | Yes — `github.com/RLE-Foundation/Plasticine`, MIT |
| 2 | 2602.06453 | On the Plasticity and Stability for Post-Training Large Language Models | v1, 6 Feb 2026 | 10 | None found |
| 3 | 2606.09932 | When RL Fails after SFT: Rejuvenating Model Plasticity for Robust SFT-to-RL Handoff | v1, 7 Jun 2026 | 27 | None found |
| 4 | 2606.24752 | Can Scale Save Us From Plasticity Loss in Large Language Models? | v1, 23 Jun 2026 | 17 | None found |

**Authors and affiliations (as printed on the paper, p. 1 of each).**

- **Plasticine** — Mingqi Yuan, Qi Wang, Guozheng Ma, Caihao Sun (equal contribution), Bo Li, Xin Jin, Yunbo Wang, Xiaokang Yang, Wenjun Zeng, Dacheng Tao, Jiayu Chen. HK PolyU, SJTU, EIT Ningbo, NTU, HKU, INFIFORCE. Footer reads "©2026 Plasticine Team. License: CC-BY 4.0", and the layout is the JMLR/MLOSS template; **the arXiv listing states no acceptance venue, so publication status is not reported.** Comments field: "21 pages, 7 figures."
- **Paper 2** — Wenwen Qiang, Ziyin Gu, Jiahuan Zhou, Jie Hu, Jingyao Wang (corresponding), Changwen Zheng, Hui Xiong. ISCAS / UCAS, Wangxuan Institute (PKU), Meituan, HKUST(GZ) and HKUST. Header on p. 1 reads "Preprint. February 9, 2026." ICML template. **Not reported: venue, code, appendices.**
- **Paper 3** — Runze Liu*, Jiashun Liu*, Xu Wan, Yuqian Fu, Ling Pan (* equal). HKUST, Zhejiang University, State Key Laboratory of Multimodal AI Systems (CASIA). **Not reported: venue, code.**
- **Paper 4** — J. Fernando Hernandez-Garcia*, Tomás Figliolia*, Beren Millidge. Zyphra, San Francisco. IEEE-style template. **Not reported: venue, code.**

**Version note.** Plasticine v2 (Feb 2026) is the version reviewed. The arXiv ID Tommy supplied resolves to the same paper; the v1→v2 delta could not be diffed from the abstract page alone (**unverified**), but v2's stated scope — "over 13 mitigation methods, 6 evaluation metrics" — matches the appendix contents reviewed here, so the review is internally consistent with v2.

**Repository verification (GitHub API, 2026-07-26).** `RLE-Foundation/Plasticine`: MIT licence, 44 stars, 3 forks, created 2025-04-07, last push 2026-02-09, not archived, 31 files. The URL is printed in the paper's own abstract (p. 1), so the link is author-endorsed, not inferred. For Papers 2–4 a text search of the full extracted PDF text for `github.com`, `anonymous.4open`, `huggingface.co` and `*.github.io` returned **no repository of the authors' own**; Paper 3's only two hits are third-party tools it consumes (`THUDM/slime` and `huggingface/Math-Verify`).

---

## 3. Paper-by-paper reviews

### 3.1 Plasticine (arXiv:2504.17490v2)

#### 3.1.1 Research question and contribution

**Stated problem.** Deep RL agents lose the ability to adapt as training progresses, and "this field lacks unified benchmarks and evaluation protocols" (Abstract, p. 1). The paper's three framing questions are explicit (§1, p. 2): how to reliably measure plasticity loss; how well existing mitigations work; and how plasticity loss varies across domains and scenarios.

**Type of contribution.** Benchmark/framework construction. There is no hypothesis to test and essentially no empirical claim: the main body is four pages, and the experiments (Appendix G, pp. 15–17) show exactly three figures, one environment each, with "more detailed benchmark experiment results" deferred to a Weights & Biases space (footnote 1, p. 15). **Interpretation:** this is a software artefact paper. Treating it as evidence about *which* interventions work would be a misreading; treating it as the reference implementation of the measurement conventions is exactly right.

#### 3.1.2 Training transition

Three progressively non-stationary regimes (§2.3, pp. 3–4):

- **Standard online RL** — ALE, non-stationarity arising only from policy drift and bootstrapping. Frames stacked to (84,84,4) (Appendix E.1, p. 12).
- **Continual Procgen** — a new procedurally generated level is sampled every 2M steps and treated as a distinct task; 10 tasks, level offset 20, (64,64,3) observations, easy mode (Appendix E.2 p. 13, Table 2 p. 16).
- **Continual DMC** — chained tasks, e.g. Dog Stand → Walk → Run → Trot, 1M environment steps each (Appendix E.3, p. 13).

Source and target stages are the *same continuing run*: there is no branch, no checkpoint ladder, no reset of optimizer state at a handoff, and no independent future-training stage. Metrics are logged inline every 10 PPO iterations.

#### 3.1.3 Plasticity definition

**Conceptual (§2.1, p. 3):** "Plasticity loss occurs when neural networks gradually develop optimization pathologies during non-stationary RL training processes, causing them to lose their inherent learnability."

**Operational:** there is none in outcome terms. The paper substitutes a metric panel — "Current methodologies typically involve monitoring multiple plasticity-related indicators from various perspectives and analyzing them holistically" (§2.2, p. 3) — and explicitly concedes that "accurately quantifying neural network plasticity remains an open research question."

**Classification:** representation diversity + avoidance of dead units + optimisation-ease proxies. It measures **proxies, not actual future learning.** For our purposes this is the single most important thing to say about Plasticine in the comparison table: it standardises the *predictor* side and says nothing about the *outcome* side.

#### 3.1.4 Plasticity metrics — full specification

Six metrics, defined in Appendix D (pp. 10–11) and implemented in `plasticine_metrics/`. Because the paper under-specifies the measurement conditions, the code was read directly; the two sources are separated below.

**Ratio of Dormant Units (RDU).** Paper, Eq. (5) p. 10: score `s_i^ℓ = E_x|h_i^ℓ(x)| / ((1/H_ℓ) Σ_k E_x|h_k^ℓ(x)|)`; neuron is τ-dormant if `s_i^ℓ ≤ τ`; RDU is the percentage of dormant neurons. Attributed to Sokar et al. 2023 (ReDo).

From `plasticine_metrics/units.py` (**inference from code**):
- *Activation site*: forward hooks on activation **modules** — `nn.ReLU`, `nn.Tanh`, `nn.Sigmoid`, `CReLU4Linear/Conv2d`, `DFF4Linear/Conv2d`. Post-activation output.
- *Pooling*: `score = layer_activations.abs().mean(dim=0)` for linear layers; `mean(dim=(0,2,3))` for conv (batch + spatial).
- *Normalisation*: `normalized_score = score / (score.mean() + 1e-9)` — per layer.
- *Threshold*: `dormant_mask = normalized_score <= tau`; if `tau == 0.0` it falls back to `torch.isclose(..., 0)`.
- *Aggregation*: `torch.mean(torch.stack(rdu))` — an **unweighted mean over layers**, not a global neuron-count fraction. Layers with few units are weighted equally with wide layers.
- *Probe data*: a single tensor passed by the caller; in `plasticine/ppo_continual_procgen_plasticine.py:356` this is `b_obs`, **the current on-policy rollout batch**, not a frozen probe set.
- *Threshold value*: `tau=0.025`, both for measurement and for ReDo resets (same file, line 356).
- *Mode*: wrapped in `torch.no_grad()`, but `model.eval()` is **never called**. Harmless for CleanRL agents (no dropout/BatchNorm); not harmless for an LLM.
- *dtype*: unspecified; inherits the tensor's dtype (float32 in these agents).

**Fraction of Active Units (FAU).** Eq. (6), p. 10: fraction of units with `a_n(x) > 0`. Code (`units.py`) makes it activation-specific: ReLU family → `1 − mean(features == 0)`; tanh → `1 − mean(|f| > 0.99)`; sigmoid → `1 − mean(f < 0.01 or f > 0.99)`. Requires a 2-D `[batch, features]` matrix with `batch ≥ features`.

**Stable Rank (SR).** Eq. (7), p. 10: `SR(F) = min{k : Σ_{i≤k} σ_i / Σ_j σ_j > 0.99}`. **Critical naming warning:** this is *not* the classical stable rank `‖F‖_F² / ‖F‖_2²`. It is the `srank_δ` of Kumar et al. 2022 with δ = 0.01. Code confirms: `sum(cumsum(σ)/sum(σ) < 0.99) + 1`. If we ever report a "stable rank" we must say which one.

**Effective Rank (ER).** Eq. (8), p. 10: `ER(F) = exp(H(p_1..p_q))`, `H = −Σ p_k log p_k`, `p_k = σ_k / ‖σ‖_1`, `q = min(n,m)`. Attributed to Roy & Vetterli 2007. Code (`rank.py`):
- Input must be 2-D `[batch_size, num_features]`, with `assert features.size(0) >= features.size(1)`.
- `torch.linalg.svdvals(features)` — **no mean-centering**, no scaling.
- `probs = σ / Σ|σ|`, then `probs = probs[probs > 0]`, then `exp(−Σ p log p)`.
- Higher ER = more plasticity (more dimensions carrying signal). No dtype control; float32 in practice.

**Weight Difference (D.5, p. 11).** L2 difference between two model states; used against a stored copy (`compute_l2_norm_difference(agent, agent_copy)`), with a bare `try/except` because plasticity injection changes the architecture (`ppo_continual_procgen_plasticine.py:361-364`).

**Gradient Norm (Eq. 9, p. 11).** Global L2 over all parameter gradients.

**Sensitivity analysis: none.** The paper reports no ablation over τ, layer choice, probe size, pooling or dtype. **This is a real gap** — it is presented as a measurement standard but never characterises the stability of its own measurements.

#### 3.1.5 Future-learning evaluation

**None.** No training is ever launched from an earlier checkpoint. All metrics are logged online in the same run that produced them. There is no fixed future budget, no per-checkpoint baseline, no optimizer-state protocol, and no forgetting-vs-inability distinction. Distance from our design: maximal on the outcome side, minimal on the predictor side.

#### 3.1.6 Prediction

Category 1 only: **descriptive association, logged concurrently.** No predictor/target pair, no temporal ordering, no statistics, no causal claim. The metrics are plotted next to return curves; nothing is fit.

#### 3.1.7 Interventions

Thirteen-plus methods across five families (§2.1 p. 3, Appendix C pp. 7–9):

| Family | Methods | What changes | When applied |
| --- | --- | --- | --- |
| Reset | Shrink-and-Perturb (Eq. 1), Plasticity Injection, ReDo, Resetting Layer | Weights re-initialised in whole or part | Periodic intervals (ReDo/SnP also available per-step as "Soft SnP") |
| Normalisation | LayerNorm; Normalize-and-Project (NaP) | Norm before every nonlinearity; periodic rescale of layer weights to initial norms | Continuous + periodic projection |
| Regularisation | L2, Regenerative Regularisation, Parseval (Eq. 2) | Extra loss terms on ‖θ‖, ‖θ−θ_init‖, ‖WWᵀ−sI‖_F | Every update |
| Activation | CReLU (Eq. 3), Deep Fourier Features (Eq. 4) | Concatenated ReLU(±x) or sin/cos | Architecture-level |
| Optimizer | Fast TRAC, KRON | Meta-optimizer / Kronecker-factored curvature | Every update |

ReDo is the one that matters for us: it resets neurons whose normalised score `≤ τ`, at fixed intervals, using a mini-batch of training data — in the reference script, ten times per task (`iteration % (num_iterations // 10) == 0`) at `tau=0.025`.

**Compute overhead, improvement magnitudes and per-method ablations are not reported in the paper** (the three figures show unlabelled comparative curves for one environment each). **Could ReDo be ported to RLVR?** Yes mechanically — but see §5.2: on Qwen2.5-0.5B nothing is dormant at τ = 0.025, so ReDo would be a no-op. Reporting that as a finding is legitimate; reporting it as "plasticity preserved" is not.

#### 3.1.8 Released setup

| Artefact | Status | Path |
| --- | --- | --- |
| Code repository | Released, MIT | `github.com/RLE-Foundation/Plasticine` |
| Metric implementations | Released | `plasticine_metrics/{rank,units,norm,metrics}.py` |
| Training scripts | Released | `plasticine/{ppo_continual_procgen,ppo_continual_dmc,c51_atari}_{base,plasticine}.py`, `scripts/` |
| Environments | Released | `plasticine_envs/{procgen,dmc}_wrappers.py` |
| Dependencies | Released, per-env | `requirements/requirements-{ale,dmc,procgen}.txt` |
| Hyperparameters | In paper | Tables 1–3, pp. 15–17 |
| Full results | **Off-paper** | W&B space only (footnote 1, p. 15) |
| Model checkpoints | Not released | — |
| Analysis notebooks | Not released | — |

**Reproducibility assessment: complete and runnable for the training loops, but the paper's own empirical claims are effectively un-auditable** because the results live in an external dashboard rather than in the paper. Repo is small (31 files) and readable — a strength for us, since we only want the metric functions.

---

### 3.2 On the Plasticity and Stability for Post-Training LLMs (arXiv:2602.06453v1)

#### 3.2.1 Research question and contribution

**Stated problem.** GRPO training is unstable, manifesting as a trade-off between "reasoning plasticity and general capability retention" (Abstract, p. 1). The claimed root cause is a *geometric* conflict: the gradient of the reward term and the gradient of the KL term point in opposing directions and destructively interfere.

**Hypothesis.** That resolving this conflict probabilistically — rather than by hard projection à la PCGrad — improves both stability and reasoning performance.

**Contributions (§1, p. 2):** the conflict diagnosis; PCR, a Bayesian soft-projection rule; an MLP-only hybrid implementation; and an MMSE-optimality theorem.

**Type:** causal intervention on the optimizer, with a small diagnostic component. **It is not a measurement paper and not a benchmark.**

#### 3.2.2 Training transition

A single continuous RLVR run. Source stage = an already-distilled reasoning model; target = the same run, later. **There is no second stage, no checkpoint ladder, no branch, no future-task transfer.** The only "transition" studied is the within-run evolution of gradient geometry across training steps (Fig. 1(c), p. 4 — layer index × training step heatmap of cos(g_pla, g_sta), 0–500 steps).

Training data changes only in the sense that the base models were previously fine-tuned elsewhere: DeepScaleR-1.5B-Preview "was previously fine-tuned on 40k math QA pairs" and is "further trained on 919 AIME problems spanning 1989–2023"; DeepSeek-R1-Distill-Qwen-1.5B is trained on "a random subset of 4,000 QA pairs from NuminaMath" (§6.1, p. 7).

#### 3.2.3 Plasticity definition

**Conceptual:** plasticity = capacity to acquire new reasoning skill; stability = retention of general knowledge and linguistic coherence (§1, p. 1).

**Operational (Eq. 5, p. 3):** `L_pla(θ) = −E[(1/n)Σ_i (1/T_i)Σ_j S_{i,j}(θ)]`, the negative token-level clipped surrogate gain. `L_sta(θ) = E[(1/n)Σ_i (1/T_i)Σ_j K_{i,j}(θ)]`, the token-level KL penalty (Eq. 6). The corresponding gradients are `g_pla` (Eq. 8) and `g_sta` (Eq. 9).

**Classification.** This is *not* a plasticity metric in the sense the rest of the literature uses the word. It is a decomposition of the GRPO loss into its two existing terms and a relabelling of them. Plasticity here is neither a capacity, nor a representation property, nor an outcome — it is the reward-seeking gradient direction. **Measured proxy for downstream performance: yes, and only indirectly; it can never be zero-or-degraded in the sense Dohare et al. mean.**

**Technical concern (interpretation).** §3.1, p. 2 states "π_ref is typically set to the old policy π_θold." In GRPO as defined by Shao et al. 2024, π_ref is the *frozen initial/SFT policy*; setting it to π_θold turns the KL term into a trust region on the current update rather than an anchor to pretrained knowledge. If the implementation actually does this, the "stability gradient" cannot be doing the general-capability preservation the paper attributes to it. This cannot be checked — there is no code and no appendix.

#### 3.2.4 Plasticity metrics

The full inventory:

| Metric | Definition | Where | Direction |
| --- | --- | --- | --- |
| Layer-wise cosine similarity cos(g_pla, g_sta) | Standard cosine between the two gradient vectors, per Transformer layer, per step | §3.3 p. 3; Fig. 1(c) p. 4 | Negative = conflict = bad |
| Gradient norm | Global gradient norm over training, "as a proxy for optimization smoothness" | §6.4 p. 8; Fig. 3 p. 7 | Large swings = unstable |
| Projection strength α distribution | Density of the derived α per module family | Fig. 5(a) p. 8 | Higher in MLP = more conflict |
| cos(g_final, g_sta) | Post-arbitration alignment | Fig. 5(b) p. 8 | Non-negative = conflict resolved |
| Pass@1 / MMLU / WikiText-2 PPL | Task outcomes, swept against KL coefficient β | Fig. 1(a)(b) p. 4; Table 1 p. 7 | — |

**No effective rank. No dormant neurons. No rank, sparsity, dead-unit, Hessian, NTK, parameter-displacement, representation-drift or entropy measurement.** Nothing here transfers to our metric panel.

**Unspecified for every metric:** activation site (n/a), layer selection for the heatmap beyond "0–32", probe dataset, probe size, pooling, evaluation mode, dtype. "Detailed setup in Appendix B" (§3.3, p. 3) — **Appendix B does not exist in the released paper.** No sensitivity analysis of any kind.

#### 3.2.5 Future-learning evaluation

**None.** No training starts from an earlier checkpoint; no fixed future budget; no baseline-controlled improvement; no optimizer-state protocol; no seeds; no uncertainty; no forgetting-vs-inability separation. Table 1 reports post-training deltas relative to each *base model's* pre-training score, which is a one-step improvement, not a checkpoint ladder. Distance from our design: **very far.**

#### 3.2.6 Prediction

**None.** Category 1/2 at most (concurrent association between gradient conflict and instability). No predictor variable is measured before an outcome it is claimed to predict. Causality is argued theoretically (Theorem 5.1) — but the proof is in "Appendix G," which does not exist.

#### 3.2.7 Interventions

**PCR (Probabilistic Conflict Resolution).**

- *What changes:* the aggregated gradient before the optimizer step. Decompose `μ_pla = μ_pla^⊥ + μ_pla^∥` w.r.t. `μ_sta` (Eqs. 11–12, p. 5). Retain a fraction `k = λ_pla/(λ_pla + λ_sta)` of the conflicting component (Prop. 4.1, Eq. 13, p. 5). Final update `g_final = μ_pla − α·(μ_pla·μ_sta/‖μ_sta‖²)·μ_sta` with `α = 1 − k = λ_sta/(λ_pla+λ_sta)` (Eq. 15, p. 5).
- *When:* every optimizer step.
- *Information required:* both gradient components separately, plus a variance estimate per component. Covariance is approximated as isotropic `Σ ≈ σ²I`, "where the scalar variance is estimated by the trace of intra-group gradients" (§4.1, p. 4). **The exact estimator is not given** and Appendix C/E, cited for the explanation, do not exist.
- *Compute overhead:* stated qualitatively only — PCR is applied to MLP layers only because "applying it element-wise to every single parameter in a billion-parameter LLM is computationally too expensive" (§4.5, p. 6). Fig. 4(b) plots Pass@1 against a relative "Training Time (N×)" axis with no absolute numbers.
- *Improvement:* Table 1 (p. 7) shows average gains of +3.2 / +3.6 / +3.5 points over three base models, roughly +1 point over the strongest baseline (GCPO). HumanEval: "almost 1.2%" (§6.2, p. 7).
- *Ablations:* fixed α ∈ {0.2, 0.5, 0.8} vs auto (Fig. 4a); layer subset — MLP-only vs all layers vs "Transformer Layers" (Fig. 4b, the three-way distinction is never defined); learning-rate scale (Fig. 4c).
- *Could it be incorporated into an RLVR pipeline?* In principle yes — it is a gradient-surgery hook that needs the two loss terms differentiated separately. In our pipeline `beta = 0.0`, so **`g_sta` is identically zero and PCR is a no-op.** It becomes relevant only if we run the KL β>0 baseline that is already logged as an open team question.

#### 3.2.8 Released setup

**Nothing.** No repository, no configs, no scripts, no checkpoints, no data splits, no logging code, no analysis. And the paper is missing its own appendices, so even the paper-internal record is incomplete. There is no group size `n`, no number of RL steps (Fig. 3's x-axis runs to 1k steps but the training length is never stated), no seed count, no error bars, and no rollout configuration. Reported: learning rate 1e-6, weight decay 0.01, batch size 256, token budget 16,384, A100 clusters (§6.1, p. 7).

**Internal consistency issues found while reading** (all **explicitly stated facts** about the text, offered as verification flags):
- §3.3, p. 3 says "The heatmap in Figure 1(b) reveals…"; the heatmap is Figure 1(c) — Figure 1(b) is the Pareto frontier.
- Eq. 16 (p. 6) says the MLP branch is computed "via Eq. 17"; the paper has no Eq. 17.
- §4.1, p. 4: "The CLT states that the distribution of the sample mean approaches a multivariate Gaussian as the sample size implies" — a corrupted sentence.

**Reproducibility assessment: no official release; key training code, appendices and proofs missing. This paper cannot be used as an RLVR pipeline.** Its ideas are usable; its artefacts do not exist.

---

### 3.3 When RL Fails after SFT (arXiv:2606.09932v1)

#### 3.3.1 Research question and contribution

**Stated problem.** "Checkpoints with excessive SFT often show limited improvement during RL" (Abstract, p. 1). Which SFT checkpoint should initialise RL, and can a bad one be repaired?

**Hypothesis.** That over-trained SFT models are not merely overfit but *rigid*: their parameters have moved far from base in a small number of coordinates, their token distributions have collapsed, and consequently RL cannot reshape them.

**Contributions (§1, p. 3):** (1) naming the SFT-to-RL handoff dilemma; (2) a multi-perspective diagnosis; (3) `Rejuvenation`, a post-hoc repair; (4) validation on math and agentic tasks.

**Type:** diagnostic measurement **plus** causal intervention. It is the only one of the four that does both on an LLM RL pipeline.

#### 3.3.2 Training transition

**base → SFT ladder → GRPO RL.** Precisely:

- Source stage: SFT on a 100k subset of a 500k math corpus, EvoLM-4B base (chosen because it is "pre-trained on a controlled corpus without evaluation data contamination", §4.1, p. 9). LlamaFactory, AdamW β=(0.9,0.999), constant LR 3e-6, **no warmup, no decay, no weight decay** — deliberately, "so that excessive SFT can manifest its full effect" (Appendix B, p. 20).
- Checkpoints sampled during the source stage: **yes** — epochs 0.5, 1, 2, 4, 8, 16, 32 (Figs. 2, 4, 5 x-axes). ModSFT = epoch 2, OverSFT = epoch 32. "All variants share the same data, the same context window, and the same data ordering, so any difference between ModSFT and OverSFT is purely a function of training duration" (Appendix B, p. 20). **This is exactly the control our Stage-A ladder needs.**
- Target stage: **GRPO RL launched independently from each checkpoint.** verl backend; actor initialised from the SFT checkpoint; reference policy = "the same SFT model frozen at step 0" — i.e. the KL anchor is *per-checkpoint*, not a global anchor.
- Optimizer state: **not reported.** Each RL run is a fresh verl launch, so a fresh AdamW state is the natural inference, but the paper never says so. Mark as **unclear from the available sources.**
- Data/reward/task change at the transition: yes — SFT is next-token imitation on math solutions; RL is binary-ish rule-based reward (+1 correct, −1 otherwise) via Math-Verify on the boxed answer.
- Agentic variant: Qwen3-8B, τ-bench Retail, slime + Megatron-LM + SGLang, Qwen3-4B-Instruct-2507 as user simulator.

#### 3.3.3 Plasticity definition

**Conceptual and operational are the same, and it is stated in a footnote (p. 2):** "We use model plasticity to refer to the ability of SFT models to undergo reward-driven improvement in subsequent RL. A plastic model should remain responsive to RL update and such updates can effectively translate into task performance gains."

**Classification:** ability to improve future reward/accuracy; task-specific adaptability; and — via the phrase "responsive to RL update … effectively translate" — explicitly *not* just gradient magnitude. The paper operationalises the outcome as **RL gain in percentage points** (Fig. 5 right-hand axes) and the predictors as parameter drift, entropy and gradient/update dynamics.

This is conceptually the closest match to our RQ1 of any of the four.

#### 3.3.4 Plasticity metrics

**Pre-RL (measured on the SFT checkpoint, before any RL update):**

| Metric | Computation | Site | Where | Direction |
| --- | --- | --- | --- | --- |
| Element-wise parameter delta vs base | `θ_SFT − θ_Base`, visualised as a heat map per weight matrix | `layers.0.self_attn.v_proj`, `.k_proj`, `layers.27.*` | Fig. 3 p. 5; Figs. 11–13 pp. 26–27 | Large sparse spikes = low plasticity |
| Mean-squared weight difference; mean-squared weight | Scalar aggregates over the same deltas | Whole model | Fig. 2 p. 4 | Higher = worse |
| Training loss | SFT cross-entropy | — | Table 1 p. 5 | Near-zero = saturated |
| Token entropy | Mean per-token predictive entropy (teacher-forced) | Output distribution | Table 1 p. 5 | Lower = less plastic |
| Pass@K − Pass@1 gap | Diversity headroom | MATH-500 | Table 1 p. 5 | Smaller = collapsed |
| Per-position entropy gap vs base | `|H_OverSFT(t) − H_Base(t)|`, Top-N positions selected | Response positions | §3.4.1 p. 7 | Used for targeting, not scoring |

Table 1 values (p. 5): UnderSFT loss 0.178 / entropy 0.281 / Pass@1 9.7 / gap 13.7; ModSFT 0.111 / 0.184 / 15.2 / 15.3; OverSFT 0.002 / **0.024** / 15.8 / 11.2. Note the tension the paper itself highlights: OverSFT has the *best* greedy Pass@1 and the *worst* diversity — "high SFT scores are not necessarily reliable predictors of post-RL performance" (§2, p. 3).

**During-RL:**

| Metric | Computation | Where | Finding |
| --- | --- | --- | --- |
| Mean gradient norm | Averaged over RL training | Fig. 4 p. 6 | OverSFT **higher** (≈0.34 vs ≈0.23) |
| Weight update magnitude | Averaged RL-induced parameter movement | Fig. 4 p. 6 | OverSFT **lower** (≈6.2e-9 vs ≈7.0e-9) |

**The paradox is the paper's central diagnostic result** (§3.2.1, p. 6): big gradients, small movement, small gain. **Interpretation:** this is a strong argument that gradient norm alone — a dashboard signal our proposal treats as a baseline — is not merely uninformative but *anti-informative* here. Worth citing directly in our Research Doc.

**Not measured anywhere in this paper:** effective rank, stable rank, participation ratio, singular-value spectrum, activation variance/sparsity, dormant units, FAU, Hessian/sharpness (the word "sharp" is used descriptively of the parameter landscape, never measured), NTK, representation similarity/drift, confidence calibration. The neuron-level machinery is *attribution*, not *activity* — see §3.3.7.

**Probe/measurement conventions:** entropy is measured teacher-forced on "a calibration set of prompt–response pairs" (§3.4.1, p. 7) whose size, source and dtype are **not reported**. Layer selection for the parameter visualisations is illustrative (layers 0 and 27). No pooling, dtype, evaluation-mode or threshold conventions are given. **No sensitivity analysis on any measurement**; the only sweeps are over intervention hyperparameters (α, ρ).

#### 3.3.5 Future-learning evaluation

This is the section that matters most for us. Scoring it against the prompt's checklist:

- **New training from multiple earlier checkpoints?** **Yes.** Seven SFT epochs → seven independent GRPO runs (Fig. 5, p. 6).
- **Future task: same, new, or new domain?** New objective on largely the same domain (math SFT → math RL), plus a genuinely out-of-distribution *evaluation* suite (GPQA-Diamond, ARC-Challenge, MMLU-Pro). The OOD sets are evaluation-only, not a second training stage.
- **Fixed future budget?** **Not reported for the math setting.** Appendix B (pp. 20–21) gives batch sizes, LR, clipping, KL coefficient and "Validation is run every 50 steps," but never the total number of RL steps. For agentic RL: "up to 200 rollouts, save and evaluate every 20." **The math result — the headline result — has no stated budget.** For a paper about fixed-budget adaptability this is a material omission.
- **Baseline-controlled improvement?** **Partially, and this is where it gets dangerous.** Table 2 (p. 10) reports both pre-RL and post-RL scores, so the delta is computable per checkpoint (ModSFT 9.1 → 17.0 = +7.9; OverSFT 10.0 → 14.9 = +4.9; Rejuvenation 5.4 → 17.5 = **+12.1**). But the Rejuvenation row's pre-RL average is 5.4 — *far below* OverSFT's 10.0. The intervention destroys a large amount of pre-RL capability and then recovers more during RL. **The headline "+12.1" is inflated by a depressed baseline.** The authors do not flag this. **Direct lesson for us: "improvement from each checkpoint's own starting accuracy" is manipulable by anything that lowers the start. We must report the endpoint level alongside the delta, always.**
- **Optimizer state restored or reinitialised?** **Unclear from the available sources.**
- **Shared future data, order, seeds, hyperparameters across checkpoints?** **Yes, stated explicitly:** "All RL hyper-parameters, data, prompts, and seeds are kept identical across initializations to ensure a fair comparison" (Appendix B, p. 21).
- **Outcome quantity?** Endpoint score and endpoint delta (percentage points). No AUC, no convergence speed, no sample efficiency.
- **Multiple future-training seeds?** **No.** One seed, shared. This is one Stage-A trajectory family with one Stage-B run each — *not* independent full-pipeline replicates.
- **Uncertainty reported?** **No.** No error bars, no confidence intervals, no variance, anywhere in Tables 2–7 or Figs. 4–5.
- **Forgetting vs inability distinguished?** Partially and indirectly: the ID/OOD split shows OverSFT+RL trails on both, and the pre-RL rows show OverSFT has *not* forgotten ID skill. But no controlled test isolates the two.

**Closeness to our GSM8K→SVAMP design.** Structurally the nearest analogue that exists: a ladder of source-stage checkpoints, an identical downstream RL recipe branched from each, and a per-checkpoint improvement readout. Differences that matter: their source stage is SFT (ours is RLVR); their ladder is spaced by epochs over a fixed corpus (ours by GRPO update count); their downstream task is nearly in-distribution (ours crosses GSM8K→SVAMP); their budget is unstated (ours is pre-registered at 50 updates); they run one seed (we should run ≥3 Stage-B seeds); and they never test whether any pre-RL measurement *predicts* the post-RL gain.

**Data-integrity flag (explicitly stated fact).** The same configuration is reported twice with different numbers: Table 2 and Table 4 give Rejuvenation GSM8K = 49.1, ID Avg = 19.7; Table 7 gives "OverSFT+Rejuvenation" GSM8K = 48.6, ID Avg = 19.6. Other columns are identical. Minor, but it means at least one table was not regenerated from the same run.

#### 3.3.6 Prediction

**Category 3 (retrospective explanation), edging toward category 5 in structure but not in analysis.**

- *Predictor variables:* pre-RL token entropy, parameter-drift magnitude, Pass@K−Pass@1 gap.
- *Target variable:* post-RL score / RL gain in points.
- *Temporal ordering:* correct — the predictors are measured before RL starts.
- *Number of checkpoints:* 7 (math), 2 named (agentic).
- *Statistical method:* **none.** No correlation, no regression, no test.
- *Train/test split:* none.
- *Reported correlation or predictive accuracy:* **none reported.**
- *Generalisation across seeds/tasks/models/hardware:* the fusion+reset *intervention* transfers from EvoLM-4B/math to Qwen3-8B/agentic (Tables 2 and 3), which is real evidence the *mechanism* generalises. The *measurements* are never tested as predictors, so their generalisation is untested.
- *Causality:* the interventions (Tables 4–6) give genuine causal evidence that reducing parameter drift and relaxing over-confident neurons improves subsequent RL. That is causal evidence about a *fix*, not about a *predictor*.

**We must not cite this paper as evidence that pre-RL metrics predict post-RL adaptability.** It shows the correlation exists visually across seven checkpoints; it never quantifies it.

#### 3.3.7 Interventions

**Rejuvenation = base-anchored fusion + attribution-guided neuron reset** (Algorithm 1, p. 20).

**(a) Base-anchored linear fusion** (§3.3, Eq. 1, p. 6). `θ_fuse = α·θ_OverSFT + (1−α)·θ_Base`, applied to *all* parameters including attention, MLP, RMSNorm weights, final norm and `lm_head`. Rationale is the task-vector view (Ilharco et al. 2023): preserve direction, shrink magnitude. Requires: the base model. Cost: one element-wise interpolation. Sweep (Table 5, p. 11): α ∈ {0.40, 0.45, 0.50, 0.55, 0.60}; α = 0.5 is the default and best on average (Avg 17.0); small α favours OOD, large α favours ID.

**(b) Attribution-guided neuron reset** (§3.4, pp. 6–8).
1. Select target tokens: run base and OverSFT teacher-forced, take the Top-N positions by per-position entropy gap `|H_Over(t) − H_Base(t)|`.
2. Decompose the target logit into residual-stream contributions under a **frozen final-RMSNorm approximation** (Eq. 2, p. 7); the authors track the reconstruction error `|z̃ − z|` to confirm tightness, and explicitly warn the scores are "attribution signals for ranking rather than as a complete causal explanation."
3. Exact per-neuron contributions for the residual writers `o_proj` and `down_proj` (Eq. 3, p. 8); a gradient×activation proxy for `q/k/v/up/gate_proj` (pre-RoPE for q/k, so "local attribution proxies rather than exact causal decompositions").
4. Reset set `S` = union over selected tokens and calibration examples (Eq. 4); overwrite with base values, `θ_reset = ω·θ_Base + (1−ω)·θ_fuse` (Eq. 5) — rows for q/k/v/up/gate, columns for o/down.
- Sweep (Table 6, p. 12): ρ ∈ {0.5%, 1%, 2%, 4%}; ρ = 1% default. 4% starts erasing useful SFT behaviour.
- Where the reset lands: concentrated in `k_proj` and `v_proj` of the last layers (Fig. 10, p. 25), and stable across calibration prompts.

**Compute overhead (Appendix C, p. 24):** one-shot, post-hoc; **3 minutes for EvoLM-4B and under 5 minutes for Qwen3-8B on a single H800** — negligible against SFT or RL.

**Ablation (Table 4, p. 11):** OverSFT+RL 14.9 → +Fusion 17.0 → +Fusion+Reset 17.5. Fusion alone recovers to the ModSFT+RL level (17.0); the reset adds the OOD margin.

**Stated limitation (Conclusion, p. 12):** both operations require access to the base model.

**Could it be incorporated into an RLVR pipeline?** Yes, and cheaply — fusion is a `state_dict` interpolation and needs no framework support at all. **For us it is an obvious Experiment-2 candidate but not part of RQ1:** our Stage A is RLVR at lr 1e-6 with β=0, which produces nothing like the epoch-32 SFT drift that motivates fusion. Applying it would be testing a different hypothesis.

#### 3.3.8 Released setup

| Artefact | Status |
| --- | --- |
| Code repository | **None found** (no link in the paper; web search returns no author-endorsed repo) |
| Training scripts | Not released; recipes described in Appendix B (pp. 20–21) |
| Checkpoints (SFT ladder, rejuvenated) | Not released |
| Calibration set for DLA | Not released, not described (size/source not reported) |
| Data splits | Not released ("a 100k subset sampled from a 500k mathematical SFT corpus" — corpus not named) |
| Metric implementations | Not released |
| Third-party dependencies (documented) | LlamaFactory (SFT), slime, verl (RL), vLLM, SGLang, Megatron-LM, Math-Verify |

**Reproducibility assessment: analysis and recipe described, no code released.** A competent team could re-implement the RL stage from Appendix B on verl in a few days — the hyperparameters really are complete for RL — but the SFT corpus, the DLA calibration set, the math RL step budget and the reset implementation would all have to be guessed.

---

### 3.4 Can Scale Save Us From Plasticity Loss in LLMs? (arXiv:2606.24752v1)

#### 3.4.1 Research question and contribution

**Stated problem.** Plasticity loss is well documented in small networks and vision, "rarely in natural-language domains" (Abstract, p. 1). Does it occur in GPT-style Transformers on realistic text, and does scale fix it?

**Hypotheses.** (1) Plasticity loss occurs in GPT-style LMs on natural language. (2) Its onset scales predictably with model size. (3) It is not exclusive to abrupt task changes.

**Type:** diagnostic measurement + scaling behaviour. No intervention is proposed or tested (mitigations are discussed as future work, §V-B end).

**Notable for us:** the first author, J. Fernando Hernandez-Garcia, is a co-author of Dohare et al. (2024, *Nature*), the canonical loss-of-plasticity paper. The methodological lineage is direct.

#### 3.4.2 Training transition

**Continual pretraining → held-out probing, repeated.** Precisely (§IV-A, pp. 3–4):

- Task sequence: English → Chinese → French → Japanese → Spanish → German → Portuguese → Russian, ordered "to reduce similarity between consecutive languages." Data: CulturaX (167 languages, 6.3T tokens); 100B train / 1B eval per language; each task instance draws a **fresh** 5B tokens.
- A **cycle** = 8 task instances = 40B tokens. Models trained for up to 48 cycles (Fig. 2).
- **Probe:** held-out Vietnamese — chosen as "the only austroasiatic language in the training set … to reduce similarity between the probing distribution and the continual pretraining data." 20B train / 1B eval. Each probe trains **5B tokens on a copy of the checkpoint, and the parameter updates are discarded** before pretraining resumes.
- Checkpoints sampled during the source stage: **yes**, at every cycle boundary (later, at selected intra-cycle boundaries).
- Future learning starts independently from each checkpoint: **yes**, by construction (copy + discard).
- **Optimizer state: reset at the start of each task** (§IV-B, p. 4), explicitly so "the plasticity degradations we observed were due to inherent plasticity loss in the weights rather than simply stale optimizer states." LR warm-up is also restarted per task. **This is the cleanest handling of optimizer state in the four papers and the protocol we should copy.**
- Second setting: **Multilingual Stationary Learning Problem** — the same 8 languages mixed stationarily, probed every 5B tokens (§V-B, p. 6). Plasticity loss appears there too.

#### 3.4.3 Plasticity definition

**Conceptual (§III, p. 3):** the ability to learn from new data; distinguished sharply from stability/forgetting — "plasticity loss is about the network becoming incapable of learning new information, rather than forgetting old information" (§I, p. 1).

**Operational (§I, p. 1):** "we operationalize plasticity loss as a degradation in a model's ability to improve on a target distribution under a fixed training budget."

The authors also justify the probe design against the alternative: online measurement as each new task arrives "requires tasks to be equally difficult to accurately assess plasticity loss," so they use a held-out probe instead (§III, p. 3). **That argument applies verbatim to our SVAMP probe and is worth citing in our proposal.**

**Classification:** intrinsic learning capacity, measured as fixed-budget improvement on a held-out target. **This is our definition.**

#### 3.4.4 Plasticity metrics

**Primary outcome — probing AUC.** During each probe, validation loss is measured on **1,280 randomly sampled sequences of length 2,048, every 95 training steps**, over the 5B-token probe. The plasticity score is the **area under the validation-loss curve** vs probe step; lower AUC = faster adaptation. Cross-checkpoint comparability is handled by normalising to the model's own first cycle (Eq. 1, p. 4): `100 × (AUC_k/AUC_1 − 1)`. In the extended-probing analysis they instead divide by the AUC of training on Vietnamese alone (Appendix B, p. 13), which is a *from-scratch* reference rather than a *first-cycle* reference.

**AUC is a strictly better outcome statistic than an endpoint delta** — it integrates the whole learning curve, so it captures both speed and level, and it is far less noisy than a single final evaluation. Our `eval_every: 10` over a 50-update Stage B already gives us 5–6 points; that is enough to compute an AUC.

**Correlates (all measured, all reported as insufficient):**

| Metric | Exact computation | Probe data | Threshold | Where |
| --- | --- | --- | --- | --- |
| Average parameter magnitude | Mean absolute value over **all non-embedding parameters, weights and biases pooled globally** — explicitly "not computed separately per layer before averaging" | n/a | n/a | §V-B p. 6; Fig. 5 top row |
| Dormant-unit fraction | `h̄_i^l = (1/(|D|·m)) Σ_x Σ_j h_i^l(x,j)` (Eq. 3); `s_i^l = h̄_i^l / ((1/N_l) Σ_k h̄_k^l)` (Eq. 4); unit is ε-dormant if `s ≤ ε` | **256 sequences × 2,048 tokens from the Vietnamese validation set** — held out, never seen in pretraining | **ε = 0.01** | §V-B p. 7; Figs. 5 (bottom), 6 |
| Attention-head entropy | `−Σ p_i log p_i` over normalised attention weights, averaged over sequences and query positions | **512 sequences × 2,048** from the **Russian** validation set; query positions **256–2,048** | `H_max ≈ 6.9215` nats (Eq. 5); **lazy** if `> 0.9·H_max`, **collapsed** if `< 0.1·H_max` | §V-B pp. 7–8; Fig. 7 |

**Site and pooling for dormancy:** MLP layers only; averaged over *all* token positions and all sequences. **Direction:** higher dormant fraction = less plasticity (nominally).

**Honest caveat the authors add (§V-B, p. 7), which we should reproduce verbatim in our own reporting:** "In networks with smooth activations such as GeLU, dormancy should not be interpreted as exact inactivity, but rather as a diagnostic indicating that some units contribute comparatively little to the layer's representation."

**Sensitivity analysis:** none on ε, layer choice, probe size or dtype. But there *is* a layer-resolved breakdown (Fig. 6), which is effectively a sensitivity result: dormancy is wildly non-uniform — >95% of units in layer 8 of the 53M model, ~80% in layer 10 of the 106M model. **Network-averaged dormancy would have hidden this entirely.**

**No effective rank, stable rank, participation ratio or spectral measurement anywhere.** That is a genuine gap in the LLM plasticity literature — and it is the gap our Q metric occupies.

#### 3.4.5 Future-learning evaluation

- **New training from multiple earlier checkpoints?** **Yes** — at every cycle boundary, up to 48 cycles.
- **Future task:** new domain (a held-out language), deliberately chosen for low transfer.
- **Fixed budget?** **Yes** — exactly 5B tokens per probe, identical across checkpoints and model sizes.
- **Initial differences controlled?** **Yes**, two ways: normalising each model to its own first-cycle AUC (Eq. 1), and in the extended analysis dividing by the from-scratch Vietnamese AUC.
- **Optimizer state?** **Reset per task, deliberately and documented.**
- **Shared data/order/seeds/hyperparameters across checkpoints?** Probe data is subsampled from the same 20B Vietnamese pool; hyperparameters match the pretraining config including warm-up. Whether the *same* 5B subsample is used at every probe is **not reported** ("5 billion training tokens randomly subsampled") — a possible noise source.
- **Outcome:** AUC of the validation-loss curve. Not endpoint, not convergence step, not sample efficiency.
- **Multiple seeds?** **No.** One run per model size; no uncertainty is reported on any probe curve. They compensate with 8 model sizes and heavy smoothing (moving average, window 3), which is a different kind of replication.
- **Forgetting vs inability distinguished?** **Yes, structurally** — the probe measures learning of *new* data on a *discarded copy*, so forgetting of the pretraining languages cannot contaminate the measurement. This is the cleanest separation in the four papers.

**Closeness to our design.** On protocol, very close: fixed-budget adaptation on a held-out target, launched from a ladder of source-stage checkpoints, updates discarded, per-checkpoint normalisation. On substance, distant: pretraining rather than RLVR, loss rather than reward, 5B tokens rather than 50 GRPO updates, and no activation-metric prediction.

#### 3.4.6 Prediction

Two distinct things must be separated here.

**(a) The scaling law — genuine prediction, wrong axis for us.** `T = 1.3×10⁻⁵ · P^0.8269` (Eq. 2, p. 5), where `T` = task-instance number at onset, `P` = non-embedding parameters.
- Predictor: model size. Target: onset of plasticity loss. Temporal ordering: n/a (both are properties of a completed run).
- Samples: **8** (Table IV, p. 14): 5M→6, 12M→6, 27M→16, 39M→18, 53M→54, 83M→54, 106M→62, 314M→118.
- Method: log-log OLS, compared against linear, linear-log and exponential forms with **leave-one-out cross-validation** (Table V, p. 16). Log-log: R² 0.8969, in-sample RMSE 11.524, **out-of-sample RMSE 17.864**; linear: R² 0.8914 but out-of-sample RMSE **31.113**. The LOO comparison is what makes the model-selection claim defensible.
- Generalisation: extrapolated to Springer et al. (2025) — predicts onset at ~1.8T tokens for a 1B model and ~9T for 7B, consistent with their reported observations (§V-A, pp. 5–6). This is external corroboration, though of a coarse kind.
- Honesty: the caption of Fig. 3 (p. 6) says "a qualitatively decent but not perfect fit with several arguable outliers at 12M and 53M." With 8 points, two acknowledged outliers, standard error on the exponent of 0.1027 (i.e. 0.827 ± 0.10), and a definition of "onset" that depends on a 3-point moving average of a noisy curve, **this law should be read as a trend, not a calibrated predictor.**

**(b) The internal correlates — explicitly not predictive.** The paper is unusually rigorous about this and the failures are worth listing, because they are precisely the failure modes our RQ1 will face:
- The 106M model "starts showing deterioration in performance on the eighth cycle … [but] its average parameter magnitude decreases steadily between the eighth and twentieth cycles."
- For the 53M model, "probing performance improves between the first and seventh cycles, even though average parameter magnitude steadily increases."
- The 12M model "exhibited the most severe plasticity loss and the performance continued to deteriorate after cycle 5" **without an increasing dormant fraction.**
- Collapsed-head percentage rises for 53M and *falls* for 106M; lazy-head increase precedes onset in 106M and follows it in 12M.
- Summary (§V-B, p. 8): "None of the three metrics perfectly tracks the onset or severity of plasticity loss across all models." And §I, p. 2: "we do not yet manage to find a 'smoking gun'."

**Category: 1 and 2 for the correlates (descriptive, concurrent, and explicitly negative); a bounded form of 6 for the scaling law (validated by LOO-CV across model sizes, not across seeds or tasks).** No causal claim is made anywhere: "Our measurements are correlational and do not provide a mechanistic explanation" (§V-B, p. 8).

#### 3.4.7 Interventions

**None implemented or tested.** §V-B closes with a survey of candidate mitigations mapped onto the observed pathologies: higher weight decay (citing Han et al. 2026) and weight clipping (Elsayed et al. 2024) for parameter growth; Continual Backprop, ReDo, Self-Normalized Resets and GraMa for dormant units; Shrink-and-Perturb, UPGD and Selective Weight Reinitialization for attention pathologies; learnable attention-sink tokens/registers for diffuse attention. The paper states outright that "whether these methods can prevent the attention-pathology patterns observed here remains an open empirical question."

#### 3.4.8 Released setup

**Nothing released.** No repository, no configs, no checkpoints, no data-processing scripts, no metric implementations. The paper is unusually complete on *specification* — Table I gives every architecture (layers, hidden dim, heads, effective and total params); Table II gives every learning rate with both interpolated and grid-searched values and the resulting losses; Table III gives the per-language transfer AUCs; Table IV the onsets; Table V the four model fits with standard errors and both RMSEs; Fig. 8 is a full architecture diagram — but nothing is executable.

The data are public (CulturaX, HuggingFace) and the architecture is a vanilla pre-norm GPT with absolute positional embeddings, tied embeddings and GeLU (Fig. 8, p. 13), so re-implementation is feasible in principle. The compute is not: 48 cycles × 40B tokens = ~1.9T tokens for the largest model, plus 5B tokens per probe at dozens of probe points. **Reproducibility assessment: no official release; fully specified but far outside our compute budget.**

---

## 4. Main comparison table

Split into four panels so each fits the page. Same row order throughout.

### Panel A — setting and transition

| Paper | Models | Training algorithm | Training transition | Checkpoint design |
| --- | --- | --- | --- | --- |
| Plasticine | CleanRL PPO / C51 agents (CNN encoders) | PPO (on-policy), C51 (off-policy value) | Within-run non-stationarity: standard ALE; Continual Procgen (new level / 2M steps); Continual DMC (chained tasks, 1M steps each) | None — metrics logged inline every 10 iterations |
| PCR (2602.06453) | DeepScaleR-1.5B-Preview, DeepSeek-R1-Distill-Qwen-1.5B/7B, Qwen2-7B-Instruct | GRPO (+ PCR gradient surgery) | Single continuous RLVR run; no second stage | None |
| When RL Fails after SFT | EvoLM-4B (math), Qwen3-8B (agentic) | SFT (LlamaFactory/slime) then GRPO (verl/slime) | base → SFT ladder → RL, branched per checkpoint | **7 SFT checkpoints: epochs 0.5/1/2/4/8/16/32**, same data and ordering |
| Can Scale Save Us | GPT-style pre-norm decoders, 5M–314M non-emb. params, 8 sizes | AdamW next-token pretraining | Continual multilingual pretraining (8 langs × 5B tok/cycle) + stationary-mixture control | Every cycle boundary, up to 48 cycles; later intra-cycle after Zh/De/Ru |

### Panel B — definition and metrics

| Paper | Plasticity definition (operational) | Metrics |
| --- | --- | --- |
| Plasticine | None in outcome terms; a six-metric panel stands in for a definition | RDU (Eq. 5, τ=0.025 in code), FAU (Eq. 6), "Stable Rank" = srank at 99% singular-mass (Eq. 7), Effective Rank = exp(entropy of L1-normalised singular values) (Eq. 8), weight difference, gradient norm |
| PCR | Plasticity = gradient of the GRPO surrogate-gain term (Eq. 5/8); stability = gradient of the KL term (Eq. 6/9) | Layer-wise cos(g_pla, g_sta); gradient norm; distribution of projection strength α; cos(g_final, g_sta); Pass@1 / MMLU / WikiText-2 PPL. **No activation or rank metric** |
| When RL Fails after SFT | "Ability of SFT models to undergo reward-driven improvement in subsequent RL" (fn. 1, p. 2); operationalised as RL gain in points | Pre-RL: element-wise parameter delta vs base, mean-squared weight difference, weight magnitude, token entropy, Pass@K−Pass@1 gap. During RL: mean gradient norm, weight-update magnitude. **No rank, no dormancy** |
| Can Scale Save Us | "Degradation in a model's ability to improve on a target distribution under a fixed training budget" (§I) | Outcome: AUC of probe validation-loss curve. Correlates: global mean |θ|; dormant MLP units (Eqs. 3–4, ε=0.01, 256×2048 held-out Vietnamese tokens); attention entropy (512×2048 Russian, lazy >0.9·H_max, collapsed <0.1·H_max). **No rank metric** |

### Panel C — future learning and prediction

| Paper | Future-learning evaluation | Fixed budget? | Baseline-controlled? | Seeds | Prediction |
| --- | --- | --- | --- | --- | --- |
| Plasticine | None | n/a | n/a | "multiple runs" (count not stated) | None — concurrent description only |
| PCR | None | n/a | Deltas vs each base model's own score (one step, not a ladder) | Not reported | None |
| When RL Fails after SFT | **Yes** — GRPO branched from 7 SFT checkpoints; ID + OOD eval suites | **Math: not reported.** Agentic: up to 200 rollouts, eval every 20 | Partially — pre-RL and post-RL both tabulated, but the winning method's pre-RL baseline is much lower, inflating its delta | **1**, shared across initialisations | Retrospective explanation only; **no correlation, test or split reported** |
| Can Scale Save Us | **Yes** — 5B-token Vietnamese probe on a discarded copy at every cycle | **Yes**, 5B tokens exactly | **Yes** — normalised to own first-cycle AUC, and to from-scratch Vietnamese AUC | **1 per model size**; no uncertainty reported | Scaling law for **onset vs model size** (8 points, log-log, LOO-CV, R²=0.897, LOO-RMSE 17.9). Internal correlates explicitly **fail** to track onset |

### Panel D — interventions, release, and relevance to us

| Paper | Interventions | Released setup | Main limitation | Relevance to our project |
| --- | --- | --- | --- | --- |
| Plasticine | 13+ methods in 5 families (reset / normalisation / regularisation / activation / optimizer); ReDo at τ=0.025 | **Full code, MIT.** Metrics, agents, env wrappers, requirements, hyperparameter tables. Full results only on W&B | No LLM, no RLVR; no sensitivity analysis of its own metrics; benchmark results deferred off-paper | **Measurement-convention baseline.** Our reference implementation for effective rank and dormancy formulas — but its probe (current rollout batch, train mode, layer-averaged) is weaker than ours |
| PCR | PCR soft projection on MLP layers only; +3.2/+3.6/+3.5 avg points over base models | **Nothing.** No code; **and the arXiv version is missing Appendices B–L**, which it cites for setup, algorithm and both proofs | Unverifiable setup; group size, step count, seeds all unreported; internal inconsistencies (Fig. 1(b)/(c), "Eq. 17") | Low. Its "plasticity" is a loss term, not a capacity. With our β=0, PCR is a no-op. Relevant only if we run the KL β>0 baseline |
| When RL Fails after SFT | **Rejuvenation** = base-anchored fusion (α=0.5) + attribution-guided neuron reset (ρ=1%); 3–5 min on one H800 | **Nothing.** RL hyperparameters are complete (App. B) but no code, checkpoints, corpus name or calibration set | No budget stated for the headline math result; single seed; no uncertainty; delta inflated by a depressed pre-RL baseline; Table 4 vs 7 mismatch | **Highest structural relevance.** Nearest existing checkpoint-ladder→future-RL design; its grad-norm paradox is direct evidence against dashboard baselines; its verl recipe is a concrete reference config |
| Can Scale Save Us | None implemented; mitigations surveyed only | **Nothing.** But fully specified: architectures, LRs, transfer AUCs, onsets, model fits with SEs | ~1.9T tokens of compute; no seeds; single-run curves; onset defined via a 3-point moving average | **Highest methodological relevance.** Our operational definition, the discarded-copy probe protocol, the AUC outcome, the optimizer-reset rule, and the strongest available prior evidence that activation correlates **do not** reliably predict future learning |

### 4.1 The differences that matter, in prose

**Three of the four never measure future learning at all.** Plasticine and PCR observe a single training trajectory and describe it. That distinction — trajectory description versus branch-and-measure — is the axis our project sits on, and only Papers 3 and 4 are on the right side of it.

**"Plasticity" names four different things across these papers.** In Plasticine it is a property of the *representation* (rank, dormancy). In PCR it is a *term in a loss function*. In Paper 3 it is a *disposition of a checkpoint* toward future reward-driven improvement. In Paper 4 it is a *measured outcome* — fixed-budget improvement on a held-out target. Our proposal uses Paper 4's sense for the outcome and Plasticine's sense for the predictor, and the whole point of RQ1 is that the link between those two senses is unestablished. **We should say this explicitly in the Research Doc**: we are not measuring "plasticity" twice; we are asking whether a representation-level quantity forecasts an outcome-level one.

**The one paper with the closest experimental structure is the weakest statistically.** Paper 3 has the ladder, the shared recipe and the per-checkpoint gains — and one seed, no error bars, no correlation coefficient, and an unstated budget for its headline result. **This is an opportunity.** A pre-registered version of that design, with a stated budget, ≥3 Stage-B seeds and an actual predictive statistic, is a genuine contribution even at 0.5B scale.

**The one paper with the cleanest methodology reports a negative result on exactly our predictors.** Paper 4 measured dormant units and parameter magnitude against fixed-budget adaptability across eight model sizes and found neither tracks it. It did *not* measure effective rank. Our Q metric is therefore not yet refuted — but we should be calibrated: a competent team looking for activation-based early warning in LMs recently looked and did not find it. **Our null hypothesis deserves real respect, and our write-up should be prepared to report a null cleanly rather than fishing.**

**A fifth paper, not on Tommy's list, is closer to our RQ than any of the four.** *SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR* (arXiv:2606.18487, v1 16 Jun 2026, v2 22 Jun 2026; Aphale & Liu) runs SFT-depth ladders on Qwen2.5-Coder-3B (five depths × three seeds) and DeepSeek-Coder-6.7B (four depths × three seeds), measures **pre-RL entropy** as a predictor of GRPO outcome, and reports an actual statistic: ρ = +0.69, with peak GRPO pass@10 falling from 0.806 to 0.481 across the Qwen ladder. It also gives a theoretical reason a checkpoint can be untrainable under binary-reward GRPO — expected within-group advantage variance `p(1−p)(g−1)/g` vanishes when the group pass rate `p` falls below a critical `p*(g)`, e.g. `p*(8) = 0.083`. **Two direct consequences for us:** (i) this is the multi-seed, statistically analysed version of Paper 3's design and should be read next; (ii) `p*(g)` is a concrete, checkable failure mode for our own sparse-reward gate — with `num_generations = 8`, a Stage-B SVAMP pass rate below ~8% means most groups carry no gradient signal at all, which would look like "no plasticity" but is actually "no reward variance." **I did not verify its code availability beyond the abstract page (none listed), and I have not read its full text — flagging it as required follow-up reading, not as a reviewed source.**

---

## 5. Measurement-convention recommendations

Below, **[Established]** marks a convention taken directly from one of the four papers or its released code; **[Ours]** marks my recommendation, which the papers do not settle.

### 5.1 Effective rank

| Decision | Recommendation | Basis |
| --- | --- | --- |
| Formula | `ER = exp(−Σ p_k log p_k)`, `p_k = σ_k / Σ_j σ_j`, drop `p_k = 0` before the log | **[Established]** Plasticine Eq. 8 p. 10 and `rank.py`; Roy & Vetterli 2007. Already what `src/metrics.py:spectrum_metrics` does |
| Activation tensor | Residual-stream hidden state at the **output of decoder block ℓ** (`hidden_states[ℓ+1]`) | **[Ours]** Plasticine uses the encoder's penultimate feature vector; the LLM analogue is the residual stream. Already our convention |
| Layer selection | Fixed `{4, 12, 22}` of 24, pre-registered, never re-chosen after seeing results | **[Ours]**, reinforced by Paper 4 Fig. 6: pathology concentrates in *single* layers, so a network average can hide everything. **Also report per-layer, never only the mean** |
| Token pooling | Last non-padding token per prompt (one row per prompt) as primary | **[Ours]**. Paper 4 pools over *all* token positions for dormancy; for rank, one row per prompt keeps rows independent, which matters for the SVD |
| Padding | Right-padding; index the last valid position via the attention mask; padded positions never enter the matrix | **[Ours]** — already implemented in `collect_probe_activations` |
| Sample unit | One row per probe prompt | **[Ours]** |
| Probe dataset | Frozen `data/probe_set_ids.json`, identical across all checkpoints and all runs, drawn from the Stage-A task family | **[Ours]**, and **stricter than Plasticine**, which measures on the current rollout batch (`ppo_continual_procgen_plasticine.py:356`) — a moving probe that makes cross-time comparison unsound. Paper 4 uses a *held-out* probe (Vietnamese validation) for dormancy, which is stricter still |
| **Probe size** | **Raise the primary probe to `n ≥ 2·d`, i.e. ≥ 1792 prompts for Qwen2.5-0.5B (d = 896).** Minimum acceptable is `n ≥ d`. Also report ER at `n ∈ {512, 1024, 2048}` once, as a sample-size sensitivity curve | **[Established, as a hard constraint]** — Plasticine's `rank.py` asserts `features.size(0) >= features.size(1)`. **[Ours]** for the 2·d margin. **Our current `probe_questions = 512` violates this**: the activation matrix is 512×896, so at most 512 singular values are non-zero and ER is truncated by sampling, not by the representation. The config already contains `sensitivity_probe_questions: 2048` — promote it |
| Centering | Report **both** centered and uncentered ER; treat centered as primary | **[Ours]**. Plasticine does **not** centre. Our artefacts show why it matters: `metrics_ckpt0.json` has `anisotropy_uncentered ≈ 0.995` and `anisotropy_centered ≈ −0.001` at layer 4 — an enormous shared mean direction. Uncentered, that direction becomes one dominant singular value and depresses ER for a reason unrelated to representational collapse. Reporting both makes us comparable to Plasticine *and* interpretable |
| SVD procedure | `numpy.linalg.svd(A, compute_uv=False)` on the centered matrix; no scaling, no whitening | **[Ours]** — matches Plasticine's `svdvals` up to centering |
| Numerical precision | Cast activations to **float32** for accumulation and **float64** for the SVD, regardless of model dtype; record the model dtype in the artefact | **[Ours]**, and stricter than Plasticine (float32 throughout). Already implemented and recorded in `measurement_contract`. **Non-negotiable given the fp16/bf16 history in `eaaj-pilot-win4070`** |
| Additional spectral statistics | Keep `erank_norm = ER/d`, participation ratio `(Σλ)²/Σλ²` with `λ = σ²`, and top-k variance shares at k ∈ {1, 8, 32} | **[Ours]** — participation ratio weights the spectrum quadratically and moves for different reasons than ER; the pair is more informative than either alone |
| Cross-checkpoint comparability | Identical probe set, identical prompt order, identical `max_prompt_length`, identical layers, identical dtype path, identical pooling. Any change invalidates the series and forces a re-measure of **all** checkpoints | **[Ours]** — proposal §8; no paper states such a rule |
| Sensitivity checks to run once | (a) probe size 512 / 1024 / 2048; (b) centered vs uncentered; (c) float32 vs float64 SVD; (d) all 24 layers at one checkpoint, to confirm {4,12,22} is representative | **[Ours]** — none of the four papers does any of this, which is itself worth a sentence in the Research Doc |

### 5.2 Dormant-neuron fraction

| Decision | Recommendation | Basis |
| --- | --- | --- |
| What counts as a "neuron" | For Qwen2's gated MLP, the units of the **intermediate representation** `act_fn(gate_proj(x)) ⊙ up_proj(x)` — i.e. the input to `down_proj` (4,864 units per block) | **[Ours]**, forced by architecture. Plasticine hooks `nn.ReLU`/`nn.Tanh` modules, which do not exist in Qwen2. Paper 4 measures "MLP units" in a standard non-gated GeLU MLP. **Our gated choice must be stated explicitly whenever we cite either paper's numbers** — they are not the same object |
| Activation site | Forward **pre-hook** on `mlp.down_proj`, capturing `args[0]` | **[Ours]** — already implemented |
| Magnitude statistic | Mean absolute activation per unit, over all non-padding token positions of all probe prompts | **[Established]** — Paper 4 Eq. 3 pools over sequences *and* token positions; Plasticine pools over batch. Ours matches Paper 4 |
| Normalisation baseline | Divide by the mean of the per-unit means **within the same layer** | **[Established]** — Sokar et al. 2023; Plasticine Eq. 5; Paper 4 Eq. 4. Unanimous |
| Threshold | Report at **τ ∈ {0.01, 0.025, 0.1}** and publish the full normalised-score distribution, not just the fraction | **[Established]** for 0.01 (Paper 4, ε=0.01) and 0.025 (Plasticine code and ReDo default); **[Ours]** for 0.1 and for publishing the distribution |
| Averaging across tokens and prompts | Uniform over all non-padding positions (a long prompt contributes more positions) | **[Established]** Paper 4 Eq. 3. **[Ours]:** also record the per-prompt-normalised variant once, to confirm length does not drive the result |
| Aggregation across layers | **Per layer, always.** If a scalar is needed, use a neuron-count-weighted fraction, not Plasticine's unweighted mean over layers | **[Ours]**, motivated by Paper 4 Fig. 6 (one layer at >95% while the network average stays low) and by Plasticine's `torch.mean(torch.stack(rdu))`, which weights a 512-unit layer equally with a 4,864-unit layer |
| Probe-set requirements | Same frozen probe as effective rank. Paper 4's stronger option — a **held-out** probe from a distribution the model never trained on — is worth adding as a secondary measurement | **[Established]** Paper 4 §V-B (Vietnamese validation, 256×2048); **[Ours]** for reusing the Stage-A probe as primary |
| Evaluation mode / dtype | `model.eval()`, `torch.no_grad()`, float32 accumulation, model dtype recorded | **[Ours]**, stricter than Plasticine, which never calls `.eval()` |

**Expected failure mode — and it has already happened.** In `outputs/local_cuda_grpo_gsm8k_6a075c15808e/measurements/metrics_ckpt0.json`, every layer reports `dormant_frac_tau0.025 = 0.0` **and** `dormant_frac_tau0.1 = 0.0`, with `dormant_score_min` of 0.437 (layer 4), 0.250 (layer 12) and 0.129 (layer 22), and medians near 0.9. There is no unit anywhere close to any literature threshold.

This is **structural, not evidential**:
1. Qwen2's MLP uses SiLU, which is smooth and non-zero for all finite inputs, so the gated product `act_fn(gate)⊙up` is essentially never exactly zero. Paper 4 warns about precisely this for GeLU (§V-B, p. 7).
2. The ReDo score is *relative to the layer mean*. In a well-conditioned pretrained LLM, per-unit mean activations are concentrated within roughly one order of magnitude, so `s_i` clusters around 1 and never approaches 0.025.
3. Paper 4 does observe large dormant fractions in an equivalent formulation — but on models trained for hundreds of billions of tokens with GeLU and no gating. Our 200 GRPO updates at lr 1e-6 will not move a pretrained network into that regime.

**Required reporting rule (proposal §7 compliance).** Dormant fraction must be reported as **"identically zero at all thresholds — the metric has no resolution in this setting,"** never as "dormancy did not increase, so plasticity was preserved." A constant zero is a measurement floor, not a finding. Concretely, in every artefact and slide: report `dormant_score_min` and the 1st/5th percentile of `s` alongside the fraction, so the reader can see *how far* the distribution sits from the threshold. If we want a dormancy-family signal with actual resolution on this model, the defensible substitute is the **low-activity quantile** — e.g. the 5th percentile of `s_i`, or the fraction below `0.5` — declared in advance as a descriptive statistic, explicitly *not* the ReDo dormant fraction.

**Threshold-sensitivity analysis.** Because the fraction is degenerate, the informative sensitivity analysis is not over τ but over the score distribution: plot the empirical CDF of `s_i` per layer per checkpoint and look for leftward mass migration. That is a real, reportable signal even when every threshold-crossing count is zero.

---

## 6. RLVR pipeline candidate table

Search scope: the four papers' own stacks, plus the major open RLVR frameworks, checked against the GitHub API on 2026-07-26. Popularity was not used as a criterion; file trees and example configs were inspected.

**Panel A — model, algorithm, data, branching.**

| Candidate | Model support | RL algorithm | Math datasets | Checkpoint branching | Two-stage training |
| --- | --- | --- | --- | --- | --- |
| **TRL `GRPOTrainer` 1.6.0** (our current base) | Qwen2.5 explicitly listed as tested; any HF causal LM | GRPO, plus `loss_type` ∈ {grpo, dapo, dr_grpo, sapo}; `beta`, `num_generations`, `num_iterations`, `scale_rewards` | Any HF dataset; reward is a user-supplied Python callable receiving `prompts`, `completions`, `completion_ids`, `trainer_state` + dataset columns | Native: HF `Trainer` `save_steps` + `resume_from_checkpoint`; branching = point a new run's model path at a checkpoint dir | Not built in — two `GRPOConfig`s and two scripts (we already do this) |
| **verl** (`verl-project/verl`) | Broad: Qwen2.5/Qwen3, DeepSeek, Llama; `examples/sft/gsm8k/run_qwen2_5_0_5b_fsdp.sh` exists for **SFT**; GRPO GSM8K shells start at 0.6B | GRPO, PPO, and many variants; **used by Paper 3** | `examples/data_preprocess/gsm8k.py`; `verl/utils/reward_score/gsm8k.py` gives exact-answer scoring out of the box | Supported via its checkpoint manager, but the config surface is large and Ray-mediated | Requires re-running the driver with a new `actor.model.path`; no first-class branching abstraction |
| **OpenRLHF** | Broad HF models | PPO, GRPO/DAPO-family, RLOO | Supported via custom reward models/functions | Ray-based; checkpointing supported | Not first-class |
| **Open-R1** | Qwen family, DeepSeek-R1 reproduction | GRPO via TRL underneath | MATH/AIME-oriented recipes | Inherits TRL/HF | Recipe-driven, not branch-driven |
| **EasyR1** | veRL-based, multimodal focus | GRPO and variants | Math recipes present | Inherits verl | Not first-class |
| **Paper 3's stack (verl + LlamaFactory + Math-Verify)** | EvoLM-4B, Qwen3-8B | GRPO with a fully published config (App. B, pp. 20–21) | Math-Verify on boxed answers, reward +1/−1 | Per-checkpoint launch, as we need | **Yes — this is the design we want** |
| **Plasticine** | CleanRL CNN agents | PPO, C51 | None (RL environments only) | n/a | n/a |

**Panel B — hooks, compute, docs, licence, work required.**

| Candidate | Metric hooks | Compute burden | Documentation | License | Required modifications |
| --- | --- | --- | --- | --- | --- |
| **TRL `GRPOTrainer` 1.6.0** | None built in; but plain `nn.Module` forward hooks work because the policy is an ordinary HF model in-process | Lowest of all candidates. Runs on one L4/T4; vLLM optional (`use_vllm`, `vllm_mode`) | Extensive HF docs + docstrings | Apache-2.0 | Stage-A/B orchestration, frozen splits, probe measurement — **already written in `eaaj-pilot`** |
| **verl** | Policy lives inside Ray workers behind FSDP/Megatron shards — activation hooks require reaching into a worker; **materially harder than in-process** | High: Ray + FSDP2/Megatron + vLLM/SGLang. Paper 3 used 8×H800 | Good docs (`docs/examples/gsm8k_example.rst`), fast-moving | Apache-2.0 | Re-implement all orchestration in verl's config idiom, plus a sharded-model hook path |
| **OpenRLHF** | Same sharding problem as verl | High: Ray-centric, designed for multi-node | Good | Apache-2.0 | Same as verl, without a paper-endorsed GSM8K GRPO reference |
| **Open-R1** | Inherits TRL (in-process hooks work) | Moderate–high; recipes target multi-GPU | Good | Apache-2.0 | **Last push 2026-04-02** — least actively maintained of the candidates. Adds a recipe layer we do not need |
| **EasyR1** | Inherits verl's sharding | High | Moderate | Apache-2.0 | Multimodal orientation is orthogonal to our need |
| **Paper 3's stack** | None released | 8×H800 per run | Recipe published, **no code** | n/a (nothing released) | Everything: it is a recipe, not an artefact |
| **Plasticine** | **Best-in-class and directly reusable as formulas**: `plasticine_metrics/{rank,units}.py` | Trivial | Good README + appendices | **MIT** | Not an RLVR pipeline at all — harvest the metric functions only |

**On not confusing frameworks with pipelines.** None of the four papers releases a *pipeline* for our experiment. verl is the framework Paper 3 used, and its Appendix B config is the most complete published GRPO recipe of the four — but a recipe on 8×H800 for a 4B model is a reference, not something we can run.

---

## 7. Pipeline scores

0–5 per criterion. Only the three candidates that could plausibly host our experiment are scored; Plasticine and Paper 3's stack are excluded because neither is a runnable RLVR pipeline for us (Plasticine has no LLM support; Paper 3 released no code).

| Criterion | TRL 1.6 / `eaaj-pilot` | verl | OpenRLHF |
| --- | --- | --- | --- |
| 1. Compatibility with Qwen2.5-0.5B | **5** | 4 | 4 |
| 2. Compatibility with GRPO / RLVR | **5** | **5** | 4 |
| 3. Support for GSM8K and SVAMP | **5** | 4 | 3 |
| 4. Ease of saving and branching from checkpoints | **5** | 3 | 3 |
| 5. Ease of Stage-A → Stage-B transitions | **5** | 2 | 2 |
| 6. Ease of adding activation measurements | **5** | 2 | 2 |
| 7. Reproducibility | **5** | 4 | 3 |
| 8. Compute accessibility | **5** | 2 | 1 |
| 9. Code maintainability | 4 | 3 | 3 |
| 10. Match to our scientific design | **5** | 3 | 2 |
| **Total (/50)** | **49** | **32** | **27** |

**Evidence for each score — TRL / `eaaj-pilot`.**
1. TRL's GRPO docs list Qwen2.5 among tested models; our runs already train it (`outputs/local_cuda_grpo_gsm8k_*`).
2. `GRPOConfig` exposes `num_generations`, `beta`, `loss_type`, `scale_rewards`, `num_iterations` — everything the pre-registered recipe needs; `beta = 0.0` is the default.
3. Both datasets are pinned by revision in `pilot_config.json` (`openai/gsm8k`, `ChilleD/SVAMP`) with frozen splits already built.
4. HF `Trainer` `save_steps` + `resume_from_checkpoint`; branching is a directory path. We already produce `metrics_ckpt{0,25,50,100,200}.json`.
5. `src/adaptation.py` and `scripts/run_stageb_seed_repeat.py` exist and run.
6. The policy is an ordinary in-process HF module: `src/metrics.py:collect_probe_activations` registers a forward hook on the decoder block and a forward **pre**-hook on `mlp.down_proj`. This is impossible-to-cheap in-process and hard behind FSDP/Ray.
7. Pinned requirements, model and dataset revisions pinned by SHA, seeds fixed, `measurement_contract` written into every artefact, run manifests recording torch/CUDA.
8. Runs on Colab L4/T4 within the ~300 compute-unit budget, and locally on CPU/MPS for development.
9. **4, not 5** — the codebase now carries three execution strata (cpu / mps / cuda-win4070) plus phase-following logic, which is real complexity we maintain ourselves.
10. Built directly from proposal §5–§9.

**Evidence — verl.** Score 5 on GRPO is earned: Paper 3's published recipe (App. B) is a verl config, and `verl/utils/reward_score/gsm8k.py` implements exact-answer scoring. Score 2 on Stage-A→B and on hooks reflects that the policy lives inside Ray workers behind FSDP2/Megatron sharding; a per-checkpoint frozen-probe measurement pass would mean either a separate out-of-framework measurement script (which is what we already have) or intrusive worker-side code. Score 2 on compute: the GRPO GSM8K shells in-tree are multi-GPU and start at 0.6B.

**Evidence — OpenRLHF.** Ray-first, multi-node design; no GSM8K-with-exact-reward reference config found in the tree comparable to verl's; no paper in this review uses it.

**Major risks in this scoring.**
- **TRL API churn.** TRL is under heavy development (last push 2026-07-27). Our pin at `trl==1.6.0` is the mitigation, and it must not be bumped mid-experiment.
- **TRL's GRPO is not the fastest.** Without vLLM, generation dominates wall-clock. This is a throughput risk, not a validity risk.
- **Score 6 assumes in-process access stays true.** If we ever enable `use_vllm` with `vllm_mode="server"`, generation moves out of process — but the *measurement* pass loads the checkpoint separately anyway, so hooks remain unaffected. **Stated as an assumption, verified only for our current single-process path.**

**Unsupported assumptions I am flagging rather than hiding.**
- I did not benchmark verl on an L4 with a 0.5B model. The compute-accessibility score is inferred from the in-tree example configs and Paper 3's reported hardware, not measured.
- I read verl's file tree and GSM8K/GRPO example paths, not its checkpoint-manager source. The "branching is possible but config-heavy" claim is an inference from structure.
- OpenRLHF was assessed from repository metadata and its stated design, not from a file-tree inspection at the same depth as verl.

---

## 8. Final recommended pipeline

### Recommended base pipeline

**Continue with `eaaj-pilot` on Hugging Face TRL 1.6.0 `GRPOTrainer`**, driven by the single pre-registered recipe in `eaaj-pilot/pilot_config.json` and executed through `scripts/run_local_pipeline.py` / notebooks `01`–`04`. Borrow **formulas only** from Plasticine (`plasticine_metrics/rank.py`, `plasticine_metrics/units.py`, MIT) and **protocol** from Paper 4 (discarded-copy probe, fixed budget, optimizer reset, AUC outcome).

### Why it is the closest match

1. **No released alternative implements our experiment.** Three of four papers released nothing. The fourth released a deep-RL benchmark with no LLM support.
2. **The hard parts are ours regardless of framework.** Frozen GSM8K/SVAMP splits, a checkpoint ladder at 0/25/50/100/200, independent Stage-B branches, a frozen probe set, eval-mode activation hooks, float64 SVD, and a per-artefact measurement contract — none of this exists in TRL, verl or OpenRLHF. Switching frameworks would mean rewriting all of it in a harder environment.
3. **In-process hooks are a decisive advantage.** Our Q measurement is a forward pre-hook on `mlp.down_proj` and a forward hook on a decoder block. Under verl's FSDP2/Megatron sharding this becomes a substantial engineering task; in TRL it is 40 lines that already work.
4. **Compute fits the budget.** Paper 3's math RL used 8×H800; Paper 4 used ~1.9T tokens. We have ~300 Colab compute units and an RTX 4070 Laptop. TRL at 0.5B is the only option in this list that fits.
5. **We already have measured artefacts.** Five checkpoints with Q metrics exist under `outputs/local_cuda_grpo_gsm8k_6a075c15808e/measurements/`. Switching frameworks discards a working, audited measurement chain in exchange for nothing the science requires.

### Components we can reuse unchanged

| Component | Source | Status |
| --- | --- | --- |
| Model loading, tokenizer, revision pinning | `pilot_config.json` + notebooks | Reuse as-is |
| GRPO trainer, generation, clipping, advantage computation | TRL 1.6.0 `GRPOTrainer` | Reuse as-is |
| Exact-answer reward | `src/reward.py` (unit-tested, `tests/test_reward.py`) | Reuse as-is |
| Dataset preprocessing and frozen splits | `src/data.py`, `data/probe_set_ids.json` | Reuse as-is |
| Checkpoint saving | HF `Trainer` `save_steps` | Reuse as-is |
| Stage-B adaptation driver | `src/adaptation.py`, `scripts/run_stageb_seed_repeat.py` | Reuse as-is |
| Evaluation | `src/evaluate.py` | Reuse as-is |
| Effective rank / spectral metrics | `src/metrics.py:spectrum_metrics` | Reuse; **change probe size only** |
| Activation collection | `src/metrics.py:collect_probe_activations` | Reuse as-is |
| Distributed execution | Not used (single GPU) | n/a |

### External components to import

| From | What | How |
| --- | --- | --- |
| Plasticine (MIT) | The **uncentered** effective-rank variant and the srank-at-99% definition, as secondary metrics reported alongside ours, for literature comparability | ~15 lines added to `src/metrics.py`, with attribution |
| Paper 4 | Probe protocol: measure on a **copy**, discard updates, reset optimizer at each branch, report **AUC** of the Stage-B learning curve as the primary outcome alongside the endpoint delta | Protocol change in `src/adaptation.py` + `src/analysis.py` |
| Paper 3 | Reference GRPO hyperparameters for sanity-checking ours (8 responses/prompt, symmetric clip 0.2, constant LR 1e-6, grad clip 1.0) | Documentation only |
| Paper 3 | The **gradient-norm paradox** as a documented reason our dashboard baselines may be anti-informative | Research Doc citation |

---

## 9. Required modifications

Difficulty: **trivial** = config edit; **small** = < 1 day; **moderate** = 1–3 days including tests; **large** = > 3 days or a design change.

| # | Modification | Status today | Difficulty | Why |
| --- | --- | --- | --- | --- |
| 1 | Frozen GSM8K and SVAMP splits | **Done** — dataset revisions pinned by SHA in `pilot_config.json`; 512/256/100 questions fixed | trivial | Already enforced; only needs a hash written into the manifest (see #12) |
| 2 | Stage-A checkpoints at 0/25/50/100/200 | **Done** — `checkpoint_steps` in config, artefacts exist | trivial | — |
| 3 | Branch Stage B independently from every checkpoint | **Done** — `src/adaptation.py` | trivial | — |
| 4 | Reset/restore optimizer state under a fixed protocol | **Partly** — each Stage-B run is a fresh `GRPOTrainer`, so AdamW state is fresh, but this is incidental rather than asserted | **small** | Paper 4 resets deliberately and says why (§IV-B). We should assert it in code and record `optimizer_state: "fresh"` in the manifest. Paper 3 leaves this unstated — we should not |
| 5 | Fixed 50-update Stage-B budget | **Done** — `budget_updates: 50` | trivial | This is where we are *stronger* than Paper 3, whose math budget is unreported |
| 6 | Baseline SVAMP evaluation before adaptation | **Done** — `eval_every: 10` starting at update 0 | trivial | Essential: Paper 3's inflated `+12.1` shows what happens without it |
| 7 | Full learning-curve logging | **Done** — 6 eval points per Stage-B run | trivial | Enables the AUC outcome (#7b) |
| 7b | **Add AUC of the Stage-B learning curve as a co-primary outcome** | **Not done** | **small** | Paper 4's outcome statistic; integrates the curve, less noise than an endpoint. Trapezoid over existing eval points; ~30 lines in `src/analysis.py` |
| 8 | Multiple Stage-B seeds | **Partly** — `scripts/run_stageb_seed_repeat.py` exists | **small** (code) / **moderate** (compute) | **≥3 seeds per checkpoint is the single biggest scientific upgrade over Paper 3 and Paper 4, both of which run one.** Cost: 5 checkpoints × 3 seeds × 50 updates. Must be budgeted in `compute_log.md` before launching |
| 9 | Effective-rank activation hooks | **Done**, but see #9b | trivial | — |
| 9b | **Raise the primary probe to n ≥ 1792 (ideally 2048) and record an n-sensitivity curve** | **Not done — this is a defect** | **small** | `d = 896`; the current 512-prompt probe yields a rank-truncated 512×896 matrix. Plasticine asserts `n ≥ d`. Config already has `sensitivity_probe_questions: 2048`. **All existing ER numbers must be re-measured after this change** |
| 10 | Dormant-neuron measurement | **Done**, but degenerate | **small** | Add τ = 0.01 (Paper 4's value); emit the score CDF and 1st/5th percentiles; add the mandated "no resolution in this setting" wording to the report template |
| 11 | Strict completion validation | **Partly** — `src/preflight.py`, the notebook-01 sparse-reward gate, and the win4070 step-25 update sentinel | **moderate** | Add a post-hoc assertion that each recorded checkpoint corresponds to the intended number of *completed* optimizer updates, and that Stage-B ran exactly 50. Paper 3's unstated budget is the cautionary case; our own v1 bf16 no-op run (`eaaj-pilot-win4070/WIN4070_RUN_ANALYSIS.md`) is the local one |
| 12 | Artefact provenance and configuration hashing | **Partly** — `measurement_contract` is embedded; revisions pinned | **small** | Hash `pilot_config.json` and the frozen split ID lists; write `config_sha256`, `probe_set_sha256`, `trl_version`, `torch_version`, `model_dtype`, `git_commit` into every metrics and evaluation artefact. Makes cross-machine comparability auditable rather than assumed |
| 13 | Report **both** centered and uncentered effective rank | **Not done** (centered only) | trivial | Comparability with Plasticine's convention; our own artefacts show a 0.995 uncentered anisotropy, so the two will differ substantially |
| 14 | Per-layer reporting rule, never network-mean-only | **Done** for storage, **not enforced** in reporting | trivial | Paper 4 Fig. 6: a single layer can hit 95% dormancy while the mean stays flat |
| 15 | Guard against the `p*(g)` degenerate-advantage regime | **Partly** — notebook 01 has a sparse-reward gate for Stage A | **small** | With `num_generations = 8`, a Stage-B pass rate below ~8% makes most groups reward-constant and the gradient zero (Aphale & Liu, 2606.18487). **Log the fraction of zero-variance groups per Stage-B run**; without it, "checkpoint cannot adapt" is confounded with "no reward variance" |

**Nothing in this list is large.** That is itself the strongest argument for the recommendation: the residual work on our own pipeline is a handful of small tasks, whereas porting to verl would start with a large one.

---

## 10. Minimum viable implementation plan

The smallest scientifically defensible experiment, given that Papers 3 and 4 both fail on seeds and Paper 3 fails on budget disclosure.

| Element | Specification | Rationale |
| --- | --- | --- |
| Model | Qwen2.5-0.5B (base), revision `060db64…` | Already pinned; base-vs-Instruct remains an open team question and must stay logged, not silently decided |
| Stage-A data | 512 frozen GSM8K training questions | Pre-registered |
| Stage-A algorithm | GRPO, `num_generations = 8`, lr 1e-6, `beta = 0`, temp 0.7, 200 updates | Pre-registered; group size matches Paper 3's 8 responses/prompt |
| Reward | Binary exact-answer on the parsed final answer | Pre-registered; equivalent in spirit to Paper 3's Math-Verify +1/−1 |
| Checkpoints | 0, 25, 50, 100, 200 updates | 5 points. Paper 3 used 7 (on a log-ish epoch ladder); 5 is the floor for a rank correlation to mean anything |
| Measurement | Effective rank (centered **and** uncentered), participation ratio, top-k variance shares, anisotropy pair, dormant score distribution at τ ∈ {0.01, 0.025, 0.1}, weight norms — at layers {4, 12, 22}, **probe n = 2048**, eval mode, float32 accumulation, float64 SVD | §5. The probe-size fix is mandatory before any ER number is quoted |
| Stage-B | Identical 50-update GRPO on 256 frozen SVAMP questions, launched independently from each checkpoint, fresh optimizer, identical data order | Pre-registered; matches Paper 3's "identical across initializations" control and Paper 4's optimizer-reset rule |
| Stage-B seeds | **3 per checkpoint** (15 runs) | The decisive upgrade over both papers. Without ≥3 we cannot separate checkpoint effect from Stage-B noise, and a 5-point correlation on noisy singletons is uninterpretable |
| Evaluation | 100 held-out SVAMP questions, evaluated at updates 0, 10, 20, 30, 40, 50 | Update-0 eval is the per-checkpoint baseline; the full curve enables AUC |
| Primary outcome | **Both** endpoint delta from the checkpoint's own update-0 SVAMP accuracy **and** AUC of the Stage-B accuracy curve | Paper 4 uses AUC; Paper 3's inflated delta shows why the endpoint level must be reported next to the delta |
| Primary analysis | Spearman rank correlation between each Q metric at a checkpoint and the mean Stage-B outcome across seeds (n = 5 checkpoints), with the seed-level spread shown. Dashboard baselines (reward slope, KL, grad norm, entropy) analysed identically | n = 5 cannot support a p-value worth trusting. **Report the correlation with its confidence interval and state plainly that 5 points is descriptive, not confirmatory** |
| Hardware | One Colab L4 (or the local RTX 4070 Laptop, fp32 master + bf16 autocast + 8-bit paged AdamW per `WIN4070_RERUN_GUIDE.md`). Stage-A ~200 updates; Stage-B 15 × 50 = 750 updates plus 90 evaluations | Fits the ~300 compute-unit budget. Every run logged in `compute_log.md` with GPU, duration and units before/after |

**Pre-registered stopping and honesty rules.** (a) If Stage A's reward is constant within every sampled prompt group, stop at the notebook-01 gate and ask the team — do not add a shaping reward. (b) If the fraction of zero-variance Stage-B groups exceeds ~50%, the run measures reward degeneracy, not plasticity, and must be reported as such. (c) If the dormant fraction is zero everywhere — which §5.2 predicts — report it as a metric with no resolution, not as preserved plasticity. (d) If the Q-vs-adaptability correlation is null, that is the result; Paper 4 found the same for its correlates and said so.

---

## 11. Risks, missing evidence, and unresolved questions

### Risks to our scientific claim

1. **The literature's prior is against us.** Paper 4 tested two of our predictor families (dormancy, parameter magnitude) against fixed-budget adaptability across 8 model sizes and found neither tracks it. It did not test effective rank — that is our opening — but we should design and write for a possible null.
2. **n = 5 checkpoints.** Any correlation over 5 points is descriptive. Paper 4 needed 8 model sizes plus LOO-CV to defend a 2-parameter fit and still called it "not perfect." We must not over-claim.
3. **Reward degeneracy masquerading as plasticity loss.** The `p*(g)` argument (Aphale & Liu) gives a concrete mechanism by which a checkpoint appears unable to learn purely because group reward variance vanished. Mitigation is modification #15.
4. **Effective rank is not a licence for a causal claim.** Per the prompt's own instruction and Paper 4's stance: a decrease in effective rank does not prove impaired adaptability. Our language must stay correlational.
5. **Probe-size bias in existing artefacts.** Every effective-rank number measured so far used n = 512 < d = 896. Those numbers are internally comparable to each other but sample-truncated, and they should not be compared to any literature value or quoted as absolute ranks until re-measured.
6. **Baseline manipulation in delta-based outcomes.** Paper 3's `+12.1` is the worked example. Our update-0 SVAMP baseline protects us — provided we always report the endpoint level next to the delta.

### Missing evidence I could not obtain

- **Paper 2's appendices do not exist** in the arXiv v1 PDF or HTML, despite ~12 cross-references. Its experimental setup, algorithm listing and both proofs are unverifiable. Its group size, RL step count, seed count and rollout configuration are unreported.
- **Paper 3's math RL step budget is not reported.** For a paper whose subject is future-learning capacity, the budget of the future learning is the most important missing number.
- **Papers 2, 3 and 4 released no code.** Claims about their implementations are limited to what the text states.
- **Plasticine's benchmark results live in a W&B space,** not in the paper. I read the code and the metric definitions; I did not audit the dashboard.
- **Plasticine v1 → v2 diff not examined.** The review is of v2.
- **Optimizer-state handling at Paper 3's RL branches is unstated.** A fresh verl launch implies a fresh optimizer, but the paper does not say so.
- **I have not read arXiv:2606.18487 in full** — only its abstract page. Everything I attribute to it is from that page and must be verified before it is cited in the Research Doc.
- **verl on a single L4 with a 0.5B model was not benchmarked.** The compute-accessibility judgement is inferred from example configs and Paper 3's reported hardware.

### Unresolved questions for the team (Slack, per CLAUDE.md — do not decide silently)

1. **Base vs Instruct.** Still open in `pilot_config.json`. Paper 3 shows the SFT/instruct history of a checkpoint materially changes its RL trainability — evidence that this choice is not cosmetic.
2. **GRPO vs SFT for Stage B.** Paper 4's probe is supervised next-token learning; Paper 3's is RL. Both are defensible operationalisations of "adaptability" and they may not agree.
3. **Is SVAMP too close to GSM8K?** Paper 4 chose Vietnamese *specifically* to minimise transfer, and reported that transfer differences between adjacent languages were large enough to swamp the plasticity effect until they restricted probing to similar-transfer languages (§V-A, p. 5; Appendix B, Table III). **If SVAMP transfers strongly from GSM8K, our Stage-B signal could be dominated by transfer, exactly as theirs was.** This raises the priority of the question considerably.
4. **KL β > 0 baseline.** Paper 2 exists entirely because of the β trade-off, and its Fig. 1(a) shows an extremely sharp Pareto frontier in β. If we ever run β > 0, PCR becomes relevant; at β = 0 it is a no-op.
5. **Do we adopt AUC as a co-primary outcome?** My recommendation is yes (§9 #7b), but it changes the pre-registered outcome definition and therefore needs Tommy's sign-off.
6. **Do we re-measure all existing checkpoints at the larger probe size?** My recommendation is yes, before any Q number is reported outside the team.

---

## 12. Source list

**Primary sources — the four papers (downloaded 2026-07-26, stored in `lit review/task3_core_papers/`).**

1. Yuan, M., Wang, Q., Ma, G., Sun, C., Li, B., Jin, X., Wang, Y., Yang, X., Zeng, W., Tao, D., Chen, J. *Plasticine: Accelerating Research in Plasticity-Motivated Deep Reinforcement Learning.* arXiv:2504.17490v2, 10 Feb 2026. 21 pp. — `https://arxiv.org/abs/2504.17490`
2. Qiang, W., Gu, Z., Zhou, J., Hu, J., Wang, J., Zheng, C., Xiong, H. *On the Plasticity and Stability for Post-Training Large Language Models.* arXiv:2602.06453v1, 6 Feb 2026. 10 pp. (appendices cited but absent) — `https://arxiv.org/abs/2602.06453`
3. Liu, R., Liu, J., Wan, X., Fu, Y., Pan, L. *When RL Fails after SFT: Rejuvenating Model Plasticity for Robust SFT-to-RL Handoff.* arXiv:2606.09932v1, 7 Jun 2026. 27 pp. — `https://arxiv.org/abs/2606.09932`
4. Hernandez-Garcia, J. F., Figliolia, T., Millidge, B. *Can Scale Save Us From Plasticity Loss in Large Language Models?* arXiv:2606.24752v1, 23 Jun 2026. 17 pp. Zyphra. — `https://arxiv.org/abs/2606.24752`

**Primary sources — code (inspected 2026-07-26).**

5. `github.com/RLE-Foundation/Plasticine` — MIT, 44 stars, last push 2026-02-09. Files read: `plasticine_metrics/rank.py`, `plasticine_metrics/units.py`, `plasticine_metrics/metrics.py`, `plasticine/ppo_continual_procgen_plasticine.py` (metric block, lines 351–372).
6. `github.com/huggingface/trl` — Apache-2.0; `docs/source/grpo_trainer.md` read for `GRPOConfig` parameters and reward-function signature.
7. `github.com/verl-project/verl` — Apache-2.0 (redirected from `volcengine/verl`); file tree inspected for GSM8K and GRPO examples, incl. `verl/utils/reward_score/gsm8k.py`, `examples/data_preprocess/gsm8k.py`, `verl/experimental/one_step_off_policy/shell/grpo_0.6b_gsm8k_fsdp2_2_6.sh`.
8. `github.com/OpenRLHF/OpenRLHF` — Apache-2.0; repository metadata only.
9. `github.com/huggingface/open-r1` — Apache-2.0; repository metadata only; last push 2026-04-02.
10. `github.com/hiyouga/EasyR1` — Apache-2.0; repository metadata only.

**Follow-up reading identified during the review (not reviewed here).**

11. Aphale, S., Liu, K. *SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR.* arXiv:2606.18487, v1 16 Jun 2026, v2 22 Jun 2026. **Abstract page only.** — `https://arxiv.org/abs/2606.18487`
12. Han, T., Bordt, S., Zhang, H., Kakade, S. *Weight Decay Improves Language Model Plasticity.* ICML 2026 (cited by both Paper 3 and Paper 4). — arXiv:2602.11137
13. Springer, J. M. et al. *Overtrained Language Models Are Harder to Fine-Tune.* ICML 2025 (Paper 4's main point of comparison at billion-parameter scale).
14. Sokar, G., Agarwal, R., Castro, P. S., Evci, U. *The Dormant Neuron Phenomenon in Deep Reinforcement Learning.* ICML 2023 — the origin of the dormancy score used by both Plasticine (Eq. 5) and Paper 4 (Eqs. 3–4).
15. Roy, O., Vetterli, M. *The Effective Rank: A Measure of Effective Dimensionality.* EUSIPCO 2007 — the origin of the effective-rank formula.

**Internal artefacts consulted.**

16. `eaaj-pilot/pilot_config.json` — the pre-registered recipe.
17. `eaaj-pilot/src/metrics.py` — our measurement implementation.
18. `eaaj-pilot/outputs/local_cuda_grpo_gsm8k_6a075c15808e/measurements/metrics_ckpt0.json` — the artefact establishing `d = 896`, `n_probe = 512`, and the identically-zero dormant fractions.
19. `eaaj-pilot/requirements.txt` — `trl==1.6.0`, `transformers==5.13.0`.
20. `eaaj-pilot-win4070/WIN4070_RUN_ANALYSIS.md` — the local precedent for a silently no-op training run.
