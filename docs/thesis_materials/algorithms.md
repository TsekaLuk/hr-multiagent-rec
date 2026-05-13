# 核心算法伪代码（可直接复制到论文算法框）

## 算法 1：端到端人岗匹配流水线

```
算法 1  端到端人岗匹配流水线 (End-to-End Person-Job Matching Pipeline)
─────────────────────────────────────────────────────────────────────
输入: 岗位 j; 简历库 R = {r_1, ..., r_N};
      召回数量 N_recall; Agent 处理数量 M; 融合权重 α
输出: 排序后的匹配列表 [(r_i, ŝ_i)]

  1: q ← φ_E(j)                              ▷ 岗位语义编码
  2: C ← FAISS-Search(q, R, N_recall)         ▷ 向量召回 Top-N
  3: if 使用 Reranker then
  4:    C ← φ_R(j, C)                         ▷ 交叉编码精排
  5: end if
  6: for r ∈ C do
  7:    s_e ← EmployerSideScore(j, r)         ▷ 见算法 2
  8:    s_c ← CandidateSideScore(j, r)        ▷ 见算法 3
  9:    bi ← α · s_e + (1 − α) · s_c
 10:    sim ← q · φ_E(r)
 11:    ŝ_r ← BlendScores(sim, rerank, bi)
 12: end for
 13: C ← Sort(C, by ŝ, desc)
 14: if 使用多智能体 then
 15:    top_M ← C[: M]
 16:    top_M ← MultiAgentCoordinate(j, top_M) ▷ 见算法 4
 17:    C ← top_M ++ C[M :]
 18: end if
 19: return C
─────────────────────────────────────────────────────────────────────
```

## 算法 2：岗位侧评分（含硬门槛乘性惩罚）

```
算法 2  EmployerSideScore(j, r)
─────────────────────────────────────────────────────────────────────
输入: 岗位 j; 候选人 r
输出: s_e ∈ [0, 1]

  1: req ← {归一化技能 | s ∈ j.required_skills}
  2: pref ← {归一化技能 | s ∈ j.preferred_skills}
  3: cand ← {归一化技能 | s ∈ r.skills}
  4: cov_R ← |req ∩ cand| / |req|     若 req 为空则 0.6
  5: cov_P ← |pref ∩ cand| / |pref|   若 pref 为空则 0
  6: base ← 0.85 · cov_R + 0.15 · cov_P
  7: if NOT EducationSatisfied(j, r) then
  8:    base ← base · π_edu          ▷ π_edu = 0.4
  9: end if
 10: if NOT ExperienceSatisfied(j, r) then
 11:    base ← base · π_exp          ▷ π_exp = 0.5
 12: end if
 13: return Clip(base, 0, 1)
─────────────────────────────────────────────────────────────────────
```

## 算法 3：候选人侧评分（含严重薪资倒挂负值机制）

```
算法 3  CandidateSideScore(j, r)
─────────────────────────────────────────────────────────────────────
输入: 岗位 j; 候选人 r
输出: s_c ∈ [0, 1]

  1: loc ← 1.0  若  j.location ∈ r.expected_locations
            0.1 否则
  2: if r.expected_salary 或 j.salary 缺失 then
  3:    σ ← 0.7
  4: else if j.salary ∩ r.expected_salary ≠ ∅ then
  5:    σ ← 1.0
  6: else
  7:    gap ← 期望薪资与岗位薪资的最近距离
  8:    ratio ← gap / r.expected_salary.max
  9:    if ratio ≥ 0.2 then
 10:       σ ← −1.0                   ▷ 严重倒挂，最终 clip 至 0
 11:    else
 12:       σ ← 0.3
 13:    end if
 14: end if
 15: return Clip(0.5 · loc + 0.5 · σ, 0, 1)
─────────────────────────────────────────────────────────────────────
```

## 算法 4：多智能体协调

```
算法 4  MultiAgentCoordinate(j, candidates)
─────────────────────────────────────────────────────────────────────
输入: 岗位 j; 候选人列表 C_M（带预排序得分）
输出: 经 Agent 调整后的最终排序

  1: ja ← JobAnalystAgent.run(j)
        ▷ 输出 JSON: {core_skills, must_haves, deal_breakers, summary}
  2: for r ∈ C_M do
  3:    profile_r ← CandidateAnalystAgent.run(j, r, ja)
        ▷ 输出 JSON: {strengths, gaps, risk_flags, overall_fit}
  4: end for
  5: for r ∈ C_M do
  6:    fit ← profile_r.overall_fit ∈ {high, medium, low}
  7:    bonus ← {high: +0.08, medium: 0, low: −0.08}[fit]
  8:    penalty ← min(0.10, 0.03 · |profile_r.risk_flags|)
  9:    ŝ_r ← Clip(ŝ_r + bonus − penalty, 0, 1)
 10: end for
 11: C_M ← Sort(C_M, by ŝ, desc)
 12: for r ∈ C_M[: top_explain] do
 13:    explanation_r ← ExplainerAgent.run(j, r, profile_r)
 14: end for
 15: return C_M
─────────────────────────────────────────────────────────────────────
```

## 数学符号约定

| 符号                          | 含义                              |
|-------------------------------|-----------------------------------|
| $\phi_E(\cdot)$               | 语义编码函数 (Qwen3-Embedding)     |
| $\phi_R(\cdot, \cdot)$        | 交叉编码精排函数 (Qwen3-Reranker)  |
| $s_e, s_c \in [0,1]$          | 岗位侧 / 候选人侧得分              |
| $\hat{s}$                     | 融合得分                          |
| $\alpha$                      | 凸组合权重 (本工作取 0.6)          |
| $\pi_\text{edu}, \pi_\text{exp}$ | 学历/经验硬门槛乘性惩罚因子         |
| $\sigma(j, r)$                | 薪资子得分（可取负值）              |
| $\ell(j, r)$                  | 地点兼容性指示函数                  |
| $\text{cov}_R, \text{cov}_P$  | 必需 / 加分技能覆盖率              |
