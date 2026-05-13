# 数据流图（论文第 3 章）

## 从原始招聘数据到匹配排序

```
原始数据                清洗 + 标准化           语义编码         向量检索      匹配评分        Agent 协同     最终输出
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Tianchi 智联招聘 ─┐                                                                                          
                 ├── Pydantic Schema ───┐                                                                    
Job-SDF 技能时序 ──┤   (Resume, Job)    │                                                                    
                 │                       │                                                                    
ESCO 技能本体 ────┤   ⇩ 字段验证          │     ⇩ Qwen3-Emb       ⇩ FAISS         ⇩ 双向         ⇩ 4 Agent    
                 │                       ├─→  φ_E(text)  ─────→ Top-N candidates ─→ (s_e, s_c) ─→ 重排序  ─→  Top-K 排序
合成 Tianchi-style ┘                     │     dim=1024                                  ⇩                    + 解释
                                         │                                          ⇩ Qwen3-Reranker          
                                         │                                          φ_R(j, r)                
                                         │                                          交叉编码                  
                                         │                                                                    
                                         └──→ Skill 归一化 (大小写、空格、ESCO 映射) ──────────────────────────
```

## 数据契约边界

```
                     ┌────────────────────────────────────────────┐
   外部数据源 ───→   │  Pydantic Validator (hr_rec.data.schemas)  │
   (CSV / JSON /     │  ├── Resume      ┌──────────────────────┐  │
    LLM 输出)        │  ├── Job          │ 不变量检查：           │  │
                     │  ├── Skill        │  • salary.min ≤ max  │  │
                     │  ├── SalaryRange  │  • dates 顺序         │  │
                     │  └── MatchScore   │  • 单位区间 [0, 1]     │  │
                     │                   │  • Unicode 保留        │  │
                     │                   │  • 重复技能去重         │  │
                     │                   └──────────────────────┘  │
                     │                                              │
                     │  失败 → 抛 ValidationError, 拒绝入库          │
                     └────────────────────────────────────────────┘
                                          │
                                          ▼ 通过验证后
                              ┌─────────────────────────┐
                              │  内部 Python 对象        │
                              │  (immutable dataclasses) │
                              └─────────────────────────┘
```

## 实验数据生成（确定性 + 可复现）

```
随机种子 seed=42  →  hashlib.sha256(...) → 子种子                            
        │                                                                    
        ├─→ make_resume("seed-i") → Resume 对象 ┐                              
        │                                       ├─→ build_corpus(n_jobs, n_resumes)
        ├─→ make_job("seed-i") → Job 对象       ┤                              
        │                                                                    
        └─→ make_ground_truth_pairs(jobs, resumes) → [(job_id, resume_id, rel)]
              │
              ├─ rel=2 (强): ≥3 技能重叠 ∧ 地点 ∧ 薪资重叠
              ├─ rel=1 (弱): ≥2 技能重叠
              └─ rel=0 (无): 其他
```
