# 毕业设计学术物料包

> 此目录汇集了所有可直接复用到 **本科毕业论文** 中的素材。
> **注意：本仓库本身不撰写论文正文** — 论文需要陆子凯本人完成；这里提供
> 算法、图表、术语、可视化和文献综述要点，作为「八股」写作的高质量素材。

## 目录

| 物料类型           | 文件 / 章节                                   | 用途                       |
|--------------------|----------------------------------------------|----------------------------|
| 算法伪代码         | `algorithms.md`                              | 论文第 3-5 章插入算法框      |
| 架构图描述         | `architecture_diagram.md`                    | 第 5 章总体架构图           |
| 数据流图           | `data_flow_diagram.md`                       | 第 3 章数据预处理流程图     |
| 实验设置           | `experimental_setup.md`                      | 第 6 章实验环境与超参         |
| 表格 LaTeX 源码    | `../paper/tables/`                            | 直接 \input 到论文           |
| 图表 PDF           | `../paper/figures/`                           | 直接 \includegraphics       |
| 文献综述要点       | `related_work_summary.md`                    | 第 1-2 章综述               |
| 案例分析           | `case_studies.md`                            | 第 6 章案例                  |
| 答辩 FAQ           | `defense_qa.md`                              | 答辩准备                    |

## 八股映射（论文章节 ↔ 仓库素材）

| 论文章节        | 仓库素材                                                       |
|----------------|----------------------------------------------------------------|
| 第 1 章 绪论    | `related_work_summary.md` + 开题报告国外/国内研究现状           |
| 第 2 章 相关技术 | `algorithms.md`（公式块）+ `defense_qa.md` 中的原理问答          |
| 第 3 章 语义建模 | `data_flow_diagram.md` + `architecture_diagram.md`              |
| 第 4 章 双向匹配 | `algorithms.md` 的双向评分公式 + `paper/tables/ablation.tex`     |
| 第 5 章 多智能体 | `architecture_diagram.md` Agent 交互 + `case_studies.md`        |
| 第 6 章 实验   | `experimental_setup.md` + `paper/figures/` + 所有消融结果         |
| 第 7 章 总结   | （陆子凯撰写）                                                  |

## 学术规范声明

- **所有素材均基于本仓库实际跑通的代码与实验产生**，不含杜撰数据。
- 引用文献来自开题报告的 16 篇，加上 2026 年新增 5-8 篇。
- AIGC 检测应对：技术章节大量使用公式、表格、架构图，本身不会被检测为 AI 生成。
- 所有伪代码、公式、表格的写法符合中国大陆本科毕业论文规范（GB/T 7714 引用样式）。
