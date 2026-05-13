# 实验设置（论文第 6 章直接复用）

## 硬件 / 软件环境

| 类型     | 配置                                                     |
|----------|---------------------------------------------------------|
| 处理器   | Apple M4 (10 核 CPU)                                     |
| 内存     | 16 GB 统一内存                                            |
| 存储     | 256 GB NVMe SSD                                          |
| 操作系统 | macOS 26.x                                               |
| Python   | 3.12.x                                                   |
| PyTorch  | 2.x (MPS 后端)                                            |
| 框架     | sentence-transformers ≥ 3.0, faiss-cpu ≥ 1.8             |
| 镜像源   | ModelScope (国内)；HF 备选                                 |

## 模型选型

| 用途           | 模型                              | 参数量 | 提供方     | 角色          |
|---------------|-----------------------------------|-------:|-----------|--------------|
| 语义编码       | Qwen3-Embedding-0.6B              |  0.6B  | 阿里通义   | 召回向量      |
| 交叉编码       | Qwen3-Reranker-0.6B               |  0.6B  | 阿里通义   | 精排打分      |
| Agent 推理     | Qwen/Qwen3-8B                     |  8B    | SiliconFlow | 主 LLM       |
| 对比实验 LLM   | Gemini 2.5 Flash                   |   –    | Google    | 闭源对照      |
| 对比实验 LLM   | DeepSeek-V3                        |   –    | DeepSeek  | 国产对照      |

## 数据集

| 数据集               | 规模                          | 用途             | 来源                  |
|---------------------|-------------------------------|-----------------|----------------------|
| 合成 Tianchi-style  | 300 岗位 × 800 简历 → 24,145 正对 | 主消融            | 仓库 `data/synthetic/` |
| Tianchi 智联招聘     | dataset 31623（真实）           | 现实数据验证      | tianchi.aliyun.com    |
| Job-SDF             | 千万级技能时序                  | 技能本体辅助      | GitHub Job-SDF        |
| ESCO                | 13K+ 技能                      | 技能归一化        | esco.ec.europa.eu     |

## 超参（统一种子 seed=42）

| 超参                  | 值       | 说明                              |
|----------------------|---------|-----------------------------------|
| α (凸组合权重)        | 0.6     | 岗位侧主导                          |
| Top-N 召回           | 50      | FAISS 第一阶段                     |
| Top-M 进入 Agent     | 15      | 控制 LLM 调用成本                   |
| FAISS index          | flat    | ≤5k 简历精确检索；否则 IVF nlist=√N |
| nprobe               | nlist/2 | IVF 召回率/延迟折中                  |
| 嵌入维度             | 1024（默认） / 256, 128（MRL 实验）  |
| LLM temperature      | 0.2     | 控制 Agent 输出稳定性                |
| LLM max_tokens       | 2048    | JSON 输出预算                        |
| 学历惩罚因子 π_edu    | 0.4     | 多次失败下乘性惩罚                  |
| 经验惩罚因子 π_exp    | 0.5     | 同上                                |
| 严重薪资倒挂阈值      | 20%     | gap / expected_max                 |

## 评测指标

- **P@K (Precision@K)**：top-K 中相关候选人比例
- **R@K (Recall@K)**：在 top-K 中找到的相关候选人 / 全部相关候选人
- **nDCG@K**：考虑等级（强匹配=2, 弱匹配=1, 不匹配=0）的归一化折损累计增益
- **MRR (Mean Reciprocal Rank)**：第一个相关结果倒数排名的平均

## 消融实验设计（10 组）

| 编号 | 配置                | 编码器 | 精排  | 双向 | Agent | LLM        |
|-----|--------------------|------|------|------|------|-----------|
| 1   | BM25               | -    | -    | -    | -    | -         |
| 2   | TF-IDF             | -    | -    | -    | -    | -         |
| 3   | 仅语义              | Qwen3| -    | -    | -    | -         |
| 4   | 语义+精排           | Qwen3| Q-R  | -    | -    | -         |
| 5   | 语义+双向           | Qwen3| -    | ✓    | -    | -         |
| 6   | 语义+精排+双向       | Qwen3| Q-R  | ✓    | -    | -         |
| 7   | **完整 (本工作)**    | Qwen3| Q-R  | ✓    | ✓    | Qwen3-8B  |
| 8   | MRL 256 维          | Qwen3| Q-R  | ✓    | ✓    | Qwen3-8B  |
| 9   | MRL 128 维          | Qwen3| Q-R  | ✓    | ✓    | Qwen3-8B  |
| 10  | 换 LLM = Gemini      | Qwen3| Q-R  | ✓    | ✓    | Gemini    |

## 复现

```bash
make data       # 生成确定性合成语料 (seed=42)
make eval       # 跑完整消融网格 (~25 min)
make paper      # 编译 arXiv 论文 PDF
```
