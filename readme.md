# 超声图像的智能筛查与分级任务技术说明文档

---

## 一、团队信息

- **团队名称**：川海  
- **A榜排名**：第 22 名

---

## 二、项目概述

### 项目核心任务

本项目旨在完成乳腺癌超声图像的六分类任务，标签反映乳腺病灶的良恶性概率。该任务面临类别分布极度不均衡与类别间特征相似度高等挑战，要求模型具有较强的泛化能力和对少数类的敏感性。

### 解决思路

为应对上述挑战，本项目从损失函数、数据增强、采样策略等多个方面设计了综合方案：

- 主干网络：采用 Swin Transformer（`swin_base_patch4_window12_384`）作为图像主干，结合其在视觉任务中的局部建模与全局感知能力；
- 损失函数：引入 RW-LDAM-DRW Loss，融合类别敏感 margin、类别频次重加权与延迟启用机制，以增强对尾类样本的判别能力；
- 数据增强：基于 Albumentations 实现多种图像增强策略（翻转、旋转、遮挡等），提升模型鲁棒性；
- 类别平衡：结合手动设定的类别过采样比例，有效缓解数据不平衡问题；
- 模型稳定性：引入 Exponential Moving Average（EMA）机制，平滑模型参数，提升推理稳定性；
- 评估方法：采用 K 折交叉验证，结合多模型融合策略，提高最终模型性能评估的可靠性。

---

## 三、代码结构说明

```plaintext
Mycode1/
├── cfgs/                  # 数据增强和损失函数等配置文件
├── checkpoints/           # 三个checkpoint的模型文件
├── code/                  # 核心训练与模型文件
│   ├── jimm/              # Swin模型实现
│   ├── utils/             # 工具函数
│   └── ...                # checkpoint对应的训练和推理代码
├── results/               # 结果保存的目录
└── ...                    # 运行脚本
```

---

## 四、环境配置步骤

- 已经在镜像内python环境下默认安装。如失效，按照environment.yml重建conda环境

---

## 五、运行步骤说明

- 首先进入项目文件夹 `cd /root/workspace/Mycode1`

- 如需完整训练，则运行 `/root/workspace/Mycode1/run_checkpoint1.sh` ，其他checkpoint同理

- 输入输出说明：输入位于 `/root/workspace/TrainSet` 和 `/root/workspace/TestSetA` ；输出位于 `/root/workspace/Mycode1/results/对应的checkpoint`

- 如只执行推理，则运行 `/root/workspace/Mycode1/run_checkpoint1_testonly.sh`

- 输入输出说明：输入路径与训练时相同 ；输出路径变为 `/root/workspace/Mycode1/checkpoints/对应的checkpoint`

- B榜测试时，如果使用需更改 `/root/workspace/Mycode1/cfgs/basicv2.yml` 下的 `test_path`；`run_checkpoint1.sh` 和 `run_checkpoint2.sh` 下的 `test_dir` 以及 `run_checkpoint3.sh` 下的 `dataroot`

---

## 六、Checkpoint说明

### Checkpoint1:

- **模型结构**：Swin Transformer，输入尺寸 384×384
- **训练轮次**：100 轮
- **训练方式**：4折交叉验证
- **损失函数**：
  - 使用 RW-LDAM-DRW（Re-Weighted Label-Distribution-Aware Margin Loss + Deferred Re-Weighting）
  - 第 60 轮后启用类别重加权，缓解类别不平衡
- **数据增强与采样**：
  - 使用增强配置文件：`enhanced_custom_transforms_384.yml`
  - 应用类别过采样策略：在每个batch中，把少数类采样到一定倍数。
- **学习率策略**：
  - Warmup：前 5 个 epoch 学习率线性增长
  - Cosine Annealing：剩余轮次使用 cosine 衰减至 `final_lr=5e-6`
- **优化器**：AdamW，初始学习率 `1e-4`
- **EMA**：启用 Exponential Moving Average，衰减率 `0.999`
- **其他参数**：
  - 批大小：22
  - 随机种子：42，确保可复现性
- **A榜性能**：0.8895

### Checkpoint2:

- **模型结构**：Swin Transformer，输入尺寸 384×384
- **训练轮次**：100 轮
- **训练方式**：4折交叉验证
- **损失函数**：
  - 使用 RW-LDAM-DRW（Re-Weighted Label-Distribution-Aware Margin Loss + Deferred Re-Weighting）
  - 第 60 轮后启用类别重加权，缓解类别不平衡
- **数据增强与采样**：
  - 使用增强配置文件：`enhanced_custom_transforms_384.yml`
  - 应用类别过采样策略：使用手动设置的过采样比例
- **学习率策略**：
  - Warmup：前 5 个 epoch 学习率线性增长
  - Cosine Annealing：剩余轮次使用 cosine 衰减至 `final_lr=5e-6`
- **优化器**：AdamW，初始学习率 `1e-4`
- **EMA**：启用 Exponential Moving Average，衰减率 `0.999`
- **其他参数**：
  - 批大小：22
  - 随机种子：42，确保可复现性
- **A榜性能**：0.8835

### Checkpoint3:

- **模型结构**：Swin Transformer，输入尺寸 384×384
- **训练轮次**：100 轮
- **训练方式**：无交叉验证，默认数据集划分
- **损失函数**：
  - 使用 CrossEntropyLoss
- **数据增强与采样**：
  - 使用增强配置文件：`utils/transforms.py`
- **学习率策略**：
  - Warmup：前 5 个 epoch 学习率线性增长
- **优化器**：AdamW，初始学习率 `1e-4`
- **其他参数**：
  - 批大小：16
  - 随机种子：42，确保可复现性
- **A榜性能**：0.8650


---

## 七、其他补充说明