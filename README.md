# OPE: Optimized Private Expert Model

## 1. 项目简介 (Introduction)

**OPE (Optimized Private Expert)** 是对 **PLE (Progressive Layered Extraction)** 模型的优化版本。在原始 PLE 模型的基础上，我们为每个专家塔单独分配了输入，实现了如下优化：

- **✅ 特征筛选**：利用 PLE 模型计算特征重要性，筛选出优选特征。
- **✅ 独立嵌入 (Embedding)**：将优选特征独立于全特征进行嵌入，提升特征表示效果。
- **✅ 输入优化**：全特征与优选特征分别作为 OPE 网络的输入，分别进入任务专家塔和共享专家塔进行训练。
- **✅ 性能提升**：相比 PLE 模型，OPE 在离线 AUC 评估上表现更优。

---

## 2. 模型结构 (Model Architecture)

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1ujOe9c4M2krGTgOY9eRMyPyc3A71a9I7" width="45%">
  <img src="https://drive.google.com/uc?export=view&id=1BFr9xGFewjCMJyIyq3X4dy3a14z20U59" width="48%">
</div>

### PLE（左） 与 OPE（右） 模型对比

- **PLE 模型**：将专家塔分为 **任务特定专家塔** 和 **共享专家塔**，并通过 **门控网络** 控制各专家对输出的贡献。
- **OPE 模型**：在此基础上，为每个任务的专家塔单独分配优选特征作为输入。这些优选特征独立于原始特征进行嵌入，能够更好地提取任务特征，从而有效缓解任务间的负迁移问题。

#### 特征优选方法

我们采用 **特征重要性排序** 方法进行特征筛选，具体步骤如下：

1. 在训练好的 PLE 模型上，计算全特征验证集上所有任务的 AUC，得到 `baseline_auc`。
2. 依次掩盖每个特征，再次计算所有任务的 AUC，得到 `feature_auc`。
3. 计算特征重要性：  
   $$
   \text{importance} = \text{baseline_auc} - \text{feature_auc}
   $$
4. 根据特征重要性排序，分别选出对每个任务最重要的 n 个特征。

 **💡 提示**：该方法能有效筛选出对任务贡献最大的特征，从而提升 OPE 模型在多任务学习中的表现。


---

## 3. 训练与评估 (Training and Evaluation)

### 环境依赖

- **✅ Python 3.x**
- **✅ PyTorch**
- **✅ NumPy**
- **✅ Pandas**
- **✅ Scikit-Learn**

### 训练 PLE 和 OPE 模型

在终端或命令行中依次执行以下命令：

```bash
cd OPE
pip install -r requirements.txt
python Train_PLE.py
python Train_OPE.py
```
⚠️ 注意：

请确保所有依赖项已正确安装。
根据需要调整训练参数以获得最佳效果。

---

## 4. 结果对比 (Results Comparison)

### 离线 AUC 评估

| Model | Task 1 AUC | Task 2 AUC |
|-------|------------|------------|
| PLE   | 0.xx       | 0.xx       |
| OPE   | 0.xx       | 0.xx       |

🚀 实验结果：实验结果表明，相比 PLE 模型，OPE 通过优化特征输入方式显著提高了 AUC 评分，验证了改进方法的有效性。
