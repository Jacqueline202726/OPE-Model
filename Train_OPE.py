import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import roc_auc_score
import csv
from Models import PLE
from Models_OPE import OPE
from Train_PLE import device, test, batch_size, data_preparation, getTensorDataset, val_loader, train_label_tmp, validation_label_tmp, test_label_tmp 


def map_features_to_columns(train_df, validation_data, label_columns, categorical_columns):
    """
    构造原始特征到 One-Hot 后列的映射。

    参数：
    - train_df: 原始训练数据集
    - validation_data: 验证数据集
    - label_columns: 目标标签列
    - categorical_columns: 需要进行 One-Hot 编码的分类特征列

    返回：
    - feature_to_columns: 一个字典，映射原始特征到 One-Hot 后的列
    """
    original_features = list(train_df.drop(label_columns, axis=1).columns)
    feature_to_columns = {}
    for feature in original_features:
        if feature in categorical_columns:
            cols = [col for col in validation_data.columns if col.startswith(feature + "_")]
            if cols:
                feature_to_columns[feature] = cols
        else:
            if feature in validation_data.columns:
                feature_to_columns[feature] = [feature]
    return feature_to_columns

def generate_masked_validation_data(validation_data, feature_to_columns, categorical_columns):
    """
    生成掩盖特征的验证集列表。

    参数：
    - validation_data: 验证数据集
    - feature_to_columns: 一个字典，映射原始特征到 One-Hot 后的列
    - categorical_columns: 需要进行 One-Hot 编码的分类特征列

    返回：
    - masked_validation_data_list: 包含元组 (原始特征名称, 掩盖该特征后的验证集) 的列表
    """
    masked_validation_data_list = []
    for feature, cols in feature_to_columns.items():
        masked_validation = validation_data.copy()
        if feature not in categorical_columns:
            masked_validation[cols] = masked_validation[cols[0]].mean()
        else:
            masked_validation[cols] = 0
        masked_validation_data_list.append((feature, masked_validation))
    
    return masked_validation_data_list

def compute_feature_importance(model_path, val_loader, masked_validation_data_list, validation_label_tmp, batch_size, device):
    """
    计算特征重要性。

    参数：
    - model_path: 训练好的模型路径
    - val_loader: 原始验证集的 DataLoader
    - masked_validation_data_list: 包含 (原始特征名称, 掩盖该特征后的验证集) 的列表
    - validation_label_tmp: 验证集对应的标签
    - batch_size: DataLoader 的批量大小
    - device: 设备 (CPU 或 GPU)

    返回：
    - importance_results: 一个包含特征重要性的列表
    """
    # 初始化模型
    model = PLE(num_CGC_layers=4, input_size=499, num_specific_experts=4, num_shared_experts=4, 
                experts_out=32, experts_hidden=32, towers_hidden=8)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 计算基准 AUC
    baseline_auc1, baseline_auc2 = test(val_loader)

    # 计算特征重要性
    importance_results = []
    for feature, masked_validation in masked_validation_data_list:
        masked_loader = DataLoader(
            dataset=getTensorDataset(masked_validation.to_numpy(), validation_label_tmp),
            batch_size=batch_size
        )
        auc1_masked, auc2_masked = test(masked_loader)
        importance_results.append({
            "Feature": feature,
            "Importance1": baseline_auc1 - auc1_masked,
            "Importance2": baseline_auc2 - auc2_masked
        })
    
    return importance_results

def get_top_features(importance_results, top_n=5):
    """
    从特征重要性结果中提取前 top_n 个重要特征。

    参数：
    - importance_results: 特征重要性列表
    - top_n: 需要选择的最重要特征数量

    返回：
    - top_features_task1: 任务1的前 top_n 重要特征列表
    - top_features_task2: 任务2的前 top_n 重要特征列表
    """
    importance_df = pd.DataFrame(importance_results)
    top5_task1 = importance_df.sort_values(by="Importance1", ascending=False).head(top_n)
    top5_task2 = importance_df.sort_values(by="Importance2", ascending=False).head(top_n)
    
    top_features_task1 = top5_task1["Feature"].tolist()
    top_features_task2 = top5_task2["Feature"].tolist()

    return top_features_task1, top_features_task2

def extract_features_from_dataframe(df, top_features, feature_to_columns):
    """
    根据给定的特征列表，从数据集中提取对应的列。

    参数：
    - df: pandas DataFrame，输入数据集
    - top_features: 重要特征列表
    - feature_to_columns: 原始特征到 One-Hot 处理后列的映射字典

    返回：
    - extracted_df: 提取后的 DataFrame
    """
    extracted_df = pd.DataFrame()
    for feature in top_features:
        if feature in feature_to_columns:
            extracted_df = pd.concat([extracted_df, df[feature_to_columns[feature]]], axis=1)
        else:
            print(f"Warning: {feature} 不在映射中！")
    return extracted_df

def create_dataloader(full_data, task1_data, task2_data, labels, batch_size, shuffle):
    """
    构造 TensorDataset 和 DataLoader。

    参数：
    - full_data: 完整输入数据
    - task1_data: 任务1的特征数据
    - task2_data: 任务2的特征数据
    - labels: 目标标签
    - batch_size: DataLoader 批量大小
    - shuffle: 是否打乱数据

    返回：
    - DataLoader 对象
    """
    tensor_full = torch.Tensor(full_data.to_numpy().astype(np.float32))
    tensor_task1 = torch.Tensor(task1_data.to_numpy().astype(np.float32))
    tensor_task2 = torch.Tensor(task2_data.to_numpy().astype(np.float32))
    tensor_y = torch.Tensor(labels)

    dataset = TensorDataset(tensor_full, tensor_task1, tensor_task2, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def test_ope(model, loader):
    t1_pred, t2_pred, t1_target, t2_target = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, x1, x2, y in loader:
            x, x1, x2, y = x.to(device), x1.to(device), x2.to(device), y.to(device)
            yhat = model(x, x1, x2)
            y1, y2 = y[:, 0], y[:, 1]
            yhat_1, yhat_2 = yhat[0], yhat[1]
            loss = loss_fn(yhat_1, y1.view(-1, 1)) + loss_fn(yhat_2, y2.view(-1, 1))
            t1_pred += list(yhat_1.cpu().numpy())
            t2_pred += list(yhat_2.cpu().numpy())
            t1_target += list(y1.cpu().numpy())
            t2_target += list(y2.cpu().numpy())
    auc_1 = roc_auc_score(t1_target, t1_pred)
    auc_2 = roc_auc_score(t2_target, t2_pred)
    return auc_1, auc_2

# ---------------------------
# 重要特征提取
# ---------------------------
# 所有列标签
column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']
# 目标列标签
label_columns = ['income_50k', 'marital_stat'] 
# 类别列标签（与数值列标签相对）
categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                        'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                        'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                        'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                        'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                        'vet_question']
# 数据预处理
train_df = pd.read_csv('census-income.data.gz', delimiter=',', header=None, index_col=None, names=column_names)
test_df = pd.read_csv('census-income.test.gz', delimiter=',', header=None, index_col=None, names=column_names)
train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preparation()
# 提取所有特征对应的 one-hot 之后的列标签
feature_to_columns = map_features_to_columns(train_df, validation_data, label_columns, categorical_columns)
# 生成 (原始特征名称, 掩盖该特征后的验证集) 的列表
masked_validation_data_list = generate_masked_validation_data(validation_data, feature_to_columns, categorical_columns)
# 调用训练后的PLE模型，计算对于两个任务的特征重要性
importance_results = compute_feature_importance(
    model_path="model.pth",
    val_loader=val_loader,
    masked_validation_data_list=masked_validation_data_list,
    validation_label_tmp=validation_label_tmp,
    batch_size=batch_size,
    device=device
)
top_features_task1, top_features_task2 = get_top_features(importance_results, top_n=5)

# 训练集
task1_train_data = extract_features_from_dataframe(train_data, top_features_task1, feature_to_columns)
task2_train_data = extract_features_from_dataframe(train_data, top_features_task2, feature_to_columns)
train_loader = create_dataloader(train_data, task1_train_data, task2_train_data, train_label_tmp, batch_size, shuffle=True)

# 验证集
task1_val_data = extract_features_from_dataframe(validation_data, top_features_task1, feature_to_columns)
task2_val_data = extract_features_from_dataframe(validation_data, top_features_task2, feature_to_columns)
val_loader = create_dataloader(validation_data, task1_val_data, task2_val_data, validation_label_tmp, batch_size, shuffle=False)

# 测试集
task1_test_data = extract_features_from_dataframe(test_data, top_features_task1, feature_to_columns)
task2_test_data = extract_features_from_dataframe(test_data, top_features_task2, feature_to_columns)
test_loader = create_dataloader(test_data, task1_test_data, task2_test_data, test_label_tmp, batch_size, shuffle=False)

# ---------------------------
# 训练过程
# ---------------------------
model_ope = OPE(
    input_size_full=499,
    input_size_task1=task1_train_data.shape[1],  # tensor_task1.shape[1],
    input_size_task2=task2_train_data.shape[1],  # tensor_task2.shape[1],
    emb_dim_full=128,
    emb_dim_task1=64,
    emb_dim_task2=64,
    num_CGC_layers=4,
    num_specific_experts=4,
    num_shared_experts=4,
    experts_out=32,
    experts_hidden=32,
    towers_hidden=8
)
model_ope = model_ope.to(device)

lr = 1e-4
n_epochs = 100
loss_fn = nn.BCELoss(reduction='mean')
optimizer = optim.Adam(model_ope.parameters(), lr=lr, weight_decay=1e-5)
losses = []

with open("OPE_results.csv", "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Epoch", "Train Loss", "Val Task1 AUC", "Val Task2 AUC"])
    for epoch in range(n_epochs):
        model_ope.train()
        epoch_loss = []
        print("Epoch: {}/{}".format(epoch, n_epochs))
        for x, x1, x2, y in train_loader:
            x, x1, x2, y = x.to(device), x1.to(device), x2.to(device), y.to(device)
            y_hat = model_ope(x, x1, x2)
            y1, y2 = y[:, 0], y[:, 1]
            loss = loss_fn(y_hat[0], y1.view(-1, 1)) + loss_fn(y_hat[1], y2.view(-1, 1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss.append(loss.item())
        avg_loss = np.mean(epoch_loss)
        auc1, auc2 = test_ope(model_ope, val_loader)
        print('Epoch {} - train loss: {:.5f}, val task1 auc: {:.5f}, val task2 auc: {:.5f}'
              .format(epoch, avg_loss, auc1, auc2))
        csvwriter.writerow([epoch, avg_loss, auc1, auc2])
    auc1, auc2 = test_ope(model_ope, test_loader)
    print('Test Task1 AUC: {:.3f}, Test Task2 AUC: {:.3f}'.format(auc1, auc2))
    torch.save(model_ope.state_dict(), "model_ope.pth")
