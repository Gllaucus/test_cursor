import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
# 设置显示选项，防止省略
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行（谨慎使用，数据多时会刷屏）
pd.set_option('display.width', None)        # 自动换行，适应终端宽度
pd.set_option('display.colheader_justify', 'left')  # 左对齐列名
pd.set_option('display.float_format', '{:.2f}'.format)  # 浮点数显示格式（可选）

# ================== 设置随机种子（确保结果可复现）==================
def set_seed(seed=42):
    """设置所有随机种子以确保实验可复现"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU
        # 确保CUDA运算的确定性（可能牺牲性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ 随机种子已设置为: {seed}")

# 调用种子设置
set_seed(42)

# 读取CSV文件
file_path = r'E:\研一\ftec人工智能\final\data\neural_network_input_with_geographic_score.csv'
df = pd.read_csv(file_path)

# 显示数据的基本信息
print("数据集形状:", df.shape)
print("\n前几行数据:")
print(df.head())

# 检查各列的数据类型
print("\n数据类型:")
print(df.dtypes)

# 检查缺失值
print("\n缺失值数量:")
print(df.isnull().sum())

# （可选）检查是否有完全重复的行
print("\n重复行数量:", df.duplicated().sum())

# （可选）描述性统计
print("\n描述性统计:")
print(df.describe())

# ================== 第一步：使用 dataset_split 划分数据集 ==================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 确保 dataset_split 列存在
assert 'dataset_split' in df.columns, "❌ 数据中缺少 'dataset_split' 列！"

# 确保 date 是 datetime 类型（用于提取 month）
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# 特征列表（必须与后续一致）
selected_features = [
    'servicecharge_scaled',
    'picturecount_scaled',
    'telekomuploadspeed_scaled',
    'age_scaled',
    'floor_scaled',
    'convenience_shap_sum',
    'housing_quality_score',
    'geographic_score',
    'month_sin',
    'month_cos'
]

# 检查所有特征是否存在
missing_feats = [f for f in selected_features if f not in df.columns]
if missing_feats:
    raise KeyError(f"缺失特征: {missing_feats}")

# 根据 dataset_split 划分
df_train = df[df['dataset_split'] == 'train']
df_val   = df[df['dataset_split'] == 'val']
df_test  = df[df['dataset_split'] == 'test']

print(f"📌 使用预定义 dataset_split 划分:")
print(f"   - 训练集: {len(df_train)} 样本")
print(f"   - 验证集: {len(df_val)} 样本")
print(f"   - 测试集: {len(df_test)} 样本")

# ================== 对指定特征进行标准化（仅使用训练集统计量）==================
features_to_standardize = ['convenience_shap_sum', 'housing_quality_score', 'geographic_score']

for feat in features_to_standardize:
    # 仅从训练集计算均值和标准差
    mean_val = df_train[feat].mean()
    std_val = df_train[feat].std()

    # 避免除零（如果 std 为 0，则跳过标准化）
    if std_val == 0:
        print(f"⚠️ 警告: {feat} 的标准差为 0，跳过标准化")
        continue

    # 对所有数据集应用相同的标准化
    df_train[feat] = (df_train[feat] - mean_val) / std_val
    df_val[feat] = (df_val[feat] - mean_val) / std_val
    df_test[feat] = (df_test[feat] - mean_val) / std_val

print("✅ 已对以下特征完成标准化（基于训练集统计量）:", features_to_standardize)


# 提取特征和标签
X_train = df_train[selected_features].values.astype('float32')
y_train = df_train['totalrent'].values.astype('float32')

X_val = df_val[selected_features].values.astype('float32')
y_val = df_val['totalrent'].values.astype('float32')

X_test = df_test[selected_features].values.astype('float32')
y_test = df_test['totalrent'].values.astype('float32')

# 显示统计信息
print(f"\n特征数量: {len(selected_features)}")
print(f"训练集 y 均值: {y_train.mean():.2f} €, 标准差: {y_train.std():.2f} €")
print(f"验证集 y 均值: {y_val.mean():.2f} €")
print(f"测试集 y 均值: {y_test.mean():.2f} €")

# 可视化训练集 y 分布
print("\n【训练集 y 统计】")
print(f"均值: {y_train.mean():.2f}")
print(f"标准差: {y_train.std():.2f}")
print(f"最小值: {y_train.min():.2f}")
print(f"最大值: {y_train.max():.2f}")
print(f"中位数: {np.median(y_train):.2f}")
print(f"1% 分位数: {np.percentile(y_train, 1):.2f}")
print(f"99% 分位数: {np.percentile(y_train, 99):.2f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(y_train, bins=100, alpha=0.7)
plt.title('Train y Distribution')
plt.xlabel('Rent (€)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(y_train, bins=100, alpha=0.7, log=True)
plt.title('Train y Distribution (Log Scale)')
plt.xlabel('Rent (€)')
plt.ylabel('Count (log)')
plt.tight_layout()
plt.show()

# ================== 第二步：转换为 PyTorch 张量 并 创建 DataLoader ==================
import torch
from torch.utils.data import DataLoader, TensorDataset


# 转为 tensor 时使用 scaled 版本
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# X 不需要再标准化（您说已在 CSV 中处理）
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)


# 创建 DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)

print("✅ train_loader, val_loader, test_loader 已创建！")
print(f"训练集 batch 数: {len(train_loader)}")
print(f"每个 batch 大小: {next(iter(train_loader))[0].shape}")


# ================== 第三步：定义神经网络模型 ==================


class RentPredictor(nn.Module):
    def __init__(self, input_dim):
        super(RentPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),


            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)  # 输出一个值：房租
        )

    def forward(self, x):
        return self.network(x)

# 初始化模型
input_dim = X_train.shape[1]
model = RentPredictor(input_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
    )

# 使用 CPU 或 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
model.to(device)



print("✅ 模型已创建并移动到设备")


if torch.cuda.is_available():
    print(f"当前使用的 GPU 设备: {device}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU 索引: {torch.cuda.current_device()}")
    print(f"当前 GPU 名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA 不可用，正在使用 CPU")

# ================== 第四步：训练模型（带进度条）==================
from tqdm import tqdm
import time

epochs = 300
history = {
    'train_loss': [],
    'val_loss': [],
    'train_mae': [],
    'val_mae': []
}

print("开始训练模型...\n")

# 外层进度条：遍历每个 epoch
for epoch in tqdm(range(epochs), desc="Overall Progress", total=epochs):
    model.train()
    train_loss = 0.0
    train_mae = 0.0

    # 内层进度条：显示当前 epoch 的 batch 进度
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
    for inputs, targets in train_pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 累加损失和 MAE
        train_loss += loss.item() * inputs.size(0)
        train_mae += torch.mean(torch.abs(outputs - targets)).item() * inputs.size(0)

        # 实时更新进度条信息
        train_pbar.set_postfix({
            'loss': f"{loss.item():.4f}"
        })

    train_loss /= len(train_loader.dataset)
    train_mae /= len(train_loader.dataset)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
    with torch.no_grad():
        for inputs, targets in val_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            val_mae += torch.mean(torch.abs(outputs - targets)).item() * inputs.size(0)

            val_pbar.set_postfix({
                'val_loss': f"{loss.item():.4f}"
            })

    val_loss /= len(val_loader.dataset)
    val_mae /= len(val_loader.dataset)

    # 记录历史
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_mae'].append(train_mae)
    history['val_mae'].append(val_mae)

    # 每 10 个 epoch 打印详细信息（也可改为每 1 个）
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:03d}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")

# ================== 第六步：测试集评估 ==================
model.eval()
test_loss = 0.0
test_mae = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, targets).item() * inputs.size(0)
        test_mae += torch.mean(torch.abs(outputs - targets)).item() * inputs.size(0)

test_loss /= len(test_loader.dataset)
test_mae /= len(test_loader.dataset)

print(f"\n【测试集结果】")
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# 示例预测
sample_inputs = X_test_tensor[:5].to(device)
predictions = model(sample_inputs).detach().cpu().numpy()

print("\n前5个真实值 vs 预测值:")
for i in range(5):
    print(f"真实值: {y_test[i]:.2f} € → 预测值: {predictions[i][0]:.2f} €")

# ================== 第七步：可视化训练过程 ==================
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 绘制训练和验证损失曲线
plt.figure(figsize=(15, 5))

# 子图1：Loss (MSE)
plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss', color='blue')
plt.plot(history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training & Validation Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：MAE
plt.subplot(1, 3, 2)
plt.plot(history['train_mae'], label='Train MAE', color='green')
plt.plot(history['val_mae'], label='Validation MAE', color='red')
plt.title('Training & Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图3：配置信息
plt.subplot(1, 3, 3)
info_text = (
    f"Model: {model.__class__.__name__}\n"
    f"Optimizer: Adam\n"
    f"LR: 0.001\n"
    f"Batch Size: 64\n"
    f"Epochs: {epochs}\n"
    f"Input Dim: {input_dim}"
)
plt.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
plt.axis('off')
plt.title('Training Config')

plt.tight_layout()
plt.show()

# ================== 第八步：测试集预测 + 残差分析（修正版）==================
model.eval()
y_test_pred = []  # 存储原始预测值（已经是 €）
y_test_true = []  # 存储原始真实值（已经是 €）

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        y_test_pred.extend(outputs.cpu().numpy().flatten())  # 直接取值
        y_test_true.extend(targets.cpu().numpy().flatten())  # 直接取值

# 转为 numpy
y_test_pred = np.array(y_test_pred)
y_test_true = np.array(y_test_true)

# ✅ 直接使用原始值计算指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test_true, y_test_pred)
mae = mean_absolute_error(y_test_true, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_true, y_test_pred)

print(f"\n【测试集详细评估指标（原始欧元单位）】")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f} €")
print(f"MAE:  {mae:.2f} €")
print(f"R²:   {r2:.4f}")

torch.save(model.state_dict(), "rent_mlp_featured.pt")

# ================== 第九步：绘制残差图 ==================
plt.figure(figsize=(15, 5))

# 残差图用原始值
residuals = y_test_true - y_test_pred

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test_pred, residuals, alpha=0.6)
plt.axhline(0, color='r', ls='--')
plt.title('Residual Plot (Original Scale)')
plt.xlabel('Predicted Rent (€)')
plt.ylabel('Residual (€)')

plt.subplot(1, 3, 2)
plt.scatter(y_test_true, y_test_pred, alpha=0.6)
plt.plot([y_test_true.min(), y_test_true.max()],
         [y_test_true.min(), y_test_true.max()], 'r--')
plt.title('Predicted vs True (€)')
plt.xlabel('True Rent (€)')
plt.ylabel('Predicted Rent (€)')

plt.subplot(1, 3, 3)
plt.hist(residuals, bins=50, alpha=0.7)
plt.axvline(0, color='r', ls='--')
plt.title('Residual Distribution')
plt.xlabel('Error (€)')
plt.tight_layout()
plt.show()

# ================== 第十步：打印 10 个真实 vs 预测 示例（原始欧元单位）==================
print("\n" + "="*50)
print("           最终预测结果：真实值 vs 预测值")
print("="*50)

n_show = 10
for i in range(n_show):
    true_rent = y_test_true[i]
    pred_rent = y_test_pred[i]
    error = true_rent - pred_rent
    print(f"样本 {i+1:2d} | "
          f"真实租金: {true_rent:8.2f} € | "
          f"预测租金: {pred_rent:8.2f} € | "
          f"误差: {error:8.2f} €")

mae_top10 = mean_absolute_error(y_test_true[:n_show], y_test_pred[:n_show])
print("-" * 50)
print(f"前 {n_show} 个样本的平均绝对误差 (MAE): {mae_top10:.2f} €")
print("="*50)

# ================== 第十一步：Permutation Importance（排列重要性）==================
from copy import deepcopy
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def compute_metric(y_true, y_pred, metric='mae'):
    """计算评估指标"""
    if metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'rmse':
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    elif metric == 'r2':
        return r2_score(y_true, y_pred)
    else:
        raise ValueError("Unsupported metric")

def permutation_importance(model, X_tensor, y_true_numpy, feature_names, metric='mae', device='cpu'):
    """
    计算每个特征的排列重要性
    Args:
        model: 训练好的 PyTorch 模型
        X_tensor: 测试集特征 (torch.Tensor)
        y_true_numpy: 测试集标签 (numpy array)
        feature_names: 特征名称列表
        metric: 使用的评估指标
        device: 运行设备
    Returns:
        importance_df: 排序后的特征重要性 DataFrame
    """
    model.eval()
    X_tensor = X_tensor.to(device)

    # 基准性能（原始预测）
    with torch.no_grad():
        y_pred_baseline = model(X_tensor).cpu().numpy().flatten()
    baseline_score = compute_metric(y_true_numpy, y_pred_baseline, metric=metric)

    num_features = X_tensor.shape[1]
    importance_scores = []

    print(f"\n📊 开始计算排列重要性，共 {num_features} 个特征...\n")

    for i in tqdm(range(num_features), desc="Permuting Features"):
        X_permuted = X_tensor.clone()
        # 打乱第 i 个特征
        idx = torch.randperm(X_permuted.size(0))
        X_permuted[:, i] = X_permuted[:, i][idx]

        # 预测
        with torch.no_grad():
            output = model(X_permuted)
            y_pred_shuffled = output.cpu().numpy().flatten()

        # 计算打乱后的误差
        shuffled_score = compute_metric(y_true_numpy, y_pred_shuffled, metric=metric)

        # 重要性 = 性能下降程度（越大表示该特征越重要）
        importance = shuffled_score - baseline_score
        importance_scores.append(importance)

    # 创建结果 DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    return importance_df

# 使用与模型输入完全一致的特征列表（10个）
selected_features = [
    'servicecharge_scaled',
    'picturecount_scaled',
    'telekomuploadspeed_scaled',
    'age_scaled',
    'floor_scaled',
    'convenience_shap_sum',
    'housing_quality_score',
    'geographic_score',
    'month_sin',
    'month_cos'
]

# 计算排列重要性（使用 MAE 作为衡量标准）
importance_df = permutation_importance(
    model=model,
    X_tensor=X_test_tensor,
    y_true_numpy=y_test_true,
    feature_names=selected_features,
    metric='mae',      # 使用 MAE：单位为 €，解释性强
    device=device
)

# === 输出结果 ===
print("\n" + "="*60)
print("           🏆 排列重要性排序（基于测试集）")
print("="*60)
print(importance_df.to_string(index=True, float_format='%.6f'))

# === 可视化 ===
plt.figure(figsize=(10, 6))
sns.barplot(
    data=importance_df,
    x='importance',
    y='feature',
    palette='Blues_r'
)
plt.title('Permutation Importance (Test Set, ΔMAE)', fontsize=14)
plt.xlabel('Importance (Increase in MAE after Shuffling)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# （可选）保存结果
# importance_df.to_csv('permutation_importance_8features.csv', index=False)
# print("\n✅ 排列重要性已保存至 'permutation_importance_8features.csv'")

# （可选）保存结果
# importance_df.to_csv('permutation_importance_8features.csv', index=False)
# print("\n✅ 排列重要性已保存至 'permutation_importance_8features.csv'")