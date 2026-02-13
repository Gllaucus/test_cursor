"""
基于 ftec_nn_featured.py 的 MLP 模型封装为 Flask + 网页。

使用方式（在终端，已在 conda 环境 py39 下）：
1. conda activate py39
2. pip install flask
3. 在 ftec_nn_featured.py 训练结束后加上：torch.save(model.state_dict(), "rent_mlp_featured.pt")
4. 运行训练脚本一次，生成权重文件。
5. python rent_flask_app.py
6. 浏览器打开 http://127.0.0.1:5000
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, jsonify, request

# ========= 配置：与 ftec_nn_featured.py 保持一致 =========

FILE_PATH = r"E:\研一\ftec人工智能\final\data\neural_network_input_with_geographic_score.csv"
MODEL_PATH = "rent_mlp_featured.pt"

# 与脚本中 selected_features 完全一致
SELECTED_FEATURES: List[str] = [
    "servicecharge_scaled",
    "picturecount_scaled",
    "telekomuploadspeed_scaled",
    "age_scaled",
    "floor_scaled",
    "convenience_shap_sum",
    "housing_quality_score",
    "geographic_score",
    "month_sin",
    "month_cos",
]

# 与脚本中 features_to_standardize 一致：推理时也要用训练集统计量做标准化
FEATURES_TO_STANDARDIZE = [
    "convenience_shap_sum",
    "housing_quality_score",
    "geographic_score",
]


# ========= 1. 定义与 ftec_nn_featured.py 相同的网络结构 =========


class RentPredictor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
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
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(x)


# ========= 2. 读取 CSV，推断特征统计量 & 标准化参数 =========


if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"找不到数据文件：{FILE_PATH}\n"
        f"请确认路径是否与 ftec_nn_featured.py 中一致，或修改 rent_flask_app.py 里的 FILE_PATH。"
    )

df = pd.read_csv(FILE_PATH)

if "dataset_split" not in df.columns:
    raise KeyError("数据中缺少 'dataset_split' 列，无法复用训练时的标准化逻辑。")

# 与原脚本一致：从 date 列计算 month_sin 和 month_cos（如果 CSV 里没有的话）
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    if "month_sin" not in df.columns or "month_cos" not in df.columns:
        df["month"] = df["date"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df_train = df[df["dataset_split"] == "train"].copy()

# 训练集特征统计（方便在前端展示均值/最小值等）
feature_stats = df_train[SELECTED_FEATURES].describe().T

# 为需要标准化的特征记录训练集的均值和标准差
standardize_params: Dict[str, Dict[str, float]] = {}
for feat in FEATURES_TO_STANDARDIZE:
    if feat not in df_train.columns:
        raise KeyError(f"缺失特征 {feat}，请检查 CSV。")
    mean_val = float(df_train[feat].mean())
    std_val = float(df_train[feat].std())
    if std_val == 0:
        std_val = 1.0  # 避免除以 0，意味着该特征几乎常数
    standardize_params[feat] = {"mean": mean_val, "std": std_val}


# ========= 3. 加载训练好的模型权重 =========


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RentPredictor(input_dim=len(SELECTED_FEATURES)).to(device)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"找不到模型权重文件 {MODEL_PATH}。\n"
        f"请在 ftec_nn_featured.py 训练结束后，加上：\n"
        f"    torch.save(model.state_dict(), '{MODEL_PATH}')\n"
        f"并重新运行训练脚本生成该文件。"
    )

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()


# ========= 4. Flask 应用 =========


app = Flask(__name__)


@app.get("/schema")
def get_schema():
    """
    返回当前 MLP 所需的特征列表（selected_features），
    以及训练集上的均值/最小/最大值，方便在网页上生成输入框。
    """
    features: List[Dict[str, float]] = []
    for name in SELECTED_FEATURES:
        stats = feature_stats.loc[name]
        features.append(
            {
                "name": name,
                "mean": float(stats.get("mean", np.nan)),
                "min": float(stats.get("min", np.nan)),
                "max": float(stats.get("max", np.nan)),
                "standardized": bool(name in FEATURES_TO_STANDARDIZE),
            }
        )
    return jsonify({"features": features})


@app.post("/api/predict")
def api_predict():
    """
    接收 JSON，键为 selected_features 中的特征名（与脚本一致），值为“原始”数值。
    特殊处理：
    - 如果提供了 "date"（字符串，如 "2024-01-15"），会自动计算 month_sin 和 month_cos
    - convenience_shap_sum / housing_quality_score / geographic_score 会按训练集均值/方差做标准化；
    - 其他特征直接作为模型输入；
    - 没有提供的字段用训练集均值填充。
    返回：{ "rent": 1234.56 }
    """
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "请求体必须是 JSON 对象"}), 400

    # 如果用户提供了 date，自动计算 month_sin 和 month_cos
    if "date" in data and data["date"]:
        try:
            date_val = pd.to_datetime(data["date"])
            month = date_val.month
            month_sin_val = float(np.sin(2 * np.pi * month / 12))
            month_cos_val = float(np.cos(2 * np.pi * month / 12))
            # 如果用户没单独填这两个值，就用计算出的值覆盖
            if "month_sin" not in data or not data["month_sin"]:
                data["month_sin"] = month_sin_val
            if "month_cos" not in data or not data["month_cos"]:
                data["month_cos"] = month_cos_val
        except Exception as e:
            return jsonify({"error": f"无法解析 date 字段：{e}"}), 400

    row_vals: List[float] = []
    for name in SELECTED_FEATURES:
        raw_val = data.get(name, None)
        if raw_val is None or raw_val == "":
            # 如果用户没填，就直接用训练集该特征（已处理后的）均值
            val_raw = float(feature_stats.loc[name]["mean"])
        else:
            try:
                val_raw = float(raw_val)
            except (TypeError, ValueError):
                return jsonify({"error": f"特征 {name} 的值无法转换为数字：{raw_val!r}"}), 400

        # 对需要标准化的特征，套用与训练完全相同的 mean/std
        if name in standardize_params:
            p = standardize_params[name]
            val = (val_raw - p["mean"]) / p["std"]
        else:
            val = val_raw

        row_vals.append(val)

    x = torch.tensor([row_vals], dtype=torch.float32, device=device)
    with torch.no_grad():
        y = model(x).cpu().numpy().reshape(-1)[0]

    return jsonify({"rent": float(y)})


@app.get("/")
def index():
    """
    返回一个简单网页：自动从 /schema 拿特征名生成输入表单，
    提交到 /api/predict，显示 MLP 预测的月租金。
    """
    html = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <title>房租预测小助手（MLP/Flask）</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <style>
      body {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 0;
        padding: 16px;
        background: #020617;
        color: #e5e7eb;
      }
      .wrap {
        max-width: 960px;
        margin: 0 auto;
        background: rgba(15,23,42,0.96);
        border-radius: 16px;
        padding: 18px 18px 20px;
        box-shadow: 0 20px 45px rgba(0,0,0,0.65);
        border: 1px solid rgba(148,163,184,0.4);
      }
      h1 {
        margin-top: 0;
        font-size: 20px;
      }
      .sub {
        font-size: 13px;
        color: #9ca3af;
        margin-bottom: 10px;
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 10px 14px;
        margin-bottom: 14px;
        max-height: 360px;
        overflow: auto;
        padding-right: 4px;
      }
      .field label {
        font-size: 12px;
        display: block;
        margin-bottom: 2px;
      }
      .field small {
        color: #6b7280;
        font-size: 11px;
        margin-left: 4px;
      }
      .field input {
        width: 100%;
        padding: 6px 8px;
        border-radius: 8px;
        border: 1px solid #4b5563;
        background: #020617;
        color: #e5e7eb;
      }
      .field input:focus {
        outline: none;
        border-color: #a855f7;
        box-shadow: 0 0 0 1px rgba(168,85,247,0.6);
      }
      .btn {
        border-radius: 999px;
        border: none;
        padding: 8px 16px;
        cursor: pointer;
        font-weight: 700;
        color: #f9fafb;
        background: linear-gradient(135deg,#8b5cf6,#22c55e);
      }
      .btn[disabled] {
        opacity: .6;
        cursor: default;
      }
      .result {
        margin-top: 12px;
        border-top: 1px dashed #4b5563;
        padding-top: 10px;
      }
      .price {
        font-size: 22px;
        font-weight: 800;
      }
      .muted {
        font-size: 12px;
        color: #9ca3af;
      }
      .error {
        font-size: 12px;
        color: #fecaca;
        margin-top: 6px;
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>房租预测小助手（基于 ftec_nn_featured.py 的 MLP）</h1>
      <div class="sub">
        下方所有输入字段来自脚本中的 selected_features，你可以只修改关心的字段，其余用训练集均值自动填充。
      </div>
      <div id="fields" class="grid"></div>
      <button id="btnPredict" class="btn">预测月租金</button>
      <div class="result" id="result" style="display:none;">
        <div class="price" id="priceText"></div>
        <div class="muted" id="detailText"></div>
        <div class="error" id="errorText" style="display:none;"></div>
      </div>
    </div>

    <script>
      const fieldsBox = document.getElementById("fields");
      const btn = document.getElementById("btnPredict");
      const resultBox = document.getElementById("result");
      const priceText = document.getElementById("priceText");
      const detailText = document.getElementById("detailText");
      const errorText = document.getElementById("errorText");

      async function loadSchema() {
        const resp = await fetch("/schema");
        const data = await resp.json();
        const feats = data.features || [];
        fieldsBox.innerHTML = "";
        for (const f of feats) {
          const div = document.createElement("div");
          div.className = "field";
          const label = document.createElement("label");
          label.textContent = f.name + (f.standardized ? "  (会自动标准化)" : "");
          const small = document.createElement("small");
          const mean = isFinite(f.mean) ? f.mean.toFixed(2) : "未知";
          small.textContent = "均值: " + mean;
          label.appendChild(small);
          const input = document.createElement("input");
          input.type = "number";
          input.step = "any";
          input.value = isFinite(f.mean) ? f.mean : "";
          input.dataset.name = f.name;
          div.appendChild(label);
          div.appendChild(input);
          fieldsBox.appendChild(div);
        }
      }

      async function predict() {
        const inputs = fieldsBox.querySelectorAll("input");
        const payload = {};
        inputs.forEach((inp) => {
          const name = inp.dataset.name;
          const v = inp.value.trim();
          if (v !== "") {
            payload[name] = Number(v);
          }
        });

        btn.disabled = true;
        btn.textContent = "预测中...";
        errorText.style.display = "none";
        errorText.textContent = "";

        try {
          const resp = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const data = await resp.json();
          resultBox.style.display = "block";
          if (!resp.ok) {
            errorText.style.display = "block";
            errorText.textContent = data.error || ("请求失败，状态码 " + resp.status);
            priceText.textContent = "";
            detailText.textContent = "";
            return;
          }
          const rent = data.rent;
          priceText.textContent = rent.toLocaleString("zh-CN") + " 元/月";
          detailText.textContent = "该数值完全由你的 PyTorch MLP 模型计算得到（输入特征来自当前表单）。";
        } catch (err) {
          resultBox.style.display = "block";
          errorText.style.display = "block";
          errorText.textContent = "请求后端时出错：" + err;
          priceText.textContent = "";
          detailText.textContent = "";
        } finally {
          btn.disabled = false;
          btn.textContent = "预测月租金";
        }
      }

      btn.addEventListener("click", predict);
      window.addEventListener("load", loadSchema);
    </script>
  </body>
</html>
    """
    return html


if __name__ == "__main__":
    print("✅ Flask 服务已启动，浏览器打开 http://127.0.0.1:5000 即可使用房租预测小助手。")
    app.run(host="127.0.0.1", port=5000, debug=True)

