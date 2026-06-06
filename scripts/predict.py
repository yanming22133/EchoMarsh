"""
EchoMarsh 模型预测 — 批量推理，充分利用 GPU
用法:
    python scripts/predict.py
"""
import os, sys, pickle, glob
import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.backbone.model_factory import ModelFactory
from core.trainer.stream_dataset import process_csv, build_samples

DATA_DIR = os.path.join(project_root, "data", "stocks")
CKPT_DIR = os.path.join(project_root, "models", "checkpoints")
SEQ_LEN, PRED_LEN = 120, 5
INCLUDE_CODES = ('60', '00')

# 加载模型
model, device = ModelFactory.create_model('transformer', ts_feature_dim=32, meta_feature_dim=7)
model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_echomarsh_model.pth"),
                                 map_location=device, weights_only=True))
model.eval()

# 加载 scaler
with open(os.path.join(CKPT_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
norm_indices = [0,1,2,3,4,5,6,7,8]  # 9 个 NORM_COLS

# 获取文件列表
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if INCLUDE_CODES:
    files = [f for f in files if os.path.basename(f).startswith(INCLUDE_CODES)]
print(f"主板: {len(files)} 只股票")

# 逐文件处理，但攒到 512 个样本再批量推理
batch_ts, batch_meta, batch_codes = [], [], []
results = []

for idx, f in enumerate(files):
    code = os.path.basename(f)[:6]
    data = process_csv(f, SEQ_LEN, PRED_LEN, '2024-01-01', '2026-05-14')
    if data is None:
        continue
    feat_raw, targets, meta = data
    samples = build_samples(feat_raw, targets, meta, SEQ_LEN, scaler, norm_indices)
    if not samples:
        continue
    x, meta_arr, _ = samples[-1]
    batch_ts.append(x)
    batch_meta.append(meta_arr)
    batch_codes.append(code)

    # 满 512 个或最后一批 → 批量推理
    if len(batch_ts) >= 512 or idx == len(files) - 1:
        with torch.no_grad():
            ts = torch.from_numpy(np.array(batch_ts).astype(np.float32)).to(device)
            mt = torch.from_numpy(np.array(batch_meta).astype(np.float32)).to(device)
            with torch.amp.autocast(device_type=device.type):
                out = model(ts, mt)
            rets = out[:, 0].cpu().numpy()      # 1d_ret 用于排序
            probs = torch.sigmoid(out[:, 4]).cpu().numpy()  # limit_up_logit
        for c, r, p in zip(batch_codes, rets, probs):
            results.append((c, r, p))
        batch_ts, batch_meta, batch_codes = [], [], []

    if (idx + 1) % 500 == 0:
        print(f"  处理中: {idx+1}/{len(files)} 只股票")

# 排序输出
results.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 70)
print("  Top 20 推荐（预期收益最高）")
print("=" * 70)
print(f"  {'代码':<8} {'预期收益%':<12} {'涨停概率':<10}")
for i, (code, ret, prob) in enumerate(results[:20]):
    star = " ⭐" if prob > 0.1 else ""
    print(f"  {code:<8} {ret:<+10.4f}   {prob:.4f}{star}")

print(f"\n  ... 共 {len(results)} 只股票")

# 涨停概率排序
results_by_prob = sorted(results, key=lambda x: x[2], reverse=True)
print("\n" + "=" * 70)
print("  Top 10 涨停潜力股")
print("=" * 70)
print(f"  {'代码':<8} {'预期收益%':<12} {'涨停概率':<10}")
for code, ret, prob in results_by_prob[:10]:
    print(f"  {code:<8} {ret:<+10.4f}   {prob:.4f}")

# 统计
all_rets = np.array([r[1] for r in results])
all_probs = np.array([r[2] for r in results])
print(f"\n{'='*70}")
print(f"  统计: 均值收益={all_rets.mean():.4f}%  "
      f"涨停概率均值={all_probs.mean():.4f}  "
      f"max={all_probs.max():.4f}")
print(f"  预期收益>0: {(all_rets>0).sum()}/{len(all_rets)}")
