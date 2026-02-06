import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 設定路徑 (請確認這指向你的輸出目錄)
OUTPUT_DIR = "outputs/power_sensitivity" 
JSON_PATH = os.path.join(OUTPUT_DIR, "power_results.json")
SAVE_PATH = os.path.join(OUTPUT_DIR, "sensitivity_analysis_fitted.pdf") 

def plot_dual_axis():
    if not os.path.exists(JSON_PATH):
        print(f"❌ Cannot find {JSON_PATH}")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # 1. 整理數據並排序
    data.sort(key=lambda x: x['power'])
    
    powers = np.array([d['power'] for d in data])
    accs = np.array([d['acc'] for d in data])
    brisques = np.array([d['brisque'] for d in data])

    # === 建立平滑擬合曲線 (僅供視覺展示) ===
    # 使用 3 次多項式擬合 (Polyfit Degree 3)
    x_smooth = np.linspace(powers.min(), powers.max(), 300)
    
    # 擬合 Accuracy
    z1 = np.polyfit(powers, accs, 3)
    p1 = np.poly1d(z1)
    y1_smooth = p1(x_smooth)

    # 擬合 BRISQUE
    z2 = np.polyfit(powers, brisques, 3)
    p2 = np.poly1d(z2)
    y2_smooth = p2(x_smooth)

    # === [修正] 自動尋找最佳點 (Sweet Spot) 使用「原始數據」 ===
    # 避免擬合曲線造成的偏移，直接評估真實測量的點
    
    # 1. 正規化 (0~1)
    # 使用原始數據的 min/max
    norm_acc = (accs - accs.min()) / (accs.max() - accs.min() + 1e-8)
    # BRISQUE 原值越小越好。正規化後 0 代表最小(好)，1 代表最大(差)
    norm_bri = (brisques - brisques.min()) / (brisques.max() - brisques.min() + 1e-8)
    
    # 2. 計算距離 (Euclidean Distance to Ideal Point)
    # 理想點: Acc=Max (Norm=1.0), BRISQUE=Min (Norm=0.0)
    distances = np.sqrt((1.0 - norm_acc)**2 + (0.0 - norm_bri)**2)
    
    # 3. 找到距離最小的點索引
    best_idx = np.argmin(distances)
    p_target = powers[best_idx]
    
    # 取得該點的真實數值
    y_acc_best = accs[best_idx]
    y_bri_best = brisques[best_idx]

    # 2. 設定畫布
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # 3. 繪製左軸 (Accuracy)
    color_acc = '#1f77b4' # 專業藍
    ax1.set_xlabel('Mask Power ($p$)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Bit Accuracy (%)', color=color_acc, fontsize=14, fontweight='bold')
    
    # A. 畫原始散點
    ax1.scatter(powers, accs, color=color_acc, alpha=0.5, s=60, label='Raw Accuracy')
    # B. 畫擬合趨勢線
    line1 = ax1.plot(x_smooth, y1_smooth, color=color_acc, linewidth=3, alpha=0.8, label='Accuracy Trend')
    
    ax1.tick_params(axis='y', labelcolor=color_acc, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # 4. 繪製右軸 (Quality / BRISQUE)
    ax2 = ax1.twinx()
    color_qual = '#d62728' # 專業紅
    ax2.set_ylabel('BRISQUE Score (Lower is Better)', color=color_qual, fontsize=14, fontweight='bold')
    
    # A. 畫原始散點
    ax2.scatter(powers, brisques, color=color_qual, marker='s', alpha=0.5, s=60, label='Raw BRISQUE')
    # B. 畫擬合趨勢線
    line2 = ax2.plot(x_smooth, y2_smooth, color=color_qual, linestyle='--', linewidth=3, alpha=0.8, label='BRISQUE Trend')
    
    ax2.tick_params(axis='y', labelcolor=color_qual, labelsize=12)

    # 5. 標註 Sweet Spot
    # 畫垂直線標示
    plt.axvline(x=p_target, color='green', linestyle='-.', linewidth=2, alpha=0.8)
    
    # 特別圈出最佳點 (在原始散點上)
    # Accuracy 點
    ax1.plot(p_target, y_acc_best, 'o', color='green', markersize=12, markerfacecolor='none', markeredgewidth=2)
    # BRISQUE 點
    ax2.plot(p_target, y_bri_best, 's', color='green', markersize=12, markerfacecolor='none', markeredgewidth=2)

    # 標籤文字
    info_text = (f" Optimal Balance\n"
                 f" $p={p_target}$\n"
                 f" Acc: {y_acc_best:.1f}%\n"
                 f" BRISQUE: {y_bri_best:.1f}")
    
    # 智慧調整文字位置
    text_x = p_target + 0.5 if p_target < (powers.max() + powers.min())/2 else p_target - 3.0
    
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="green", alpha=0.9, linewidth=1.5)
    ax1.text(text_x, (min(accs) + max(accs))/2, info_text, 
             fontsize=12, fontweight='bold', color='green', bbox=bbox_props)

    # 6. 合併圖例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    from matplotlib.lines import Line2D
    lines.append(Line2D([0], [0], color='green', linestyle='-.', linewidth=2))
    labels.append(f'Selected Power ($p={p_target}$)')
    
    ax1.legend(lines, labels, loc='center right', fontsize=11, frameon=True, shadow=True, fancybox=True)

    plt.title('Parameter Sensitivity Analysis: Mask Power ($p$)', fontsize=16, pad=15)
    plt.tight_layout()
    
    # 存檔
    plt.savefig(SAVE_PATH)
    plt.savefig(SAVE_PATH.replace('.pdf', '.png'), dpi=300)
    print(f"✅ Fitted Sensitivity Plot (Raw Data Metric) saved to {SAVE_PATH}")

if __name__ == "__main__":
    plot_dual_axis()