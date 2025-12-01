import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 設定你要比較的圖片路徑 (請替換成你實際生成的圖片路徑)
# 建議比較同一張 Index 的圖片 (因為種子固定，內容應該一樣)
IMG_IDX = "00000.png"

# 路徑 A: V1 (No Mask) 的 Stego 圖
PATH_V1 = os.path.join("outputs", "flicker8k_v3_nomask", "stego", IMG_IDX)
# 路徑 B: V2 (Masked) 的 Stego 圖
PATH_V2 = os.path.join("outputs", "flicker8k_v3_mask", "stego", IMG_IDX)
# 路徑 C: 原始 Cover 圖 (V1 和 V2 的 Cover 應該是一樣的)
PATH_COVER = os.path.join("outputs", "flicker8k_v3_mask", "cover", IMG_IDX)

def load_img(path):
    if not os.path.exists(path):
        print(f"❌ 找不到圖片: {path}")
        return None
    # 讀取並轉為 RGB
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show_diff():
    cover = load_img(PATH_COVER)
    stego_v1 = load_img(PATH_V1)
    stego_v2 = load_img(PATH_V2)

    if cover is None or stego_v1 is None or stego_v2 is None:
        return

    # 計算殘差 (Residual)
    # diff = |Stego - Cover|
    # 為了讓肉眼看得到，我們把差異放大 50 倍
    SCALE = 50
    
    diff_v1 = cv2.absdiff(stego_v1, cover) * SCALE
    diff_v2 = cv2.absdiff(stego_v2, cover) * SCALE

    # 繪圖
    plt.figure(figsize=(15, 10))

    # 第一列：原圖與 Stego
    plt.subplot(2, 3, 1)
    plt.title("Cover Image (Base)")
    plt.imshow(cover)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Stego V1 (No Mask)")
    plt.imshow(stego_v1)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Stego V2 (Masked)")
    plt.imshow(stego_v2)
    plt.axis('off')

    # 第二列：差異圖 (Residual Maps)
    plt.subplot(2, 3, 5)
    plt.title(f"Residual V1 (No Mask)\nAmplified {SCALE}x")
    plt.imshow(diff_v1)
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title(f"Residual V2 (Masked)\nAmplified {SCALE}x")
    plt.imshow(diff_v2)
    plt.axis('off')

    plt.tight_layout()
    
    # 【關鍵修改】不使用 plt.show()，改為保存圖片
    output_filename = "diff_comparison_result.png"
    plt.savefig(output_filename, dpi=150)
    
    print(f"✅ 圖片已保存為: {output_filename}")
    print("👉 請在你的檔案瀏覽器中打開這張圖片查看結果。")
    print("------------------------------------------------")
    print("預期結果解讀：")
    print("1. 下排左圖 (V1 No Mask): 雜訊點應該均勻散佈在整張圖，包括天空。")
    print("2. 下排右圖 (V2 Masked): 天空區域應該是黑色的(乾淨)，雜訊集中在建築/紋理處。")

if __name__ == "__main__":
    show_diff()