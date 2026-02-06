import json
from datasets import load_dataset

# ================= 參數設定 =================
TARGET_COUNT = 500

KEYWORDS_ANIME = [
    "anime", "manga", "waifu", "flat color", 
    "cel shaded", "cel shading", "studio ghibli", 
    "makoto shinkai", "illustration", "2d"
]

KEYWORDS_ART = [
    "oil painting", "thick impasto", "brush strokes", 
    "van gogh", "monet", "concept art", 
    "artstation", "matte painting", "highly detailed painting",
    "greg rutkowski"
]
# ==========================================

def fetch_and_save_prompts():
    print(f"🚀 開始讀取 DiffusionDB parquet 檔...")
    print(f"🎯 目標收集：動漫風格 {TARGET_COUNT} 筆 / 油畫風格 {TARGET_COUNT} 筆")

    try:
        # 直接讀本地 parquet
        dataset = load_dataset(
            "parquet",
            data_files="/nfs/Workspace/stt/mas_GRDH/diffusiondb/metadata-large.parquet",
            split="train",
            streaming=True
        )
    except Exception as e:
        print(f"❌ 資料集載入失敗: {e}")
        return

    anime_prompts = []
    art_prompts = []
    seen_prompts = set()

    print("🔍 開始掃描資料集...")

    for i, item in enumerate(dataset):
        # DiffusionDB prompt 欄位通常是 'prompt'
        prompt_text = item.get('prompt') or item.get('Prompt') or item.get('text')
        if not prompt_text:
            continue

        clean_prompt = " ".join(prompt_text.strip().split())
        prompt_lower = clean_prompt.lower()

        if clean_prompt in seen_prompts:
            continue

        # 篩選動漫風格
        if len(anime_prompts) < TARGET_COUNT and any(k in prompt_lower for k in KEYWORDS_ANIME):
            anime_prompts.append(clean_prompt)
            seen_prompts.add(clean_prompt)
            if len(anime_prompts) % 50 == 0:
                print(f"✅ [Anime] 進度: {len(anime_prompts)}/{TARGET_COUNT}")

        # 篩選油畫風格
        if len(art_prompts) < TARGET_COUNT and any(k in prompt_lower for k in KEYWORDS_ART):
            art_prompts.append(clean_prompt)
            seen_prompts.add(clean_prompt)
            if len(art_prompts) % 50 == 0:
                print(f"🎨 [Art] 進度: {len(art_prompts)}/{TARGET_COUNT}")

        # 兩邊都收集滿就停止
        if len(anime_prompts) >= TARGET_COUNT and len(art_prompts) >= TARGET_COUNT:
            print("\n✨ 兩種類型皆已收集完成！")
            break

        # 安全閥，避免無限抓取
        if i > 100000:
            print("⚠️ 掃描超過 100,000 筆資料，提前停止。")
            break

    # ================= 存檔 =================
    print("\n💾 正在寫入檔案...")

    with open("prompts_anime.txt", "w", encoding="utf-8") as f:
        for p in anime_prompts:
            f.write(p + "\n")
    
    with open("prompts_art.txt", "w", encoding="utf-8") as f:
        for p in art_prompts:
            f.write(p + "\n")

    print(f"🎉 成功！檔案已建立於目錄下：")
    print(f"1. prompts_anime.txt (共 {len(anime_prompts)} 筆)")
    print(f"2. prompts_art.txt   (共 {len(art_prompts)} 筆)")

if __name__ == "__main__":
    fetch_and_save_prompts()
