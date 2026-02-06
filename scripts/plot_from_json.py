import os
import json
import matplotlib.pyplot as plt
import numpy as np

# === Global Settings ===
# Path to your new JSON file
DEFAULT_JSON_PATH = "../outputs/analysis_report/all_convergence_data.json"
TARGET_SEED = 46
OUTPUT_FILENAME = f"convergence_plot_seed_{TARGET_SEED}.png"

def smooth_curve(scalars, weight=0.7):
    """ Exponential Moving Average for smoothing curves """
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    json_path = DEFAULT_JSON_PATH

    # 1. Check if file exists
    if not os.path.exists(json_path):
        print(f"❌ Error: File not found at {json_path}")
        if os.path.exists("all_convergence_data.json"):
            json_path = "all_convergence_data.json"
            print(f"✅ Found file in current directory: {json_path}")
        else:
            print("Please check the file path.")
            return

    # 2. Load Data
    print(f"📂 Loading data from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return

    history = []
    prompt_text = ""

    # 3. Parse Data Structure
    # Case A: New structure (Dict with "results" list)
    if isinstance(data, dict) and "results" in data:
        print("ℹ️ Detected 'results' list format.")
        found = False
        for entry in data["results"]:
            # Check if seed matches (handle both int and str types)
            if str(entry.get("seed")) == str(TARGET_SEED):
                history = entry["history"]
                prompt_text = entry.get("prompt", "")
                found = True
                print(f"✅ Found data for Seed {TARGET_SEED} (Prompt: '{prompt_text}')")
                break
        
        if not found:
            print(f"❌ Error: Seed {TARGET_SEED} not found in 'results'.")
            # Print available seeds to help debugging
            available_seeds = [str(e.get("seed")) for e in data["results"]]
            print(f"   Available seeds: {available_seeds}")
            return

    # Case B: Old structure (Dict where keys are seeds)
    elif isinstance(data, dict):
        key = str(TARGET_SEED)
        if key in data:
            print(f"✅ Found data for Seed {TARGET_SEED} (Old Format).")
            history = data[key]
        else:
            print(f"❌ Error: Seed {TARGET_SEED} not found as key.")
            print(f"   Available keys: {list(data.keys())}")
            return

    # Case C: Simple List (Single run)
    elif isinstance(data, list):
        print("ℹ️ Detected single-run list format.")
        history = data

    else:
        print("❌ Error: Unknown JSON structure.")
        return

    # 4. Extract metrics
    try:
        iters = [x['iter'] for x in history]
        recon_loss = [x['recon_loss'] for x in history]
        reg_loss = [x['reg_loss'] for x in history]
    except KeyError as e:
        print(f"❌ Data Error: Missing key {e} in history data.")
        return

    # Smooth the curve
    recon_smooth = smooth_curve(recon_loss, weight=0.7)

    # 5. Plotting
    print("🎨 Plotting...")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Left Y-Axis: Reconstruction Loss (Blue) ---
    color_recon = 'tab:blue'
    color_recon_raw = 'lightblue'
    
    ax1.set_xlabel('Optimization Iterations', fontsize=12)
    ax1.set_ylabel('Reconstruction Loss (MSE)', color=color_recon, fontweight='bold', fontsize=12)
    
    ax1.plot(iters, recon_loss, color=color_recon_raw, alpha=0.4, linewidth=1, label='Recon Loss (Raw)')
    ax1.plot(iters, recon_smooth, color=color_recon, linewidth=3, label='Recon Loss (Trend)')
    
    ax1.tick_params(axis='y', labelcolor=color_recon)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Right Y-Axis: Regularization Loss (Red) ---
    ax2 = ax1.twinx()
    color_reg = 'tab:red'
    
    ax2.set_ylabel('Regularization Loss (Decoding)', color=color_reg, fontweight='bold', fontsize=12)
    ax2.plot(iters, reg_loss, color=color_reg, marker='x', markersize=6, linewidth=2, linestyle='--', label='Reg Loss')
    ax2.tick_params(axis='y', labelcolor=color_reg)

    # --- Title & Legend ---
    title_text = ""
    if prompt_text:
        title_text += ""
    plt.title(title_text, fontsize=14, pad=20)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True)

    # 6. Save File
    output_dir = os.path.dirname(json_path)
    if not output_dir: output_dir = "."
    save_path = os.path.join(output_dir, OUTPUT_FILENAME)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"🎉 Plot saved successfully to: {save_path}")

if __name__ == "__main__":
    main()