import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# --- Configuration ---
# List your result directories here
BASELINE_DIRS = ["averaged_test/c0r1", "averaged_test/c0r2", "averaged_test/c0r3", "averaged_test/c0r4", "averaged_test/c0r5", "averaged_test/c0r6", "averaged_test/c0r7", "averaged_test/c0r8", "averaged_test/c0r9", "averaged_test/c0r10"]
PHYSICS_DIRS  = ["averaged_test/c1.5r1", "averaged_test/c1.5r2", "averaged_test/c1.5r3", "averaged_test/c1.5r4", "averaged_test/c1.5r5", "averaged_test/c1.5r6", "averaged_test/c1.5r7", "averaged_test/c1.5r8", "averaged_test/c1.5r9", "averaged_test/c1.5r10"]

OUTPUT_DIR = "averaged_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_group_results(dir_list):
    """Loads and aggregates results from a list of directories."""
    all_test_acc = []
    all_cms = []
    
    for d in dir_list:
        path = os.path.join(d, "experiment_results.json")
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
            all_test_acc.append(data['training_history']['test_accuracy'])
            all_cms.append(data['final_results']['confusion_matrix_test'])
            
    return np.array(all_test_acc), np.array(all_cms)

# 1. Load and Average
print("Loading results...")
acc_base_raw, cms_base_raw = load_group_results(BASELINE_DIRS)
acc_phys_raw, cms_phys_raw = load_group_results(PHYSICS_DIRS)

# Calculate Means and Stdevs
mean_acc_base = np.mean(acc_base_raw, axis=0)
std_acc_base  = np.std(acc_base_raw, axis=0)

mean_acc_phys = np.mean(acc_phys_raw, axis=0)
std_acc_phys  = np.std(acc_phys_raw, axis=0)

# Calculate Mean Confusion Matrices
mean_cm_base = np.mean(cms_base_raw, axis=0)
mean_cm_phys = np.mean(cms_phys_raw, axis=0)
diff_cm = mean_cm_phys - mean_cm_base

# Calculate Delta Mean
delta_mean = mean_acc_phys - mean_acc_base

# --- Plotting ---
plt.style.use('default')

# FIGURE 1: Smooth Accuracy Comparison
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
epochs = range(len(mean_acc_base))

# Baseline plot with shadow
plt.plot(epochs, mean_acc_base, label="Baseline (k=0)", color='black', linestyle='--')
plt.fill_between(epochs, mean_acc_base - std_acc_base, mean_acc_base + std_acc_base, color='black', alpha=0.1)

# Physics plot with shadow
plt.plot(epochs, mean_acc_phys, label="Physics (k=1.5)", color='royalblue', linewidth=2)
plt.fill_between(epochs, mean_acc_phys - std_acc_phys, mean_acc_phys + std_acc_phys, color='royalblue', alpha=0.2)

plt.title("Averaged Test Accuracy (5 Runs)\nShaded area = 1 Standard Deviation")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(alpha=0.3)

# FIGURE 2: Averaged Delta
plt.subplot(1, 2, 2)
plt.axhline(0, color='black', linewidth=1)
plt.plot(epochs, delta_mean, color='darkgreen', linewidth=1.5)
plt.fill_between(epochs, delta_mean, 0, where=(delta_mean > 0), color='green', alpha=0.3)
plt.fill_between(epochs, delta_mean, 0, where=(delta_mean < 0), color='red', alpha=0.3)
plt.title("Mean Accuracy Gain (Physics - Baseline)")
plt.xlabel("Epoch"); plt.ylabel("Delta Accuracy")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/averaged_accuracy_delta.png")

# FIGURE 3: Mean Differential Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(diff_cm, annot=True, fmt='.1f', cmap='RdBu', center=0)
plt.title("Mean Differential Confusion Matrix\n(Average k=1.5) - (Average k=0)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.savefig(f"{OUTPUT_DIR}/averaged_diff_cm.png")

# --- Final Printout ---
print("\n" + "="*30)
print(f"RESULTS AVERAGED OVER {len(acc_base_raw)} RUNS")
print(f"Mean Baseline Acc: {mean_acc_base[-1]*100:.2f}% (+/- {std_acc_base[-1]*100:.2f}%)")
print(f"Mean Physics Acc:  {mean_acc_phys[-1]*100:.2f}% (+/- {std_acc_phys[-1]*100:.2f}%)")
print(f"NET STABLE GAIN:   {(mean_acc_phys[-1] - mean_acc_base[-1])*100:+.2f}%")
print("="*30)

plt.show()