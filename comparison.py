import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# --- Configuration ---
DIR_CTRL = "c0.0"
DIR_PHYS = "c1.5"
OUTPUT_DIR = "comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(directory):
    path = os.path.join(directory, "experiment_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find results at {path}")
    with open(path, 'r') as f:
        return json.load(f)

# 1. Load Data
log_data_0 = load_data(DIR_CTRL)
log_data_25 = load_data(DIR_PHYS)

# 2. Extract Histories
test_acc_0 = np.array(log_data_0['training_history']['test_accuracy'])
test_acc_25 = np.array(log_data_25['training_history']['test_accuracy'])
train_acc_0 = np.array(log_data_0['training_history']['train_accuracy'])
train_acc_25 = np.array(log_data_25['training_history']['train_accuracy'])

# 3. Calculate Accuracy Delta
delta_test = test_acc_25 - test_acc_0
delta_train = train_acc_25 - train_acc_0

# 4. Extract Confusion Matrices
cm_te_0 = np.array(log_data_0['final_results']['confusion_matrix_test'])
cm_te_25 = np.array(log_data_25['final_results']['confusion_matrix_test'])
cm_delta = cm_te_25 - cm_te_0

# --- FIGURE 1: Accuracy Comparison & Delta ---
plt.figure(figsize=(14, 6))

# Plot A: Absolute Accuracies
plt.subplot(1, 2, 1)
plt.plot(test_acc_0, label="Linear (k=0)", color='black', linestyle='--', alpha=0.7)
plt.plot(test_acc_25, label="Physics (k=2.5)", color='royalblue', linewidth=2)
plt.title("Test Accuracy: Physics vs. Linear")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)

# Plot B: The "Physics Bonus" (Delta)
plt.subplot(1, 2, 2)
plt.axhline(0, color='black', linewidth=1)
plt.fill_between(range(len(delta_test)), delta_test, 0, 
                 where=(delta_test > 0), color='green', alpha=0.3, label='Physics Gain')
plt.fill_between(range(len(delta_test)), delta_test, 0, 
                 where=(delta_test < 0), color='red', alpha=0.3, label='Physics Loss')
plt.plot(delta_test, color='darkgreen', linewidth=1.5)
plt.title(f"Accuracy Delta (k={log_data_25['system_parameters']['k_coulomb']} minus k=0)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy Difference")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png")

# --- FIGURE 2: Differential Confusion Matrix ---
plt.figure(figsize=(10, 8))
# Use a diverging colormap: 
# Blue = Physics did BETTER (caught more correct or made fewer mistakes)
# Red = Physics did WORSE (missed more or made more mistakes)
sns.heatmap(cm_delta, annot=True, fmt='d', cmap='RdBu', center=0)

plt.title("Differential Confusion Matrix (Physics - Linear)\nPositive on Diagonal = Physics Win | Positive off Diagonal = Physics Error")
plt.xlabel("Predicted Digit")
plt.ylabel("True Digit")
plt.savefig(f"{OUTPUT_DIR}/differential_confusion.png")

# --- TEXT SUMMARY ---
final_gain = (test_acc_25[-1] - test_acc_0[-1]) * 100
print(f"--- Analysis Complete ---")
print(f"Linear Final Acc:  {test_acc_0[-1]*100:.2f}%")
print(f"Physics Final Acc: {test_acc_25[-1]*100:.2f}%")
print(f"Net Physics Gain:  {final_gain:+.2f}%")

plt.show()