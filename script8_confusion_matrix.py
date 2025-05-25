import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 8

# Data based on the problem description:
# Putative SST-Calb2 cell (Predicted Positive by the method)
#   - Actual Calb2 positive (True Positive, TP):    18
#   - Actual Calb2 negative (False Positive, FP):   17

# Putative SST-other cell (Predicted Negative by the method)
#   - Actual Calb2 positive (False Negative, FN):   11
#   - Actual Calb2 negative (True Negative, TN):    38

tp = 18
fp = 17
fn = 11
tn = 38

cm_data = np.array([[tn, fp],
                    [fn, tp]])

y_true = np.concatenate([np.ones(tp), np.zeros(fp), np.ones(fn), np.zeros(tn)])
y_pred = np.concatenate([np.ones(tp), np.ones(fp), np.zeros(fn), np.zeros(tn)])
f1 = f1_score(y_true, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20},
            xticklabels=['Putative SST-O', 'Putative SST-Calb2'],
            yticklabels=['Calb2 Negative', 'Calb2 Positive'])
plt.xlabel('predicted label')
plt.ylabel('molecular label')
plt.title(f'Confusion Matrix\nF1 Score (Calb2 positive class): {f1:.3f}')
plt.savefig("confusion_matrix.png", dpi=300)