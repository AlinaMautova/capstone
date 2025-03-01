import numpy as np

def calculate_metrics(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    precision = []
    recall = []
    f1_score = []
    
    total_correct = np.trace(confusion_matrix)
    total_samples = np.sum(confusion_matrix)
    accuracy = total_correct / total_samples
    
    for i in range(num_classes):
        TP = confusion_matrix[i, i]  # True Positives
        FP = np.sum(confusion_matrix[:, i]) - TP  # False Positives
        FN = np.sum(confusion_matrix[i, :]) - TP  # False Negatives
        
        rec = TP / (TP + FP) if (TP + FP) > 0 else 0
        prec = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        recall.append(round(prec, 3))
        precision.append(round(rec, 3))
        f1_score.append(round(f1, 3))
    
    return round(accuracy, 3), recall, precision, f1_score

# Given confusion matrix
confusion_matrix = np.array([
    [30, 20, 10],
    [50, 60, 10],
    [20, 20, 80]
])

# Calculate metrics
accuracy, precision, recall, f1_score = calculate_metrics(confusion_matrix)

# Display results
print("Accuracy:", accuracy)
print("\nClass-wise Metrics:")
print("Class | Precision | Recall | F1-score")
for i, cls in enumerate(['a', 'b', 'c']):
    print(f"{cls}    | {precision[i]}       | {recall[i]}   | {f1_score[i]}")
