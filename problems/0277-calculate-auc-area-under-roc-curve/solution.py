def calculate_auc(y_true: list[int], y_scores: list[float]) -> float:
    # Count total positive and negative ground truth instances
    positives = sum(y_true)
    negatives = len(y_true) - positives

    # Edge case: If all labels belong to a single class, ROC cannot be defined
    if positives == 0 or negatives == 0:
        return 0.0

    # Sort instances in descending order of predicted score
    sorted_pairs = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)

    # Track points along the ROC curve starting at (FPR=0, TPR=0)
    fpr_points = [0.0]
    tpr_points = [0.0]

    current_tp = 0
    current_fp = 0
    i = 0
    n = len(sorted_pairs)

    # Group predictions with the same score to handle ties correctly
    while i < n:
        score = sorted_pairs[i][0]
        while i < n and sorted_pairs[i][0] == score:
            if sorted_pairs[i][1] == 1:
                current_tp += 1
            else:
                current_fp += 1
            i += 1

        fpr_points.append(current_fp / negatives)
        tpr_points.append(current_tp / positives)

    # Calculate area using trapezoidal numerical integration
    auc = 0.0
    for j in range(1, len(fpr_points)):
        delta_fpr = fpr_points[j] - fpr_points[j - 1]
        avg_tpr = (tpr_points[j] + tpr_points[j - 1]) / 2.0
        auc += avg_tpr * delta_fpr

    return round(auc, 4)