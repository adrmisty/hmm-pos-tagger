# utils_eval.py = evaluation and analysis utility functions
# adriana r.f. (@adrmisty)
# vera s.g. (@verasenderowiczg)
# emiel v.j. (@emielvanderghinste)
# nov-2026

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score

def compute_metrics(gold_tags, pred_tags):
    """
    Calculates macro-averaged metrics and accuracy.
    Returns a dictionary of scores.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_tags, pred_tags, average='macro', zero_division=0
    )
    accuracy = accuracy_score(gold_tags, pred_tags)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# (!!!) specific error analysis
def analyze_tag_errors(gold_data: list, predictions: list, target_tag: str, error_type: str, mispredicted_as: str):
    """Prints specific sentences where the model failed for a given tag."""
    
    log_msg = f"Specific ({target_tag} -> {mispredicted_as})" if error_type == 'specific' else f"{error_type.upper()} ({target_tag})"
    print(f"\n--- Error analysis for: {target_tag} (Mode: {log_msg}) ---")
    
    count = 0
    for gold_sent, pred_sent in zip(gold_data, predictions):
        errs = []
        words = [w for w, _ in gold_sent]

        for i, ((gw, gt), (pw, pt)) in enumerate(zip(gold_sent, pred_sent)):
            is_err = False
            reason = ""
            
            if error_type == 'recall' and gt == target_tag and pt != target_tag:
                is_err, reason = True, f"GOLD: {gt} | PRED: {pt}"
            elif error_type == 'precision' and pt == target_tag and gt != target_tag:
                is_err, reason = True, f"PRED: {pt} | GOLD: {gt}"
            elif error_type == 'specific' and gt == target_tag and pt == mispredicted_as:
                is_err, reason = True, f"GOLD: {target_tag} | PRED: {pt}"

            if is_err: errs.append({'i': i, 'w': gw, 'r': reason})

        if errs:
            count += len(errs)
            print(f"\n[Errors: {len(errs)}] Sentence: {' '.join(words)}")
            for e in errs:
                print(f"  > '{e['w']}' (Idx {e['i']}) : {e['r']}")

    print(f"\n--- Total tag errors: {count} ---")

def plot_confusion_matrix(test_data_tagged: list, predictions: list, model_name: str = "unhinged-hmm-model"):
    """Generates a heatmap as a confusion matrix and a text report for the model performance."""
    
    gold_tags = [tag for sent in test_data_tagged for _, tag in sent] # flat lists
    pred_tags = [tag for sent in predictions for _, tag in sent]
    tags = sorted(list(set(gold_tags)))

    # (!!!) all metrics from sklearn: P, R, F1
    cm = confusion_matrix(gold_tags, pred_tags, labels=tags)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_tags, pred_tags, labels=tags, zero_division=0
    )
    
    print("\n=== Precision / Recall / F1-score per tag ===\n")
    print(classification_report(gold_tags, pred_tags, labels=tags, digits=4))

    # (!!!) confusion matrix heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=tags, yticklabels=tags,
        cbar_kws={'label': 'Number of Instances'}
    )
    
    accuracy = cm.trace() / cm.sum() if cm.sum() > 0 else 0.0
    plt.xlabel(f"Predicted tag\nAccuracy: {accuracy:.4f}", fontsize=14)
    plt.ylabel("Gold tag", fontsize=14)
    plt.title(f"Confusion matrix [{model_name}]", fontsize=16)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    _plot_metrics_table(tags, precision, recall, f1, support, model_name)

# --------------------------------------------------------------------------------------

def _plot_metrics_table(tags, precision, recall, f1, support, model_name):
    """Auxiliary: renders a text table of metrics as a matplotlib figure."""
    lines = [f"{'TAG':<6}{'P':>6}{'R':>6}{'F1':>6}{'N':>6}", "-" * 30]
    for tag, p, r, f, n in zip(tags, precision, recall, f1, support):
        lines.append(f"{tag:<6}{p:>6.2f}{r:>6.2f}{f:>6.2f}{int(n):>6}")
    
    lines.append("-" * 30)
    lines.append(f"{'MACRO':<6}{precision.mean():>6.2f}{recall.mean():>6.2f}{f1.mean():>6.2f}{'':>6}")

    plt.figure(figsize=(6, 8))
    plt.axis("off")
    plt.title(f"Detailed metrics\n{model_name}", fontsize=12)
    plt.text(0.0, 1.0, "\n".join(lines), family="monospace", va="top", fontsize=10)
    plt.tight_layout()
    plt.show()