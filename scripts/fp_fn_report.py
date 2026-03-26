from ultralytics import YOLO

def get_metrics():
    model = YOLO("model/best.pt")

    results = model.val(
        data="dataset/data.yaml",
        imgsz=416,
        batch=4,
        device="cpu",
        save_json=False
    )

    precision = results.results_dict["metrics/precision(B)"]
    recall = results.results_dict["metrics/recall(B)"]
    map50 = results.results_dict["metrics/mAP50(B)"]
    map50_95 = results.results_dict["metrics/mAP50-95(B)"]

    total_ground_truth = int(sum(results.nt_per_class))
    estimated_true_positives = int(round(recall * total_ground_truth))
    estimated_false_negatives = total_ground_truth - estimated_true_positives

    if precision > 0:
        estimated_false_positives = int(round((estimated_true_positives / precision) - estimated_true_positives))
    else:
        estimated_false_positives = 0

    return {
        "Precision": precision,
        "Recall": recall,
        "mAP50": map50,
        "mAP50-95": map50_95,
        "False Positives": estimated_false_positives,
        "False Negatives": estimated_false_negatives
    }