CLASS_WEIGHTS = {
    "Gun": 10,
    "Knife": 8,
    "Scissors": 5,
    "Pliers": 3,
    "Wrench": 2
}

def calculate_risk(detections):
    if not detections:
        return 0, "Low"

    # Get highest-risk class present (ignore confidence)
    highest_class = None
    highest_weight = 0

    for class_name, confidence in detections:
        weight = CLASS_WEIGHTS.get(class_name, 1)
        if weight > highest_weight:
            highest_weight = weight
            highest_class = class_name

    # Risk score is directly the class weight (out of 10)
    score = highest_weight

    # Risk level based on class severity
    if highest_class == "Gun":
        level = "High"
    elif highest_class == "Knife":
        level = "High"
    elif highest_class in ["Scissors"]:
        level = "Medium"
    else:
        level = "Low"

    return score, level