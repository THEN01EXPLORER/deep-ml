
from collections import Counter

def balance_undersample(data):
    if not data:
        return []

    counts = Counter(label for sample, label in data)
    min_count = min(counts.values())

    kept = Counter()
    result = []

    for item in data:
        sample, label = item

        if kept[label] < min_count:
            result.append(item)
            kept[label] += 1

    return result
