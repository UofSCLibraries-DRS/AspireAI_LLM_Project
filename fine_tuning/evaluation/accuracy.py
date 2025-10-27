import pandas as pd
import json
import os
import csv
import concurrent
from tqdm import tqdm
from gaico.metrics import (
    BLEU,
    ROUGE,
    JSDivergence,
    JaccardSimilarity,
    LevenshteinDistance,
    BERTScore,
)


def calculate_accuracy_from_results(results_path: str):
    df = pd.read_csv(results_path)

    metric_classes = {
        "bleu": BLEU(),
        "rouge": ROUGE(),
        "js_div": JSDivergence(),
        "jaccard": JaccardSimilarity(),
        "levenshtein": LevenshteinDistance(),
        "bert_score": BERTScore(),
    }

    response_names = df.columns[2:]

    print(response_names)

    # Initialize metric classes
    metric_classes = {
        "bleu": BLEU(),
        "rouge": ROUGE(),
        "js_div": JSDivergence(),
        "jaccard": JaccardSimilarity(),
        "levenshtein": LevenshteinDistance(),
        "bert_score": BERTScore(),
    }

    def calculate_metrics(ground_truth, prediction):
        return {
            "BLEU": metric_classes["bleu"].calculate(ground_truth, prediction),
            "ROUGE-L": metric_classes["rouge"]
            .calculate(ground_truth, prediction)
            .get("rougeL", 0),
            "JSD": metric_classes["js_div"].calculate(ground_truth, prediction),
            "Jaccard": metric_classes["jaccard"].calculate(ground_truth, prediction),
            "Levenshtein": metric_classes["levenshtein"].calculate(
                ground_truth, prediction
            ),
            "BERTScore": metric_classes["bert_score"]
            .calculate(ground_truth, prediction)
            .get("f1", 0),
        }

    def process_row(row):
        ground_truth = row["answer"]
        return {
            response_name: calculate_metrics(ground_truth, row[response_name])
            for response_name in response_names
        }

    # Convert DataFrame to list of dictionaries
    data = df.to_dict("records")

    # Use concurrent.futures for parallelization
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks
        future_to_row = {executor.submit(process_row, row): row for row in data}

        # Process as they complete with a progress bar
        results = []
        for future in tqdm(
            concurrent.futures.as_completed(future_to_row),
            total=len(data),
            desc="Processing",
        ):
            results.append(future.result())

    # Restructure the results
    results = {
        response_name: [row[response_name] for row in results]
        for response_name in response_names
    }

    metric_names = results[next(iter(results))][0].keys()

    # Compute average for each metric
    all_metrics = {m: [] for m in metric_names}
    for resp_list in results.values():
        for row_metrics in resp_list:
            for m in metric_names:
                v = row_metrics.get(m)
                if v is not None:
                    all_metrics[m].append(float(v))

    averages = {
        m: (sum(vals) / len(vals) if vals else None) for m, vals in all_metrics.items()
    }

    json_path = f"{dir}/accuracy/json/{prompt_name}.json"

    with open(json_path, "w") as f:
        json.dump(averages, f, indent=4)

    print(averages)

    csv_path = "./summary_v2.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    csv_fieldnames = ["model_name", "prompt_name"] + list(metric_names)

    csv_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        if not csv_exists:
            writer.writeheader()

        # Build the row dict
        row = {"model_name": model_name, "prompt_name": prompt_name}
        # insert metric values; convert to native types (float) so csv writes them nicely
        for m in metric_names:
            row[m] = averages.get(m)

        writer.writerow(row)

    print("Appended row to CSV:", csv_path)
