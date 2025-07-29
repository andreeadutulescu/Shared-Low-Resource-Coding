import subprocess
import json
from typing import List, Dict
from tqdm import tqdm

def check_r_syntax(code_snippets, timeout=3.0):
    results = []
    for code in tqdm(code_snippets):
        # Safely encode the R string literal via JSON so quotes/newlines are handled
        r_literal = json.dumps(code)
        cmd = [
            "Rscript", "-e",
            (
                f"res <- tryCatch(parse(text={r_literal}), "
                "error=function(e) { cat(e$message); quit(status=1) })"
            )
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if proc.returncode == 0:
                results.append({"code": code, "ok": "True", "error": ""})
            else:
                # R errors are printed to stdout
                err = proc.stdout.strip() or proc.stderr.strip()
                results.append({"code": code, "ok": "False", "error": err})
        except subprocess.TimeoutExpired:
            results.append({
                "code": code,
                "ok": "False",
                "error": f"Parsing timed out after {timeout} seconds"
            })
    return results
            

if __name__ == "__main__":
    
    dataset = json.load(open("Low-Resource-Coding/translated_datasets/unified_data_with_r_code_finetuning_train.json", "r"))
    samples = [data['r_code'] for data in dataset]

    report = check_r_syntax(samples)
    for entry, data in zip(report, dataset):
        data['r_code_check'] = entry
    with open("Low-Resource-Coding/translated_datasets/unified_data_with_r_code_finetuning_train_checked.json", "w") as f:
        json.dump(dataset, f, indent=4)
