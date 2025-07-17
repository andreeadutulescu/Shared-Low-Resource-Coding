import json
import subprocess
import tempfile
import os
from tqdm import tqdm

LANGUAGE = 'r' # 'r' or 'python'
MODEL = 'finetuned_gemma_ok' # deepseek gpt qwen llama translationsand_code_help phi4 codestral
FILENAME = f"{MODEL}_{LANGUAGE}_humaneval_test_generated"
# FILENAME = 'translations_help_r_humaneval_test_generated'

print(f"Evaluating {MODEL} {LANGUAGE}")
# print(f"Evaluating {FILENAME}")

if LANGUAGE == 'r':
    COMMAND = 'Rscript'
    SUFFIX = '.R'
elif LANGUAGE == 'python':
    COMMAND = 'python3.11'
    SUFFIX = '.py'


def run_code(r_code, timeout=5):
    tmp_file_name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix=SUFFIX, delete=False) as tmp_file:
            tmp_file_name = tmp_file.name
            tmp_file.write(r_code)
        
        result = subprocess.run(
            [COMMAND, tmp_file_name],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return (result.returncode, result.stdout, result.stderr)

    except subprocess.TimeoutExpired:
        return (None, None, "TimeoutExpired")

    except Exception as e:
        return (None, None, str(e))

    finally:
        if tmp_file_name and os.path.exists(tmp_file_name):
            os.remove(tmp_file_name)

dataset = json.load(open(f"Low-Resource-Coding/new_datasets/{FILENAME}.json", 'r'))

correct = 0
incorrect = 0
TIMEOUT = 5

for data in tqdm(dataset):
    if data['code']:
        if LANGUAGE == 'r':
            whole_code = data['code'] + '\n' + data['test']
        elif LANGUAGE == 'python':
            whole_code = data['code'] + '\n' + data['test'] + '\n' + f"check({data['entry_point']})\n"
    
        returncode, stdout, stderr = run_code(whole_code, TIMEOUT)
    else:
        returncode = 1
        stderr = "No code provided"
        
    if returncode == 0:
        correct += 1
        data['execution_status'] = 'CORRECT'
    else:
        incorrect += 1
        data['execution_status'] = stderr

print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {correct / (correct + incorrect)}")

json.dump(dataset, open(f"Low-Resource-Coding/new_datasets/errors_{FILENAME}.json", 'w'), indent=4)