import json
import zipfile
import glob
import os
import shutil
import re

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

def postprocess_text_classification(preds, labels):
    preds = [str(pred).strip() for pred in preds]
    labels = [str(label).strip() for label in labels]
    return preds, labels


def normalize_news_category_prediction(pred):
    text = str(pred)
    lower_text = text.lower()
    marker = "category:"
    if marker in lower_text:
        pos = lower_text.rfind(marker)
        text = text[pos + len(marker):]
    text = text.strip().splitlines()[0].strip().strip('"').strip()
    return text

def postprocess_text_generation(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def create_metric_f1_accuracy(all_labels):
    def create_mapping(x):
        try:
            return all_labels.index(x)
        except:
            return -1
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x) for x in decoded_preds]
        decoded_labels = [create_mapping(x) for x in decoded_labels]
        result_acc = accuracy_score(decoded_labels, decoded_preds)
        result_f1 = f1_score(
            decoded_labels,
            decoded_preds,
            labels=list(range(len(all_labels))),
            average="macro",
            zero_division=0,
        )
        result = {"accuracy": result_acc, "f1": result_f1}
        return result
    return compute_metrics

def create_metric_mae_rmse():
    def create_mapping(x, y):
        x_str = str(x).strip()
        try:
            return float(x_str)
        except:
            score_match = re.search(r"score\s*:\s*([1-5])", x_str, flags=re.IGNORECASE)
            if score_match:
                return float(score_match.group(1))
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):
                return 1.0
            else:
                return 5.0
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x, y) for x, y in zip(decoded_preds, decoded_labels)]
        decoded_labels = [create_mapping(x, x) for x in decoded_labels]
        result_mae = mean_absolute_error(decoded_labels, decoded_preds)
        result_rmse = mean_squared_error(decoded_labels, decoded_preds) ** 0.5
        result = {"MAE": result_mae, "RMSE": result_rmse}
        return result
    return compute_metrics

def create_metric_rouge():
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_generation(decoded_preds, decoded_labels)
        preds = decoded_preds
        labels = decoded_labels
        if not labels:
            return {"rouge-1": 0.0, "rouge-L": 0.0}
        rouge_1_total = 0.0
        rouge_l_total = 0.0
        for pred_text, label_list in zip(preds, labels):
            label_text = label_list[0] if label_list else ""
            headline_match = re.search(r"(?is).*headline:\s*(.*)$", pred_text)
            title_match = re.search(r"(?is).*title:\s*(.*)$", pred_text)
            if title_match:
                pred_text = title_match.group(1).strip()
            elif headline_match:
                pred_text = headline_match.group(1).strip()
            pred_tokens = pred_text.split()
            label_tokens = label_text.split()
            if not label_tokens:
                continue
            label_counts = {}
            for token in label_tokens:
                label_counts[token] = label_counts.get(token, 0) + 1
            overlap = 0
            for token in pred_tokens:
                if label_counts.get(token, 0) > 0:
                    overlap += 1
                    label_counts[token] -= 1
            rouge_1_total += overlap / len(label_tokens)

            m = len(label_tokens)
            n = len(pred_tokens)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if label_tokens[i - 1] == pred_tokens[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
            rouge_l_total += dp[m][n] / m if m else 0.0

        count = len(labels)
        rouge_1 = rouge_1_total / count if count else 0.0
        rouge_l = rouge_l_total / count if count else 0.0
        return {"rouge-1": rouge_1, "rouge-L": rouge_l}
    return compute_metrics

class LaMPEvaluation(object):
    
    def __init__(self, all_golds_zip_file_addr = None, single_gold_json_file_addr = None, extract_addr = "./tmp") -> None:
        assert all_golds_zip_file_addr or single_gold_json_file_addr, "The golds should be provided for all datasets or at least one."
        assert not (all_golds_zip_file_addr and single_gold_json_file_addr), "The golds should be provided using zip file or json file not both."
        self.tasks_golds = dict()
        self.extract_addr = extract_addr
        self.evaluate_all_is_possible = False
        if all_golds_zip_file_addr:
            os.makedirs(self.extract_addr, exist_ok=True)
            with zipfile.ZipFile(all_golds_zip_file_addr, 'r') as zobj:
                zobj.extractall(path = extract_addr)
            for file_addr in glob.glob(os.path.join(self.extract_addr, "**/*.json"), recursive=True):
                with open(file_addr) as file:
                    task = json.load(file)
                    self.tasks_golds[task['task']] = task['golds']
            self._empty_dir(self.extract_addr)
            self.evaluate_all_is_possible = True
        if single_gold_json_file_addr:
            with open(single_gold_json_file_addr) as file:
                    task = json.load(file)
                    self.tasks_golds[task['task']] = task['golds']
    
    def _empty_dir(self, directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    def _get_all_gold_ids(self, task_name):
        return set([sample['id'] for sample in self.tasks_golds[task_name]])
    
    def _get_all_ids(self, input):
        return set([sample['id'] for sample in input])
    
    def evaluate_all(self, predicts_zipfile_addr):
        assert self.evaluate_all_is_possible, "You did not provide golds for all tasks."
        with zipfile.ZipFile(predicts_zipfile_addr, 'r') as zobj:
            zobj.extractall(path = self.extract_addr)
        results_raw = dict()
        all_task_names = set()
        for file_addr in glob.glob(os.path.join(self.extract_addr, "**/*.json"), recursive=True):
            with open(file_addr) as file:
                preds = json.load(file)
            all_task_names.add(preds['task'])
            results_raw[preds['task']] = self._evaluate_task(preds['golds'], preds['task'])
        self._empty_dir(self.extract_addr)
        assert len(all_task_names) == 7, "The provided results do not cover all the tasks in the benchmark."
        return results_raw

    def evaluate_task(self, predicts_json_addr, task_name):
        with open(predicts_json_addr) as file:
            preds = json.load(file)
        assert preds['task'] == task_name, "The provided task_name and the results do not match."
        assert preds['task'] in self.tasks_golds.keys(), "The provided golds cannot be used to evaluate this task."
        return self._evaluate_task(preds['golds'], task_name)

    def _evaluate_task(self, predictions, task_name):
        golds_dict = {y['id']:y['output'] for y in self.tasks_golds[task_name]}
        if task_name == "LaMP_2N":
            preds_dict = {x['id']: normalize_news_category_prediction(x['output']) for x in predictions}
        else:
            preds_dict = {x['id']:x['output'] for x in predictions}
        
        gold_ids = self._get_all_gold_ids(task_name)
        pred_ids = self._get_all_ids(predictions)

        assert gold_ids == pred_ids, "Predictions ids and gold ids do not match. {}".format(gold_ids-pred_ids)

        if task_name in ["LaMP_1", "LaMP_2N", "LaMP_2M"]:
            metric = create_metric_f1_accuracy(self._get_labels(task_name))
        elif task_name == "LaMP_3":
            metric = create_metric_mae_rmse()
        else:
            metric = create_metric_rouge()
        
        gold_ids = list(gold_ids)
        golds = [golds_dict[id] for id in gold_ids]
        preds = [preds_dict[id] for id in gold_ids]
        return metric(preds, golds)
    
    def _get_labels(self, task_name):
        if task_name == "LaMP_1":
            return ["[1]", "[2]"]
        elif task_name == "LaMP_2N":
            return ["food & drink", "sports", "education", "parents", "religion", "travel", "business", "crime", "science & technology", "culture & arts", "entertainment", "politics", "women", "style & beauty", "healthy living"]
        elif task_name == "LaMP_2M":
            return ["sci-fi", "based on a book", "comedy", "action", "twist ending", "dystopia", "dark comedy", "classic", "psychology", "fantasy", "romance", "thought-provoking", "social commentary", "violence", "true story"]        
        elif task_name == "LaMP_3":
            return ["1", "2", "3", "4", "5"]
        else:
            raise ValueError("Invalid task_name")