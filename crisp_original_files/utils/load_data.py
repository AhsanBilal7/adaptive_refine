import re
import json 
import random
import os
from .config import *


# ---------------------------------------------------------------------------
# HuggingFace dataset identifiers
# Map each local dataset name → (hf_repo, hf_subset, hf_split)
# ---------------------------------------------------------------------------
HF_DATASET_MAP = {
    # Math / arithmetic
    'gsm8k':            ('openai/gsm8k',                    'main',             'test'),
    'gsmic':            ('openai/gsm8k',                    'main',             'test'),   # GSM-IC variant
    'gsmp':             ('openai/gsm8k',                    'main',             'test'),   # GSM-Plus variant
    'math':             ('HuggingFaceH4/MATH-500',           'default',              'test'),
    'aime24':           ('math-ai/aime24',           None,               'test'),
    'aqua':             ('aqua_rat',                        'raw',              'validation'),
    'olympiadbench':    ('GAIR/OlympiadBench',              'OE_TO_maths_en_COMP', 'test'),
    's1k':              ('simplescaling/s1K',               None,               'train'),
    's1k-1.1':          ('simplescaling/s1K-1.1',           None,               'train'),
    'deepmath':         ('zwhe99/DeepMath-103K',            None,               'train'),

    # Science / graduate
    'gpqa_diamond':     ('Idavidrein/gpqa',                 'gpqa_diamond',     'train'),
    'gpqa_experts':     ('Idavidrein/gpqa',                 'gpqa_experts',     'train'),
    'gpqa_extended':    ('Idavidrein/gpqa',                 'gpqa_extended',    'train'),
    'gpqa_main':        ('Idavidrein/gpqa',                 'gpqa_main',        'train'),

    # Commonsense / NLI
    'csqa':             ('commonsense_qa',                  None,               'validation'),
    'wino':             ('winogrande',                      'winogrande_l',     'validation'),
    'siqa':             ('social_i_qa',                     None,               'validation'),

    # Symbolic / other
    'lastletter':       ('ChilleD/LastLetterConcat',        None,               'test'),
    'coinflip':         ('skrishna/coin_flip',              None,               'validation'),
    'mirage':           ('MIRAGE-Reasoning/MIRAGE',         None,               'test'),
    'addition':         (None, None, None),   # no public HF dataset; generate locally
    'product':          (None, None, None),
}


def _load_from_huggingface(dataset: str, nsamples: int, mode: str) -> list:
    """
    Fall-back loader: pull the dataset from HuggingFace Hub when the local
    .jsonl file is absent.  Returns a list of raw dicts in the same format
    that the local JSONL loader would produce.
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace fallback. "
            "Install it with:  pip install datasets"
        ) from exc

    if dataset not in HF_DATASET_MAP:
        raise ValueError(
            f"Dataset '{dataset}' is not registered in HF_DATASET_MAP. "
            "Add an entry or provide a local data file."
        )

    repo, subset, hf_split = HF_DATASET_MAP[dataset]

    if repo is None:
        raise ValueError(
            f"Dataset '{dataset}' has no HuggingFace source. "
            "Please provide a local data file."
        )

    # Map local mode names ('dev', 'train', 'test') to HF split names
    split_map = {
        'dev':   hf_split or 'validation',
        'train': 'train',
        'test':  'test',
    }
    split = split_map.get(mode, hf_split or mode)

    print(f"[load_data] Local file not found. Downloading '{dataset}' "
          f"from HuggingFace: {repo} (subset={subset}, split={split}) …")

    load_kwargs = dict(path=repo, split=split, trust_remote_code=True)
    if subset:
        load_kwargs['name'] = subset

    hf_ds = hf_load(**load_kwargs)
    items = [dict(row) for row in hf_ds]
    return items


# ---------------------------------------------------------------------------
# Existing helpers (unchanged)
# ---------------------------------------------------------------------------

def add_bias_sentence(prompt, bias_sentence):
    pattern = r"(#|##) Reasoning"
    matches = list(re.finditer(pattern, prompt))
    if not matches:
        return prompt + bias_sentence + '\n'
    last_match = matches[-1].start()
    return prompt[:last_match] + bias_sentence + '\n' + prompt[last_match:]


def load_prompt(dataset, method, n_examples=3):
    prompt_file = f'./prompts/{method}/{dataset}_{n_examples}.txt'
    with open(prompt_file, 'r') as fin:
        lines = [line.strip() for line in fin.readlines()]
    return '\n'.join(lines)


def format_prompt(full_prompt, item):
    fields = re.findall(r'\{\{\w+\}\}', full_prompt)
    for field in fields:
        if field[2:-2] not in item.keys():
            continue
        value = item[field[2:-2]]
        if isinstance(value, list):
            value = '\n'.join(value)
        full_prompt = full_prompt.replace(field, value)
    return full_prompt


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except Exception:
        return None


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx:right_brace_idx + 1] if right_brace_idx is not None else None


# ---------------------------------------------------------------------------
# Core dataset loader
# ---------------------------------------------------------------------------

def load_dataset(dataset, nsamples, mode='dev'):
    """
    Load dataset items from the local JSONL file when it exists; otherwise
    fall back to HuggingFace Hub automatically.
    """
    data_file = f'./data/{dataset}/{mode}.jsonl'

    # ------------------------------------------------------------------ #
    # Primary path: local JSONL                                           #
    # ------------------------------------------------------------------ #
    if os.path.exists(data_file):
        with open(data_file, 'r') as fin:
            items = [json.loads(line) for line in fin]
    # ------------------------------------------------------------------ #
    # Fallback path: HuggingFace Hub                                      #
    # ------------------------------------------------------------------ #
    else:
        items = _load_from_huggingface(dataset, nsamples, mode)

    # ------------------------------------------------------------------ #
    # Per-dataset normalisation (unchanged from original)                 #
    # ------------------------------------------------------------------ #
    for idx, item in enumerate(items):
        if 'id' not in item:
            item['id'] = f'{dataset.upper()}_Q{idx}'

        if dataset in ['gsm8k', 'gsmic', 'gsmp']:
            question = item['question']
            parts = item['answer'].split('####')
            item['question'] = question
            item['reason']   = parts[0].strip()
            item['answer']   = str(eval(parts[1].strip().replace(',', '')))

        elif dataset in ['gpqa_diamond', 'gpqa_experts', 'gpqa_extended', 'gpqa_main']:
            question   = item['Question']
            label_dic  = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            label_num  = random.randint(0, 3)
            label      = label_dic[label_num]
            options    = [f"{item[f'Incorrect Answer {i+1}'].strip()}" for i in range(3)]
            options.insert(label_num, f"{item['Correct Answer'].strip()}")
            options    = [f"({label_dic[i]}) {options[i]}" for i in range(4)]
            item['question'] = question
            item['answer']   = label
            item['options']  = options
            item['reason']   = item['Explanation']

        elif dataset == 'aqua':
            item['answer'] = item['correct']
            item['reason'] = item['rationale']


        elif dataset == 'math':
            item['question'] = item['problem']
            item['reason']   = item['solution']
            item['answer']   = remove_boxed(last_boxed_only_string(item['solution']))
        
        elif dataset == 'aime24':
            item['question'] = item['problem']
            item['answer']   = remove_boxed(last_boxed_only_string(item['solution']))

        elif dataset == 'olympiadbench':
            item['reason'] = item['solution'][0]
            item['answer'] = item['final_answer'][0]

        elif dataset == 'csqa':
            question       = item['question']['stem']
            label          = item['answerKey']
            options        = [f"({tup['label']}) {tup['text']}"
                              for tup in item['question']['choices']]
            item['question'] = question
            item['id']       = f'{dataset.upper()}_Q{idx}'
            item['answer']   = label
            item['options']  = options

        elif dataset == 'wino':
            question      = item['sentence']
            label_dic     = {'1': 'A', '2': 'B'}
            label         = label_dic[item['answer']]
            options       = [f"(A) {item['option1']}", f"(B) {item['option2']}"]
            item['question'] = question
            item['answer']   = label
            item['options']  = options

        elif dataset == 'siqa':
            item['question'] = item['context'] + ' ' + item['question']

    random.shuffle(items)

    if nsamples == 50:
        return items[200:250]
    return items[:nsamples] if nsamples > 0 else items


# ---------------------------------------------------------------------------
# Answer extraction helpers (unchanged)
# ---------------------------------------------------------------------------

def extract_logic(answer):
    pattern1 = r"The answer is:?\s*([A-E])"
    pattern2 = r"The answer is:?\s*(True|False|Unknown)"
    pattern4 = r"(True|False|Unknown)"

    option = None
    match = re.search(pattern1, answer, re.IGNORECASE)
    if match:
        option = match.group(1)

    if not option:
        match = re.search(pattern2, answer, re.IGNORECASE)
        if match:
            option = match.group(1)

    if not option:
        match = re.search(pattern4, answer, re.IGNORECASE)
        if match:
            option = match.group(1)

    return option


def isnum(string):
    if string.isdigit():
        return True
    try:
        float(string)
        return True
    except ValueError:
        return False


def extract_answer(output, dataset, **kwargs):
    if not output:
        return None
    pred = None

    if dataset in ['addition', 'product', 'gsmic', 'gsmp', 'gsm8k']:
        if '# Answer:' in output or 'answer is:' in output:
            pred = output.split(':')[-1].strip().strip('.').strip()
        else:
            pred = output
        pred = pred.replace(',', '')
        if not isnum(pred):
            pred = re.findall(r'\d*\.?\d+', pred)
            if not pred:
                return None
            pred = pred[-1].strip()

    elif dataset in ['lastletter', 'coinflip']:
        answer  = output.split(':')[-1]
        matches = re.findall(r'[a-zA-Z]+', answer)
        pred    = matches[0] if matches else None

    elif dataset in ['math', 'olympiadbench', 'aime24', 's1k', 's1k-1.1', 'deepmath']:
        match_string = last_boxed_only_string(output)
        pred = remove_boxed(match_string) if match_string else None

    elif dataset == 'mirage':
        if kwargs['task'] == 'inductive':
            pred = extract_rule(output, kwargs['context'])
        else:
            pred = extract_fact(output, kwargs['context'])

    else:
        match_string = last_boxed_only_string(output)
        if match_string:
            pred = remove_boxed(match_string).strip()
        else:
            matches = re.findall(r'[A-E]$|\([A-E]\)|[A-E]\.|[A-E]\,', output)
            option  = [m.strip('()., ') for m in matches]
            pred    = option[-1] if option else None

    return pred if pred else extract_logic(output.split(':')[-1])


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json_data(path):
    if not os.path.exists(path):
        print(f'path {path} not exists')
        return [None]
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def write_json_data(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------------
# DataLoader & InterventionData (unchanged except minor style)
# ---------------------------------------------------------------------------

class DataLoader(object):
    def __init__(self, dataset, n_samples) -> None:
        self.dataset   = dataset
        self.n_samples = n_samples

    def load_data(self, method, n_examples, mode='dev'):
        data   = load_dataset(self.dataset, self.n_samples, mode=mode)
        prompt = load_prompt(self.dataset, method, n_examples)
        for item in data:
            item['raw_question'] = item['question']
            if 'context' in item and item['context']:
                item['raw_question'] = item['context'] + ' ' + item['raw_question']
            if 'options' in item:
                item['raw_question'] = item['raw_question'] + ' ' + ' '.join(item['options'])
            item['question'] = format_prompt(prompt, item)
        return data


class InterventionData:
    def __init__(self, msg, tokenizer) -> None:
        self.question  = None
        self.cot       = None
        self.pred      = None
        self.load_data(msg)

        self.question_end         = None
        self.cot_end              = None
        self.cot_input_ids        = None
        self.pred_ids             = None
        self.tokenize_data(tokenizer)

        self.cot_intervention_idx = {}
        self.get_intervention_idx()

    def load_data(self, msg):
        self.question = msg['question']
        self.cot      = msg['response'].split('\n# Answer:')[0]
        self.pred     = msg['answer']

    def tokenize_data(self, tokenizer):
        cot_input          = self.question + self.cot + f'\n# Answer:\nThe answer is: {self.pred}'
        self.question_end  = len(tokenizer(self.question, return_tensors="pt").input_ids[0])
        self.cot_end       = len(tokenizer(self.question + self.cot, return_tensors="pt").input_ids[0])
        self.cot_input_ids = tokenizer(cot_input, return_tensors="pt").input_ids
        pred_len           = len(tokenizer(self.pred, return_tensors="pt").input_ids[0])
        self.pred_ids      = self.cot_input_ids[:, -pred_len]

    def get_intervention_idx(self):
        interval_length = max(self.cot_end - self.question_end, 1)
        start = self.question_end
        for cnt in range(1, 11):
            end = interval_length * cnt // 10 + self.question_end
            if end == start:
                self.cot_intervention_idx[cnt] = list(range(start, start + 1))
            else:
                self.cot_intervention_idx[cnt] = list(range(start, end))
            start = end