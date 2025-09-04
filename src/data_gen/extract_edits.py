import os
import re
import json
import argparse
from tqdm import tqdm

def extract(generated_eidts):
    lines = generated_eidts.strip().split('\n')

    edit_list = []
    word_list = []

    for line in lines:
        match = re.match(r"(\d+)\.\s(.+)", line)
        if match:
            index = match.group(1)
            edit_content = match.group(2)

            word_match = re.search(r"\[(.*?)\]", edit_content)
            if word_match:
                word = word_match.group(1)
                edit_content_clean = re.sub(r"\[(.*?)\]", word, edit_content).strip()
                edit_list.append(edit_content_clean)
                word_list.append(word)
    return edit_list, word_list


def extract_edits(dataset, dataset_name):
    extract_edits_dataset = []
    extract_edits_failed = []
    extract_edits_failed_idx = []
    
    if dataset_name == 'esnli':
        for data in tqdm(dataset, desc="Processing"):
            premise = data['premise']
            hypothesis = data['hypothesis']
            
            premise_generated_edits = data['edit_gen_premise']
            hypothesis_generated_edits = data['edit_gen_hypothesis']
            
            premise_edit_list = []
            hypothesis_edit_list = []
            premise_word_list = []
            hypothesis_word_list = []
            
            premise_edit_list, premise_word_list = extract(premise_generated_edits)
            hypothesis_edit_list, hypothesis_word_list = extract(hypothesis_generated_edits)    

            if len(premise_edit_list) != len(hypothesis_edit_list):
                extract_edits_failed.append(data)
                extract_edits_failed_idx.append(data['idx'])
                continue

            eidx = 0
            for i, premise_edit in enumerate(premise_edit_list):
                extract_edits_dataset.append({
                    'idx': data['idx'],
                    'eidx': eidx,
                    'premise': premise_edit,
                    'hypothesis': hypothesis,
                    "choices": data['choices'],
                    'edit_pos': 'premise',
                    'edit_word': premise_word_list[i],
                })
                eidx += 1
            for j, hypothesis_edit in enumerate(hypothesis_edit_list):
                extract_edits_dataset.append({
                    'idx': data['idx'],
                    'eidx': eidx,
                    'premise': premise,
                    'hypothesis': hypothesis_edit,
                    "choices": data['choices'],
                    'edit_pos': 'hypothesis',
                    'edit_word': hypothesis_word_list[j],
                })
                eidx += 1
    elif dataset_name == 'comve':
        for data in tqdm(dataset, desc="Processing"):
            sentence0 = data['sentence0']
            sentence1 = data['sentence1']
            
            sentence0_generated_edits = data['edit_gen_sentence0']
            sentence1_generated_edits = data['edit_gen_sentence1']            
            
            sentence0_edit_list = []
            sentence1_edit_list = []
            sentence0_word_list = []
            sentence1_word_list = []
            
            sentence0_edit_list, sentence0_word_list = extract(sentence0_generated_edits)
            sentence1_edit_list, sentence1_word_list = extract(sentence1_generated_edits)
            
            if len(sentence0_edit_list) != len(sentence1_edit_list):
                extract_edits_failed.append(data)
                extract_edits_failed_idx.append(data['idx'])
                continue
            
            eidx = 0
            for i, sentence0_edit in enumerate(sentence0_edit_list):
                extract_edits_dataset.append({
                    'idx': data['idx'],
                    'eidx': eidx,
                    'sentence0': sentence0_edit,
                    'sentence1': sentence1,
                    'choices': data['choices'],
                    'edit_pos': 'sentence0',
                    'edit_word': sentence0_word_list[i],
                })
                eidx += 1
            for j, sentence1_edit in enumerate(sentence1_edit_list):
                extract_edits_dataset.append({
                    'idx': data['idx'],
                    'eidx': eidx,
                    'sentence0': sentence0,
                    'sentence1': sentence1_edit,
                    'choices': data['choices'],
                    'edit_pos': 'sentence1',
                    'edit_word': sentence1_word_list[j],
                })
                eidx += 1
    elif dataset_name == 'ecqa':
        for data in tqdm(dataset, desc="Processing"):
            question_generated_edits = data['edit_gen_question']

            question_edit_list = []
            question_word_list = []

            question_edit_list, question_word_list = extract(question_generated_edits)

            if len(question_edit_list) != 20:
                extract_edits_failed.append(data)
                extract_edits_failed_idx.append(data['idx'])
                continue

            eidx = 0
            for i, question_edit in enumerate(question_edit_list):
                extract_edits_dataset.append({
                    'idx': data['idx'],
                    'eidx': eidx,
                    'question': question_edit,
                    'choices': data['choices'],
                    'edit_pos': 'question',
                    'edit_word': question_word_list[i],
                })
                eidx += 1

    return extract_edits_dataset, extract_edits_failed, extract_edits_failed_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', "--dir", type=str, default='data')
    parser.add_argument('-d', '--dataset', type=str, default='esnli', choices=['comve', 'ecqa', 'esnli'])
    parser.add_argument('-dt','--dataset_type', type=str, default='org', choices=['org', 'failed', 'final'])
    args = parser.parse_args()

    dir = args.dir
    dataset_name = args.dataset
    dataset_type = args.dataset_type

    edits_gen_path = os.path.join(dir, 'counterfactual', dataset_name, f"gen_{dataset_type}.json")
    with open(edits_gen_path, 'r') as f:
        edits_gen = json.load(f)

    extract_edits_dataset, extract_edits_failed, extract_edits_failed_idx = extract_edits(edits_gen, dataset_name)

    edits_ext = {
        'total': len(extract_edits_dataset),
        'num_failed': len(extract_edits_failed_idx),
        'extract_edits_failed_idx': extract_edits_failed_idx,
        'extract_edits_failed': extract_edits_failed,
        'extract_edits_dataset': extract_edits_dataset
    }

    edits_ext_path = os.path.join(dir, 'counterfactual', dataset_name, f"ext_{dataset_type}.json")
    
    output_dir = os.path.dirname(edits_ext_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(edits_ext_path, 'w') as f:
        json.dump(edits_ext, f, indent=4)
    
main()
