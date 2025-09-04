import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

def generate(prompt, model_version="gpt-4o-2024-08-06"):
    completion = client.chat.completions.create(
        model=model_version,
        messages=[
            {"role": "system", "content": "You are a helpful assistant!"},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


def generate_edits(dataset, dataset_name):
    edits_dataset = []
    
    if dataset_name in ['esnli', 'comve']:
        current_dir = os.path.dirname(__file__)
        prompt_prefix_path = os.path.join(current_dir, 'edit_prompt_10.txt')
        with open(prompt_prefix_path, 'r') as f:
            prompt_prefix = f.read()

        if dataset_name == 'esnli':
            for data in tqdm(dataset, desc="Processing"):
                premise = data['premise']
                hypothesis = data['hypothesis']

                premise_prompt = prompt_prefix + premise + "\nOutput:\n"
                hypothesis_prompt = prompt_prefix + hypothesis + "\nOutput:\n"

                premise_generated_edits = ""
                hypothesis_generated_edits = ""
                
                premise_generated_edits = generate(premise_prompt)
                hypothesis_generated_edits = generate(hypothesis_prompt)
                
                data['edit_gen_premise'] = premise_generated_edits
                data['edit_gen_hypothesis'] = hypothesis_generated_edits
                
                edits_dataset.append(data)
        elif dataset_name == 'comve':
            for data in tqdm(dataset, desc="Processing"):
                sentence0 = data['sentence0']
                sentence1 = data['sentence1']

                sentence0_prompt = prompt_prefix + sentence0 + "\nOutput:\n"
                sentence1_prompt = prompt_prefix + sentence1 + "\nOutput:\n"

                sentence0_generated_edits = ""
                sentence1_generated_edits = ""
                
                sentence0_generated_edits = generate(sentence0_prompt)
                sentence1_generated_edits = generate(sentence1_prompt)
                
                data['edit_gen_sentence0'] = sentence0_generated_edits
                data['edit_gen_sentence1'] = sentence1_generated_edits
                
                edits_dataset.append(data)                
    elif dataset_name in ['ecqa']:
        current_dir = os.path.dirname(__file__)
        prompt_prefix_path = os.path.join(current_dir, 'edit_prompt_20.txt')
        with open(prompt_prefix_path, 'r') as f:
            prompt_prefix = f.read()

        if dataset_name == 'ecqa':
            for data in tqdm(dataset, desc="Processing"):
                question = data['question']

                question_prompt = prompt_prefix + question + "\nOutput:\n"

                question_generated_edits = ""
                
                question_generated_edits = generate(question_prompt)
                
                data['edit_gen_question'] = question_generated_edits
                
                edits_dataset.append(data) 
    return edits_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', "--dir", type=str, default='data')
    parser.add_argument('-d', '--dataset', type=str, default='esnli', choices=['comve', 'ecqa', 'esnli'])
    parser.add_argument('-dt','--dataset_type', type=str, default='org', choices=['org', 'failed'])
    parser.add_argument('-n', '--num_data', type=int, default=3)
    args = parser.parse_args()

    dir = args.dir
    dataset_name = args.dataset
    dataset_type = args.dataset_type
    num_data = args.num_data

    if dataset_type == 'org':
        dataset_path = os.path.join('data/formatted', dataset_name, 'test.json')
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)[:num_data]
    elif dataset_type == 'failed':
        dataset_path = os.path.join(dir, 'counterfactual', dataset_name, "ext_org.json")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)['extract_edits_failed']

    edits_dataset = generate_edits(dataset, dataset_name)

    if dataset_type == 'org':
        edits_output_path = os.path.join(dir, 'counterfactual', dataset_name, "gen_org.json")
    elif dataset_type == 'failed':
        edits_output_path = os.path.join(dir, 'counterfactual', dataset_name, "gen_failed.json")
    
    output_dir = os.path.dirname(edits_output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(edits_output_path, 'w') as f:
        json.dump(edits_dataset, f, indent=4)
    
main()
