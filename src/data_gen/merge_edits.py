import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', "--dir", type=str, default='data')
    parser.add_argument('-d', '--dataset', type=str, default='esnli', choices=['comve', 'ecqa', 'esnli'])
    args = parser.parse_args()
    
    dir = args.dir
    dataset_name = args.dataset
    
    edits_gen_org = os.path.join(dir, 'counterfactual', dataset_name, "gen_org.json")
    edits_gen_failed = os.path.join(dir, 'counterfactual', dataset_name, f"gen_failed.json")
    with open(edits_gen_org, 'r') as f:
        gen_org = json.load(f)
    with open(edits_gen_failed, 'r') as f:
        gen_failed = json.load(f)
    
    edits_ext_org = os.path.join(dir, 'counterfactual', dataset_name, f"ext_org.json")
    with open(edits_ext_org, 'r') as f:
        ext_org = json.load(f)
        
    edits_gen_final = os.path.join(dir, 'counterfactual', dataset_name, f"gen_final.json")
    output_dir = os.path.dirname(edits_gen_final)
    os.makedirs(output_dir, exist_ok=True)
        
    if ext_org['num_failed'] != 0:
        for data in gen_org:
            if data['idx'] in ext_org['extract_edits_failed_idx']:
                for new_data in gen_failed:
                    if data['idx'] == new_data['idx']:
                        data.update(new_data)
                        
    with open(edits_gen_final, 'w') as f:
        json.dump(gen_org, f, indent=4)
    
main()
