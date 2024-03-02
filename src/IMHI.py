import os
import argparse
import pandas as pd

import torch
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, precision_score, recall_score

def load_instruction_test_data():
    test_data = {}
    for root, ds, fs in os.walk("../test_data/test_instruction"):
        for fn in fs:
            data = pd.read_csv(os.path.join(root, fn))
            texts = data['query'].to_list()
            labels = data['gpt-3.5-turbo'].to_list()
            test_data[fn.split('.')[0]] = [texts, labels]
    return test_data

def load_complete_test_data():
    test_data = {}
    for root, ds, fs in os.walk("../test_data/test_complete"):
        for fn in fs:
            data = pd.read_csv(os.path.join(root, fn))
            texts = data['query'].to_list()
            labels = data['gpt-3.5-turbo'].to_list()
            test_data[fn.split('.')[0]] = [texts, labels]
    return test_data

def load_expert_data():
    test_data = {}
    for root, ds, fs in os.walk("../human_evaluation/test_instruction_expert"):
        for fn in fs:
            data = pd.read_csv(os.path.join(root, fn))
            texts = data['query'].to_list()
            labels = data['gpt-3.5-turbo'].to_list()
            test_data[fn.split('.')[0]] = [texts, labels]
    return test_data

def generate_response(model, tokenizer, test_data, device, batch_size):
    generated_text = {}
    goldens = {}

    model.to(device)

    for dataset_name in test_data.keys():
        #if dataset_name not in ['DR', 'dreaddit']:
        #    continue
        print('Generating for dataset: {}'.format(dataset_name))
        queries, golden = test_data[dataset_name]
        goldens[dataset_name]  = golden
        responses = []

        for i in range(0, len(queries), batch_size):
            batch_data = queries[i: min(i+batch_size, len(queries))]
            #print(batch_data[:2])
            inputs = tokenizer(batch_data, return_tensors="pt", padding=True)
            #print(inputs)
            #final_input = inputs.input_ids
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            #print(final_input)
            generate_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=2048)
            for j in range(generate_ids.shape[0]):
                truc_ids = generate_ids[j][len(input_ids[j]) :]
                response = tokenizer.decode(truc_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
                responses.append(response)
            print(i)
        generated_text[dataset_name] = responses

    return generated_text, goldens

def save_output(generated_text, goldens, output_path):
    if not os.path.exists("../model_output/"):
        os.mkdir("../model_output/")
    if not os.path.exists("../model_output/"+output_path):
        os.mkdir("../model_output/"+output_path)
    for dataset_name in generated_text.keys():
        output = {'goldens': goldens[dataset_name], 'generated_text': generated_text[dataset_name]}
        output = pd.DataFrame(output, index=None)
        output.to_csv("{}/{}/{}.csv".format('../model_output',
                                         output_path, dataset_name), index=False, escapechar='\\')

def calculate_f1(generated, goldens, output_path):
    for dataset_name in generated.keys():
        golden = goldens[dataset_name]
        outputs = generated[dataset_name]

        output_label = []
        golden_label = []
        count = 0

        for ref, output in zip(golden, outputs):
            ref_an = ref.split("Reasoning:")[0]
            output_an = output.split("Reasoning:")[0]
            #print(output)
            #output_an = str(output)[:70]

            if dataset_name == 'swmh':
                if 'no mental' in output_an.lower():
                    output_label.append(0)
                elif 'suicide' in output_an.lower():
                    output_label.append(1)
                elif 'depression' in output_an.lower():
                    output_label.append(2)
                elif 'anxiety' in output_an.lower():
                    output_label.append(3)
                elif 'bipolar' in output_an.lower():
                    output_label.append(4)
                else:
                    count += 1
                    output_label.append(0)
                    #print(output)

                if 'no mental' in ref_an.lower():
                    golden_label.append(0)
                elif 'suicide' in ref_an.lower():
                    golden_label.append(1)
                elif 'depression' in ref_an.lower():
                    golden_label.append(2)
                elif 'anxiety' in ref_an.lower():
                    golden_label.append(3)
                elif 'bipolar' in ref_an.lower():
                    golden_label.append(4)
                else:
                    output_label = output_label[:-1]

            elif dataset_name == 't-sid':
                if 'depression' in output_an.lower():
                    output_label.append(2)
                elif 'suicide' in output_an.lower():
                    output_label.append(1)
                elif 'ptsd' in output_an.lower():
                    output_label.append(3)
                elif 'no mental' in output_an.lower():
                    output_label.append(0)
                else:
                    count += 1
                    #print(output)
                    output_label.append(0)
                    #print(output)

                if 'depression' in ref_an.lower():
                    golden_label.append(2)
                elif 'suicide or self-harm' in ref_an.lower():
                    golden_label.append(1)
                elif 'ptsd' in ref_an.lower():
                    golden_label.append(3)
                elif 'no mental disorders' in ref_an.lower():
                    golden_label.append(0)

            elif dataset_name in ['CLP', 'DR', 'dreaddit', 'loneliness', 'Irf', 'MultiWD']:
                if 'yes' in output_an.lower():
                    output_label.append(1)
                elif 'no' in output_an.lower():
                    output_label.append(0)
                else:
                    count += 1
                    output_label.append(0)
                    #print(output)

                if 'yes' in ref_an.lower():
                    golden_label.append(1)
                elif 'no' in ref_an.lower():
                    golden_label.append(0)

            elif dataset_name == 'SAD':
                if 'school' in output_an.lower():
                    output_label.append(0)
                elif 'financial' in output_an.lower():
                    output_label.append(1)
                elif 'family' in output_an.lower():
                    output_label.append(2)
                elif 'social' in output_an.lower():
                    output_label.append(3)
                elif 'work' in output_an.lower():
                    output_label.append(4)
                elif 'health' in output_an.lower():
                    output_label.append(5)
                elif 'emotional' in output_an.lower():
                    output_label.append(6)
                elif 'decision' in output_an.lower():
                    output_label.append(7)
                elif 'other' in output_an.lower():
                    output_label.append(8)
                else:
                    count += 1
                    output_label.append(8)
                    #print(output_an)

                if 'school' in ref_an.lower():
                    golden_label.append(0)
                elif 'financial problem' in ref_an.lower():
                    golden_label.append(1)
                elif 'family issues' in ref_an.lower():
                    golden_label.append(2)
                elif 'social relationships' in ref_an.lower():
                    golden_label.append(3)
                elif 'work' in ref_an.lower():
                    golden_label.append(4)
                elif 'health issues' in ref_an.lower():
                    golden_label.append(5)
                elif 'emotional turmoil' in ref_an.lower():
                    golden_label.append(6)
                elif 'everyday decision making' in ref_an.lower():
                    golden_label.append(7)
                elif 'other causes' in ref_an.lower():
                    golden_label.append(8)

            elif dataset_name == 'CAMS':
                if 'none' in output_an.lower():
                    output_label.append(0)
                elif 'bias' in output_an.lower():
                    output_label.append(1)
                elif 'jobs' in output_an.lower():
                    output_label.append(2)
                elif 'medication' in output_an.lower():
                    output_label.append(3)
                elif 'relationship' in output_an.lower():
                    output_label.append(4)
                elif 'alienation' in output_an.lower():
                    output_label.append(5)
                else:
                    count += 1
                    output_label.append(0)
                    #print(output_an)

                if 'none' in ref_an.lower():
                    golden_label.append(0)
                elif 'bias or abuse' in ref_an.lower():
                    golden_label.append(1)
                elif 'jobs and career' in ref_an.lower():
                    golden_label.append(2)
                elif 'medication' in ref_an.lower():
                    golden_label.append(3)
                elif 'relationship' in ref_an.lower():
                    golden_label.append(4)
                elif 'alienation' in ref_an.lower():
                    golden_label.append(5)

        avg_accuracy = round(accuracy_score(golden_label, output_label) * 100, 2)
        weighted_f1 = round(f1_score(golden_label, output_label, average='weighted') * 100, 2)
        micro_f1 = round(f1_score(golden_label, output_label, average='micro') * 100, 2)
        macro_f1 = round(f1_score(golden_label, output_label, average='macro') * 100, 2)
        # recall = round(recall_score(f_labels, outputs, average='weighted')*100, 2)
        result = "Dataset: {}, average acc:{}, weighted F1 {}, micro F1 {}, macro F1 {}, OOD count: {}\n".format(dataset_name,
                                                                                             avg_accuracy, weighted_f1,
                                                                                             micro_f1, macro_f1, count)
        print(result)
        with open("{}/{}/results.txt".format('../model_output', output_path), 'a+') as f:
            f.write(result)
        print(count)

def main(model_path: str, model_output_path: str, batch_size: int, test_dataset: str, rule_calculate: bool,
         llama: bool, device: str, lora: bool, cuda: bool, lora_path: str):
    if llama:
        model = LlamaForCausalLM.from_pretrained(model_path)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side='left')
    else:
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if lora:
        model = PeftModel.from_pretrained(model, lora_path)

    tokenizer.pad_token = tokenizer.unk_token

    if test_dataset == 'IMHI':
        test_data = load_instruction_test_data()
    elif test_dataset == 'IMHI-completion':
        test_data = load_complete_test_data()
    elif test_dataset == 'expert':
        test_data = load_expert_data()
    generated_text, goldens = generate_response(model, tokenizer, test_data, device, batch_size)
    save_output(generated_text, goldens, model_output_path)
    if rule_calculate:
        calculate_f1(generated_text, goldens, model_output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='The IMHI benchmark.')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model_output_path', type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--test_dataset', type=str, choices=['IMHI', 'IMHI-completion', 'expert'])
    parser.add_argument('--rule_calculate', action='store_true')
    parser.add_argument('--llama', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora_path', type=str)

    args = parser.parse_args()
    args = vars(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args['cuda'] is True else "cpu")
    args['device'] = device

    main(**args)