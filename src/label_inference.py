from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import argparse
import pandas as pd
import torch
import os

def get_dict(dataset_name):
    if dataset_name == 'CAMS':
        labels = {0: 'none', 1: 'bias', 2: 'job', 3: 'medication', 4: 'relation', 5: 'alienation'}
        num_labels = 6
    elif dataset_name == 'CLP':
        labels = {0: 'no', 1: 'yes'}
        num_labels = 2
    elif dataset_name == 'DR':
        labels = {0: 'no', 1: 'yes'}
        num_labels = 2
    elif dataset_name == 'dreaddit':
        labels = {0: 'no', 1: 'yes'}
        num_labels = 2
    elif dataset_name == 'Irf':
        labels = {0: 'no', 1: 'yes'}
        num_labels = 2
    elif dataset_name == 'loneliness':
        labels = {0: 'no', 1: 'yes'}
        num_labels = 2
    elif dataset_name == 'MultiWD':
        labels = {0: 'no', 1: 'yes'}
        num_labels = 2
    elif dataset_name == 'SAD':
        labels = {0: 'school', 1: 'financial',2: 'family', 3: 'social',4: 'work', 5: 'health',6: 'emotional', 7: 'everyday',8: 'other'}
        num_labels = 9
    elif dataset_name == 'swmh':
        labels = {0: 'depression', 1: 'suicide', 2: 'anxiety', 3: 'bipolar', 4: 'no mental'}
        num_labels = 5
    elif dataset_name == 't-sid':
        labels = {0: 'depression', 1: 'suicide', 2: 'ptsd', 3: 'control'}
        num_labels = 4
    else:
        raise Exception("ERROR! please choose the correct dataset!")

    return labels, num_labels

def main(model_path: str, data_path: str, data_output_path: str, batch_size: int, device: str, cuda: bool):

    if not os.path.exists(data_output_path):
        os.mkdir(data_output_path)
    for root, ds, fs in os.walk(data_path):
        for fn in fs:
            data = pd.read_csv(os.path.join(root, fn))
            goldens = data['goldens'].to_list()
            generated_text = data['generated_text'].to_list()

            dataset_name = fn.split('.')[0]
            labels, num_labels = get_dict(dataset_name)
            print('Generating for {} dataset.'.format(dataset_name))

            mentalbert = BertForSequenceClassification.from_pretrained(os.path.join(model_path, dataset_name), num_labels=num_labels).to(device)
            tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, dataset_name))

            all_labels = []
            for i in range(0, len(generated_text), batch_size):
                batch_data = generated_text[i: min(i + batch_size, len(generated_text))]
                inputs = tokenizer(batch_data, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                    device)
                outputs = mentalbert(**inputs)[0]
                outputs = outputs.cpu().detach().numpy()
                for j in range(0, len(batch_data)):
                    all_labels.append(np.argmax(outputs[j]))
            final_labels = []
            for num in all_labels:
                final_labels.append(labels[num])

            output = {'goldens': goldens, 'generated_text': generated_text, 'labels': final_labels}
            output_file = pd.DataFrame(output, index=None)
            output_file.to_csv(os.path.join(data_output_path, '{}.csv'.format(dataset_name)), index=False, escapechar='\\')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The IMHI benchmark.')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--data_output_path', type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    args = vars(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args['cuda'] is True else "cpu")
    args['device'] = device

    main(**args)