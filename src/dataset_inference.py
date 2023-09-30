from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import argparse
import pandas as pd
import torch
import csv

def main(model_path: str, dataset_name: str, input_csv: str, model_output_path: str, batch_size: int, device: str, cuda: bool):
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
        print("ERROR! please choose correct dataset!")

    df_train = pd.read_csv(input_csv, names=['sentence'], header=None)  ## use your own customized dataset and column index
    sentences = df_train['sentence'].tolist()

    mentalbert = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    batch_size = batch_size
    all_labels = []
    for i in range(0, len(sentences), batch_size):
        batch_data = sentences[i: min(i + batch_size, len(sentences))]
        inputs = tokenizer(batch_data, return_tensors="pt", padding=True,truncation=True,max_length=512).to(device)
        outputs = mentalbert(**inputs)[0].to(device)
        for j in range(0, len(batch_data)):
            all_labels.append(np.argmax(outputs.cpu().detach().numpy()[j]))

    # labels = {0:'0', 1:'1',2:'2',3:'3',4:'4',5:'5'}
    # for idx, sent in enumerate(sentences):
    #     print(sent, '----', labels[np.argmax(outputs.detach().numpy()[idx])])

    with open(model_output_path, 'a+', newline='', encoding='utf-8') as ff:
        csv_write = csv.writer(ff)
        name = ['sentences', 'label']
        csv_write.writerow(name)
        for i, j in zip(sentences, all_labels):
            data_row = []
            data_row.append(i)
            data_row.append(labels[j])
            csv_write.writerow(data_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The IMHI benchmark.')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--input_csv', type=str, default='')
    parser.add_argument('--model_output_path', type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dataset_name', type=str, choices=['CAMS', 'CLP', 'DR','dreaddit','Irf','loneliness','MultiWD','SAD','swmh','t-sid'])
    args = parser.parse_args()
    args = vars(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args['cuda'] is True else "cpu")
    args['device'] = device

    main(**args)