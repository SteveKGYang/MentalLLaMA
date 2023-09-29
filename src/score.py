import os
import pandas as pd
import evaluate
import pickle
import argparse
import torch

from BARTScore.bart_score import BARTScorer
from GPTScore.gpt3_score import gpt3score

def rouge(gen_dir_name):
    rouge = evaluate.load('rouge')
    score_results = {}

    for root, ds, fs in os.walk("../model_output/{}".format(gen_dir_name)):
        for fn in fs:
            data = pd.read_csv(os.path.join(root, fn))
            dname = fn.split('.')[0]
            predictions = data['generated_text'].to_list()
            references = data['goldens'].to_list()

            result = rouge.compute(predictions=predictions, references=references)
            score_results[dname] = [result['rouge1'], result['rouge2'], result['rougeL']]
            print('Results for {} dataset: {}'.format(dname, score_results[dname]))
    pickle.dump(score_results,
                open('../quality_evaluation_results/rouge_score_{}.pkl'.format(gen_dir_name), 'wb+'))
    return score_results

def bleu(gen_dir_name):
    score_results = {}
    rouge = evaluate.load('bleu')

    for root, ds, fs in os.walk("../model_output/{}".format(gen_dir_name)):
        for fn in fs:
            data = pd.read_csv(os.path.join(root, fn))
            dname = fn.split('.')[0]
            predictions = data['generated_text'].to_list()
            references = data['goldens'].to_list()

            result = rouge.compute(predictions=predictions, references=references)
            score_results[dname] = result['bleu']
            print('Results for {} dataset: {}'.format(dname, score_results[dname]))
    pickle.dump(score_results,
                open('../quality_evaluation_results/bleu_score_{}.pkl'.format(gen_dir_name), 'wb+'))
    return score_results

def GPTScore(gen_dir_name):
    GPT_model = input('Which GPT-based model will you use?')
    api_key = input('Your api key: ')
    score_results = {}
    results = []
    for root, ds, fs in os.walk("../model_output/{}".format(gen_dir_name)):
        for fn in fs:
            data = pd.read_csv(os.path.join(root, fn))
            dname = fn.split('.')[0]
            predictions = data['generated_text'].to_list()
            references = data['goldens'].to_list()
            for prediction, reference in zip(predictions, references):
                score = gpt3score(reference, prediction, gpt3model=GPT_model, api_key=api_key)
                results.append(score)
            score_results[dname] = sum(results) / len(results)
            print('Results for {} dataset: {}'.format(dname, score_results[dname]))
    pickle.dump(score_results,
                open('../quality_evaluation_results/GPT3_score_{}.pkl'.format(gen_dir_name), 'wb+'))
    return score_results

def BERTScore(gen_dir_name):
    score_results = {}
    bert_score = evaluate.load('bertscore')
    model_type = input('Which BERT-based model will you use?')

    for root, ds, fs in os.walk("../model_output/{}".format(gen_dir_name)):
        for fn in fs:
            data = pd.read_csv(os.path.join(root, fn))
            dname = fn.split('.')[0]
            predictions = data['generated_text'].to_list()
            references = data['goldens'].to_list()

            f_n = []
            for p in predictions:
                f_n.append(str(p))

            result = bert_score.compute(predictions=f_n, references=references, model_type=model_type, batch_size=16)
            overall = sum(result['f1'])/len(result['f1'])
            print('Results for {} dataset: {}'.format(dname, overall))
            score_results[dname] = overall

    pickle.dump(score_results,
                open('../quality_evaluation_results/bert_score_{}.pkl'.format(gen_dir_name), 'wb+'))
    return score_results

def BARTscore(gen_dir_name, device):
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path='bart_score.pth')
    score_results = {}

    for root, ds, fs in os.walk("../model_output/{}".format(gen_dir_name)):
        for fn in fs:
            data = pd.read_csv(os.path.join(root, fn))
            dname = fn.split('.')[0]
            predictions = data['generated_text'].to_list()
            references = data['goldens'].to_list()

            f_n = []
            for p in predictions:
                f_n.append(str(p))

            result = bart_scorer.score(f_n, references)
            overall = sum(result) / len(result)
            print('Results for {} dataset: {}'.format(dname, overall))
            score_results[dname] = overall

    pickle.dump(score_results,
                open('../quality_evaluation_results/bart_score_{}.pkl'.format(gen_dir_name), 'wb+'))
    return score_results

def main(gen_dir_name: str, cuda: bool, device: str, score_method: str):
    if not os.path.exists("../quality_evaluation_results/"):
        os.mkdir("../quality_evaluation_results/")

    if score_method == 'bart_score':
        BARTscore(gen_dir_name, device)
    elif score_method == 'bert_score':
        BERTScore(gen_dir_name)
    elif score_method == 'GPT3_score':
        GPTScore(gen_dir_name)
    elif score_method == 'bleu':
        bleu(gen_dir_name)
    elif score_method == 'rouge':
        rouge(gen_dir_name)


if __name__ == '__main__':
    #IMHI_BARTscore('GPT4_expert')

    parser = argparse.ArgumentParser(
        description='The BART-score evaluation.')
    parser.add_argument('--gen_dir_name', type=str)
    parser.add_argument('--score_method', type=str, default='bart_score', choices=['bart_score', 'GPT3_score', 'bert_score', 'bleu', 'rouge'])
    parser.add_argument('--cuda', action='store_true')

    args = parser.parse_args()
    args = vars(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args['cuda'] is True else "cpu")
    args['device'] = device

    main(**args)