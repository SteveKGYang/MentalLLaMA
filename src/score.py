import os
import pandas as pd
import evaluate
import pickle
import argparse
import torch

from BARTScore.bart_score import BARTScorer
from GPTScore.gpt3_score import gpt3score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_data(file_name):
    if 'gpt3' in file_name:
        model_name = 'curie-instruct-beta'
    else:
        model_name = 'gpt-3.5-turbo'

    origin_data = pd.read_csv(file_name)

    posts = origin_data['text'].to_list()
    responses = origin_data[model_name].to_list()
    fs = origin_data['fluency'].to_list()
    res = origin_data['reliability'].to_list()
    cs = origin_data['completeness'].to_list()
    os = origin_data['overall'].to_list()
    return posts, responses, fs, res, cs, os


def rouge(file_name):
    references, predictions, fs, res, cs, overalls = load_data(os.path.join('final_data', file_name))
    #nreferences, npredictions, fs, res, cs, overalls = load_data(os.path.join('final_data', 'chatgpt_false_data.csv'))
    #references += nreferences
    #predictions += npredictions

    rouge = evaluate.load('rouge')
    rouge_score_results = {'rouge1':[], 'rouge2':[], 'rougeL':[], 'overall':[]}
    for prediction, reference in zip(predictions, references):
        result = rouge.compute(predictions=[prediction], references=[reference])
        rouge_score_results['rouge1'].append(result['rouge1'])
        rouge_score_results['rouge2'].append(result['rouge2'])
        rouge_score_results['rougeL'].append(result['rougeL'])

    result = rouge.compute(predictions=predictions, references=references)
    rouge_score_results['overall'] = [result['rouge1'], result['rouge2'], result['rougeL']]
    pickle.dump(rouge_score_results, open(os.path.join('results', 'rouge_{}.pkl'.format(file_name)), 'wb+'))
    return rouge_score_results


def bleu(file_name):
    references, predictions, fs, res, cs, overalls = load_data(os.path.join('final_data', file_name))
    #nreferences, npredictions, fs, res, cs, overalls = load_data(os.path.join('final_data', 'chatgpt_false_data.csv'))
    #references += nreferences
    #predictions += npredictions

    rouge = evaluate.load('bleu')
    rouge_score_results = {'bleu1':[], 'overall':None}
    for prediction, reference in zip(predictions, references):
        result = rouge.compute(predictions=[prediction], references=[reference])
        rouge_score_results['bleu1'].append(result['bleu'])

    result = rouge.compute(predictions=predictions, references=references)
    rouge_score_results['overall'] = result['bleu']
    pickle.dump(rouge_score_results, open(os.path.join('results', 'bleu_{}.pkl'.format(file_name)), 'wb+'))
    print(rouge_score_results['overall'])
    return rouge_score_results


def BERTScore(file_name, model_type):
    references, predictions, fs, res, cs, overalls = load_data(os.path.join('final_data', file_name))

    bert_score = evaluate.load('bertscore')
    bert_score_results = {}
    result = bert_score.compute(predictions=predictions, references=references, model_type=model_type, batch_size=16)

    bert_score_results['f1'] = result['f1']
    bert_score_results['overall'] = sum(result['f1'])/len(result['f1'])
    pickle.dump(bert_score_results, open(os.path.join('results', 'bertscore_{}_{}.pkl'.format(model_type, file_name)), 'wb+'))
    return bert_score_results

def GPTScore(file_name, GPT_model, api_key):
    references, predictions, fs, res, cs, overalls = load_data(os.path.join('final_data', file_name))
    bert_score_results = {'results':[], 'overall':None}

    for prediction, reference in zip(predictions, references):
        score = gpt3score(reference, prediction, gpt3model=GPT_model, api_key=api_key)
        bert_score_results['results'].append(score)

    bert_score_results['overall'] = sum(bert_score_results['results']) / len(bert_score_results['results'])
    pickle.dump(bert_score_results,
                open(os.path.join('results', 'gptscore_{}_{}.pkl'.format(GPT_model, file_name)), 'wb+'))
    return bert_score_results

def calculate_pearsonr(file_name, auto_metric):
    pearsonr_metric = evaluate.load("pearsonr")

    predictions = pickle.load(open(os.path.join('results', '{}_{}.pkl'.format(auto_metric, file_name)), 'rb'))
    _, _, fs, res, cs, oos = load_data(os.path.join('final_data', file_name))

    references = {'fluency':[], 'reliability':[], 'completeness':[], 'overall':[]}
    for f, r, c, o in zip(fs, res, cs, oos):
        nf = f[1:-1].split(',')
        nr = r[1:-1].split(',')
        nc = c[1:-1].split(',')
        no = o[1:-1].split(',')
        sum = 0
        for n in nf:
            sum += int(n)
        references['fluency'].append(sum / 3)
        sum = 0
        for n in nr:
            sum += int(n)
        references['reliability'].append(sum / 3)
        sum = 0
        for n in nc:
            sum += int(n)
        references['completeness'].append(sum / 3)
        sum = 0
        for n in no:
            sum += int(n)
        references['overall'].append(sum / 3)
    for key in predictions.keys():
        if key != 'overall':
            for ref_key in references.keys():
                results = pearsonr_metric.compute(predictions=predictions[key], references=references[ref_key])
                print('Pearson corelation for {} and {} is {}'.format(key, ref_key, results['pearsonr']))
        else:
            print('The value is: {}'.format(predictions[key]))


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

def main(gen_dir_name: str, cuda: bool, device: str, score_mathod: str):
    if not os.path.exists("../quality_evaluation_results/"):
        os.mkdir("../quality_evaluation_results/")

    if score_mathod == 'bart_score':
        BARTscore(gen_dir_name, device)


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