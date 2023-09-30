<p align="center" width="100%">
<img src="https://i.postimg.cc/0Nd8VxbL/logo.png"  width="100%" height="100%">
</p>

<div>
<div align="left">
    <a href='https://stevekgyang.github.io/' target='_blank'>Kailai Yang<sup>1,2</sup>&emsp;
    <a href='https://www.zhangtianlin.top/' target='_blank'>Tianlin Zhang<sup>1,2</sup>&emsp;
    <a target='_blank'>Shaoxiong Ji<sup>3</sup></a>&emsp;
    <a target='_blank'>Ziyan Kuang<sup>4</sup></a>&emsp;
    <a target='_blank'>Qianqian Xie<sup>1,2</sup></a>&emsp;
    <a href='https://research.manchester.ac.uk/en/persons/sophia.ananiadou' target='_blank'>Sophia Ananiadou<sup>1,2</sup></a>&emsp;
    <a href='https://jimin.chancefocus.com/' target='_blank'>Jimin Huang<sup>5</sup></a>
</div>
<div>
<div align="left">
    <sup>1</sup>National Centre for Text Mining&emsp;
    <sup>2</sup>The University of Manchester&emsp;
    <sup>3</sup>University of Helsinki&emsp;
    <sup>4</sup>Jiangxi Normal University&emsp;
    <sup>5</sup>Wuhan University&emsp;
</div>

<div align="left">
    <img src='https://i.postimg.cc/Kj7RzvNr/nactem-hires.png' alt='NaCTeM' height='85px'>&emsp;
    <img src='https://i.postimg.cc/nc2Jy6FN/uom.png' alt='UoM University Logo' height='85px'>&emsp;
    <img src='https://i.postimg.cc/cJD3HsRY/helsinki.jpg' alt='helsinki Logo' height='85px'>&emsp;
    <img src='https://i.postimg.cc/T3tjyqGp/Jiangxi.png' alt='helsinki Logo' height='85px'>
    <img src='https://i.postimg.cc/CLtkBwz7/57-EDDD9-FB0-DF712-F3-AB627163-C2-1-EF15655-13-FCA.png' alt='Wuhan University Logo' height='85px'>
</div>

![](https://black.readthedocs.io/en/stable/_static/license.svg)

## Ethical Considerations

This repository and its contents are provided for **non-clinical research only**
. None of the material constitutes actual diagnosis or advice, and help-seeker should get assistance
from professional psychiatrists or clinical practitioners. No warranties, express or implied, are offered regarding the accuracy
, completeness, or utility of the predictions and explanations. The authors and contributors are not
responsible for any errors, omissions, or any consequences arising from the use 
of the information herein. Users should exercise their own judgment and consult
professionals before making any clinical-related decisions. The use
of the software and information contained in this repository is entirely at the 
user's own risk.

The raw datasets collected to build our IMHI dataset are from public
social media platforms such as Reddit and Twitter, and we strictly
follow the privacy protocols and ethical principles to protect
user privacy and guarantee that anonymity is properly applied in
all the mental health-related texts. In addition, to minimize misuse,
all examples provided in our paper are paraphrased and obfuscated
utilizing the moderate disguising scheme.

In addition, recent studies have indicated LLMs may introduce some potential
bias, such as gender gaps. Meanwhile, some incorrect prediction results, inappropriate explanations, and over-generalization
also illustrate the potential risks of current LLMs. Therefore, there
are still many challenges in applying the model to real-scenario
mental health monitoring systems.

*By using or accessing the information in this repository, you agree to indemnify, defend, and hold harmless the authors, contributors, and any affiliated organizations or persons from any and all claims or damages.*

## Introduction

This project presents our efforts towards interpretable mental health analysis
with large language models (LLMs). In early works we comprehensively evaluate the zero-shot/few-shot 
performances of the latest LLMs such as ChatGPT and GPT-4 on generating explanations
for mental health analysis. Based on the findings, we build the Interpretable Mental Health Instruction (IMHI)
dataset with 105K instruction samples, the first multi-task and multi-source instruction-tuning dataset for interpretable mental
health analysis on social media. Based on the IMHI dataset, We propose MentalLLaMA, the first open-source instruction-following LLMs for interpretable mental
health analysis. MentalLLaMA can perform mental health
analysis on social media data and generate high-quality explanations for its predictions.
We also introduce the first holistic evaluation benchmark for interpretable mental health analysis with 19K test samples,
which covers 8 tasks and 10 test sets. Our contributions are presented in these 2 papers:

[The MentaLLaMA Paper](https://arxiv.org/abs/2309.13567) | [The Evaluation Paper](https://arxiv.org/abs/2304.03347)

## MentaLLaMA Model 

We provide 4 model checkpoints evaluated in the MentaLLaMA paper:

- [MentaLLaMA-chat-13B](https://huggingface.co/klyang/MentaLLaMA-chat-13B): This model is fine-tuned based on the Meta 
LLaMA2-chat-13B foundation model and the full IMHI instruction tuning data. The training
data covers 8 mental health analysis tasks. The model can follow instructions to make accurate mental health analysis
and generate high-quality explanations for the predictions. Due to the model size, the inference
are relatively slow.
- [MentaLLaMA-chat-7B](https://huggingface.co/klyang/MentaLLaMA-chat-7B): This model is fine-tuned based on the Meta 
LLaMA2-chat-7B foundation model and the full IMHI instruction tuning data. The training
data covers 8 mental health analysis tasks. The model can follow instructions to make mental health analysis
and generate explanations for the predictions.
- [MentalBART](https://huggingface.co/Tianlin668/MentalBART): This model is fine-tuned based on the BART-large foundation model
and the full IMHI-completion data. The training data covers 8 mental health analysis tasks. The model cannot
follow instructions, but can make mental health analysis and generate explanations in a completion-based manner.
The smaller size of this model allows faster inference and easier deployment.
- [MentalT5](https://huggingface.co/Tianlin668/MentalT5): This model is fine-tuned based on the T5-large foundation model
and the full IMHI-completion data. The model cannot
follow instructions, but can make mental health analysis and generate explanations in a completion-based manner.
The smaller size of this model allows faster inference and easier deployment.

You can use the MentaLLaMA models in your Python project with the Hugging Face Transformers library. 
Here is a simple example of how to load the model:

```python
from transformers import LlamaTokenizer, LlamaForCausalLM
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
```

In this example, LlamaTokenizer is used to load the tokenizer, and LlamaForCausalLM is used to load the model. The `device_map='auto'` argument is used to automatically
use the GPU if it's available. `MODEL_PATH` denotes your model save path.

After loading the models, you can generate a response. Here is an example:

```python
prompt = 'Consider this post: "work, it has been a stressful week! hope it gets better." Question: What is the stress cause of this post?'
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=2048)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

Our running of these codes on MentaLLaMA-chat-13B gets the following response:

```
Answer: This post shows the stress cause related to work. Reasoning: The post explicitly mentions work as being stressful and expresses a hope that it gets better. This indicates that the poster is experiencing stress in relation to their work, suggesting that work is the primary cause of their stress in this instance.
```

## The IMHI Benchmark

### Test data
We introduce the first holistic evaluation benchmark for interpretable mental health analysis with 19K test samples
. We collect raw test data from 10 existing datasets covering 8 mental health analysis tasks, and transfer them into
test data for interpretable mental health analysis. Statistic about the 10 test sets are as follows:

| Name                                                   | Task                                  | Test samples | Data Source | Annotation        |
|--------------------------------------------------------|---------------------------------------|--------------|-------------|-------------------|
| [DR](https://aclanthology.org/W18-5903/)               | depression detection                  | 405          | Reddit      | Weak labels       |
| [CLP](https://aclanthology.org/W15-1204/)              | depression detection                  | 299          | Reddit      | Human annotations |
| [dreaddit](https://aclanthology.org/D19-6213/)         | stress detection                      | 414          | Reddit      | Human annotations |
| [SWMH](https://arxiv.org/abs/2004.07601)               | mental disorders detection            | 10,882       | Reddit      | Weak labels       |
| [T-SID](https://arxiv.org/abs/2004.07601)              | mental disorders detection            | 959          | Twitter     | Weak labels       |
| [SAD](https://dl.acm.org/doi/10.1145/3411763.3451799)  | stress cause detection                | 684          | SMS         | Human annotations |
| [CAMS](https://aclanthology.org/2022.lrec-1.686/)      | depression/suicide cause detection    | 625          | Reddit      | Human annotations |
| loneliness                                             | loneliness detection                  | 531          | Reddit      | Human annotations |
| [MultiWD](https://github.com/drmuskangarg/MultiWD)     | Wellness dimensions detection         | 2,441        | Reddit      | Human annotations |
| [IRF](https://aclanthology.org/2023.findings-acl.757/) | Interpersonal risks factors detection | 2,113        | Reddit      | Human annotations |

We currently release the test data from the following sets: DR, dreaddit, SAD, MultiWD, and IRF. The instruction
data is put under
```
/test_data/test_instruction
```
The items are easy to follow: the `query` row denotes the question, and the `gpt-3.5-turbo` row 
denotes our modified and evaluated predictions and explanations from ChatGPT. `gpt-3.5-turbo` is used as
the golden response for evaluation.

To facilitate test on models with no instruction following ability, we also release part of the test data for 
IMHI-completion. The data is put under
```
/test_data/test_complete
```
The file layouts are the same with instruction tuning data. 

## Model Evaluation

### Response Generation
To evaluate your trained model on the IMHI benchmark, first load your model and generate responses for all
test items. We use the Hugging Face Transformers library to load the model. For LLaMA-based models, you can
generate the responses with the following commands:
```
cd src
python IMHI.py --model_path MODEL_PATH --batch_size 8 --model_output_path OUTPUT_PATH --test_dataset IMHI --llama --cuda
```
`MODEL_PATH` and `OUTPUT_PATH` denote the model save path and the save path for generated responses. 
All generated responses will be put under `../model_output`. Some generated examples are shown in
```
./examples/response_generation_examples
```
You can also evaluate with the IMHI-completion
test set with the following commands:
```
cd src
python IMHI.py --model_path MODEL_PATH --batch_size 8 --model_output_path OUTPUT_PATH --test_dataset IMHI-completion --llama --cuda
```
You can also load models that are not based on LLaMA by removing the `--llama` argument.
In the generated examples, the `goldens` row denotes the reference explanations and the `generated_text`
row denotes the generated responses from your model.

### Correctness Evaluation
The first evaluation metric for our IMHI benchmark is to evaluate the classification correctness of the model
generations. If your model can generate very regular responses, a rule-based classifier can do well to assign
a label to each response. We provide a rule-based classifier in `IMHI.py` and you can use it during the response
generation process by adding the argument: `--rule_calculate` to your command. The classifier requires
the following template:

```
[label] Reasoning: [explanation]
```

However, as most LLMs are trained to generate diverse responses, a rule-based label classifier is impractical.
For example, MentaLLaMA can have the following response for an SAD query:
```
This post indicates that the poster's sister has tested positive for ovarian cancer and that the family is devastated. This suggests that the cause of stress in this situation is health issues, specifically the sister's diagnosis of ovarian cancer. The post does not mention any other potential stress causes, making health issues the most appropriate label in this case.
```
To solve this problem, in our [MentaLLaMA paper](https://arxiv.org/abs/2309.13567) we train 10 neural 
network classifiers based on [MentalBERT](https://arxiv.org/abs/2110.15621), one for each collected raw dataset. The classifiers are trained to
assign a classification label given the explanation. We release these 10 classifiers to facilitate future
evaluations on IMHI benchmark.

The models can be downloaded as follows: [CAMS](https://huggingface.co/Tianlin668/CAMS), [CLP](https://huggingface.co/Tianlin668/CLP), [DR](https://huggingface.co/Tianlin668/DR),
[dreaddit](https://huggingface.co/Tianlin668/dreaddit), [Irf](https://huggingface.co/Tianlin668/Irf), [loneliness](https://huggingface.co/Tianlin668/loneliness), [MultiWD](https://huggingface.co/Tianlin668/MultiWD),
[SAD](https://huggingface.co/Tianlin668/SAD), [swmh](https://huggingface.co/Tianlin668/swmh), [t-sid](https://huggingface.co/Tianlin668/t-sid),

You can obtain the labels as follows: ...

### Explanation Quality Evaluation
The second evaluation metric for the IMHI benchmark is to evaluate the quality of the generated explanations.
The results in our 
[evaluation paper](https://arxiv.org/abs/2304.03347) show that [BART-score](https://arxiv.org/abs/2106.11520) is
moderately correlated with human annotations in 4 human evaluation aspects, and outperforms other automatic evaluation metrics. Therefore,
we utilize BART-score to evaluate the quality of the generated explanations. Specifically, you should first
generate responses using the `IMHI.py` script and obtain the response dir as in `examples/response_generation_examples`.
Firstly, download the [BART-score](https://github.com/neulab/BARTScore) directory and put it under `/src`, then
download the [BART-score checkpoint](https://drive.google.com/file/d/1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m/view?usp=sharing).
Then score your responses with BART-score using the following commands:
```
cd src
python score.py --gen_dir_name DIR_NAME --score_method bart_score --cuda
```
`DIR_NAME` denotes the dir name of your geenrated responses and should be put under `../model_output`. 
We also provide other scoring methods. You can change `--score_method` to 'GPT3_score', 'bert_score', 'bleu', 'rouge'
to use these metrics. For [GPT-score](https://github.com/jinlanfu/GPTScore), you need to first download
the project and put it under `/src`.

## Human Annotations

We release our human annotations on AI-generated explanations to facilitate future research on aligning automatic evaluation
tools for interpretable mental health analysis. Based on these human evaluation results, we tested various existing
automatic evaluation metrics on correlation with human preferences. The results in our 
[evaluation paper](https://arxiv.org/abs/2304.03347) show that 
BART-score is moderately correlated with human annotations in all 4 aspects.

### Quality Evaluation
In our [evaluation paper](https://arxiv.org/abs/2304.03347), we manually labeled a subset of the AIGC results for the DR dataset in 4 aspects:
fluency, completeness, reliability, and overall. The annotations are released in this dir:
```
/human_evaluation/DR_annotation
```
where we labeled 163 ChatGPT-generated explanations for the depression detection dataset DR. The file `chatgpt_data.csv`
includes 121 explanations that correctly classified by ChatGPT. `chatgpt_false_data.csv`
includes 42 explanations that falsely classified by ChatGPT. We also include 121 explanations that correctly 
classified by InstructionGPT-3 in `gpt3_data.csv`.

### Expert-written Golden Explanations
In our [MentaLLaMA paper](https://arxiv.org/abs/2309.13567), we invited one domain expert major in quantitative psychology
to write an explanation for 350 selected posts (35 posts for each raw dataset). The golden set is used to accurately
evaluate the explanation-generation ability of LLMs in an automatic manner. To facilitate future research, we
release the expert-written explanations for the following datasets: DR, dreaddit, SWMH, T-SID, SAD, CAMS, 
loneliness, MultiWD, and IRF (35 samples each). The data is released in this dir:
```
/test_data/test_instruction_expert
```
The expert-written explanations are processed to follow the same format as other test datasets to facilitate
model evaluations. You can test your model on the expert-written golden explanations with similar commands
as in response generation. For example, you can test LLaMA-based models as follows:
```
cd src
python IMHI.py --model_path MODEL_PATH --batch_size 8 --model_output_path OUTPUT_PATH --test_dataset expert --llama --cuda
```


## Citation

If you use MentaLLaMA in your work, please cite our paper.

```
@misc{yang2023mentalllama,
      title={MentalLLaMA: Interpretable Mental Health Analysis on Social Media with Large Language Models}, 
      author={Kailai Yang and Tianlin Zhang and Ziyan Kuang and Qianqian Xie and Sophia Ananiadou},
      year={2023},
      eprint={2309.13567},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

MentaLLaMA is licensed under [MIT]. Please find more details in the [MIT](LICENSE) file.
