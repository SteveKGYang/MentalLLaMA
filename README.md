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
dataset with 105K instruction samples, the first multi-task and multisource instruction-tuning dataset for interpretable mental
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
We introduce the first holistic evaluation benchmark for interpretable mental health analysis with 19K test samples
. We collect raw test data from 10 existing datasets covering 8 mental health analysis tasks, and transfer them into
test data for interpretable mental health analysis. Statistic about the 10 test sets are as follows:

| Data                                                                                                                        | Task                                  | Test samples | Data Source | Annotation        |
|-----------------------------------------------------------------------------------------------------------------------------|---------------------------------------|--------------|-------------|-------------------|
| [DR](https://aclanthology.org/W18-5903/)                                                                                    | depression detection                  | 405          | Reddit      | Weak labels       |
| [CLP](https://aclanthology.org/W15-1204/)                                                                                   | depression detection                  | 299          | Reddit      | Human annotations |
| [dreaddit](https://aclanthology.org/D19-6213/)                                                                              | stress detection                      | 414          | Reddit      | Human annotations |
| [SWMH](https://arxiv.org/abs/2004.07601)                                                                                    | mental disorders detection            | 10,882       | Reddit      | Weak labels       |
| [T-SID](https://arxiv.org/abs/2004.07601)                                                                                   | mental disorders detection            | 959          | Twitter     | Weak labels       |
| [SAD](https://dl.acm.org/doi/10.1145/3411763.3451799)                                                                       | stress cause detection                | 684          | SMS         | Human annotations |
| [CAMS](https://aclanthology.org/2022.lrec-1.686/)                                                                           | depression/suicide cause detection    | 625          | Reddit      | Human annotations |
| loneliness                                                                                                                  | loneliness detection                  | 531          | Reddit      | Human annotations |
| [MultiWD](https://github.com/drmuskangarg/MultiWD) | Wellness dimensions detection         | 2,441        | Reddit      | Human annotations |
| [IRF](https://aclanthology.org/2023.findings-acl.757/)                                                                                   | Interpersonal risks factors detection | 2,113        | Reddit      | Human annotations |

We currently release the test data from the following sets: DR, dreaddit, SAD, MultiWD, and IRF. The instruction
data is put under
```
/test_data/test_instruction
```
The items are easy to follow: the `query` row denotes the question, and the `gpt-3.5-turbo` row 
denotes our modified and evaluated predictions and explanations from ChatGPT. `gpt-3.5-turbo` is used as
the golden response for evaluation.

## Human Annotations

We release our human annotations on AI-generated explanations to facilitate future research on aligning automatic evaluation
tools for interpretable mental health analysis. Based on these human evaluation results, we tested various existing
automatic evaluation metrics on correlation with human preferences. The results in our 
[evaluation paper](https://arxiv.org/abs/2304.03347) show that the scores by 
[BART-score](https://github.com/neulab/BARTScore) are moderately correlated with human annotations.

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
/test_data/expert_examples
```
In each file, the `Post` row denotes the examined post, the `Expert_Response` row denotes the expert-written
predictions and explanations. In `Irf.csv` and `MultiWD.csv`, the `Aspect` row denotes the examined aspect/wellness dimension.

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
