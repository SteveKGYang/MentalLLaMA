# MentaLLaMA


<p align="center" width="100%">
<img src="https://i.postimg.cc/nc2Jy6FN/uom.png"  width="100%" height="100%">
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
    <img src='https://i.postimg.cc/nc2Jy6FN/uom.png' alt='UoM University Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/cJD3HsRY/helsinki.jpg' alt='helsinki Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/T3tjyqGp/Jiangxi.png' alt='helsinki Logo' height='100px'>
    <img src='https://i.postimg.cc/CLtkBwz7/57-EDDD9-FB0-DF712-F3-AB627163-C2-1-EF15655-13-FCA.png' alt='Wuhan University Logo' height='100px'>
</div>

![](https://black.readthedocs.io/en/stable/_static/license.svg)

**Introduction**

This project presents our efforts towards interpretable mental health analysis
with large language models (LLMs). We comprehensively evaluate the zero-shot/few-shot 
performances of the latest LLMs such as ChatGPT and GPT-4 on generating explanations
for mental health analysis. We build the Interpretable Mental Health Instruction (IMHI)
dataset with 105K samples, the first multi-task and multisource instruction-tuning dataset for interpretable mental
health analysis on social media. Based on the IMHI dataset, We propose MentalLLaMA, the first open-source instructionfollowing large language model for interpretable mental
health analysis. MentalLLaMA can perform mental health
analysis on social media data and generate high-quality explanations for its predictions.
We also introduce the first holistic evaluation benchmark for interpretable mental health analysis with 19K test samples,
which covers 8 tasks and 10 test sets. Our contributions are presented in these 2 papers:

[The evaluation Paper](https://arxiv.org/abs/2304.03347) | [The MentaLLaMA paper](https://arxiv.org/abs/2309.13567)