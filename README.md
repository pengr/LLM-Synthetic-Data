# Live LLM-Synthetic-Data Papers (ICML 2025 Update)

<div align="center">

[![LICENSE](https://img.shields.io/github/license/wasiahmad/Awesome-LLM-Synthetic-Data-Generation)](https://github.com/pengr/LLM-Synthetic-Data/blob/main/LICENSE)
[![commit](https://img.shields.io/github/last-commit/wasiahmad/Awesome-LLM-Synthetic-Data-Generation?color=blue)](https://github.com/pengr/LLM-Synthetic-Data/commits/main)
[![PR](https://img.shields.io/badge/PRs-Welcome-red)](https://github.com/pengr/LLM-Synthetic-Data/pulls)
[![GitHub Repo stars](https://img.shields.io/github/stars/pengr/LLM-Synthetic-Data)](https://github.com/pengr/LLM-Synthetic-Data)

</div>

This repo, inspired by [Awesome-LLM-Synthetic-Data](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data), which focuses on **real-time, fine-grained LLM-Synthetic-Data**. 

If you find this useful, feel free to follow us and star both repos. Thanks to all the great contributors on GitHub! 

🔥 indicates an article/content that we personally highly recommend.


## Contents

* [1. Githubs](#1-githubs)
* [2. Blogs](#2-blogs)
* [3. Surveys](#3-surveys)
* [4. Methods](#4-methods)
  * [4.1. Pre-training](#41-pre-training)
  * [4.2. Continue Pre-training](#42-continue-pre-training)
  * [4.3. Instruction Tuning](#43-instruction-tuning)
  * [4.4. Alignment](#44-alignment)
  * [4.5. Model Collapse](#45-model-collapse)
  * [4.6. LLM Benchmarking](#46-llm-benchmarking)
  * [4.7. Evaluation](#47-evaluation)
  * [4.8. Distillation](#48-distillation)
* [5. Application Areas](#5-application-areas)
  * [5.1. Mathematical Reasoning](#51-mathematical-reasoning)
  * [5.2. Code Generation](#52-code-generation)
  * [5.3. Text-to-SQL](#53-text-to-sql)
  * [5.4. Reward Modeling](#54-reward-modeling)
  * [5.5. Long Context](#55-long-context)
  * [5.6. Weak-to-Strong](#56-weak-to-strong)
  * [5.7. Agent and Tool Use](#57-agent-and-tool-use)
  * [5.8. Vision and Language](#58-vision-and-language)
  * [5.9. Factuality](#59-factuality)
  * [5.10. Federated Learning](#510-federated-learning)
  * [5.11. Generative Design](#511-generative-design)
  * [5.12. Safety](#512-safety)
  * [5.13. Synthetis textbooks](#513-synthetis-textbooks)
  * [5.14. Reasoning](#514-reasoning)
* [6. Tools](#6-tools)
* [7. Datasets](#7-datasets)


## 1. Githubs:

- [https://github.com/mghiasvand1/Awesome-VLM-Synthetic-Data](https://github.com/mghiasvand1/Awesome-VLM-Synthetic-Data).🔥
- [https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data).🔥

## **2. Blogs:**

- [Synthetic data: save money, time and carbon with open source.](https://huggingface.co/blog/synthetic-data-save-costs) *Moritz Laurer*. Feb 16, 2024.
- [Synthetic data generation (Part 1).](https://cookbook.openai.com/examples/sdg1)  *Dylan Royan Almeida*. Apr 10, 2024  
- [Synthetic dataset generation techniques: Self-Instruct.](https://huggingface.co/blog/davanstrien/self-instruct) *Daniel van Strien*. May 15, 2024.
- [CodecLM: Aligning language models with tailored synthetic data](https://research.google/blog/codeclm-aligning-language-models-with-tailored-synthetic-data) *Zifeng Wang and Chen-Yu Lee.* May 30, 2024.
- [The Rise of Agentic Data Generation.](https://huggingface.co/blog/mlabonne/agentic-datagen) *Maxime Labonne.* July 15, 2024.
- [LLM-Driven Synthetic Data Generation, Curation & Evaluation.](https://cobusgreyling.medium.com/llm-driven-synthetic-data-generation-curation-evaluation-33731e33b525) *Cobus Greyling.* Aug 2, 2024.
- [Using LLMs for Synthetic Data Generation: The Definitive Guide](https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms) *Kritin Vongthongsri.* November 8, 2024.

## **3. Surveys:**

1. [Best Practices and Lessons Learned on Synthetic Data for Language Models.](https://arxiv.org/abs/2404.07503) *Ruibo Liu, Jerry Wei, Fangyu Liu, Chenglei Si, Yanzhe Zhang, Jinmeng Rao, Steven Zheng, Daiyi Peng, Diyi Yang, Denny Zhou, Andrew M. Dai.* COLM 2024.
2. [On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey.](https://arxiv.org/abs/2406.15126) *Lin Long, Rui Wang, Ruixuan Xiao, Junbo Zhao, Xiao Ding, Gang Chen, Haobo Wang.* ACL Findings 2024.
3. [Large Language Models for Data Annotation: A Survey](https://arxiv.org/abs/2402.13446) *Zhen Tan, Dawei Li, Song Wang, Alimohammad Beigi, Bohan Jiang, Amrita Bhattacharjee, Mansooreh Karami, Jundong Li, Lu Cheng, Huan Liu.* EMNLP 2024.
4. [Generative AI for Synthetic Data Generation: Methods, Challenges and the Future.](https://arxiv.org/abs/2403.04190) *Xu Guo, Yiqiang Chen.* Arxiv 2024.
5. [Comprehensive Exploration of Synthetic Data Generation: A Survey.](https://arxiv.org/abs/2401.02524) *André Bauer, Simon Trapp, Michael Stenger, Robert Leppich, Samuel Kounev, Mark Leznik, Kyle Chard, Ian Foster.* Arxiv 2024.
6. [A Survey on Bridging VLMs and Synthetic Data.](https://openreview.net/pdf?id=ThjDCZOljE) *Mohammad Ghiasvand Mohammadkhani, Saeedeh Momtazi, Hamid Beigy.* OpenReview 2025.

## **4. Methods:**

### 4.1. Pre-training

- [Phi-4 Technical Report](https://arxiv.org/pdf/2412.08905) *Microsoft Research.* Arxiv 2024.🔥
- [Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset](https://arxiv.org/abs/2412.02595) *Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro*, Arxiv 2024🔥
- [Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent](https://arxiv.org/pdf/2411.02265) *Tecent Hunyuan Team.* Arxiv 2024.🔥

### 4.2. Continue Pre-training

- [MAmmoTH2: Scaling Instructions from the Web.](https://arxiv.org/pdf/2405.03548) *Xiang Yue, Tuney Zheng, Ge Zhang, Wenhu Chen.* Neurips 2024.🔥

### 4.3. Instruction Tuning

#### A. Before ChatGPT came out
- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) *Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman.* NeurIPS 2022.
- [Generating Training Data with Language Models: Towards Zero-Shot Language Understanding](https://arxiv.org/abs/2202.04538) *Yu Meng, Jiaxin Huang, Yu Zhang, Jiawei Han.* NeurIPS 2022.
- [ZeroGen: Efficient Zero-shot Learning via Dataset Generation](https://arxiv.org/abs/2202.07922) *Jiacheng Ye, Jiahui Gao, Qintong Li, Hang Xu, Jiangtao Feng, Zhiyong Wu, Tao Yu, Lingpeng Kong.* EMNLP 2022.
- [Symbolic Knowledge Distillation: from General Language Models to Commonsense Models](https://arxiv.org/abs/2110.07178) *Peter West, Chandra Bhagavatula, Jack Hessel, Jena D. Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, Yejin Choi.* NAACL 2022.
- [Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor](https://arxiv.org/abs/2212.09689)  *Or Honovich, Thomas Scialom, Omer Levy, Timo Schick.* ACL 2023.
#### B. Rephrasing Output
- [Rephrasing theWeb A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380) *Pratyush Maini, Skyler Seto, He Bai, David Grangier, Yizhe Zhang, Navdeep Jaitly.* ACL 2024.🔥
- [Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset](https://arxiv.org/abs/2412.02595) *Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro*, Arxiv 2024
- [ToEdit: How to Synthesize Text Data to Avoid Model Collapse?](https://openreview.net/forum?id=mVCcWCjeEz) ICML 2025.
- Synthesis instruction from human-written corpus
- [LongForm: Effective Instruction Tuning with Reverse Instructions.](https://arxiv.org/abs/2304.08460) *Abdullatif Köksal, Timo Schick, Anna Korhonen, Hinrich Schütze.* EMNLP 2024 findings.🔥
- [Self-Alignment with Instruction Backtranslation.](https://arxiv.org/abs/2308.06259) *Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, Jason Weston, Mike Lewis.* ICLR 2024 Oral
#### C. Synthesis instructions and outputs using LLM
- **Seed + Prompt engineering (Principle, Templates, Constraints) + Filtering strategy**
 - *Iterative generation*
   - [Self-instruct: Aligning language models with self-generated instructions.](https://arxiv.org/abs/2212.10560) *Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi.* ACL 2023.🔥
   - [Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision.](https://arxiv.org/abs/2308.06259) *Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, Chuang Gan.* NeurIPS 2023.🔥
   - [CodecLM: Aligning Language Models with Tailored Synthetic Data.](https://arxiv.org/abs/2404.05875) *Zifeng Wang, Chun-Liang Li, Vincent Perot, Long T. Le, Jin Miao, Zizhao Zhang, Chen-Yu Lee, Tomas Pfister.* NAACL Findings 2024.🔥
 - *Non-iterative generation*
   - [Increasing Diversity While Maintaining Accuracy: Text Data Generation with Large Language Models and Human Interventions](https://arxiv.org/abs/2306.04140) *John Joon Young Chung, Ece Kamar, Saleema Amershi.* ACL 2023.🔥
   - [Large Language Model as Attributed Training Data Generator: A Tale of Diversity and Bias](https://arxiv.org/pdf/2306.15895) *Yue Yu, Yuchen Zhuang, Jieyu Zhang, Yu Meng, Alexander Ratner, Ranjay Krishna, Jiaming Shen, Chao Zhang.* NeurIPS D&B 2023.
   - [Large Language Models Can Self-Improve](https://aclanthology.org/2023.emnlp-main.67/) *Jiaxin Huang, Shixiang Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, Jiawei Han.* EMNLP 2023.
   - [TarGEN: Targeted Data Generation with Large Language Models](https://openreview.net/pdf?id=gpgMRWgv9Q) *Himanshu Gupta, Kevin Scaria, Ujjwala Anantheswaran, Shreyas Verma, Mihir Parmar, Saurabh Arjun Sawant, Chitta Baral, Swaroop Mishra.* COLM 2024.
   - [Source2Synth: Synthetic Data Generation and Curation Grounded in Real Data Sources](https://arxiv.org/abs/2409.08239) *Alisia Lupidi, Carlos Gemmell, Nicola Cancedda, Jane Dwivedi-Yu, Jason Weston, Jakob Foerster, Roberta Raileanu, Maria Lomeli.* Arxiv 2025.
   - [Generating Faithful Synthetic Data with Large Language Models: A Case Study in Computational Social Science](https://arxiv.org/abs/2305.15041) *Veniamin Veselovsky, Manoel Horta Ribeiro, Akhil Arora, Martin Josifoski, Ashton Anderson, Robert West,* Arxiv 2023.🔥
   - [MetaSynth: Meta-Prompting-Driven Agentic Scaffolds for Diverse Synthetic Data Generation](https://arxiv.org/abs/2504.12563) *Haris Riaz, Sourav Bhabesh, Vinayak Arannil, Miguel Ballesteros, Graham Horwood.* Findings of ACL 2025
   - [LAB: Large-Scale Alignment for ChatBots](https://arxiv.org/abs/2403.01081) *Shivchander Sudalairaj, Abhishek Bhandwaldar, Aldo Pareja, Kai Xu, David D. Cox, Akash Srivastava.* Arxiv 2024🔥

- **Self-play**
  - [Self-playing Adversarial Language Game Enhances LLM Reasoning](https://arxiv.org/abs/2404.10642) *Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Yong Dai, Lei Han, Nan Du.* Neurips 2024.
  - [Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models.](https://arxiv.org/abs/2406.13542) Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, Jingren Zhou. Submit to ICLR 2025.
- **Evoling (Rephrasing) instrut**
  - [WizardLM: Empowering Large Language Models to Follow Complex Instructions.](https://arxiv.org/abs/2304.12244) *Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Daxin Jiang.* ICLR 2024.🔥
  - [Making Large Language Models Better Data Creators](https://arxiv.org/pdf/2310.20111) *Dong-Ho Lee, Jay Pujara, Mohit Sewak, Ryen W. White, Sujay Kumar Jauhar.* EMNLP 2023.
- **Curating Attributes Database**
   - [Scaling Synthetic Data Creation with 1,000,000,000 Personas.](https://arxiv.org/abs/2406.20094) *Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, Dong Yu.* Arxiv 2024.🔥
   - [Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models](https://arxiv.org/abs/2402.13064) *Haoran Li, Qingxiu Dong, Zhengyang Tang, Chaojun Wang, Xingxing Zhang, Haoyang Huang, Shaohan Huang, Xiaolong Huang, Zeqiang Huang, Dongdong Zhang, Yuxian Gu, Xin Cheng, Xun Wang, Si-Qing Chen, Li Dong, Wei Lu, Zhifang Sui, Benyou Wang, Wai Lam, Furu Wei.* Arxiv 2025.
   - [Learning to Generate Instruction Tuning Datasets for Zero-Shot Task Adaptation](https://arxiv.org/abs/2402.18334) *Nihal V. Nayak, Yiyang Nan, Avi Trost, Stephen H. Bach.* ACL Findings 2024.🔥
- **No seed, no prompt templates**
  - [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464) *Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, Bill Yuchen Lin.* ICLR 2025.
- **Failure-inducing exploration to guide LLM synthetic data**
  - [Forewarned is Forearmed: Harnessing LLMs for Data Synthesis via Failure-induced Exploration.](https://openreview.net/forum?id=yitH9xAHQs) ICLR 2025.
  - [Let's Synthesize Step by Step: Iterative Dataset Synthesis with Large Language Models by Extrapolating Errors from Small Models](https://arxiv.org/abs/2310.13671) Ruida Wang, Wangchunshu Zhou, Mrinmaya Sachan, EMNLP finding 2023

### 4.4. Alignment

- [Self-Rewarding Language Models.](https://arxiv.org/abs/2401.10020) *Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston.* ICML 2024.
- [SALMON: Self-Alignment with Instructable Reward Models](https://arxiv.org/abs/2310.05910) *Zhiqing Sun, Yikang Shen, Hongxin Zhang, Qinhong Zhou, Zhenfang Chen, David Cox, Yiming Yang, Chuang Gan.* ICLR 2024.
- [Refined Direct Preference Optimization with Synthetic Data for Behavioral Alignment of LLMs](https://arxiv.org/abs/2402.08005) *V´ıctor Gallego.* Arxiv 2024.
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) *Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan.* Arxiv 2022.
- [Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision](https://arxiv.org/abs/2305.03047) *Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, Chuang Gan.* NeurIPS 2023.
- [Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models](https://arxiv.org/abs/2406.13542). *Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, Jingren Zhou.* Submit to ICLR 2025.
- [Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts](https://arxiv.org/abs/2402.16822) *Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rocktäschel, Roberta Raileanu.* NeurIPS 2024.
- [Self-Alignment with Instruction Backtranslation](https://arxiv.org/abs/2308.06259) *Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, Jason Weston, Mike Lewis.* ICLR 2024.
- [TRIDENT: Enhancing Large Language Model Safety with Tri-Dimensional Diversified Red-Teaming Data Synthesis](https://arxiv.org/pdf/2505.24672) *Xiaorui Wu, Xiaofeng Mao, Fei Li, Xin Zhang, Xuanhong Li, Chong Teng, Donghong Ji, Zhuang Li.* ACL 2025 

### 4.5. Model Collapse

- [AI models collapse when trained on recursively generated data](https://www.nature.com/articles/s41586-024-07566-y) *Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson & Yarin Gal,* Nature 2024.
- [How bad is training on synthetic data? A statistical analysis of language model collapse.](https://openreview.net/forum?id=t3z6UlV09o#discussion) *Mohamed El Amine Seddik, Suei-Wen Chen, Soufiane Hayou, Pierre Youssef, Merouane Abdelkader DEBBAH.* COLM 2024.
- [ToEdit: How to Synthesize Text Data to Avoid Model Collapse?](https://openreview.net/forum?id=mVCcWCjeEz) ICML 2025.
- [Beyond Model Collapse: Scaling Up with Synthesized Data Requires Reinforcement](https://arxiv.org/pdf/2406.07515) *Yunzhen Feqng, Elvis Dohmatob, Pu Yang, Francois Charton, Julia Kempe.* ICLR 2025.
- [A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops](https://www.arxiv.org/abs/2502.18865) *Shi Fu, Yingjie Wang, Yuzhu Chen, Xinmei Tian, Dacheng Tao.* ICLR 2025
- [Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data](https://arxiv.org/abs/2404.01413) *Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, Henry Sleight, John Hughes, Tomasz Korbak, Rajashree Agrawal, Dhruv Pai, Andrey Gromov, Daniel A. Roberts, Diyi Yang, David L. Donoho, Sanmi Koyejo.* Arxiv 2024

### 4.6. LLM Benchmarking

- [DataGen: Unified Synthetic Dataset Generation via Large Language Models](https://openreview.net/forum?id=F5R0lG74Tu) *Yue Huang, Siyuan Wu, Chujie Gao, Dongping Chen, Qihui Zhang, Yao Wan, Tianyi Zhou, Xiangliang Zhang, Jianfeng Gao, Chaowei Xiao, Lichao Sun.* ICLR 2025.

### 4.7. Evaluation

- [On the Diversity of Synthetic Data and its Impact on Training Large Language Models](https://openreview.net/forum?id=oqsQbn4XfT) *Hao Chen, Abdul Waheed, Xiang Li, Yidong Wang, Jindong Wang, Bhiksha Raj, Marah I. Abdin.* Arxiv 2025.

### 4.8. Distillation

- [FreeAL: Towards Human-Free Active Learning in the Era of Large Language Models](https://arxiv.org/abs/2311.15614), *Ruixuan Xiao, Yiwen Dong, Junbo Zhao, Runze Wu, Minmin Lin, Gang Chen, Haobo Wang.* EMNLP 2023.
- [Let's Synthesize Step by Step: Iterative Dataset Synthesis with Large Language Models by Extrapolating Errors from Small Models](https://arxiv.org/abs/2310.13671) *Ruida Wang, Wangchunshu Zhou, Mrinmaya Sachan,* EMNLP finding2023
- [Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations](https://aclanthology.org/2023.emnlp-main.647.pdf), *Zhuoyan Li, Hangxiao Zhu, Zhuoran Lu, Ming Yin.* EMNLP finding 2023.

## 5. Application Areas:

### 5.1. Mathematical Reasoning

- [Distilling LLMs' Decomposition Abilities into Compact Language Models](https://arxiv.org/abs/2402.01812) *Denis Tarasov, Kumar Shridhar.* AutoRL@ICML 2024.
- [MuggleMath: Assessing the Impact of Query and Response Augmentation on Math Reasoning](https://arxiv.org/abs/2310.05506v3) *Chengpeng Li, Zheng Yuan, Hongyi Yuan, Guanting Dong, Keming Lu, Jiancan Wu, Chuanqi Tan, Xiang Wang, Chang Zhou.* ACL 2024.
- [MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs](https://arxiv.org/abs/2402.16352) *Zimu Lu, Aojun Zhou, Houxing Ren, Ke Wang, Weikang Shi, Junting Pan, Mingjie Zhan, Hongsheng Li.* ACL 2024.
- [MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models](https://arxiv.org/abs/2309.12284) *Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, Weiyang Liu.* ICLR 2024.
- [Augmenting Math Word Problems via Iterative Question Composing](https://arxiv.org/abs/2401.09003) *Haoxiong Liu, Yifan Zhang, Yifan Luo, Andrew Chi-Chih Yao.* DPFM@ICLR 2024.
- [Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning](https://arxiv.org/abs/2403.02333) *Yiming Huang, Xiao Liu, Yeyun Gong, Zhibin Gou, Yelong Shen, Nan Duan, Weizhu Chen.* Arxiv 2024.
- [Dart-math: Difficulty-aware rejection tuning for mathematical problem-solving](https://arxiv.org/abs/2407.13690) *Yuxuan Tong, Xiwen Zhang, Rui Wang, Ruidong Wu, Junxian He.* NeurIPS 2024.
- [RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold](https://arxiv.org/abs/2406.14532) *Amrith Setlur, Saurabh Garg, Xinyang Geng, Naman Garg, Virginia Smith, Aviral Kumar.* NeurIPS 2025

### 5.2. Code Generation

- [Program Synthesis with Large Language Models](https://arxiv.org/pdf/2108.07732) *Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, Charles Sutton.* Arxiv 2021.
- [CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning](https://arxiv.org/abs/2207.01780) *Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, Steven C.H. Hoi.* NeurIPS 2022.
- [InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback](https://arxiv.org/abs/2306.14898) *John Yang, Akshara Prabhakar, Karthik Narasimhan, Shunyu Yao.* Arxiv 2023.
- [Language Models Can Teach Themselves to Program Better](https://arxiv.org/abs/2207.14502) *Patrick Haluptzok, Matthew Bowers, Adam Tauman Kalai.* ICLR 2023.
- [CODEGEN: AN OPEN LARGE LANGUAGE MODEL FOR CODE WITH MULTI-TURN PROGRAM SYNTHESIS](https://arxiv.org/pdf/2203.13474). ICLR2023.
- [Code Alpaca: An Instruction-following LLaMA Model trained on code generation instructions](https://github.com/sahil280114/codealpaca) *Sahil Chaudhary*. GitHub 2023.
- [Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models](https://arxiv.org/abs/2407.21077) *Somshubra Majumdar, Vahid Noroozi, Sean Narenthiran, Aleksander Ficek, Jagadeesh Balam, Boris Ginsburg.* Arxiv 2024.
- [Magicoder: Empowering Code Generation with OSS-Instruct](https://arxiv.org/abs/2312.02120) *Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, Lingming Zhang.* ICML 2024.
- [WaveCoder: Widespread And Versatile Enhancement For Code Large Language Models By Instruction Tuning](https://arxiv.org/abs/2312.14187) *Zhaojian Yu, Xin Zhang, Ning Shang, Yangyu Huang, Can Xu, Yishujie Zhao, Wenxiang Hu, Qiufeng Yin.* ACL 2024.
- [WizardCoder: Empowering Code Large Language Models with Evol-Instruct](https://arxiv.org/abs/2306.08568) *Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, Daxin Jiang.* ICLR 2024.
- [Learning Performance-Improving Code Edits](https://arxiv.org/abs/2302.07867) *Alexander Shypula, Aman Madaan, Yimeng Zeng, Uri Alon, Jacob Gardner, Milad Hashemi, Graham Neubig, Parthasarathy Ranganathan, Osbert Bastani, Amir Yazdanbakhsh.* ICLR 2024.
- [InverseCoder: Unleashing the Power of Instruction-Tuned Code LLMs with Inverse-Instruct](https://arxiv.org/abs/2407.05700) *Yutong Wu, Di Huang, Wenxuan Shi, Wei Wang, Lingzhe Gao, Shihao Liu, Ziyuan Nan, Kaizhao Yuan, Rui Zhang, Xishan Zhang, Zidong Du, Qi Guo, Yewen Pu, Dawei Yin, Xing Hu, Yunji Chen.* Arxiv 2024.
- [OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement](https://arxiv.org/abs/2402.14658) *Tianyu Zheng, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, Xiang Yue.* Arxiv 2024.
- [AutoCoder: Enhancing Code Large Language Model with AIEV-Instruct](https://arxiv.org/abs/2405.14906) *Bin Lei, Yuchen Li, Qiuwu Chen.* Arxiv 2024.
- [How Do Your Code LLMs Perform? Empowering Code Instruction Tuning with High-Quality Data](https://www.arxiv.org/abs/2409.03810) *Yejie Wang, Keqing He, Dayuan Fu, Zhuoma Gongque, Heyang Xu, Yanxu Chen, Zhexu Wang, Yujia Fu, Guanting Dong, Muxi Diao, Jingang Wang, Mengdi Zhang, Xunliang Cai, Weiran Xu.* Arxiv 2024.
- [SelfCodeAlign: Self-Alignment for Code Generation](https://arxiv.org/abs/2410.24198) *Yuxiang Wei, Federico Cassano, Jiawei Liu, Yifeng Ding, Naman Jain, Zachary Mueller, Harm de Vries, Leandro von Werra, Arjun Guha, Lingming Zhang.* Arxiv 2024.

### 5.3. Text-to-SQL

- [Synthesizing Text-to-SQL Data from Weak and Strong LLMs](https://arxiv.org/abs/2408.03256) *Jiaxi Yang, Binyuan Hui, Min Yang, Jian Yang, Junyang Lin, Chang Zhou.* ACL 2024.
- [Synthetic-Text-To-SQL: A synthetic dataset for training language models to generate SQL queries from natural language prompts](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) *Meyer, Yev and Emadi, Marjan and Nathawani, Dhruv and Ramaswamy, Lipika and Boyd, Kendrick and Van Segbroeck, Maarten and Grossman, Matthew and Mlocek, Piotr and Newberry, Drew.* Huggingface 2024.

### 5.4. Reward Modeling

- [West-of-N: Synthetic Preference Generation for Improved Reward Modeling.](https://arxiv.org/abs/2401.12086) *Alizée Pace, Jonathan Mallinson, Eric Malmi, Sebastian Krause, Aliaksei Severyn.* Arxiv 2024.

### 5.5. Long Context

- [Make Your LLM Fully Utilize the Context.](https://arxiv.org/abs/2404.16811) *Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou.* Arxiv 2024.
- [From Artificial Needles to Real Haystacks: Improving Retrieval Capabilities in LLMs by Finetuning on Synthetic Data](https://arxiv.org/abs/2406.19292). *Zheyang Xiong, Vasilis Papageorgiou, Kangwook Lee, Dimitris Papailiopoulos*. ICLR 2025.
- [Scaling Instruction-tuned LLMs to Million-token Contexts via Hierarchical Synthetic Data Generation](https://openreview.net/forum?id=BkwCrIsTbR) *Linda He, Jue WANG, Maurice Weber, Shang Zhu, Ben Athiwaratkun, Ce Zhang.* ICLR 2025.🔥

### 5.6. Weak-to-Strong

- [Weak-to-strong generalization: Eliciting strong capabilities with weak supervision](https://proceedings.mlr.press/v235/burns24b.html). *Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, Jeffrey Wu.* ICML 2024.
- [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models.](https://arxiv.org/abs/2401.01335) *Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu.* ICML 2024.
- [Impossible Distillation for Paraphrasing and Summarization: How to Make High-quality Lemonade out of Small, Low-quality Models](https://arxiv.org/abs/2305.16635) *Jaehun Jung, Peter West, Liwei Jiang, Faeze Brahman, Ximing Lu, Jillian Fisher, Taylor Sorensen, Yejin Choi.* NAACL 2024.

### 5.7. Agent and Tool Use

- [ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases](https://arxiv.org/abs/2306.05301). *Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, Le Sun.* Arxiv 2023.
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761). *Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom.* NeurIPS 2023.
- [GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction](https://arxiv.org/abs/2305.18752). *Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, Ying Shan.* Neurips 2023.
- [Gorilla: Large Language Model Connected with Massive APIs.](https://arxiv.org/abs/2305.15334) *Shishir G. Patil, Tianjun Zhang, Xin Wang, Joseph E. Gonzalez.* NeurIPS 2024.
- [Quality Matters: Evaluating Synthetic Data for Tool-Using LLMs](https://aclanthology.org/2024.emnlp-main.285.pdf). *Shadi Iskander, Nachshon Cohen, Zohar Karnin, Ori Shapira, Sofia Tolmach.* EMNLP 2024.
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291). *Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anandkumar.* TMLR 2024.

### 5.8. Vision and Language

- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) *Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee.* NeurIPS 2023.
- [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966) *Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, Jingren Zhou.* Arxiv 2023.
- [Enhancing Cognition and Explainability of Multimodal Foundation Models with Self-Synthesized Data](https://openreview.net/forum?id=lHbLpwbEyt) *Yucheng Shi, Quanzheng Li, Jin Sun, Xiang Li, Ninghao Liu.* ICLR 2025.
- [G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model](https://arxiv.org/abs/2312.11370) *Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, Lingpeng Kong.* ICLR 2025.
- [MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models](https://arxiv.org/abs/2304.10592) *Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny.* ICLR 2024.
- [Enhancing Large Vision Language Models with Self-Training on Image Comprehension](https://arxiv.org/abs/2405.19716) *Yihe Deng, Pan Lu, Fan Yin, Ziniu Hu, Sheng Shen, James Zou, Kai-Wei Chang, Wei Wang.* NeurIPS 2024.
- [LLaVA-OneVision: Easy Visual Task Transfer](https://arxiv.org/abs/2408.03326) *Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, Chunyuan Li.* Submit to TMLR.

### 5.9. Factuality

- [MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents](https://arxiv.org/abs/2404.10774) *Liyan Tang, Philippe Laban, Greg Durrett.* EMNLP 2024.
- [Fine-tuning Language Models for Factuality](https://arxiv.org/abs/2311.08401) *Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher D. Manning, Chelsea Finn.* ICLR 2024.

### 5.10. Federated Learning

- [Prompt Public Large Language Models to Synthesize Data for Private On-device Applications](https://arxiv.org/pdf/2404.04360). *Shanshan Wu, Zheng Xu, Yanxiang Zhang, Yuanbo Zhang, Daniel Ramage.* COLM 2024.
- [Harnessing large-language models to generate private synthetic text](https://arxiv.org/abs/2306.01684). *Alexey Kurakin, Natalia Ponomareva, Umar Syed, Liam MacDermed, Andreas Terzis.* Arxiv 2024.

### 5.11. Generative Design

- [Generative Design through Quality-Diversity Data Synthesis and Language Models.](https://arxiv.org/abs/2405.09997) *Adam Gaier, James Stoddart, Lorenzo Villaggi, Shyam Sudhakaran.* GECCO 2024.

### 5.12 Safety

- [SynthPAI: A Synthetic Dataset for Personal Attribute Inference](https://arxiv.org/abs/2406.07217) *Hanna Yukhymenko, Robin Staab, Mark Vero, Martin Vechev.* NeurIPS D&B 2024.

### 5.13 Synthetis textbooks

- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) *Ronen Eldan, Yuanzhi Li.* Arxiv 2024.
- [Phi-1: Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) *Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, Yuanzhi Li.* Arxiv 2023.

### 5.14 Reasoning
- [Enhancing Reasoning Capabilities of LLMs via Principled Synthetic Logic Corpus](https://arxiv.org/abs/2411.12498) *Terufumi Morishita, Gaku Morio, Atsuki Yamaguchi, Yasuhiro Sogawa.* NeurIPS 2024
- [LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning](https://arxiv.org/abs/2409.12929) *Jin Jiang, Yuchen Yan, Yang Liu, Jianing Wang, Shuai Peng, Xunliang Cai, Yixin Cao, Mengdi Zhang, Liangcai Gao.* ACL 2025
 
### 5.15. Knowledge-Intensive Data
- [GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation](https://arxiv.org/abs/2505.20416) *Zihong Chen, Wanli Jiang, Jinzhe Li, Zhonghang Yuan, Huanjun Kong, Wanli Ouyang, Nanqing Dong* Arxiv 2025.


### 6. Tools

- [DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows.](https://arxiv.org/abs/2402.10379) *Ajay Patel, Colin Raffel, Chris Callison-Burch.* ACL 2024.
- [AgentInstruct: Toward Generative Teaching with Agentic Flows.](https://arxiv.org/abs/2407.03502) *Arindam Mitra, Luciano Del Corro, Guoqing Zheng, Shweti Mahajan, Dany Rouhana, Andres Codas, Yadong Lu, Wei-ge Chen, Olga Vrousgos, Corby Rosset, Fillipe Silva, Hamed Khanpour, Yash Lara, Ahmed Awadallah.* Arxiv 2024.
- [Distilabel: An AI Feedback (AIF) Framework for Building Datasets with and for LLMs](https://github.com/argilla-io/distilabel). *Álvaro Bartolomé Del Canto, Gabriel Martín Blázquez, Agustín Piqueres Lajarín and Daniel Vila Suero.* GitHub 2024.
- [Fuxion: Synthetic Data Generation and Normalization Functions using Langchain + LLMs](https://github.com/tobiadefami/fuxion).

### 7. Datasets

- [Open Artificial Knowledge](https://huggingface.co/datasets/tabularisai/oak) *Vadim Borisov, Richard Schreiber.* ICML Workshop 2024.
- [PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts](https://arxiv.org/abs/2202.01279) *Stephen H. Bach, Victor Sanh, Zheng-Xin Yong, Albert Webson, Colin Raffel, Nihal V. Nayak, Abheesht Sharma, Taewoon Kim, M Saiful Bari, Thibault Fevry, Zaid Alyafeai, Manan Dey, Andrea Santilli, Zhiqing Sun, Srulik Ben-David, Canwen Xu, Gunjan Chhablani, Han Wang, Jason Alan Fries, Maged S. Al-shaibani, Shanya Sharma, Urmish Thakker, Khalid Almubarak, Xiangru Tang, Dragomir Radev, Mike Tian-Jian Jiang, Alexander M. Rush*, ACL 2022 Demo.
- [Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks](https://arxiv.org/abs/2204.07705) *Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Gary Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Maitreya Patel, Kuntal Kumar Pal, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Shailaja Keyur Sampat, Savan Doshi, Siddhartha Mishra, Sujan Reddy, Sumanta Patro, Tanay Dixit, Xudong Shen, Chitta Baral, Yejin Choi, Noah A. Smith, Hannaneh Hajishirzi, Daniel Khashabi*. EMNLP 2022.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=pengr/LLM-Synthetic-Data&type=Date)](https://star-history.com/#pengr/LLM-Synthetic-Data&Date)
