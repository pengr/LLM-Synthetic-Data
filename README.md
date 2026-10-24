# Live LLM-Synthetic-Data Papers (Updated to July,2025)

<div align="center">

[![LICENSE](https://img.shields.io/github/license/wasiahmad/Awesome-LLM-Synthetic-Data-Generation)](https://github.com/pengr/LLM-Synthetic-Data/blob/main/LICENSE)
[![commit](https://img.shields.io/github/last-commit/wasiahmad/Awesome-LLM-Synthetic-Data-Generation?color=blue)](https://github.com/pengr/LLM-Synthetic-Data/commits/main)
[![PR](https://img.shields.io/badge/PRs-Welcome-red)](https://github.com/pengr/LLM-Synthetic-Data/pulls)
[![GitHub Repo stars](https://img.shields.io/github/stars/pengr/LLM-Synthetic-Data)](https://github.com/pengr/LLM-Synthetic-Data)

</div>

<!-- This repo collects **the most live-updated, finely categorized** work on LLM-synthetic-data, such as papers, tools, datasets, blogs, and more.   -->
This repo collects <span style="color:magenta;"><strong>the most live-updated, finely categorized</strong></span> work on LLM-synthetic-data, such as papers, tools, datasets, blogs, and more.  

If you find this useful, feel free to follow us and star a ⭐. Thanks to all great GitHub contributors!  

Entries marked with 🔥 are those we highly recommend.

---

## Latest updates

- **Section 3 (Surveys):** added domain-specific synthesis surveys.  
- **Section 4 (Method):** reorganized by LLM training stages with ultra-fine subcategories for each paper (**highly recommended**).  
- **Section 5 (Analysis):** new section for synthetic-data analyses.  
- **Section 6 (Application):** expanded to 19 new sub-areas.




## Contents

* [1. Githubs](#1-githubs)
* [2. Blogs](#2-blogs)
* [3. Surveys](#3-surveys)
* [4. Methods](#4-methods)
  * [4.1 Pre-training](#41-pre-training)
  * [4.2 Continue Pre-training](#42-continue-pre-training)
  * [4.3 Instruction Tuning](#43-instruction-tuning)
  * [4.4 Alignment](#44-alignment)
  * [4.5 Refinement Learning](#45-refinment-learning)
  * [4.6 LLM Benchmarking](#46-llm-benchmarking)
  * [4.7 Using Synthetic and Real Data Jointly](#47-synthetci-and-real)
* [5. Analysis](#5-analysis)
  * [5.1 Effect of Synthetic Data](#51-effect-syntectic-data)
  * [5.2 Evaluation of Synthetic Data](#52-evaluation-synthetic-data)
* [6. Application Areas](#6-application-areas)
  * [6.1 Mathematical Reasoning](#61-mathematical-reasoning)
  * [6.2 Code Generation](#62-code-generation)
  * [6.3 Agent and Tool Use](#63-agent-and-tool-use)
  * [6.4 Vision and Language](#64-vision-and-language)
  * [6.5 Retrieval-Augmented Generation ](#65-retrieval-augmented-generation)
  * [6.6 Long Context](#66-long-context)
  * [6.7 Writing](#67-wirting)
  * [6.8 AI For Science](#68-ai-for-science)
  * [6.9 Text-to-SQL](#69-text-to-sql)
  * [6.10 Synergy between Large and Small Models](#610-synergy-between-large--small-models)
  * [6.11 Weak-to-Strong](#611-weak-to-strong)
  * [6.12 Distill Small Model](#612-distill-small-model)
  * [6.13 Multilingual Data](#613-multilingual-data)
  * [6.14 Structured Data](#614-structured-data)
  * [6.15 Natural Language Understanding](#615-natural-language-understanding)
  * [6.16 Logic Reasoning](#616-logic-reasoning)
  * [6.17 Dialogue System](#617-dialogue-system)
  * [6.18 Federated Learning](#618-federated-learning)
  * [6.19 Generative Design](#619-generative-design)
* [7. Tools](#7-tools)
* [8. Datasets](#8-datasets)


## **1. Githubs**

- [https://github.com/mghiasvand1/Awesome-VLM-Synthetic-Data](https://github.com/mghiasvand1/Awesome-VLM-Synthetic-Data).🔥
- [https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data).🔥

## **2. Blogs**

- [Synthetic data: save money, time and carbon with open source.](https://huggingface.co/blog/synthetic-data-save-costs) *Moritz Laurer*. Feb 16, 2024.
- [Synthetic data generation (Part 1).](https://cookbook.openai.com/examples/sdg1)  *Dylan Royan Almeida*. Apr 10, 2024  
- [Synthetic dataset generation techniques: Self-Instruct.](https://huggingface.co/blog/davanstrien/self-instruct) *Daniel van Strien*. May 15, 2024.
- [CodecLM: Aligning language models with tailored synthetic data](https://research.google/blog/codeclm-aligning-language-models-with-tailored-synthetic-data) *Zifeng Wang and Chen-Yu Lee.* May 30, 2024.
- [The Rise of Agentic Data Generation.](https://huggingface.co/blog/mlabonne/agentic-datagen) *Maxime Labonne.* July 15, 2024.
- [LLM-Driven Synthetic Data Generation, Curation & Evaluation.](https://cobusgreyling.medium.com/llm-driven-synthetic-data-generation-curation-evaluation-33731e33b525) *Cobus Greyling.* Aug 2, 2024.
- [Using LLMs for Synthetic Data Generation: The Definitive Guide](https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms) *Kritin Vongthongsri.* November 8, 2024.

## **3. Surveys**

- [Best Practices and Lessons Learned on Synthetic Data for Language Models.](https://arxiv.org/abs/2404.07503) *Ruibo Liu, Jerry Wei, Fangyu Liu, Chenglei Si, Yanzhe Zhang, Jinmeng Rao, Steven Zheng, Daiyi Peng, Diyi Yang, Denny Zhou, Andrew M. Dai.* COLM 2024.
- [On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey.](https://arxiv.org/abs/2406.15126) *Lin Long, Rui Wang, Ruixuan Xiao, Junbo Zhao, Xiao Ding, Gang Chen, Haobo Wang.* ACL Findings 2024.
- [Large Language Models for Data Annotation: A Survey](https://arxiv.org/abs/2402.13446) *Zhen Tan, Dawei Li, Song Wang, Alimohammad Beigi, Bohan Jiang, Amrita Bhattacharjee, Mansooreh Karami, Jundong Li, Lu Cheng, Huan Liu.* EMNLP 2024.
- [Generative AI for Synthetic Data Generation: Methods, Challenges and the Future.](https://arxiv.org/abs/2403.04190) *Xu Guo, Yiqiang Chen.* Arxiv 2024.
- [Comprehensive Exploration of Synthetic Data Generation: A Survey.](https://arxiv.org/abs/2401.02524) *André Bauer, Simon Trapp, Michael Stenger, Robert Leppich, Samuel Kounev, Mark Leznik, Kyle Chard, Ian Foster.* Arxiv 2024.
- [A Survey on Bridging VLMs and Synthetic Data.](https://openreview.net/pdf?id=ThjDCZOljE) *Mohammad Ghiasvand Mohammadkhani, Saeedeh Momtazi, Hamid Beigy.* OpenReview 2025.
- [A Survey on Data Synthesis and Augmentation for Large Language Models](https://arxiv.org/abs/2410.12896) *Ke Wang, Jiahui Zhu, Minjie Ren, Zeming Liu, Shiwei Li, Zongye Zhang, Chenkai Zhang, Xiaoyu Wu, Qiqi Zhan, Qingjie Liu, Yunhong Wang.*Arxiv 2024.
- [Synthetic Data Generation Using Large Language Models: Advances in Text and Code](https://arxiv.org/abs/2503.14023) *Mihai Nadas, Laura Diosan, Andreea Tomescu.* Arxiv 2025.
- [A Comprehensive Survey of Synthetic Tabular Data Generation](https://arxiv.org/abs/2504.16506) *Ruxue Shi, Yili Wang, Mengnan Du, Xu Shen, Yi Chang, Xin Wang.* Arxiv 2025.
- [Large Language Models for Traffic and Transportation Research: Methodologies, State of the Art, and Future Opportunities](https://arxiv.org/abs/2503.21330) *Yimo Yan, Yejia Liao, Guanhao Xu, Ruili Yao, Huiying Fan, Jingran Sun, Xia Wang, Jonathan Sprinkle, Ziyan An, Meiyi Ma, Xi Cheng, Tong Liu, Zemian Ke, Bo Zou, Matthew Barth, Yong-Hong Kuo.* Arxiv 2025.
- [Biological Sequence with Language Model Prompting: A Survey](https://arxiv.org/abs/2503.04135) *Jiyue Jiang, Zikang Wang, Yuheng Shan, Heyan Chai, Jiayi Li, Zixian Ma, Xinrui Zhang, Yu Li* Arxiv 2025.

## **4. Methods**

### **4.1. Pre-training**
- [Phi-4 Technical Report](https://arxiv.org/pdf/2412.08905) *Microsoft Research.* Arxiv 2024.🔥
- [Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset](https://arxiv.org/abs/2412.02595) *Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro*, Arxiv 2024. 🔥
- [Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent](https://arxiv.org/pdf/2411.02265) *Tecent Hunyuan Team.* Arxiv 2024.🔥
- **A. Rephrasing document**
  - [Rephrasing theWeb A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380) *Pratyush Maini, Skyler Seto, He Bai, David Grangier, Yizhe Zhang, Navdeep Jaitly.* ACL 2024.🔥
  - [ToEdit: How to Synthesize Text Data to Avoid Model Collapse?](https://openreview.net/forum?id=mVCcWCjeEz) ICML 2025.

### **4.2. Continue Pre-training**
- [Phi-1: Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) *Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, Yuanzhi Li.* Arxiv 2023.
- [MAmmoTH2: Scaling Instructions from the Web.](https://arxiv.org/pdf/2405.03548) *Xiang Yue, Tuney Zheng, Ge Zhang, Wenhu Chen.* Neurips 2024.🔥
- [Scaling Laws of Synthetic Data for Language Models](https://arxiv.org/abs/2503.19551) *Zeyu Qin, Qingxiu Dong, Xingxing Zhang, Li Dong, Xiaolong Huang, Ziyi Yang, Mahmoud Khademi, Dongdong Zhang, Hany Hassan Awadalla, Yi R. Fung, Weizhu Chen, Minhao Cheng, Furu Wei.* Arxiv 2025.

### **4.3. Instruction Tuning**

#### **4.3.1 Before ChatGPT came out**
- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) *Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman.* NeurIPS 2022.
- [Generating Training Data with Language Models: Towards Zero-Shot Language Understanding](https://arxiv.org/abs/2202.04538) *Yu Meng, Jiaxin Huang, Yu Zhang, Jiawei Han.* NeurIPS 2022.
- [ZeroGen: Efficient Zero-shot Learning via Dataset Generation](https://arxiv.org/abs/2202.07922) *Jiacheng Ye, Jiahui Gao, Qintong Li, Hang Xu, Jiangtao Feng, Zhiyong Wu, Tao Yu, Lingpeng Kong.* EMNLP 2022.
- [Symbolic Knowledge Distillation: from General Language Models to Commonsense Models](https://arxiv.org/abs/2110.07178) *Peter West, Chandra Bhagavatula, Jack Hessel, Jena D. Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, Yejin Choi.* NAACL 2022.
- [Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor](https://arxiv.org/abs/2212.09689)  *Or Honovich, Thomas Scialom, Omer Levy, Timo Schick.* ACL 2023.

#### **4.3.2 Rephrasing instructions**
- **A. Evolving Instrution**
  - [WizardLM: Empowering Large Language Models to Follow Complex Instructions.](https://arxiv.org/abs/2304.12244) *Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Daxin Jiang.* ICLR 2024.🔥
  - [Automatic Instruction Evolving for Large Language Models](https://arxiv.org/abs/2406.00770) *Weihao Zeng, Can Xu, Yingxiu Zhao, Jian-Guang Lou, Weizhu Chen.* EMNLP 2024.
  - [Optimizing Instruction Synthesis: Effective Exploration of Evolutionary Space with Tree Search](https://arxiv.org/abs/2410.10392) *Chenglin Li, Qianglong Chen, Zhi Li, Feng Tao, Yicheng Li, Hao Chen, Fei Yu, Yin Zhang.* EMNLP Findings 2024. 
  - [Infinity Instruct: Scaling Instruction Selection and Synthesis to Enhance Language Models](https://arxiv.org/abs/2506.11116) *Jijie Li, Li Du, Hanyu Zhao, Bo-wen Zhang, Liangdong Wang, Boyan Gao, Guang Liu, Yonghua Lin.* Arxiv 2025.
  - [AIDE: Attribute-Guided MultI-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](https://arxiv.org/abs/2412.06136) *Jiayu Li, Xuan Zhu, Fang Liu, Yanjun Qi.* ACL 2025.

- **B. Curating Persona Database**
   - [Scaling Synthetic Data Creation with 1,000,000,000 Personas.](https://arxiv.org/abs/2406.20094) *Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, Dong Yu.* Arxiv 2024.🔥
  - [OpenCharacter: Training Customizable Role-Playing LLMs with Large-Scale Synthetic Personas](https://arxiv.org/abs/2501.15427) *Xiaoyang Wang, Hongming Zhang, Tao Ge, Wenhao Yu, Dian Yu, Dong Yu.* Arxiv 2025.

#### **4.3.3 Instruction Inversion**
- **A. Instruction Inversion Directly From Document**
  - [LongForm: Effective Instruction Tuning with Reverse Instructions.](https://arxiv.org/abs/2304.08460) *Abdullatif Köksal, Timo Schick, Anna Korhonen, Hinrich Schütze.* EMNLP 2024 findings.🔥
  - [Self-Alignment with Instruction Backtranslation.](https://arxiv.org/abs/2308.06259) *Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, Jason Weston, Mike Lewis.* ICLR 2024 Oral
  - [From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding](https://arxiv.org/abs/2506.03968) *Chiwei Zhu, Benfeng Xu, Xiaorui Wang, Zhendong Mao.* ACL 2025.🔥

- **B. Extract document followed by Instruction Inversion**
  - [Knowledge-Instruct: Effective Continual Pre-training from Limited Data using Instructions](https://arxiv.org/abs/2504.05571) *Oded Ovadia, Meni Brief, Rachel Lemberg, Eitam Sheetrit.* Arxiv 2025.
  - [Source2Synth: Synthetic Data Generation and Curation Grounded in Real Data Sources](https://arxiv.org/abs/2409.08239) *Alisia Lupidi, Carlos Gemmell, Nicola Cancedda, Jane Dwivedi-Yu, Jason Weston, Jakob Foerster, Roberta Raileanu, Maria Lomeli.* Arxiv 2025.


#### **4.3.4 Synthesis instructions and outputs using LLM**
- **A. Rephrasing instructions and Instruction Inversion**
  - [Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction](https://arxiv.org/abs/2504.15573) *Yuxin Jiang, Yufei Wang, Chuhan Wu, Xinyi Dai, Yan Xu, Weinan Gan, Yasheng Wang, Xin Jiang, Lifeng Shang, Ruiming Tang, Wei Wang.* ACL 2025.

- **B. Synthesis from scratch without seed**
  - [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464) *Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, Bill Yuchen Lin.* ICLR 2025.
  - [LongMagpie: A Self-synthesis Method for Generating Large-scale Long-context Instructions](https://arxiv.org/abs/2505.17134) *Chaochen Gao, Xing Wu, Zijia Lin, Debing Zhang, Songlin Hu.* 
  - [TarGEN: Targeted Data Generation with Large Language Models](https://openreview.net/pdf?id=gpgMRWgv9Q) *Himanshu Gupta, Kevin Scaria, Ujjwala Anantheswaran, Shreyas Verma, Mihir Parmar, Saurabh Arjun Sawant, Chitta Baral, Swaroop Mishra.* COLM 2024.
  - [LAB: Large-Scale Alignment for ChatBots](https://arxiv.org/abs/2403.01081) *Shivchander Sudalairaj, Abhishek Bhandwaldar, Aldo Pareja, Kai Xu, David D. Cox, Akash Srivastava.* Arxiv 2024🔥


- **C. Extract Q&A Pairs**
  - [MAmmoTH2: Scaling Instructions from the Web.](https://arxiv.org/pdf/2405.03548) *Xiang Yue, Tuney Zheng, Ge Zhang, Wenhu Chen.* Neurips 2024.🔥

- **D. Knowledge-based Synthesis with seed**
 - *LLM-Generated Structural Knowledge*
   - [Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models](https://arxiv.org/abs/2402.13064) *Haoran Li, Qingxiu Dong, Zhengyang Tang, Chaojun Wang, Xingxing Zhang, Haoyang Huang, Shaohan Huang, Xiaolong Huang, Zeqiang Huang, Dongdong Zhang, Yuxian Gu, Xin Cheng, Xun Wang, Si-Qing Chen, Li Dong, Wei Lu, Zhifang Sui, Benyou Wang, Wai Lam, Furu Wei.* Arxiv 2025.
   - [TreeSynth: Synthesizing Diverse Data from Scratch via Tree-Guided Subspace Partitioning](https://www.arxiv.org/abs/2503.17195) *Sheng Wang, Pengan Chen, Jingqi Zhou, Qintong Li, Jingwei Dong, Jiahui Gao, Boyang Xue, Jiyue Jiang, Lingpeng Kong, Chuan Wu.* Arxiv 2025.
 - *LLM Self-Reflection*
   - [Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement](https://arxiv.org/abs/2501.12273) *Maosong Cao, Taolin Zhang, Mo Li, Chuyu Zhang, Yunxin Liu, Haodong Duan, Songyang Zhang, Kai Chen.* ACL 2025.
   - [DataGen: Unified Synthetic Dataset Generation via Large Language Models](https://openreview.net/forum?id=F5R0lG74Tu) *Yue Huang, Siyuan Wu, Chujie Gao, Dongping Chen, Qihui Zhang, Yao Wan, Tianyi Zhou, Xiangliang Zhang, Jianfeng Gao, Chaowei Xiao, Lichao Sun.* ICLR 2025.
   - [Beyond Sample-Level Feedback: Using Reference-Level Feedback to Guide Data Synthesis](https://arxiv.org/abs/2502.04511) *Shuhaib Mehri, Xiusi Chen, Heng Ji, Dilek Hakkani-Tür.* Arxiv 2025.
   - [GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation](https://arxiv.org/abs/2505.20416) *Zihong Chen, Wanli Jiang, Jinzhe Li, Zhonghang Yuan, Huanjun Kong, Wanli Ouyang, Nanqing Dong* Arxiv 2025.

- **E. External Knowledge**
 - *Human Rubric*
   - [Large Language Model as Attributed Training Data Generator: A Tale of Diversity and Bias](https://arxiv.org/pdf/2306.15895) *Yue Yu, Yuchen Zhuang, Jieyu Zhang, Yu Meng, Alexander Ratner, Ranjay Krishna, Jiaming Shen, Chao Zhang.* NeurIPS D&B 2023.
   - [Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision.](https://arxiv.org/abs/2308.06259) *Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, Chuang Gan.* NeurIPS 2023.🔥
 - *Data Attributes* 
   - [Learning to Generate Instruction Tuning Datasets for Zero-Shot Task Adaptation](https://arxiv.org/abs/2402.18334) *Nihal V. Nayak, Yiyang Nan, Avi Trost, Stephen H. Bach.* ACL Findings 2024.🔥
   - [CodecLM: Aligning Language Models with Tailored Synthetic Data.](https://arxiv.org/abs/2404.05875) *Zifeng Wang, Chun-Liang Li, Vincent Perot, Long T. Le, Jin Miao, Zizhao Zhang, Chen-Yu Lee, Tomas Pfister.* NAACL Findings 2024.🔥

- **F. Prompt-Based Synthesis with Seed**
 - *Few-shot Example Prompting*
   - [Self-instruct: Aligning language models with self-generated instructions.](https://arxiv.org/abs/2212.10560) *Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi.* ACL 2023.🔥
   - [Few-shot LLM Synthetic Data with Distribution Matching](https://arxiv.org/abs/2502.08661) *Jiyuan Ren, Zhaocheng Du, Zhihao Wen, Qinglin Jia, Sunhao Dai, Chuhan Wu, Zhenhua Dong.* WWW 2025.
   - [Generating Faithful Synthetic Data with Large Language Models: A Case Study in Computational Social Science](https://arxiv.org/abs/2305.15041) *Veniamin Veselovsky, Manoel Horta Ribeiro, Akhil Arora, Martin Josifoski, Ashton Anderson, Robert West,* Arxiv 2023.🔥
 - *Diverse Inference Settings*
   - [Increasing Diversity While Maintaining Accuracy: Text Data Generation with Large Language Models and Human Interventions](https://arxiv.org/abs/2306.04140) *John Joon Young Chung, Ece Kamar, Saleema Amershi.* ACL 2023.🔥
   - [Large Language Models Can Self-Improve](https://aclanthology.org/2023.emnlp-main.67/) *Jiaxin Huang, Shixiang Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, Jiawei Han.* EMNLP 2023.
 - *Fixed Prompt Templates*
   - [Making Large Language Models Better Data Creators](https://arxiv.org/pdf/2310.20111) *Dong-Ho Lee, Jay Pujara, Mohit Sewak, Ryen W. White, Sujay Kumar Jauhar.* EMNLP 2023.

- **G. Model Iteration**
   - [ALMA: Alignment with Minimal Annotation](https://arxiv.org/abs/2412.04305) *Michihiro Yasunaga, Leonid Shamis, Chunting Zhou, Andrew Cohen, Jason Weston, Luke Zettlemoyer, Marjan Ghazvininejad.* Arxiv 2024.

- **H. Agent**
   - [MetaSynth: Meta-Prompting-Driven Agentic Scaffolds for Diverse Synthetic Data Generation](https://arxiv.org/abs/2504.12563) *Haris Riaz, Sourav Bhabesh, Vinayak Arannil, Miguel Ballesteros, Graham Horwood.* Findings of ACL 2025

- **I. Learn from failure**
  - [Forewarned is Forearmed: Harnessing LLMs for Data Synthesis via Failure-induced Exploration.](https://openreview.net/forum?id=yitH9xAHQs) ICLR 2025.
  - [CDS: Knowledge Component-Driven Data Synthesis Guided by Cognitive Diagnosis Theory](https://arxiv.org/pdf/2501.07674) *Haokun Zhao, Jinyi Han, Jiaqing Liang, Yanghua Xiao, Xiaojun Meng, Jiansheng Wei.* Arxiv 2025.
  - [SIPDO: Closed-Loop Prompt Optimization via Synthetic Data Feedback](https://arxiv.org/abs/2505.19514) *Yaoning Yu, Ye Yu, Kai Wei, Haojing Luo, Haohan Wang.* Arxiv 2025.


#### **4.3.5 Generating context from instruction-answer pairs**
- [Generalizing From Short to Long: Effective Data Synthesis for Long-Context Instruction Tuning](https://arxiv.org/abs/2502.15592) *Wenhao Zhu, Pinzhen Chen, Hanxu Hu, Shujian Huang, Fei Yuan, Jiajun Chen, Alexandra Birch.* Arxiv 2025.


### **4.4. Alignment**

#### **4.4.1 Self-align**
- [Self-Rewarding Language Models.](https://arxiv.org/abs/2401.10020) *Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston.* ICML 2024.

#### **4.4.2 Human Principle Alignment**
- [Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision](https://arxiv.org/abs/2305.03047) *Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, Chuang Gan.* NeurIPS 2023.
- [Step-by-Step Mastery: Enhancing Soft Constraint Following Ability of Large Language Models](https://aclanthology.org/2025.findings-acl.1004/) *Qingyu Ren, Jie Zeng, Qianyu He, Jiaqing Liang, Yanghua Xiao, Weikang Zhou, Zeye Sun, Fei Yu.* ACL Findings 2025.

#### **4.4.3 RLAIF**
- [Phi-4 Technical Report](https://arxiv.org/pdf/2412.08905) *Microsoft Research.* Arxiv 2024.🔥
- [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464) *Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, Bill Yuchen Lin.* ICLR 2025.
- [Montessori-Instruct: Generate Influential Training Data Tailored for Student Learning](https://arxiv.org/abs/2410.14208) *Xiaochuan Li, Zichun Yu, Chenyan Xiong.* ICLR 2025.

#### **4.4.4 Safety**
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) *Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan.* Arxiv 2022.
- [SynthPAI: A Synthetic Dataset for Personal Attribute Inference](https://arxiv.org/abs/2406.07217) *Hanna Yukhymenko, Robin Staab, Mark Vero, Martin Vechev.* NeurIPS D&B 2024.
- [Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts](https://arxiv.org/abs/2402.16822) *Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rocktäschel, Roberta Raileanu.* NeurIPS 2024.
- [Refined Direct Preference Optimization with Synthetic Data for Behavioral Alignment of LLMs](https://arxiv.org/abs/2402.08005) *V´ıctor Gallego.* Arxiv 2024.
- [TRIDENT: Enhancing Large Language Model Safety with Tri-Dimensional Diversified Red-Teaming Data Synthesis](https://arxiv.org/pdf/2505.24672) *Xiaorui Wu, Xiaofeng Mao, Fei Li, Xin Zhang, Xuanhong Li, Chong Teng, Donghong Ji, Zhuang Li.* ACL 2025 
- [PoisonSwarm: Universal Harmful Information Synthesis via Model Crowdsourcing](https://arxiv.org/abs/2505.21184) *Yu Yan, Sheng Sun, Zhifei Zheng, Ziji Hao, Teli Liu, Min Liu.* Arxiv 2025.
- [AgentAlign: Navigating Safety Alignment in the Shift from Informative to Agentic Large Language Models](https://arxiv.org/abs/2505.23020) *Jinchuan Zhang, Lu Yin, Yan Zhou, Songlin Hu.* Arxiv 2025.
- [Synthesizing Privacy-Preserving Text Data via Finetuning without Finetuning Billion-Scale LLMs](https://arxiv.org/abs/2503.12347) *Bowen Tan, Zheng Xu, Eric Xing, Zhiting Hu, Shanshan Wu.* ICML 2025.
- [Scalable and Ethical Insider Threat Detection through Data Synthesis and Analysis by LLMs](https://arxiv.org/abs/2502.07045) *Haywood Gelman, John D. Hastings.* Arxiv 2025.
- [ToxiLab: How Well Do Open-Source LLMs Generate Synthetic Toxicity Data?](https://arxiv.org/abs/2411.15175) *Zheng Hui, Zhaoxiao Guo, Hang Zhao, Juanyong Duan, Lin Ai, Yinheng Li, Julia Hirschberg, Congrui Huang.* Arxiv 2024.
- [Model-based Large Language Model Customization as Service](https://arxiv.org/abs/2410.10481) *Zhaomin Wu, Jizhou Guo, Junyi Hou, Bingsheng He, Lixin Fan, Qiang Yang.* Arxiv 2024.


### **4.5. Refinement Learning**
- [SALMON: Self-Alignment with Instructable Reward Models](https://arxiv.org/abs/2310.05910) *Zhiqing Sun, Yikang Shen, Hongxin Zhang, Qinhong Zhou, Zhenfang Chen, David Cox, Yiming Yang, Chuang Gan.* ICLR 2024.
- [West-of-N: Synthetic Preference Generation for Improved Reward Modeling.](https://arxiv.org/abs/2401.12086) *Alizée Pace, Jonathan Mallinson, Eric Malmi, Sebastian Krause, Aliaksei Severyn.* Arxiv 2024.
- [LLMs Meet Finance: Fine-Tuning Foundation Models for the Open FinLLM Leaderboard](https://arxiv.org/abs/2504.13125) *Varun Rao, Youran Sun, Mahendra Kumar, Tejas Mutneja, Agastya Mukherjee, Haizhao Yang.* Arxiv 2025.
- **A. Self-playing**
  - [Self-playing Adversarial Language Game Enhances LLM Reasoning](https://arxiv.org/abs/2404.10642) *Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Yong Dai, Lei Han, Nan Du.* Neurips 2024.
  - [Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models.](https://arxiv.org/abs/2406.13542) *Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, Jingren Zhou.* ICLR 2025.
  - [Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models](https://arxiv.org/abs/2406.13542). *Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, Jingren Zhou.* Submit to ICLR 2025.


### **4.6. LLM Benchmarking**
- [DataGen: Unified Synthetic Dataset Generation via Large Language Models](https://openreview.net/forum?id=F5R0lG74Tu) *Yue Huang, Siyuan Wu, Chujie Gao, Dongping Chen, Qihui Zhang, Yao Wan, Tianyi Zhou, Xiangliang Zhang, Jianfeng Gao, Chaowei Xiao, Lichao Sun.* ICLR 2025.

### **4.7. Using synthetic and real data jointly**

- [Unveiling the Flaws: Exploring Imperfections in Synthetic Data and Mitigation Strategies for Large Language Models](https://arxiv.org/abs/2406.12397) *Jie Chen, Yupeng Zhang, Bingning Wang, Wayne Xin Zhao, Ji-Rong Wen, Weipeng Chen.* EMNLP Findings 2024.
- [Few-shot LLM Synthetic Data with Distribution Matching](https://arxiv.org/abs/2502.08661) *Jiyuan Ren, Zhaocheng Du, Zhihao Wen, Qinglin Jia, Sunhao Dai, Chuhan Wu, Zhenhua Dong.* WWW 2025.


## **5. Analysis**
### **5.1. Effect of Synthetic Data**
- [AI models collapse when trained on recursively generated data](https://www.nature.com/articles/s41586-024-07566-y) *Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson & Yarin Gal,* Nature 2024.
- [How bad is training on synthetic data? A statistical analysis of language model collapse.](https://openreview.net/forum?id=t3z6UlV09o#discussion) *Mohamed El Amine Seddik, Suei-Wen Chen, Soufiane Hayou, Pierre Youssef, Merouane Abdelkader DEBBAH.* COLM 2024.
- [Strong Model Collapse](https://arxiv.org/abs/2410.04840) *Elvis Dohmatob, Yunzhen Feng, Arjun Subramonian, Julia Kempe.* ICLR 2025.
- [Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training: A Reverse-Bottleneck Perspective](https://arxiv.org/abs/2410.01720) *Zeyu Gan, Yong Liu.* ICLR 2025.
- [Beyond Model Collapse: Scaling Up with Synthesized Data Requires Reinforcement](https://arxiv.org/pdf/2406.07515) *Yunzhen Feqng, Elvis Dohmatob, Pu Yang, Francois Charton, Julia Kempe.* ICLR 2025.
- [A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops](https://www.arxiv.org/abs/2502.18865) *Shi Fu, Yingjie Wang, Yuzhu Chen, Xinmei Tian, Dacheng Tao.* ICLR 2025
- [Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data](https://arxiv.org/abs/2404.01413) *Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, Henry Sleight, John Hughes, Tomasz Korbak, Rajashree Agrawal, Dhruv Pai, Andrey Gromov, Daniel A. Roberts, Diyi Yang, David L. Donoho, Sanmi Koyejo.* Arxiv 2024

### **5.2 Evaluation of Synthetic Data**
#### **5.2.1 Artifactuality**
- [Under the Surface: Tracking the Artifactuality of LLM-Generated Data](https://arxiv.org/abs/2401.14698) *Debarati Das, Karin De Langis, Anna Martin-Boyle, Jaehyung Kim, Minhwa Lee, Zae Myung Kim, Shirley Anugrah Hayati, Risako Owan, Bin Hu, Ritik Parkar, Ryan Koo, Jonginn Park, Aahan Tyagi, Libby Ferland, Sanjali Roy, Vincent Liu, Dongyeop Kang.* Arxiv 2024.

#### **5.2.2 Fidelity**
- [MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents](https://arxiv.org/abs/2404.10774) *Liyan Tang, Philippe Laban, Greg Durrett.* EMNLP 2024.
- [How bad is training on synthetic data? A statistical analysis of language model collapse](https://arxiv.org/abs/2404.05090) *Mohamed El Amine Seddik, Suei-Wen Chen, Soufiane Hayou, Pierre Youssef, Merouane Debbah.* COLM 2024.
- [Fine-tuning Language Models for Factuality](https://arxiv.org/abs/2311.08401) *Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher D. Manning, Chelsea Finn.* ICLR 2024. 
- [ToEdit: How to Synthesize Text Data to Avoid Model Collapse?](https://openreview.net/forum?id=mVCcWCjeEz) ICML 2025.

#### **5.2.3 Diversity**
- [Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning](https://arxiv.org/abs/2505.20161) *Jaehun Jung, Seungju Han, Ximing Lu, Skyler Hallinan, David Acuna, Shrimai Prabhumoye, Mostafa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Yejin Choi.* Arxiv 2025.
- [On the Diversity of Synthetic Data and its Impact on Training Large Language Models](https://arxiv.org/abs/2410.15226) *Hao Chen, Abdul Waheed, Xiang Li, Yidong Wang, Jindong Wang, Bhiksha Raj, Marah I. Abdin.* Arxiv 2024.
- [Evaluating the Diversity and Quality of LLM Generated Content](https://arxiv.org/abs/2504.12522) *Alexander Shypula, Shuo Li, Botong Zhang, Vishakh Padmakumar, Kayo Yin, Osbert Bastani* 	ICLR 2025 Workshop.
- [Less is More: Adaptive Coverage for Synthetic Training Data](https://arxiv.org/abs/2504.14508) *Sasan Tavakkol, Max Springer, Mohammadhossein Bateni, Neslihan Bulut, Vincent Cohen-Addad, MohammadTaghi Hajiaghayi.* Arxiv 2025.
- [Measuring diversity of synthetic prompts and data generated with fine-grained persona prompting](https://www.arxiv.org/abs/2505.17390) *Gauri Kambhatla, Chantal Shaib, Venkata Govindarajan.* Arxiv 2025. (Focus on Persona Diversity)


## **6. Application Areas**

### **6.1 Mathematical Reasoning**

- [Distilling LLMs' Decomposition Abilities into Compact Language Models](https://arxiv.org/abs/2402.01812) *Denis Tarasov, Kumar Shridhar.* AutoRL@ICML 2024.
- [MuggleMath: Assessing the Impact of Query and Response Augmentation on Math Reasoning](https://arxiv.org/abs/2310.05506v3) *Chengpeng Li, Zheng Yuan, Hongyi Yuan, Guanting Dong, Keming Lu, Jiancan Wu, Chuanqi Tan, Xiang Wang, Chang Zhou.* ACL 2024.
- [MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs](https://arxiv.org/abs/2402.16352) *Zimu Lu, Aojun Zhou, Houxing Ren, Ke Wang, Weikang Shi, Junting Pan, Mingjie Zhan, Hongsheng Li.* ACL 2024.
- [MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models](https://arxiv.org/abs/2309.12284) *Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, Weiyang Liu.* ICLR 2024.
- [Augmenting Math Word Problems via Iterative Question Composing](https://arxiv.org/abs/2401.09003) *Haoxiong Liu, Yifan Zhang, Yifan Luo, Andrew Chi-Chih Yao.* DPFM@ICLR 2024.
- [Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning](https://arxiv.org/abs/2403.02333) *Yiming Huang, Xiao Liu, Yeyun Gong, Zhibin Gou, Yelong Shen, Nan Duan, Weizhu Chen.* Arxiv 2024.
- [Dart-math: Difficulty-aware rejection tuning for mathematical problem-solving](https://arxiv.org/abs/2407.13690) *Yuxuan Tong, Xiwen Zhang, Rui Wang, Ruidong Wu, Junxian He.* NeurIPS 2024.
- [RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold](https://arxiv.org/abs/2406.14532) *Amrith Setlur, Saurabh Garg, Xinyang Geng, Naman Garg, Virginia Smith, Aviral Kumar.* NeurIPS 2025
- [VCR: A "Cone of Experience" Driven Synthetic Data Generation Framework for Mathematical Reasoning](https://ojs.aaai.org/index.php/AAAI/article/download/34645/36800) *Sannyuya Liu, Jintian Feng, Xiaoxuan Shen, Shengyingjie Liu, Qian Wan, Jianwen Sun.* AAAI 2025.
- [Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning](https://arxiv.org/abs/2403.02333) *Yiming Huang, Xiao Liu, Yeyun Gong, Zhibin Gou, Yelong Shen, Nan Duan, Weizhu Chen.* AAAI2025.
- [Synthesis by Design: Controlled Data Generation via Structural Guidance](https://arxiv.org/abs/2506.07664) *Lei Xu, Sirui Chen, Yuxuan Huang, Chaochao Lu.* Arxiv 2025.
- [LLM-based Automated Theorem Proving Hinges on Scalable Synthetic Data Generation](https://arxiv.org/abs/2505.12031) *Junyu Lai, Jiakun Zhang, Shuo Xu, Taolue Chen, Zihang Wang, Yao Yang, Jiarui Zhang, Chun Cao, Jingwei Xu.* Arxiv 2025.
- [SATURN: SAT-based Reinforcement Learning to Unleash Language Model Reasoning](https://arxiv.org/abs/2505.16368) *Huanyu Liu, Jia Li, Hao Zhu, Kechi Zhang, Yihong Dong, Ge Li.* Arxiv 2025.
- [RV-Syn: Rational and Verifiable Mathematical Reasoning Data Synthesis based on Structured Function Library](https://arxiv.org/abs/2504.20426) *Jiapeng Wang, Jinhao Jiang, Zhiqiang Zhang, Jun Zhou, Wayne Xin Zhao.* Arxiv 2025.
- [MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion](https://arxiv.org/abs/2503.16212) *Qizhi Pei, Lijun Wu, Zhuoshi Pan, Yu Li, Honglin Lin, Chenlin Ming, Xin Gao, Conghui He, Rui Yan.* ACL 2025.
- [AutoLogi: Automated Generation of Logic Puzzles for Evaluating Reasoning Abilities of Large Language Models](https://arxiv.org/abs/2502.16906) *Qin Zhu, Fei Huang, Runyu Peng, Keming Lu, Bowen Yu, Qinyuan Cheng, Xipeng Qiu, Xuanjing Huang, Junyang Lin.* Arxiv 2025.
- [CDS: Knowledge Component-Driven Data Synthesis Guided by Cognitive Diagnosis Theory](https://arxiv.org/pdf/2501.07674) *Haokun Zhao, Jinyi Han, Jiaqing Liang, Yanghua Xiao, Xiaojun Meng, Jiansheng Wei.* Arxiv 2025.
- [A Graph-Based Synthetic Data Pipeline for Scaling High-Quality Reasoning Instructions](https://arxiv.org/abs/2412.08864) *Jiankang Wang, Jianjun Xu, Xiaorui Wang, Yuxin Wang, Mengting Xing, Shancheng Fang, Zhineng Chen, Hongtao Xie, Yongdong Zhang.* Arxiv 2024.
- [Unleashing LLM Reasoning Capability via Scalable Question Synthesis from Scratch](https://arxiv.org/abs/2410.18693) *Yuyang Ding, Xinyu Shi, Xiaobo Liang, Juntao Li, Zhaopeng Tu, Qiaoming Zhu, Min Zhang.* ACL 2025.
- [Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing](https://arxiv.org/abs/2404.12253) *Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Haitao Mi, Dong Yu.* NeurIPS 2024.
- [Step-Opt: Boosting Optimization Modeling in LLMs through Iterative Data Synthesis and Structured Validation](https://arxiv.org/abs/2506.17637) *Yang Wu, Yifan Zhang, Yurong Wu, Yuran Wang, Junkai Zhang, Jian Cheng.* Arxiv 2025.
- [FLAMES: Improving LLM Math Reasoning via a Fine-Grained Analysis of the Data Synthesis Pipeline](https://arxiv.org/abs/2508.16514) *Parker Seegmiller, Kartik Mehta, Soumya Saha, Chenyang Tao, Shereen Oraby, Arpit Gupta, Tagyoung Chung, Mohit Bansal, Nanyun Peng.* EMNLP 2025.


### **6.2 Code Generation**

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
- [Veritas: Deterministic Verilog Code Synthesis from LLM-Generated Conjunctive Normal Form](https://arxiv.org/abs/2506.00005) *Prithwish Basu Roy, Akashdeep Saha, Manaar Alam, Johann Knechtel, Michail Maniatakos, Ozgur Sinanoglu, Ramesh Karri.* Arxiv 2025.
- [Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data](https://arxiv.org/abs/2506.07390) *Xin-Cheng Wen, Yijun Yang, Cuiyun Gao, Yang Xiao, Deheng Ye.* ACL Findings 2025.
- [DSCodeBench: A Realistic Benchmark for Data Science Code Generation](https://arxiv.org/abs/2505.15621) *Shuyin Ouyang, Dong Huang, Jingwen Guo, Zeyu Sun, Qihao Zhu, Jie M. Zhang.* Arxiv 2025.
- [Infinite-Instruct: Synthesizing Scaling Code instruction Data with Bidirectional Synthesis and Static Verification](https://arxiv.org/abs/2505.23177) *Wenjing Xing, Wenke Lu, Yeheng Duan, Bing Zhao, Zhenghui kang, Yaolong Wang, Kai Gao, Lei Qiao.* Arxiv 2025.
- [SATURN: SAT-based Reinforcement Learning to Unleash Language Model Reasoning](https://arxiv.org/abs/2505.16368) *Huanyu Liu, Jia Li, Hao Zhu, Kechi Zhang, Yihong Dong, Ge Li.* Arxiv 2025.
- [ChiseLLM: Unleashing the Power of Reasoning LLMs for Chisel Agile Hardware Development](https://arxiv.org/abs/2504.19144) *Bowei Wang, Jiaran Gao, Yelai Feng, Renzhi Chen, Shanshan Li, Lei Wang.* Arxiv 2025.
- [ClarifyCoder: Clarification-Aware Fine-Tuning for Programmatic Problem Solving](https://arxiv.org/abs/2504.16331) *Jie JW Wu, Manav Chaudhary, Davit Abrahamyan, Arhaan Khaku, Anjiang Wei, Fatemeh H. Fard.* Arxiv 2025.
- [UnitCoder: Scalable Iterative Code Synthesis with Unit Test Guidance](https://arxiv.org/abs/2502.11460) *Yichuan Ma, Yunfan Shao, Peiji Li, Demin Song, Qipeng Guo, Linyang Li, Xipeng Qiu, Kai Chen.* Arxiv 2025.
- [dafny-annotator: AI-Assisted Verification of Dafny Programs](https://arxiv.org/abs/2411.15143) *Gabriel Poesia, Chloe Loughridge, Nada Amin.* Arxiv 2024.
- [Mastering the Craft of Data Synthesis for CodeLLMs](https://arxiv.org/abs/2411.00005) *Meng Chen, Philip Arthur, Qianyu Feng, Cong Duy Vu Hoang, Yu-Heng Hong, Mahdi Kazemi Moghaddam, Omid Nezami, Thien Nguyen, Gioacchino Tangari, Duy Vu, Thanh Vu, Mark Johnson, Krishnaram Kenthapadi, Don Dharmasiri, Long Duong, Yuan-Fang Li.* NAACL 2025.
- [EDGE: Enhanced Grounded GUI Understanding with Enriched Multi-Granularity Synthetic Data](https://arxiv.org/abs/2410.19461) *Xuetian Chen, Hangcheng Li, Jiaqing Liang, Sihang Jiang, Deqing Yang.* Arxiv 2024. 


### **6.3 Agent and Tool Use**

- [ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases](https://arxiv.org/abs/2306.05301). *Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, Le Sun.* Arxiv 2023.
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761). *Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom.* NeurIPS 2023.
- [GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction](https://arxiv.org/abs/2305.18752). *Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, Ying Shan.* Neurips 2023.
- [Gorilla: Large Language Model Connected with Massive APIs.](https://arxiv.org/abs/2305.15334) *Shishir G. Patil, Tianjun Zhang, Xin Wang, Joseph E. Gonzalez.* NeurIPS 2024.
- [Quality Matters: Evaluating Synthetic Data for Tool-Using LLMs](https://aclanthology.org/2024.emnlp-main.285.pdf). *Shadi Iskander, Nachshon Cohen, Zohar Karnin, Ori Shapira, Sofia Tolmach.* EMNLP 2024.
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291). *Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anandkumar.* TMLR 2024.
- [A Strategic Coordination Framework of Small LLMs Matches Large LLMs in Data Synthesis](https://arxiv.org/abs/2504.12322) *Xin Gao, Qizhi Pei, Zinan Tang, Yu Li, Honglin Lin, Jiang Wu, Lijun Wu, Conghui He.* ACL 2025.
- [Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning](https://arxiv.org/abs/2505.16410) *Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu Mao, Guorui Zhou, Zhicheng Dou, Ji-Rong Wen.* Arxiv 2025.
- [ToolACE-DEV: Self-Improving Tool Learning via Decomposition and EVolution](https://arxiv.org/abs/2505.07512) *Xu Huang, Weiwen Liu, Xingshan Zeng, Yuefeng Huang, Xinlong Hao, Yuxian Wang, Yirong Zeng, Chuhan Wu, Yasheng Wang, Ruiming Tang, Defu Lian.* Arxiv 2025.
- [Advancing and Benchmarking Personalized Tool Invocation for LLMs](https://arxiv.org/abs/2505.04072) *Xu Huang, Yuefeng Huang, Weiwen Liu, Xingshan Zeng, Yasheng Wang, Ruiming Tang, Hong Xie, Defu Lian.* Arxiv 2025.
- [GraphMaster: Automated Graph Synthesis via LLM Agents in Data-Limited Environments](https://arxiv.org/abs/2504.00711) *Enjun Du, Xunkai Li, Tian Jin, Zhihan Zhang, Rong-Hua Li, Guoren Wang.* Arxiv 2025.
- [LLMSR@XLLM25: Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation](https://arxiv.org/abs/2504.16408) *Jiahao Yuan, Xingzhe Sun, Xing Yu, Jingwen Wang, Dehui Du, Zhiqing Cui, Zixiang Di.* XLLM 2025.
- [Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search](https://arxiv.org/abs/2502.00955) *Wentao Shi, Zichun Yu, Fuli Feng, Xiangnan He, Chenyan Xiong.* Arxiv 2025.
- [Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents](https://arxiv.org/abs/2411.06559) *Yu Gu, Kai Zhang, Yuting Ning, Boyuan Zheng, Boyu Gou, Tianci Xue, Cheng Chang, Sanjari Srivastava, Yanan Xie, Peng Qi, Huan Sun, Yu Su.* Submitted to TMLR 2025.
- [ToolFlow: Boosting LLM Tool-Calling Through Natural and Coherent Dialogue Synthesis](https://arxiv.org/abs/2410.18447) *Zezhong Wang, Xingshan Zeng, Weiwen Liu, Liangyou Li, Yasheng Wang, Lifeng Shang, Xin Jiang, Qun Liu, Kam-Fai Wong.* NAACL 2025.
- [Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation](https://arxiv.org/abs/2410.14251) *Shuo Tang, Xianghe Pang, Zexi Liu, Bohan Tang, Rui Ye, Tian Jin, Xiaowen Dong, Yanfeng Wang, Siheng Chen.* Arxiv 2024.
- [Self-Training Large Language Models for Tool-Use Without Demonstrations](https://arxiv.org/abs/2502.05867) *Ne Luo, Aryo Pradipta Gema, Xuanli He, Emile van Krieken, Pietro Lesci, Pasquale Minervini.* Arxiv 2025.


### **6.4 Vision and Language**

- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) *Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee.* NeurIPS 2023.
- [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966) *Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, Jingren Zhou.* Arxiv 2023.
- [Enhancing Cognition and Explainability of Multimodal Foundation Models with Self-Synthesized Data](https://openreview.net/forum?id=lHbLpwbEyt) *Yucheng Shi, Quanzheng Li, Jin Sun, Xiang Li, Ninghao Liu.* ICLR 2025.
- [G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model](https://arxiv.org/abs/2312.11370) *Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, Lingpeng Kong.* ICLR 2025.
- [MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models](https://arxiv.org/abs/2304.10592) *Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny.* ICLR 2024.
- [Enhancing Large Vision Language Models with Self-Training on Image Comprehension](https://arxiv.org/abs/2405.19716) *Yihe Deng, Pan Lu, Fan Yin, Ziniu Hu, Sheng Shen, James Zou, Kai-Wei Chang, Wei Wang.* NeurIPS 2024.
- [LLaVA-OneVision: Easy Visual Task Transfer](https://arxiv.org/abs/2408.03326) *Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, Chunyuan Li.* Submit to TMLR.
- [FUSION: Fully Integration of Vision-Language Representations for Deep Cross-Modal Understanding](https://arxiv.org/abs/2504.09925) *Zheng Liu, Mengjie Liu, Jingzhou Chen, Jingwei Xu, Bin Cui, Conghui He, Wentao Zhang.* Arxiv 2025.
- [RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-wave Point Cloud Sequence](https://arxiv.org/abs/2504.09862) *Zengyuan Lai, Jiarui Yang, Songpengcheng Xia, Lizhou Lin, Lan Sun, Renwen Wang, Jianran Liu, Qi Wu, Ling Pei.* Arxiv 2025.
- [Unicorn: Text-Only Data Synthesis for Vision Language Model Training](https://arxiv.org/abs/2503.22655) *Xiaomin Yu, Pengxiang Ding, Wenjie Zhang, Siteng Huang, Songyang Gao, Chengwei Qin, Kejian Wu, Zhaoxin Fan, Ziyue Qiao, Donglin Wang.* Arxiv 2025.
- [Unseen from Seen: Rewriting Observation-Instruction Using Foundation Models for Augmenting Vision-Language Navigation](https://arxiv.org/abs/2503.18065) *Ziming Wei, Bingqian Lin, Yunshuang Nie, Jiaqi Chen, Shikui Ma, Hang Xu, Xiaodan Liang.* Arxiv 2025.
- [Semantic to Structure: Learning Structural Representations for Infringement Detection](https://arxiv.org/abs/2502.07323) *Chuanwei Huang, Zexi Jia, Hongyan Fei, Yeshuang Zhu, Zhiqiang Yuan, Jinchao Zhang, Jie Zhou.* Arxiv 2025.
- [Theorem-Validated Reverse Chain-of-Thought Problem Generation for Geometric Reasoning](https://arxiv.org/abs/2410.17885) *Linger Deng, Linghao Zhu, Yuliang Liu, Yu Wang, Qunyi Xie, Jingjing Wu, Gang Zhang, Yingying Zhu, Xiang Bai.* Arxiv 2024.
- [Distill Visual Chart Reasoning Ability from LLMs to MLLMs](https://arxiv.org/abs/2410.18798) *Wei He, Zhiheng Xi, Wanxu Zhao, Xiaoran Fan, Yiwen Ding, Zifei Shan, Tao Gui, Qi Zhang, Xuanjing Huang.* Arxiv 2024.


### **6.5 Retrieval-Augmented Generation**
- [GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis](https://arxiv.org/abs/2505.18710) *Yi Jiang, Sendong Zhao, Jianbo Li, Haochun Wang, Bing Qin.* ACL 2025.
- [DailyQA: A Benchmark to Evaluate Web Retrieval Augmented LLMs Based on Capturing Real-World Changes](https://arxiv.org/abs/2505.17162) *Jiehan Cheng, Zhicheng Dou.* Arxiv 2025.
- [HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation](https://arxiv.org/abs/2504.12330) *Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan Meng, Ding Wang, Jun Ma.* Arxiv 2025.
- [AI-University: An LLM-based platform for instructional alignment to scientific classrooms](https://arxiv.org/abs/2504.08846) *Mostafa Faghih Shojaei, Rahul Gulati, Benjamin A. Jasperson, Shangshang Wang, Simone Cimolato, Dangli Cao, Willie Neiswanger, Krishna Garikipati.* Arxiv 2025.
- [A Collaborative Multi-Agent Approach to Retrieval-Augmented Generation Across Diverse Data](https://arxiv.org/abs/2412.05838) *Aniruddha Salve, Saba Attar, Mahesh Deshmukh, Sayali Shivpuje, Arnab Mitra Utsab.* Arxiv 2024.
- [LLM-Ref: Enhancing Reference Handling in Technical Writing with Large Language Models](https://arxiv.org/abs/2411.00294) *Kazi Ahmed Asif Fuad, Lizhong Chen.* Arxiv 2024.


### **6.6 Long Context**
- [Make Your LLM Fully Utilize the Context.](https://arxiv.org/abs/2404.16811) *Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou.* Arxiv 2024.
- [From Artificial Needles to Real Haystacks: Improving Retrieval Capabilities in LLMs by Finetuning on Synthetic Data](https://arxiv.org/abs/2406.19292). *Zheyang Xiong, Vasilis Papageorgiou, Kangwook Lee, Dimitris Papailiopoulos*. ICLR 2025.
- [Scaling Instruction-tuned LLMs to Million-token Contexts via Hierarchical Synthetic Data Generation](https://openreview.net/forum?id=BkwCrIsTbR) *Linda He, Jue WANG, Maurice Weber, Shang Zhu, Ben Athiwaratkun, Ce Zhang.* ICLR 2025.🔥
- [LongMagpie: A Self-synthesis Method for Generating Large-scale Long-context Instructions](https://arxiv.org/abs/2505.17134) *Chaochen Gao, Xing Wu, Zijia Lin, Debing Zhang, Songlin Hu.* 
- [WildLong: Synthesizing Realistic Long-Context Instruction Data at Scale](https://arxiv.org/abs/2502.16684) *Jiaxi Li, Xingxing Zhang, Xun Wang, Xiaolong Huang, Li Dong, Liang Wang, Si-Qing Chen, Wei Lu, Furu Wei.* Arxiv 2025.
- [NExtLong: Toward Effective Long-Context Training without Long Documents](https://arxiv.org/abs/2501.12766) *Chaochen Gao, Xing Wu, Zijia Lin, Debing Zhang, Songlin Hu.* ICML 2025.
- [LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs](https://arxiv.org/abs/2408.07055) *Yushi Bai, Jiajie Zhang, Xin Lv, Linzhi Zheng, Siqi Zhu, Lei Hou, Yuxiao Dong, Jie Tang, Juanzi Li.* ICLR 2025.
- [LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning](https://arxiv.org/abs/2506.18841) *Yuhao Wu, Yushi Bai, Zhiqiang Hu, Roy Ka-Wei Lee, Juanzi Li.* Arxiv 2025.🔥
- [Chunks as Arms: Multi-Armed Bandit-Guided Sampling for Long-Context LLM Preference Optimization](https://arxiv.org/abs/2508.13993) *Shaohua Duan, Pengcheng Huang, Xinze Li, Zhenghao Liu, Xiaoyuan Yi, Yukun Yan, Shuo Wang, Yu Gu, Ge Yu, Maosong Sun.* ACL 2026.


### **6.7 Writing**
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) *Ronen Eldan, Yuanzhi Li.* Arxiv 2024.
- [Weaver: Foundation Models for Creative Writing](https://arxiv.org/abs/2401.17268) *Tiannan Wang, Jiamin Chen, Qingrui Jia, Shuai Wang, Ruoyu Fang, Huilin Wang, Zhaowei Gao, Chunzhao Xie, Chuou Xu, Jihong Dai, Yibin Liu, Jialong Wu, Shengwei Ding, Long Li, Zhiwei Huang, Xinle Deng, Teng Yu, Gangan Ma, Han Xiao, Zixin Chen, Danjun Xiang, Yunxia Wang, Yuanyuan Zhu, Yi Xiao, Jing Wang, Yiru Wang, Siran Ding, Jiayang Huang, Jiayi Xu, Yilihamu Tayier, Zhenyu Hu, Yuan Gao, Chengfeng Zheng, Yueshu Ye, Yihang Li, Lei Wan, Xinyue Jiang, Yujie Wang, Siyu Cheng, Zhule Song, Xiangru Tang, Xiaohua Xu, Ningyu Zhang, Huajun Chen, Yuchen Eleanor Jiang, Wangchunshu Zhou.* Arxiv 2024.
- [LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs](https://arxiv.org/abs/2408.07055) *Yushi Bai, Jiajie Zhang, Xin Lv, Linzhi Zheng, Siqi Zhu, Lei Hou, Yuxiao Dong, Jie Tang, Juanzi Li.* ICLR 2025.
- [LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning](https://arxiv.org/abs/2506.18841) *Yuhao Wu, Yushi Bai, Zhiqiang Hu, Roy Ka-Wei Lee, Juanzi Li.* Arxiv 2025.🔥

### **6.8 AI For Science**
- [Leveraging Large Language Models for enzymatic reaction prediction and characterization](https://arxiv.org/abs/2505.05616) *Lorenzo Di Fruscia, Jana Marie Weber* Arxiv 2025.
- [Large language models to accelerate organic chemistry synthesis](https://www.nature.com/articles/s42256-025-01066-y)  *Yu Zhang, Yang Han, Shuai Chen, Ruijie Yu, Xin Zhao, Xianbin Liu, Kaipeng Zeng, Mengdi Yu, Jidong Tian, Feng Zhu, Xiaokang Yang, Yaohui Jin, Yanyan Xu.* Nature Machine Intelligence 2025.
- [SynLlama: Generating Synthesizable Molecules and Their Analogs with Large Language Models](https://arxiv.org/abs/2503.12602) *Kunyang Sun, Dorian Bagni, Joseph M. Cavanagh, Yingze Wang, Jacob M. Sawyer, Andrew Gritsevskiy, Oufan Zhang, Teresa Head-Gordon.* Arxiv 2025.
- [Causal Discovery from Data Assisted by Large Language Models](https://arxiv.org/abs/2503.13833) *Kamyar Barakati, Alexander Molak, Chris Nelson, Xiaohang Zhang, Ichiro Takeuchi, Sergei V. Kalinin.* Arxiv 2025.
- [Leveraging Large Language Models to Address Data Scarcity in Machine Learning: Applications in Graphene Synthesis](https://arxiv.org/abs/2503.04870) *Devi Dutta Biswajeet, Sara Kadkhodaei.* Arxiv 2025.
- [Agentic Mixture-of-Workflows for Multi-Modal Chemical Search](https://arxiv.org/abs/2502.19629) *Tiffany J. Callahan, Nathaniel H. Park, Sara Capponi.* Arxiv 2025.

### **6.9 Text-to-SQL**

- [Synthesizing Text-to-SQL Data from Weak and Strong LLMs](https://arxiv.org/abs/2408.03256) *Jiaxi Yang, Binyuan Hui, Min Yang, Jian Yang, Junyang Lin, Chang Zhou.* ACL 2024.
- [Synthetic-Text-To-SQL: A synthetic dataset for training language models to generate SQL queries from natural language prompts](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) *Meyer, Yev and Emadi, Marjan and Nathawani, Dhruv and Ramaswamy, Lipika and Boyd, Kendrick and Van Segbroeck, Maarten and Grossman, Matthew and Mlocek, Piotr and Newberry, Drew.* Huggingface 2024.
- [OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale](https://arxiv.org/abs/2503.02240) *Haoyang Li, Shang Wu, Xiaokang Zhang, Xinmei Huang, Jing Zhang, Fuxin Jiang, Shuai Wang, Tieying Zhang, Jianjun Chen, Rui Shi, Hong Chen, Cuiping Li.* Arxiv 2025.
- [CoddLLM: Empowering Large Language Models for Data Analytics](https://arxiv.org/abs/2502.00329) *Jiani Zhang, Hengrui Zhang, Rishav Chakravarti, Yiqun Hu, Patrick Ng, Asterios Katsifodimos, Huzefa Rangwala, George Karypis, Alon Halevy.* Arxiv 2025.


### **6.10 Synergy between Large and Small Models**
- [FreeAL: Towards Human-Free Active Learning in the Era of Large Language Models](https://arxiv.org/abs/2311.15614), *Ruixuan Xiao, Yiwen Dong, Junbo Zhao, Runze Wu, Minmin Lin, Gang Chen, Haobo Wang.* EMNLP 2023.
- [Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations](https://aclanthology.org/2023.emnlp-main.647.pdf), *Zhuoyan Li, Hangxiao Zhu, Zhuoran Lu, Ming Yin.* EMNLP finding 2023.
- [Data-Constrained Synthesis of Training Data for De-Identification](https://arxiv.org/abs/2502.14677) *Thomas Vakili, Aron Henriksson, Hercules Dalianis.* ACL 2025.
- [Symbiotic Cooperation for Web Agents: Harnessing Complementary Strengths of Large and Small LLMs](https://arxiv.org/abs/2502.07942) *Ruichen Zhang, Mufan Qiu, Zhen Tan, Mohan Zhang, Vincent Lu, Jie Peng, Kaidi Xu, Leandro Z. Agudelo, Peter Qian, Tianlong Chen.* Arxiv 2025.

### **6.11 Weak-to-Strong**

- [Weak-to-strong generalization: Eliciting strong capabilities with weak supervision](https://proceedings.mlr.press/v235/burns24b.html). *Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, Jeffrey Wu.* ICML 2024.
- [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models.](https://arxiv.org/abs/2401.01335) *Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu.* ICML 2024.
- [Impossible Distillation for Paraphrasing and Summarization: How to Make High-quality Lemonade out of Small, Low-quality Models](https://arxiv.org/abs/2305.16635) *Jaehun Jung, Peter West, Liwei Jiang, Faeze Brahman, Ximing Lu, Jillian Fisher, Taylor Sorensen, Yejin Choi.* NAACL 2024.

### **6.12 Distill Small Model**
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) *Ronen Eldan, Yuanzhi Li.* Arxiv 2024.
- [Let's Synthesize Step by Step: Iterative Dataset Synthesis with Large Language Models by Extrapolating Errors from Small Models](https://arxiv.org/abs/2310.13671) *Ruida Wang, Wangchunshu Zhou, Mrinmaya Sachan,* EMNLP finding2023
- [CorrSynth -- A Correlated Sampling Method for Diverse Dataset Generation from LLMs](https://arxiv.org/abs/2411.08553) *Suhas S Kowshik, Abhishek Divekar, Vijit Malik.* EMNLP 2024.

### **6.13 Multilingual Data**
- [CulFiT: A Fine-grained Cultural-aware LLM Training Paradigm via Multilingual Critique Data Synthesis](https://arxiv.org/abs/2505.19484) *Ruixiang Feng, Shen Gao, Xiuying Chen, Lisi Chen, Shuo Shang.* ACL 2025.
- [Command R7B Arabic: A Small, Enterprise Focused, Multilingual, and Culturally Aware Arabic LLM](https://arxiv.org/abs/2503.14603) *Yazeed Alnumay, Alexandre Barbet, Anna Bialas, William Darling, Shaan Desai, Joan Devassy, Kyle Duffy, Stephanie Howe, Olivia Lasche, Justin Lee, Anirudh Shrinivason, Jennifer Tracey.* AfricaNLP 2025.
- [Evaluating Large Language Model Capability in Vietnamese Fact-Checking Data Generation](https://arxiv.org/abs/2411.05641) *Long Truong To, Hung Tuan Le, Dat Van-Thanh Nguyen, Manh Trong Nguyen, Tri Thien Nguyen, Tin Van Huynh, Kiet Van Nguyen.* Arxiv 2024.

### **6.14 Structured Data**
- [TableDreamer: Progressive and Weakness-guided Data Synthesis from Scratch for Table Instruction Tuning](https://arxiv.org/abs/2506.08646) *Mingyu Zheng, Zhifan Feng, Jia Wang, Lanrui Wang, Zheng Lin, Yang Hao, Weiping Wang.* ACL Findings 2025.
- [A Note on Statistically Accurate Tabular Data Generation Using Large Language Models](https://www.arxiv.org/abs/2505.02659) *Andrey Sidorenko.* Arxiv 2025.
- [SLOT: Structuring the Output of Large Language Models](https://arxiv.org/abs/2505.04016) *Darren Yow-Bang Wang, Zhengyuan Shen, Soumya Smruti Mishra, Zhichao Xu, Yifei Teng, Haibo Ding.* Arxiv 2025.
- [Tabby: Tabular Adaptation for Language Models](https://arxiv.org/abs/2503.02152) *Sonia Cromp, Satya Sai Srinath Namburi GNVV, Mohammed Alkhudhayri, Catherine Cao, Samuel Guo, Nicholas Roberts, Frederic Sala.* Arxiv 2025.
- [GReaTER: Generate Realistic Tabular data after data Enhancement and Reduction](https://arxiv.org/abs/2503.15564) *Tung Sum Thomas Kwok, Chi-Hua Wang, Guang Cheng.* ICDE workshop 2025.
- [Generative adversarial networks vs large language models: a comparative study on synthetic tabular data generation](https://arxiv.org/abs/2502.14523) *Austin A. Barr, Robert Rozman, Eddie Guo.* Arxiv 2025.
- [SampleLLM: Optimizing Tabular Data Synthesis in Recommendations](https://arxiv.org/abs/2501.16125) *Jingtong Gao, Zhaocheng Du, Xiaopeng Li, Yichao Wang, Xiangyang Li, Huifeng Guo, Ruiming Tang, Xiangyu Zhao.* WWW 2025.
- [Large Language Models for Data Synthesis](https://arxiv.org/abs/2505.14752) *Yihong Tang, Menglin Kong, Lijun Sun.* Arxiv 2025.

### **6.15 Natural Language Understanding**
- [Improving Natural Language Understanding for LLMs via Large-Scale Instruction Synthesis](https://arxiv.org/abs/2502.03843) *Lin Yuan, Jun Xu, Honghao Gui, Mengshu Sun, Zhiqiang Zhang, Lei Liang, Jun Zhou.* AAAI 2025.
- [Emo Pillars: Knowledge Distillation to Support Fine-Grained Context-Aware and Context-Less Emotion Classification](https://arxiv.org/abs/2504.16856) *Alexander Shvets.* ACL Findings 2025.
- [DS2-ABSA: Dual-Stream Data Synthesis with Label Refinement for Few-Shot Aspect-Based Sentiment Analysis](https://arxiv.org/abs/2412.14849) *Hongling Xu, Yice Zhang, Qianlong Wang, Ruifeng Xu.* ACL 2025.

### **6.16 Logic Reasoning**
- [Enhancing Reasoning Capabilities of LLMs via Principled Synthetic Logic Corpus](https://arxiv.org/abs/2411.12498) *Terufumi Morishita, Gaku Morio, Atsuki Yamaguchi, Yasuhiro Sogawa.* NeurIPS 2024
- [LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning](https://arxiv.org/abs/2409.12929) *Jin Jiang, Yuchen Yan, Yang Liu, Jianing Wang, Shuai Peng, Xunliang Cai, Yixin Cao, Mengdi Zhang, Liangcai Gao.* ACL 2025


### **6.17 Dialogue System**
- [DocTalk: Scalable Graph-based Dialogue Synthesis for Enhancing LLM Conversational Capabilities](https://www.arxiv.org/abs/2507.05750) *Jing Yang Lee, Hamed Bonab, Nasser Zalmout, Ming Zeng, Sanket Lokegaonkar, Colin Lockard, Binxuan Huang, Ritesh Sarkhel, Haodong Wang.* SIGDIAL 2025.
- [Mutual Reinforcement of LLM Dialogue Synthesis and Summarization Capabilities for Few-Shot Dialogue Summarization](https://arxiv.org/abs/2502.17328) *Yen-Ju Lu, Ting-Yao Hu, Hema Swetha Koppula, Hadi Pouransari, Jen-Hao Rick Chang, Yin Xia, Xiang Kong, Qi Zhu, Simon Wang, Oncel Tuzel, Raviteja Vemulapalli.* NAACL Findings 2025.
- [Bottom-Up Synthesis of Knowledge-Grounded Task-Oriented Dialogues with Iteratively Self-Refined Prompts](https://arxiv.org/abs/2504.14375) *Kun Qian, Maximillian Chen, Siyan Li, Arpit Sharma, Zhou Yu.* NAACL 2025.

### **6.18 Federated Learning**

- [Prompt Public Large Language Models to Synthesize Data for Private On-device Applications](https://arxiv.org/pdf/2404.04360). *Shanshan Wu, Zheng Xu, Yanxiang Zhang, Yuanbo Zhang, Daniel Ramage.* COLM 2024.
- [Harnessing large-language models to generate private synthetic text](https://arxiv.org/abs/2306.01684). *Alexey Kurakin, Natalia Ponomareva, Umar Syed, Liam MacDermed, Andreas Terzis.* Arxiv 2024.

### **6.19 Generative Design**

- [Generative Design through Quality-Diversity Data Synthesis and Language Models.](https://arxiv.org/abs/2405.09997) *Adam Gaier, James Stoddart, Lorenzo Villaggi, Shyam Sudhakaran.* GECCO 2024.
- [VeriReason: Reinforcement Learning with Testbench Feedback for Reasoning-Enhanced Verilog Generation](https://arxiv.org/abs/2505.11849) *Yiting Wang, Guoheng Sun, Wanghao Ye, Gang Qu, Ang Li.* Arxiv 2025.
- [CodeV-R1: Reasoning-Enhanced Verilog Generation](https://arxiv.org/abs/2505.24183) *Yaoyu Zhu, Di Huang, Hanqi Lyu, Xiaoyun Zhang, Chongxiao Li, Wenxuan Shi, Yutong Wu, Jianan Mu, Jinghua Wang, Yang Zhao, Pengwei Jin, Shuyao Cheng, Shengwen Liang, Xishan Zhang, Rui Zhang, Zidong Du, Qi Guo, Xing Hu, Yunji Chen.* Arxiv 2025.
- [LIFT: LLM-Based Pragma Insertion for HLS via GNN Supervised Fine-Tuning](https://arxiv.org/abs/2504.21187) *Neha Prakriya, Zijian Ding, Yizhou Sun, Jason Cong.* Arxiv 2025.
- [CAD-Editor: A Locate-then-Infill Framework with Automated Training Data Synthesis for Text-Based CAD Editing](https://arxiv.org/abs/2502.03997) *Yu Yuan, Shizhao Sun, Qi Liu, Jiang Bian.* ICML 2025.
- [AnalogXpert: Automating Analog Topology Synthesis by Incorporating Circuit Design Expertise into Large Language Models](https://arxiv.org/abs/2412.19824) *Haoyi Zhang, Shizhao Sun, Yibo Lin, Runsheng Wang, Jiang Bian.* Arxiv 2024. 
 
### 5.15. Knowledge-Intensive Data
- [GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation](https://arxiv.org/abs/2505.20416) *Zihong Chen, Wanli Jiang, Jinzhe Li, Zhonghang Yuan, Huanjun Kong, Wanli Ouyang, Nanqing Dong* Arxiv 2025.

## **7. Tools**

- [DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows.](https://arxiv.org/abs/2402.10379) *Ajay Patel, Colin Raffel, Chris Callison-Burch.* ACL 2024.
- [AgentInstruct: Toward Generative Teaching with Agentic Flows.](https://arxiv.org/abs/2407.03502) *Arindam Mitra, Luciano Del Corro, Guoqing Zheng, Shweti Mahajan, Dany Rouhana, Andres Codas, Yadong Lu, Wei-ge Chen, Olga Vrousgos, Corby Rosset, Fillipe Silva, Hamed Khanpour, Yash Lara, Ahmed Awadallah.* Arxiv 2024.
- [Distilabel: An AI Feedback (AIF) Framework for Building Datasets with and for LLMs](https://github.com/argilla-io/distilabel). *Álvaro Bartolomé Del Canto, Gabriel Martín Blázquez, Agustín Piqueres Lajarín and Daniel Vila Suero.* GitHub 2024.
- [Fuxion: Synthetic Data Generation and Normalization Functions using Langchain + LLMs](https://github.com/tobiadefami/fuxion).
- [Easy Dataset: A Unified and Extensible Framework for Synthesizing LLM Fine-Tuning Data from Unstructured Documents](https://arxiv.org/abs/2507.04009) *Ziyang Miao, Qiyu Sun, Jingyuan Wang, Yuchen Gong, Yaowei Zheng, Shiqi Li, Richong Zhang.* Arxiv 2025.
- [EasyDistill: A Comprehensive Toolkit for Effective Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2505.20888) *Chengyu Wang, Junbing Yan, Wenrui Cai, Yuanhao Yue, Jun Huang.* Arxiv 2025.

## **8. Datasets**

- [Open Artificial Knowledge](https://huggingface.co/datasets/tabularisai/oak) *Vadim Borisov, Richard Schreiber.* ICML Workshop 2024.
- [PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts](https://arxiv.org/abs/2202.01279) *Stephen H. Bach, Victor Sanh, Zheng-Xin Yong, Albert Webson, Colin Raffel, Nihal V. Nayak, Abheesht Sharma, Taewoon Kim, M Saiful Bari, Thibault Fevry, Zaid Alyafeai, Manan Dey, Andrea Santilli, Zhiqing Sun, Srulik Ben-David, Canwen Xu, Gunjan Chhablani, Han Wang, Jason Alan Fries, Maged S. Al-shaibani, Shanya Sharma, Urmish Thakker, Khalid Almubarak, Xiangru Tang, Dragomir Radev, Mike Tian-Jian Jiang, Alexander M. Rush*, ACL 2022 Demo.
- [Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks](https://arxiv.org/abs/2204.07705) *Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Gary Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Maitreya Patel, Kuntal Kumar Pal, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Shailaja Keyur Sampat, Savan Doshi, Siddhartha Mishra, Sujan Reddy, Sumanta Patro, Tanay Dixit, Xudong Shen, Chitta Baral, Yejin Choi, Noah A. Smith, Hannaneh Hajishirzi, Daniel Khashabi*. EMNLP 2022.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=pengr/LLM-Synthetic-Data&type=Date)](https://star-history.com/#pengr/LLM-Synthetic-Data&Date)
