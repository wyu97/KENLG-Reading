# Knowledge-enriched Text Generation Reading-List

Here is a list of recent publications about **Knowledge-enhanced text generation**.
(Update on **July 4th, 2021**) <br>

-- We will continue to add and update related papers and codes on this page.

-- <img src="images/code.png" width="20" align=center> indicates available code and <img src="images/hot.png" width="20" align=center> indicates high citation in recent years.

## <img src="images/new.png" width="25" align=center> Survey paper

[**A Survey of Knowledge-enhanced Text Generation**](https://arxiv.org/abs/2010.04389). Wenhao Yu (ND), Chenguang Zhu (Microsoft), Zaitang Li (CUHK), Zhiting Hu (UCSD), Qingyun Wang (UIUC), Heng Ji (UIUC), Meng Jiang (ND). ACM Computing Survey, 2022 (in press).

> To the best of our knowledge, our survey is the first work that presents a comprehensive reviewof knowledge-enhanced text generation. It aims to provide NLG researchers a synthesis and pointer to related researches. Our survey also includes a detailed discussion about how NLG can benefit from recent progress in deep learning and artificial intelligence, including technologies such as graph neural network, reinforcement learning, neural topic modeling and so on.

### Please consider cite our survey if you found it is useful.

```
@article{yu2020survey,
  title={A Survey of Knowledge-Enhanced Text Generation},
  author={Yu, Wenhao and Zhu, Chenguang and Li, Zaitang and Hu, Zhiting and Wang, Qingyun and Ji, Heng and Jiang, Meng},
  journal={ACM Computing Survey (CSUR)},
  year={2020}
}
```

## <img src="images/new.png" width="25" align=center> Tutorial

We have given a tutorial on the topic of [**Knowledge-enriched Text Generation**](https://kenlg-tutorial.github.io/) in [**EMNLP 2021**](https://2021.emnlp.org/tutorials). It was held from Nov. 7th to Nov. 11th, 2021, in Dominican Republic. Check out the materials [here](https://kenlg-tutorial.github.io/).

We will also present a tutorial on a broader topic of [**Knowledge-augmented Methods for NLP**](https://kenlg-tutorial.github.io/) in [**ACL 2022**](https://www.2022.aclweb.org/). It will be held from May. 22nd to Nov. 27th, 2022, in Ireland! Stay tuned for more information!


## Basic NLG papers and codes
(For new learners, some important papers for general NLG/KENLG.)

- <img src="images/hot.png" width="20" align=center> **[Seq2Seq] Sequence to Sequence Learning with Neural Networks**
	- Ilya Sutskever (Google) et al, In NeurIPS 2014. [\[pdf\]](https://arxiv.org/pdf/1409.3215.pdf)

- <img src="images/hot.png" width="20" align=center> **[SeqAttn] Neural Machine Translation by Jointly Learning to Align and Translate** 
	- Dzmitry Bahdanau (Jacobs University) et al, In ICLR 2015. [\[pdf\]](https://arxiv.org/pdf/1409.0473.pdf)
- <img src="images/hot.png" width="20" align=center> **[CopyNet] Incorporating Copying Mechanism in Sequence-to-Sequence Learning** 
	- Jiatao Gu (The University of Hong Kong) et al, In ACL 2016. [\[pdf\]](https://arxiv.org/abs/1603.06393)
- <img src="images/hot.png" width="20" align=center> **[PointerNet] Get To The Point: Summarization with Pointer-Generator Networks**
	- Abigail See (Stanford University) et al, In ACL 2017. [\[pdf\]](https://arxiv.org/abs/1704.04368)
- <img src="images/hot.png" width="20" align=center> **[Transformer] Attention Is All You Need**
	- Ashish Vaswani (Google) et al, In NeurIPS 2017. [\[pdf\]](https://arxiv.org/abs/1706.03762)

## Pretrained language generation models
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center>  **[GPT-2] Language Models are Unsupervised Multitask Learners** 
	- Alec Radford (OpenAI) et al, OpenAI 2019. [\[pdf\]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) [\[official code (tf)\]](https://github.com/openai/gpt-2/blob/master/src/model.py) [\[huggingface (torch)\]](https://github.com/huggingface/transformers/tree/master/examples/language-modeling)

- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[UniLM] Unified Language Model Pre-training for Natural Language Understanding and Generation** 
	- Li Dong (Microsoft) et al, In NeurIPS 2019. [\[pdf\]](https://arxiv.org/abs/1905.03197) [\[official code (torch)\]](https://github.com/microsoft/unilm)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[BART] BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**
	- Mike Lewis (Facebook AI) et al, On arXiv 2019. [\[pdf\]](https://arxiv.org/abs/1910.13461) [\[fairseq (torch)\]](https://github.com/pytorch/fairseq) [\[huggingface (torch)\]](https://github.com/huggingface/transformers/tree/master/examples/seq2seq)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[T5] Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**
	- Colin Raffel (Google) et al, In JMLR 2020. [\[pdf\]](https://arxiv.org/abs/1910.10683) [\[official code (tf)\]](https://github.com/google-research/text-to-text-transfer-transformer) [\[huggingface (torch)\]](https://github.com/huggingface/transformers/tree/master/examples/seq2seq)

## Controllable generation leanrng methods
- <img src="images/hot.png" width="20" align=center>  **[Posterior Regularization] Deep Generative Models with Learnable Knowledge Constraints** 
	- Zhiting Hu (Carnegie Mellon University) et al, In NeurIPS 2018. [\[pdf\]](http://papers.nips.cc/paper/8250-deep-generative-models-with-learnable-knowledge-constraints.pdf)

- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Plug and Play] Plug and Play Language Models: A Simple Approach to Controlled Text Generation**
	- Sumanth Dathathri (Uber AI) et al, In ICLR 2020. [\[pdf\]](https://arxiv.org/abs/1912.02164) [\[code (torch)\]](https://github.com/uber-research/PPLM)
unsup_gen_for_cms_reasoning)
- <img src="images/code.png" width="20" align=center> **[Backprop-based Decoding] Back to the Future: Unsupervised Backprop-based Decoding for Counterfactual and Abductive Commonsense Reasoning** 
	- Lianhui Qin (University of Washington) et al, In EMNLP 2020. [\[pdf\]](https://arxiv.org/abs/2010.05906) [\[code (torch)\]](https://github.com/qkaren/unsup_gen_for_cms_reasoning)
- <img src="images/code.png" width="20" align=center> **[Weakly Supervision] Summarizing Text on Any Aspects: A Knowledge-Informed Weakly-Supervised Approach** 
	- Bowen Tan (Carnegie Mellon University) et al, In EMNLP 2020. [\[pdf\]](https://arxiv.org/abs/2010.06792) [\[code (torch)\]](https://github.com/tanyuqian/aspect-based-summarization)

## Topic-enhanced text generation
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Dialogue System] Topic Aware Neural Response Generation**
	- Chen Xing (Nankai University) et al, In AAAI 2017. [\[pdf\]](https://arxiv.org/pdf/1606.08340.pdf)

- <img src="images/code.png" width="20" align=center> **[Dialogue System] A Neural TopicalExpansion Framework for Unstructured Persona-oriented Dialogue Generation**
	- Minghong Xu (Shandong University) et al, In ECAI 2020. [\[pdf\]](https://arxiv.org/abs/2002.02153) [\[code (tf)\]](https://github.com/Minghong-Xu/Neural_Topical_Expansion_for_UPDS)
- **[Dialogue System] Context-Controlled Topic-Aware Neural Response Generation for Open-Domain Dialog Systems**
	- Yanxiang Ling (National University of Defense Technology) et al, In Information Processing an Management 2021. [\[pdf\]](https://reader.elsevier.com/reader/sd/pii/S0306457320308876?token=E5C148830E07DE1F323E1D2921E317DDEA7B119A796A1D7419333D78043C9DBEC842C22823F84ACCF26EF2C3950158BA) 
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Summarization] Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization**
	- Shashi Narayan (University of Edinburgh) et al, In EMNLP 2018. [\[pdf\]](https://arxiv.org/abs/1808.08745) [\[code (torch)\]](https://github.com/EdinburghNLP/XSum)
- <img src="images/hot.png" width="20" align=center> **[Summarization] Topic-Guided Variational Autoencoders for Text Generation**
	- Wenlin Wang (Duke University) et al, In NAACL 2019. [\[pdf\]](https://arxiv.org/abs/1903.07137)
- **[Summarization] Document Summarization with VHTM: Variational Hierarchical Topic-Aware Mechanism**
	- Xiyan Fu (Nankai Univeristy) et al, In AAAI 2020. [\[pdf\]](https://aaai.org/ojs/index.php/AAAI/article/view/6277)
- **[Summarization] A Topic Augmented Text Generation Model: Joint Learning of Semantics and Structural Features**
	- Peng Cui (Harbin Institute of Technology) et al, In COLING 2020. [\[pdf\]](https://arxiv.org/abs/2010.06253)
- <img src="images/code.png" width="20" align=center> **[Summarization] [Friendly Topic Assistant for Transformer Based Abstractive Summarization]**
	- Zhengjue Wang (Xidian University) et al, In EMNLP 2020. [\[pdf\]]((https://www.aclweb.org/anthology/2020.emnlp-main.35.pdf)) [\[code\]](https://github.com/BoChenGroup/TA)
- **[Machine Translation] Topic-Informed Neural Machine Translation**
	- Jian Zhang, (Dublin City University) et al, In COLING 2016. [\[pdf\]](https://www.aclweb.org/anthology/C16-1170.pdf)
- **[Machine Translation] Translating with Bilingual Topic Knowledge for Neural Machine Translation**
	- Xiangpeng Wei (Chinese Academy of Sciences) et al, In AAAI 2019. [\[pdf\]](https://www.aaai.org/ojs/index.php/AAAI/article/view/4711)
- **[Topic Transfer] A Topic Augmented Text Generation Model: Joint Learning of Semantics and Structural Features**
	- Hongyin Tang (Chinese Academy of Sciences) et al, In EMNLP 2019. [\[pdf\]](https://www.aclweb.org/anthology/D19-1513.pdf)


## Keyword-enhanced text generation
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Dialogue System] Sequence to Backward and Forward Sequences: A Content-Introducing Approach to Generative Short-Text Conversation** 
	- Lili Mou (Peking University) et al, In COLING 2016. [\[pdf\]](https://arxiv.org/pdf/1607.00970.pdf) [\[code (tf)\]](https://github.com/MaZhiyuanBUAA/Seq2BFforDialogueGeneration)

- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Dialogue System] Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory**
	- Hao Zhou (Tsinghua University) et al, In AAAI 2018. [\[pdf\]](https://arxiv.org/abs/1704.01074) [\[code (tf)\]](https://github.com/loadder/ECM-tf)
- <img src="images/hot.png" width="20" align=center> **[Dialogue System] Generating Responses with a Specific Emotion in Dialog**
	- Zhenqiao Song (Fudan University) et al, In ACL 2019. [\[pdf\]](https://www.aclweb.org/anthology/P19-1359.pdf)
- **[Summarization] Guiding Generation for Abstractive Text Summarization based on Key Information Guide Network**
	- Chenliang Li (Beijing University of Posts and Telecommunications) et al, In NAACL 2018. [\[pdf\]](https://www.aclweb.org/anthology/N18-2009.pdf)
- **[Summarization] Inferring Search Queries from Web Documents via a Graph-Augmented Sequence to Attention Network**
	- Fred X. Han (University of Alberta) et al, In WWW 2019. [\[pdf\]](https://dl.acm.org/doi/pdf/10.1145/3308558.3313746?casa_token=bIgxBamZyDkAAAAA:Oqmf3xhi_tIqBHoBZQsAHDb-OAeUBuLuAiAP4civXgx5DcJa45cMf5SjWPDAJO3U0_zJPG4oOt1aqA)
- <img src="images/code.png" width="20" align=center> **[Summarization] Coherent Comment Generation for Chinese Articles with a Graph-to-Sequence Model**
	- Wei Li (Peking University) et al, In ACL 2019. [\[pdf\]](https://arxiv.org/pdf/1906.01231.pdf) [\[code (torch)\]](https://github.com/lancopku/Graph-to-seq-comment-generation)
- **[Summarization] Keywords-Guided Abstractive Sentence Summarization**
	- Haoran Li (JD AI Research) et al, In AAAI 2020. [\[pdf\]](https://pdfs.semanticscholar.org/4e8d/103703fda8875f13c7593d80bc7428f05ded.pdf?_ga=2.124808970.1605688764.1602448023-651806684.1580066755&_gac=1.89511017.1599265589.Cj0KCQjwy8f6BRC7ARIsAPIXOjiX6Icl-pa3DjJSds7dc1teH9asRBgylw-EJHrcZfi8qlL0U9nJky4aAiD4EALw_wcB)

## Knowledge base-enhanced text generation
- <img src="images/hot.png" width="20" align=center> **[Question Answering] [Generating Natural Answers by Incorporating Copying and Retrieving Mechanisms in Sequence-to-Sequence Learning]** 
	- Shizhu He (Chinese Academy of Sciences) et al, In ACL 2017. [\[pdf\]](https://www.aclweb.org/anthology/P17-1019.pdf)

- **[Question Answering] Natural answer generation with heterogeneous memory**
	- Yao Fu (Peking University) et al, In NAACL 2018. [\[pdf\]](https://www.aclweb.org/anthology/P17-1019.pdf)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Dialogue System] Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems**
	- Andrea Madotto (Hong Kong University of Science and Technology) et al, In ACL 2019. [\[pdf\]](https://arxiv.org/abs/1804.08217) [\[code (torch)\]](https://github.com/HLTCHKUST/Mem2Seq)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Dialogue System] Global-to-local Memory Pointer Networks for Task-Oriented Dialogue** 
	- Chien-Sheng Wu (Hong Kong University of Science and Technology) et al, In ICLR 2019. [\[pdf\]](https://arxiv.org/abs/1901.04713) [\[code (torch)\]](https://github.com/jasonwu0731/GLMP)
- <img src="images/code.png" width="20" align=center> **[Dialogue System] Improving Knowledge-aware Dialogue Generation via Knowledge Base Question Answering** 
	- Jian Wang (South China University of Technology) et al, In AAAI 2020. [\[pdf\]](https://arxiv.org/abs/1912.07491) [\[code (torch)\]](https://github.com/siat-nlp/TransDG)
- **[Dialogue System] Learning to Select Knowledge for Response Generation in Dialog Systems**
	- Rongzhong Lian (Baidu) et al, In IJCAI 2019. [\[pdf\]](https://arxiv.org/abs/1902.04911)
- <img src="images/code.png" width="20" align=center> **[Dialogue System] Diverse and Informative Dialogue Generation with Context-Specific Commonsense Knowledge Awareness**
	- Sixing Wu (Peking University) et al, In ACL 2020. [\[pdf\]](https://www.aclweb.org/anthology/2020.acl-main.515.pdf) [\[code (tf)\]](https://github.com/pku-orangecat/ACL2020-ConKADI)
- <img src="images/code.png" width="20" align=center> **[Dialogue System] TopicKA: Generating Commonsense Knowledge-Aware Dialogue Responses Towards the Recommended Topic Fact**
	- Sixing Wu (Peking University) et al, In IJCAI 2020. [\[pdf\]](https://www.ijcai.org/Proceedings/2020/0521.pdf) [\[code (tf)\]](https://github.com/pku-orangecat/IJCAI2020-TopicKA)
- <img src="images/code.png" width="20" align=center> **[Content Manipulation] Learning to Select Bi-Aspect Information for Document-Scale Text Content Manipulation** 
	- Xiaocheng Feng (Harbin Institute of Technology) et al, In AAAI 2020. [\[pdf\]](https://arxiv.org/abs/2002.10210) [\[code (torch)\]](https://github.com/syw1996/SCIR-TG-Data2text-Bi-Aspect)
- <img src="images/code.png" width="20" align=center> **[Content Manipulation] Fact-based Text Editing** 
	- Hayate Iso (Nara Institute of Science and Technology) et al, In ACL 2020. [\[pdf\]](https://www.aclweb.org/anthology/2020.acl-main.17.pdf) [\[code\]](https://github.com/isomap/factedit)
- **[Summarization] Exploring Human-Like Reading Strategy for Abstractive Text Summarization]**
	- Min Yang (Chinese Academy of Sciences) et al, In AAAI 2019. [\[pdf\]](https://www.aaai.org/ojs/index.php/AAAI/article/view/4724)
- <img src="images/code.png" width="20" align=center> **[Table-to-text] Describing a Knowledge Base** 
	- Qingyun Wang (UIUC) et al, in INLG 2018. [\[pdf\]](https://www.aclweb.org/anthology/W18-6502.pdf) [[code (torch)]](https://github.com/EagleW/Describing_a_Knowledge_Base) 

## Knowledge graph-enhanced text generation
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Dialogue System] Commonsense Knowledge Aware Conversation Generation with Graph Attention** 
	- Hao Zhou (Tsinghua University) et al, In IJCAI 2018. [\[pdf\]](https://www.ijcai.org/Proceedings/2018/0643.pdf) [\[code1 (tf)\]](https://github.com/thu-coai/ccm) [\[code2 (torch)\]](https://github.com/Lyusungwon/CCM-pytorch)

- **[Dialogue System] Knowledge Aware Conversation Generation with Explainable Reasoning over Augmented Graphs** 
	- Zhibin Liu,  (Baidu) et al, In EMNLP 2019. [\[pdf\]](https://arxiv.org/abs/1903.10245)
- <img src="images/code.png" width="20" align=center> **[Dialogue System] DyKgChat: Benchmarking Dialogue Generation Grounding on Dynamic Knowledge Graphs**
	- Yi-Lin Tuan (National Taiwan University) et al, In EMNLP 2019. [\[pdf\]](https://arxiv.org/abs/1910.00610) [\[code (tf)\]](https://github.com/Pascalson/DyKGChat)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Dialogue System] Grounded Conversation Generation as Guided Traverses in Commonsense Knowledge Graphs** 
	- Houyu Zhang (Brown University) et al, In ACL 2020. [\[pdf\]](https://arxiv.org/abs/1911.02707) [\[code (torch)\]](https://github.com/thunlp/ConceptFlow)
- <img src="images/code.png" width="20" align=center> **[Dialogue System] GraphDialog: Integrating Graph Knowledge into End-to-End Task-Oriented Dialogue Systems** 
	- Shiquan Yang (University of Melbourne) et al, In EMNLP 2020. [\[pdf\]](https://arxiv.org/pdf/2010.01447.pdf) [\[code (tf)\]](https://github.com/shiquanyang/GraphDialog)
- **[Dialogue System] CARE: Commonsense-Aware Emotional Response Generation with Latent Concepts** 
	- Peixiang Zhong (Nanyang Technological University) et al, In AAAI 2021. [\[pdf\]](https://arxiv.org/abs/2012.08377) 
- **[Dialogue System] Graph-Evolving Meta-Learning for Low-Resource Medical Dialogue Generation**
    - Shuai Lin (Sun Yat-sen University) et al, In AAAI 2021. [\[pdf\]](https://arxiv.org/abs/2012.11988) [\[code (torch)\]](https://github.com/ha-lins/GEML-MDG) 
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Question Answering] Commonsense for Generative Multi-Hop Question Answering Tasks** 
	- Lisa Bauer (University of North Carolina at Chapel Hill) et al, In EMNLP 2018. [\[pdf\]](https://arxiv.org/abs/1809.06309) [\[code (tf)\]](https://github.com/yicheng-w/CommonSenseMultiHopQA)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Scientific Writing] Text Generation from Knowledge Graphs with Graph Transformers** 
	- Rik Koncel-Kedziorski (University of Washington) et al, In NAACL 2018. [\[pdf\]](https://arxiv.org/pdf/1904.02342.pdf) [\[code (torch)\]](https://github.com/rikdz/GraphWriter)
- <img src="images/code.png" width="20" align=center> **[Scientific Writing] PaperRobot: Incremental Draft Generation of Scientific Ideas** 
	- Qingyun Wang (Rensselaer Polytechnic Institute) et al, In ACL 2019. [\[pdf\]](https://arxiv.org/pdf/1905.07870.pdf) [\[code (torch)\]](https://github.com/EagleW/PaperRobot)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Story Generation] Story Ending Generation with Incremental Encoding and Commonsense Knowledge**
	- Jian Guan (Tsinghua University) et al, In AAAI 2019. [\[pdf\]](https://arxiv.org/abs/1808.10113) [\[code (tf)\]](https://github.com/JianGuanTHU/StoryEndGen)
- <img src="images/code.png" width="20" align=center> **[Story Generation] Language Generation with Multi-Hop Reasoning on Commonsense Knowledge Graph**
	- Haozhe Ji (Tsinghua University) et al, In EMNLP 2020. [\[pdf\]](https://arxiv.org/abs/2009.11692) [\[code (torch)\]](https://github.com/cdjhz/multigen)
- **[Story Generation] MEGATRON-CNTRL: Controllable Story Generation with External Knowledge Using Large-Scale Language Models**
	- Peng Xu (The Hong Kong University of Science and Technology) et al, In EMNLP 2020. [\[pdf\]](https://arxiv.org/abs/2010.00840)
- <img src="images/code.png" width="20" align=center> **[Story Generation] KG-BART: Knowledge Graph-Augmented BART for Generative Commonsense Reasoning** 
	- Ye Liu (University of Illinois at Chicago) In AAAI 2021. [\[pdf\]](https://arxiv.org/abs/2009.12677) [\[code (torch)\]](https://github.com/yeliu918/KG-BART)
- **[Machine Translation] Knowledge Graphs Enhanced Neural Machine Translation** 
	- Yang Zhao (Chinese Academy of Sciences) et al, In IJCAI 2020. [\[pdf\]](https://www.ijcai.org/Proceedings/2020/0559.pdf)
- **[Summarization] Incorporating Commonsense Knowledge into Abstractive Dialogue Summarization via Heterogeneous Graph Networks** 
	- Xiachong Feng (Harbin Institute of Technology) et al, On arXiv 2020. [\[pdf\]](https://arxiv.org/abs/2010.10044)
- <img src="images/code.png" width="20" align=center>  **[Entity Description] ENT-DESC: Entity Description Generation by Exploring Knowledge Graph** 
	- Liying Cheng (Singapore University of Technology and Design) et al, In EMNLP 2020. [\[pdf\]](https://www.aclweb.org/anthology/2020.emnlp-main.90/) [\[code\]](https://github.com/LiyingCheng95/EntityDescriptionGeneration)
- <img src="images/code.png" width="20" align=center>  **[Eassy Generation] A Sentiment-Controllable Topic-to-Essay Generator with Topic Knowledge Graph** 
	- Lin Qiao (Peking Univerisity) et al, In EMNLP findings 2020. [\[pdf\]](https://www.aclweb.org/anthology/2020.findings-emnlp.299.pdf) [\[data\]](https://pan.baidu.com/s/17pcfWUuQTbcbniT0tBdwFQ)


## Open knowledge graph-enhanced text generation <br> (Knowledge graph constructed by OpenIE)
- **[Question Answering] Using Local Knowledge Graph Construction to Scale Seq2Seq Models to Multi-Document Inputs** 
	- Angela Fan (Facebook AI Research) et al, In EMNLP 2019. [\[pdf\]](https://arxiv.org/abs/1910.08435)

- <img src="images/code.png" width="20" align=center> **[Summarization] Knowledge Graph-Augmented Abstractive Summarization with Semantic-Driven Cloze Reward** 
	- Luyang Huang (Northeastern University) et al, In ACL 2020. [\[pdf\]](https://arxiv.org/abs/2005.01159) [\[code (torch)\]](https://github.com/luyang-huang96/GraphAugmentedSum)
- **[Summarization] Boosting Factual Correctness of Abstractive Summarization with Knowledge Graph** 
	- Chenguang Zhu (Microsoft Research) et al, On arXiv 2020. [\[pdf\]](https://arxiv.org/abs/2003.08612)
- <img src="images/code.png" width="20" align=center> **[Summarization] Heterogeneous Graph Neural Networks for Extractive Document Summarization** 
	- Danqing Wang (Fudan University) et al, In ACL 2020. [\[pdf\]](https://arxiv.org/abs/2004.12393) [\[code (torch)\]](https://github.com/brxx122/HeterSUMGraph)


## Grounded text-enhanced text generation
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Dialogue System] A Knowledge-Grounded Neural Conversation Model**
	- Marjan Ghazvininejad (University of Southern California) et al, In AAAI 2018. [\[pdf\]](https://arxiv.org/abs/1702.01932) [\[data\]](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling)

- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Dialogue System] Wizard of Wikipedia: Knowledge-Powered Conversational agents** 
	- Emily Dinan (Facebook AI Research) et al, In ICLR 2019. [\[pdf\]](https://arxiv.org/pdf/1811.01241.pdf) [\[code (torch)\]](https://github.com/facebookresearch/ParlAI/tree/master/projects/wizard_of_wikipedia)
- <img src="images/code.png" width="20" align=center> **[Dialogue System] Sequential Latent Knowledge Selection for Knowledge-Grounded Dialogue** 
	- Byeongchang Kim (Seoul National University) et al, In ICLR 2020. [\[pdf\]](https://arxiv.org/abs/2002.07510) [\[code (tf)\]](https://github.com/bckim92/sequential-knowledge-transformer)
- **[Dialogue System] DeepCopy: Grounded Response Generation with Hierarchical Pointer Networks** 
	- Semih Yavuz (University of California, Santa Barbara) et al, In SIGDIAL 2019. [\[pdf\]](https://arxiv.org/abs/1908.10731)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Dialogue System] Conversing by Reading: Contentful Neural Conversation with On-demand Machine Reading** 
	- Lianhui Qin (University of Washington) et al, In ACL 2019. [\[pdf\]](https://arxiv.org/abs/1906.02738) [\[code (torch)\]](https://github.com/qkaren/converse_reading_cmr)
- <img src="images/code.png" width="20" align=center> **[Dialogue System] RefNet: A Reference-aware Network for Background Based Conversation** 
	- Chuan Meng (Shandong University) et al, In AAAI 2020. [\[pdf\]](https://arxiv.org/abs/1908.06449) [\[code (tf)\]](https://github.com/ChuanMeng/RefNet)
- <img src="images/code.png" width="20" align=center> **[Dialogue System] Thinking Globally, Acting Locally: Distantly Supervised Global-to-Local Knowledge Selection for Background Based Conversation** 
	- Pengjie Ren (University of Amsterdam) et al, In AAAI 2020. [\[pdf\]](https://arxiv.org/abs/1908.09528) [\[code (torch)\]](https://github.com/PengjieRen/GLKS)
- <img src="images/code.png" width="20" align=center> **[Dialogue System] Knowledge-Grounded Dialogue Generation with Pre-trained Language Models** 
	- Xueliang Zhao (Wangxuan Institute of Computer Technology) et al, In EMNLP 2020. [\[pdf\]](https://arxiv.org/abs/2010.08824) [\[code (torch)\]](https://github.com/zhaoxlpku/KnowledGPT)
- <img src="images/code.png" width="20" align=center> **[Dialogue System] Bridging the Gap between Prior and Posterior Knowledge Selection for Knowledge-Grounded Dialogue Generation** 
	- Xiuyi Chen (Chinese Academy of Sciences) et al, In EMNLP 2020. [\[pdf\]](https://arxiv.org/abs/2010.08824) [\[code\]](https://github.com/youngornever/bridge_latent_knowledge_selection_gap_for_conversation)
- <img src="images/code.png" width="20" align=center> **[Dialogue System] Difference-aware Knowledge Selection for Knowledge-grounded Conversation Generation** 
	- Chujie Zheng (Tsinghua University) et al, In EMNLP findings 2020. [\[pdf\]](https://arxiv.org/abs/2009.09378) [\[code (torch)\]](https://github.com/chujiezheng/DiffKS)
- **[Question Answering] Generating Well-Formed Answers by Machine Reading with Stochastic Selector Networks** 
	- Bin Bi (Alibaba) et al, In AAAI 2020. [\[pdf\]](https://aaai.org/ojs/index.php/AAAI/article/view/6238)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> **[Summarization] Retrieve, Rerank and Rewrite: Soft Template Based Neural Summarization** 
	- Ziqiang Cao (The Hong Kong Polytechnic University) et al, In ACL 2018. [\[pdf\]](https://www.aclweb.org/anthology/P18-1015.pdf) [\[data\]](https://github.com/harvardnlp/sent-summary)
- <img src="images/code.png" width="20" align=center> **[Summarization] BiSET: Bi-directional Selective Encoding with Template for Abstractive Summarization** 
	- Kai Wang (Sun Yat-sen University) et al, In ACL 2019. [\[pdf\]](https://arxiv.org/abs/1906.05012) [\[code (torch)\]](https://github.com/InitialBug/BiSET)
- **[Paraphrase] Paraphrase Generation by Learning How to Edit from Samples** 
	- Amirhossein Kazemnejad (Iran University of Science and Technology) et al, In ACL 2020. [\[pdf\]](https://www.aclweb.org/anthology/2020.acl-main.535.pdf)

## Knowledge-enhanced pretraining
- <img src="images/code.png" width="20" align=center> **[KG + LM] A Knowledge-Enhanced Pretraining Model for Commonsense Story Generation**
	- Jian Guan (Tsinghua University) et al, In TACL 2020. [\[pdf\]](https://arxiv.org/abs/1906.05012) [\[code (tf)\]](https://github.com/JianGuanTHU/CommonsenseStoryGen)

- <img src="images/code.png" width="20" align=center> **[Commonsense + LM] Abductive Commonsense Reasoning** 
	- Chandra Bhagavatula (Allen Institute for AI) et al, In ICLR 2020. [\[pdf\]](https://arxiv.org/abs/1908.05739) [\[code (torch)\]](https://github.com/allenai/abductive-commonsense-reasoning)
- <img src="images/code.png" width="20" align=center> **[Table + LM] TAPAS: Weakly Supervised Table Parsing via Pre-training** 
	- Jonathan Herzig (Google) et al, in ACL 2020. [\[pdf\]](https://www.aclweb.org/anthology/2020.acl-main.398.pdf) [[official code (tf)]](https://github.com/google-research/tapas#how-to-cite-tapas) [[video]](https://www.youtube.com/watch?v=cIUtRNhY6Rw&ab_channel=YannicKilcher)
- **[KG + LM] JAKET: Joint Pre-training of Knowledge Graph and Language Understanding** 
	- Donghan Yu (Carnegie Mellon University) et al, On arXiv 2020. [\[pdf\]](https://arxiv.org/abs/2010.00796)
- <img src="images/code.png" width="20" align=center> **[KG + Data-to-text pretraining] KGPT: Knowledge-Grounded Pre-Training for Data-to-Text Generation** 
	- Wenhu Chen (University of California, Santa Barbara) et al, In EMNLP 2020. [\[pdf\]](https://www.aclweb.org/anthology/2020.emnlp-main.697.pdf) [\[code (torch)\]](https://github.com/wenhuchen/KGPT)


## Acknowledgement

<img src="images/ack.png" width="20" align=center> This page is contributed by [Wenhao Yu](https://wyu97.github.io/)(wyu1@nd.edu) and [Qingyun Wang](https://eaglew.github.io/cv/)(qingyun4@illinois.edu).
