# Knowledge-enriched Text Generation Reading-List

Here is a list of recent publications about **Knowledge-enhanced text generation**.
(Update on **Nov 5th, 2020**) <br>

-- We will continue to add and update related papers and codes on this page.

-- <img src="images/code.png" width="20" align=center> indicates available code and <img src="images/hot.png" width="20" align=center> indicates high citation in recent years.

## <img src="images/new.png" width="25" align=center> Survey paper

[A Survey of knoweldge-enhanced Text Generation](https://arxiv.org/abs/2010.04389). Wenhao Yu (ND), Chenguang Zhu (Microsoft), Zaitang Li (CUHK), Zhiting Hu (UCSD), Qingyun Wang (UIUC), Heng Ji (UIUC), Meng Jiang (ND). arXiv. 2010.04389

> To the best of our knowledge, our survey is the first work that presents a comprehensive reviewof knowledge-enhanced text generation. It aims to provide NLG researchers a synthesis and pointer to related researches. Our survey also includes a detailed discussion about how NLG can benefit from recent progress in deep learning and artificial intelligence, including technologies such as graph neural network, reinforcement learning, neural topic modeling and so on.

## Basic NLG papers and codes
(For new learners, some important papers for general NLG/KENLG.)

- <img src="images/hot.png" width="20" align=center> [Seq2Seq] [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)
	- Ilya Sutskever (Google) et al, In NeurIPS 2014.
- <img src="images/hot.png" width="20" align=center> [SeqAttn] [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
	- Dzmitry Bahdanau (Jacobs University) et al, In ICLR 2015.
- <img src="images/hot.png" width="20" align=center> [CopyNet] [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393)
	- Jiatao Gu (The University of Hong Kong) et al, In ACL 2016.
- <img src="images/hot.png" width="20" align=center> [PointerNet] [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
	- Abigail See (Stanford University) et al, In ACL 2017.
- <img src="images/hot.png" width="20" align=center> [Transformer] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
	- Ashish Vaswani (Google) et al, In NeurIPS 2017.

## Pretrained language generation models
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center>  [GPT-2] [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
	- Alec Radford (OpenAI) et al, On OpenAI blog 2019. \[[official code](https://github.com/openai/gpt-2/blob/master/src/model.py)\](tensorflow) \[[huggingface](https://github.com/huggingface/transformers/tree/master/examples/language-modeling)\](pytorch)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [UniLM] [Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)
	- Li Dong (Microsoft) et al, In NeurIPS 2019. \[[official code](https://github.com/microsoft/unilm)\](pytorch)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [BART] [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
	- Mike Lewis (Facebook AI) et al, On arXiv 2019. \[[fairseq](https://github.com/pytorch/fairseq)\](pytorch) \[[huggingface](https://github.com/huggingface/transformers/tree/master/examples/seq2seq)\](pytorch)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [T5] [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
	- Colin Raffel (Google) et al, In JMLR 2020. \[[official code](https://github.com/google-research/text-to-text-transfer-transformer)\](tensorflow) \[[huggingface](https://github.com/huggingface/transformers/tree/master/examples/seq2seq)\](pytorch)

## Controllable generation leanrng methods
- <img src="images/hot.png" width="20" align=center>  [Posterior Regularization] [Deep Generative Models with Learnable Knowledge Constraints](http://papers.nips.cc/paper/8250-deep-generative-models-with-learnable-knowledge-constraints.pdf)
	- Zhiting Hu (Carnegie Mellon University) et al, In NeurIPS 2018.
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Plug and Play] [Plug and Play Language Models: A Simple Approach to Controlled Text Generation](https://arxiv.org/abs/1912.02164)
	- Sumanth Dathathri (Uber AI) et al, In ICLR 2020. \[[code](https://github.com/uber-research/PPLM)\](pytorch)
- [Backprop-based Decoding] [Back to the Future: Unsupervised Backprop-based Decoding for Counterfactual and Abductive Commonsense Reasoning](https://arxiv.org/abs/2010.05906)
	- Lianhui Qin (University of Washington) et al, In EMNLP 2020.
- <img src="images/code.png" width="20" align=center> [Weakly Supervision] [Summarizing Text on Any Aspects: A Knowledge-Informed Weakly-Supervised Approach](https://arxiv.org/abs/2010.06792)
	- Bowen Tan (Carnegie Mellon University) et al, In EMNLP 2020. \[[code](https://github.com/tanyuqian/aspect-based-summarization)\]


## Topic-enhanced text generation
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Dialogue System] [Topic Aware Neural Response Generation](https://arxiv.org/pdf/1606.08340.pdf)
	- Chen Xing (Nankai University) et al, In AAAI 2017.
- <img src="images/code.png" width="20" align=center> [Dialogue System] [A Neural TopicalExpansion Framework for Unstructured Persona-oriented Dialogue Generation](https://arxiv.org/abs/2002.02153)
	- Minghong Xu (Shandong University) et al, In ECAI 2020. \[[code](https://github.com/Minghong-Xu/Neural_Topical_Expansion_for_UPDS)\](tensorflow)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Summarization] [Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://arxiv.org/abs/1808.08745)
	- Shashi Narayan (University of Edinburgh) et al, In EMNLP 2018. \[[code](https://github.com/EdinburghNLP/XSum)\](pytorch)
- <img src="images/hot.png" width="20" align=center> [Summarization] [Topic-Guided Variational Autoencoders for Text Generation](https://arxiv.org/abs/1903.07137)
	- Wenlin Wang (Duke University) et al, In NAACL 2019.
- [Summarization] [Document Summarization with VHTM: Variational Hierarchical Topic-Aware Mechanism](https://aaai.org/ojs/index.php/AAAI/article/view/6277)
	- Xiyan Fu (Nankai Univeristy) et al, In AAAI 2020.
- [Summarization] [A Topic Augmented Text Generation Model: Joint Learning of Semantics and Structural Features](https://arxiv.org/abs/2010.06253)
	- Peng Cui (Harbin Institute of Technology) et al, In COLING 2020.
- [Machine Translation] [Topic-Informed Neural Machine Translation](https://www.aclweb.org/anthology/C16-1170.pdf)
	- Jian Zhang, (Dublin City University) et al, In COLING 2016.
- [Machine Translation] [Translating with Bilingual Topic Knowledge for Neural Machine Translation](https://www.aaai.org/ojs/index.php/AAAI/article/view/4711)
	- Xiangpeng Wei (Chinese Academy of Sciences) et al, In AAAI 2019.
- [Topic Transfer] [A Topic Augmented Text Generation Model: Joint Learning of Semantics and Structural Features](https://www.aclweb.org/anthology/D19-1513.pdf)
	- Hongyin Tang (Chinese Academy of Sciences) et al, In EMNLP 2019.


## Keyword-enhanced text generation
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Dialogue System] [Sequence to Backward and Forward Sequences: A Content-Introducing Approach to Generative Short-Text Conversation](https://arxiv.org/pdf/1607.00970.pdf)
	- Lili Mou (Peking University) et al, In COLING 2016. \[[code](https://github.com/MaZhiyuanBUAA/Seq2BFforDialogueGeneration)\](tensorflow)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Dialogue System] [Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory](https://arxiv.org/abs/1704.01074)
	- Hao Zhou (Tsinghua University) et al, In AAAI 2018. \[[code](https://github.com/loadder/ECM-tf)\](tensorflow)
- <img src="images/hot.png" width="20" align=center> [Dialogue System] [Generating Responses with a Specific Emotion in Dialog](https://www.aclweb.org/anthology/P19-1359.pdf)
	- Zhenqiao Song (Fudan University) et al, In ACL 2019.
- [Summarization] [Guiding Generation for Abstractive Text Summarization based on Key Information Guide Network](https://www.aclweb.org/anthology/N18-2009.pdf)
	- Chenliang Li (Beijing University of Posts and Telecommunications) et al, In NAACL 2018.
- [Summarization] [Inferring Search Queries from Web Documents via a Graph-Augmented Sequence to Attention Network](https://dl.acm.org/doi/pdf/10.1145/3308558.3313746?casa_token=bIgxBamZyDkAAAAA:Oqmf3xhi_tIqBHoBZQsAHDb-OAeUBuLuAiAP4civXgx5DcJa45cMf5SjWPDAJO3U0_zJPG4oOt1aqA)
	- Fred X. Han (University of Alberta) et al, In WWW 2019.
- <img src="images/code.png" width="20" align=center> [Summarization] [Coherent Comment Generation for Chinese Articles with a Graph-to-Sequence Model](https://arxiv.org/pdf/1906.01231.pdf)
	- Wei Li (Peking University) et al, In ACL 2019. \[[code](https://github.com/lancopku/Graph-to-seq-comment-generation)\](pytorch)
- [Summarization] [Keywords-Guided Abstractive Sentence Summarization](https://pdfs.semanticscholar.org/4e8d/103703fda8875f13c7593d80bc7428f05ded.pdf?_ga=2.124808970.1605688764.1602448023-651806684.1580066755&_gac=1.89511017.1599265589.Cj0KCQjwy8f6BRC7ARIsAPIXOjiX6Icl-pa3DjJSds7dc1teH9asRBgylw-EJHrcZfi8qlL0U9nJky4aAiD4EALw_wcB)
	- Haoran Li (JD AI Research) et al, In AAAI 2020.

## Knowledge base-enhanced text generation
- <img src="images/hot.png" width="20" align=center> [Question Answering] [Generating Natural Answers by Incorporating Copying and Retrieving Mechanisms in Sequence-to-Sequence Learning](https://www.aclweb.org/anthology/P17-1019.pdf)
	- Shizhu He (Chinese Academy of Sciences) et al, In ACL 2017.
- [Question Answering] [Natural answer generation with heterogeneous memory](https://www.aclweb.org/anthology/P17-1019.pdf)
	- Yao Fu (Peking University) et al, In NAACL 2018.
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Dialogue System] [Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems](https://arxiv.org/abs/1804.08217)
	- Andrea Madotto (Hong Kong University of Science and Technology) et al, In ACL 2019. \[[code](https://github.com/HLTCHKUST/Mem2Seq)\](pytorch)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Dialogue System] [Global-to-local Memory Pointer Networks for Task-Oriented Dialogue](https://arxiv.org/abs/1901.04713)
	- Chien-Sheng Wu (Hong Kong University of Science and Technology) et al, In ICLR 2019. \[[code](https://github.com/jasonwu0731/GLMP)\](pytorch)
- <img src="images/code.png" width="20" align=center> [Dialogue System] [Improving Knowledge-aware Dialogue Generation via Knowledge Base Question Answering](https://arxiv.org/abs/1912.07491)
	- Jian Wang (South China University of Technology) et al, In AAAI 2020. \[[code](https://github.com/siat-nlp/TransDG)\](pytorch)
- [Dialogue System] [Learning to Select Knowledge for Response Generation in Dialog Systems](https://arxiv.org/abs/1902.04911)
	- Rongzhong Lian (Baidu) et al, In IJCAI 2019.
- <img src="images/code.png" width="20" align=center> [Dialogue System] [Diverse and Informative Dialogue Generation with Context-Specific Commonsense Knowledge Awareness](https://www.aclweb.org/anthology/2020.acl-main.515.pdf)
	- Sixing Wu (Peking University) et al, In ACL 2020. \[[code](https://github.com/pku-orangecat/ACL2020-ConKADI)\](tensorflow)
- <img src="images/code.png" width="20" align=center> [Dialogue System] [TopicKA: Generating Commonsense Knowledge-Aware Dialogue Responses Towards the Recommended Topic Fact](https://www.ijcai.org/Proceedings/2020/0521.pdf)
	- Sixing Wu (Peking University) et al, In IJCAI 2020. \[[code](https://github.com/pku-orangecat/IJCAI2020-TopicKA)\]
- <img src="images/code.png" width="20" align=center> [Content Manipulation] [Learning to Select Bi-Aspect Information for Document-Scale Text Content Manipulation](https://arxiv.org/abs/2002.10210)
	- Xiaocheng Feng (Harbin Institute of Technology) et al, In AAAI 2020. \[[code](https://github.com/syw1996/SCIR-TG-Data2text-Bi-Aspect)\]
- <img src="images/code.png" width="20" align=center> [Content Manipulation] [Fact-based Text Editing](https://www.aclweb.org/anthology/2020.acl-main.17.pdf)
	- Hayate Iso (Nara Institute of Science and Technology) et al, In ACL 2020. \[[code](https://github.com/isomap/factedit)\]
- [Summarization] [Exploring Human-Like Reading Strategy for Abstractive Text Summarization](https://www.aaai.org/ojs/index.php/AAAI/article/view/4724)
	- Min Yang (Chinese Academy of Sciences) et al, In AAAI 2019.
- <img src="images/code.png" width="20" align=center> [Table-to-text] [Describing a Knowledge Base](https://www.aclweb.org/anthology/W18-6502.pdf)
	- Qingyun Wang (UIUC) et al, in INLG 2018. [[Official code]](https://github.com/EagleW/Describing_a_Knowledge_Base) (pytorch) [[Slides]](https://eaglew.github.io/files/Wikipedia.pdf)

## Knowledge graph-enhanced text generation
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Dialogue System] [Commonsense Knowledge Aware Conversation Generation with Graph Attention](https://www.ijcai.org/Proceedings/2018/0643.pdf)
	- Hao Zhou (Tsinghua University) et al, In IJCAI 2018. \[[code](https://github.com/thu-coai/ccm)\](tensorflow) \[[code](https://github.com/Lyusungwon/CCM-pytorch)\](pytorch)
- [Dialogue System] [Knowledge Aware Conversation Generation with Explainable Reasoning over Augmented Graphs](https://arxiv.org/abs/1903.10245)
	- Zhibin Liu,  (Baidu) et al, In EMNLP 2019.
- <img src="images/code.png" width="20" align=center> [Dialogue System] [DyKgChat: Benchmarking Dialogue Generation Grounding on Dynamic Knowledge Graphs](https://arxiv.org/abs/1910.00610)
	- Yi-Lin Tuan (National Taiwan University) et al, In EMNLP 2019. \[[code](https://github.com/Pascalson/DyKGChat)\](tensorflow)
- <img src="images/code.png" width="20" align=center> [Dialogue System] [Grounded Conversation Generation as Guided Traverses in Commonsense Knowledge Graphs](https://arxiv.org/abs/1911.02707)
	- Houyu Zhang (Brown University) et al, In ACL 2020. \[[code](https://github.com/thunlp/ConceptFlow)\](pytorch)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Question Answering] [Commonsense for Generative Multi-Hop Question Answering Tasks](https://arxiv.org/abs/1809.06309)
	- Lisa Bauer (University of North Carolina at Chapel Hill) et al, In EMNLP 2018. \[[code](https://github.com/yicheng-w/CommonSenseMultiHopQA)\](tensorflow)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Scientific Writing] [Text Generation from Knowledge Graphs with Graph Transformers](https://arxiv.org/pdf/1904.02342.pdf)
	- Rik Koncel-Kedziorski (University of Washington) et al, In NAACL 2018. \[[code](https://github.com/rikdz/GraphWriter)\](pytorch)
- <img src="images/code.png" width="20" align=center> [Scientific Writing] [PaperRobot: Incremental Draft Generation of Scientific Ideas](https://arxiv.org/pdf/1905.07870.pdf)
	- Qingyun Wang (Rensselaer Polytechnic Institute) et al, In ACL 2019. \[[code](https://github.com/EagleW/PaperRobot)\](pytorch)
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Story Generation] [Story Ending Generation with Incremental Encoding and Commonsense Knowledge](https://arxiv.org/abs/1808.10113)
	- Jian Guan (Tsinghua University) et al, In AAAI 2019. \[[code](https://github.com/JianGuanTHU/StoryEndGen)\](tensorflow)
- <img src="images/code.png" width="20" align=center> [Story Generation] [Language Generation with Multi-Hop Reasoning on Commonsense Knowledge Graph](https://arxiv.org/abs/2009.11692)
	- Haozhe Ji (Tsinghua University) et al, In EMNLP 2020. \[[code](https://github.com/cdjhz/multigen)\]
- [Story Generation] [KG-BART: Knowledge Graph-Augmented BART for Generative Commonsense Reasoning](https://arxiv.org/abs/2009.12677)
	- Ye Liu (University of Illinois at Chicago) On arXiv 2020.
- [Machine Translation] [Knowledge Graphs Enhanced Neural Machine Translation](https://www.ijcai.org/Proceedings/2020/0559.pdf)
	- Yang Zhao (Chinese Academy of Sciences) et al, In IJCAI 2020.
- [Summarization] [Incorporating Commonsense Knowledge into Abstractive Dialogue Summarization via Heterogeneous Graph Networks](https://arxiv.org/abs/2010.10044)
	- Xiachong Feng (Harbin Institute of Technology) et al, On arXiv 2020.

## Open knowledge graph-enhanced text generation <br> (Knowledge graph constructed by OpenIE)
- [Question Answering] [Using Local Knowledge Graph Construction to Scale Seq2Seq Models to Multi-Document Inputs](https://arxiv.org/abs/1910.08435)
	- Angela Fan (Facebook AI Research) et al, In EMNLP 2019.
- <img src="images/code.png" width="20" align=center> [Summarization] [Knowledge Graph-Augmented Abstractive Summarization with Semantic-Driven Cloze Reward](https://arxiv.org/abs/2005.01159)
	- Luyang Huang (Northeastern University) et al, In ACL 2020. \[[code](https://github.com/luyang-huang96/GraphAugmentedSum)\](pytorch)
- [Summarization] [Boosting Factual Correctness of Abstractive Summarization with Knowledge Graph](https://arxiv.org/abs/2003.08612)
	- Chenguang Zhu (Microsoft Research) et al, On arXiv 2020.
- <img src="images/code.png" width="20" align=center> [Summarization] [Heterogeneous Graph Neural Networks for Extractive Document Summarization](https://arxiv.org/abs/2004.12393)
	- Danqing Wang (Fudan University) et al, In ACL 2020. \[[code](https://github.com/brxx122/HeterSUMGraph)\](pytorch)


## Grounded text-enhanced text generation
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Dialogue System] [A Knowledge-Grounded Neural Conversation Model](https://arxiv.org/abs/1702.01932)
	- Marjan Ghazvininejad (University of Southern California) et al, In AAAI 2018. \[[data](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling)\]
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Dialogue System] [Wizard of Wikipedia: Knowledge-Powered Conversational agents](https://arxiv.org/pdf/1811.01241.pdf)
	- Emily Dinan (Facebook AI Research) et al, In ICLR 2019. \[[code](https://github.com/facebookresearch/ParlAI/tree/master/projects/wizard_of_wikipedia)\](pytorch)
- <img src="images/code.png" width="20" align=center> [Dialogue System] [Sequential Latent Knowledge Selection for Knowledge-Grounded Dialogue](https://arxiv.org/abs/2002.07510)
	- Byeongchang Kim (Seoul National University) et al, In ICLR 2020. \[[code](https://github.com/bckim92/sequential-knowledge-transformer)\](tensorflow)
- [Dialogue System] [DeepCopy: Grounded Response Generation with Hierarchical Pointer Networks](https://arxiv.org/abs/1908.10731)
	- Semih Yavuz (University of California, Santa Barbara) et al, In SIGDIAL 2019.
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Dialogue System] [Conversing by Reading: Contentful Neural Conversation with On-demand Machine Reading](https://arxiv.org/abs/1906.02738)
	- Lianhui Qin (University of Washington) et al, In ACL 2019. \[[code](https://github.com/qkaren/converse_reading_cmr)\](pytorch)
- <img src="images/code.png" width="20" align=center> [Dialogue System] [RefNet: A Reference-aware Network for Background Based Conversation](https://arxiv.org/abs/1908.06449)
	- Chuan Meng (Shandong University) et al, In AAAI 2020. \[[code](https://github.com/ChuanMeng/RefNet)\](tensorflow)
- <img src="images/code.png" width="20" align=center> [Dialogue System] [Thinking Globally, Acting Locally: Distantly Supervised Global-to-Local Knowledge Selection for Background Based Conversation](https://arxiv.org/abs/1908.09528)
	- Pengjie Ren (University of Amsterdam) et al, In AAAI 2020. \[[code](https://github.com/PengjieRen/GLKS)\](pytorch)
- [Question Answering] [Generating Well-Formed Answers by Machine Reading with Stochastic Selector Networks](https://aaai.org/ojs/index.php/AAAI/article/view/6238)
	- Bin Bi (Alibaba) et al, In AAAI 2020.
- <img src="images/code.png" width="20" align=center> <img src="images/hot.png" width="20" align=center> [Summarization] [Retrieve, Rerank and Rewrite: Soft Template Based Neural Summarization](https://www.aclweb.org/anthology/P18-1015.pdf)
	- Ziqiang Cao (The Hong Kong Polytechnic University) et al, In ACL 2018. \[[data](https://github.com/harvardnlp/sent-summary)\]
- <img src="images/code.png" width="20" align=center> [Summarization] [BiSET: Bi-directional Selective Encoding with Template for Abstractive Summarization](https://arxiv.org/abs/1906.05012)
	- Kai Wang (Sun Yat-sen University) et al, In ACL 2019. \[[code](https://github.com/InitialBug/BiSET)\](pytorch)
- [Paraphrase] [Paraphrase Generation by Learning How to Edit from Samples](https://www.aclweb.org/anthology/2020.acl-main.535.pdf)
	- Amirhossein Kazemnejad (Iran University of Science and Technology) et al, In ACL 2020.

## Knowledge-enhanced pretraining
- <img src="images/code.png" width="20" align=center> [KG + LM] [A Knowledge-Enhanced Pretraining Model for Commonsense Story Generation](https://arxiv.org/abs/1906.05012)
	- Jian Guan (Tsinghua University) et al, In TACL 2020. \[[code](https://github.com/JianGuanTHU/CommonsenseStoryGen)\](tensorflow)
- <img src="images/code.png" width="20" align=center> [Commonsense + LM] [Abductive Commonsense Reasoning](https://arxiv.org/abs/1908.05739)
	- Chandra Bhagavatula (Allen Institute for AI) et al, In ICLR 2020. \[[code](https://github.com/allenai/abductive-commonsense-reasoning)\](pytorch)
- <img src="images/code.png" width="20" align=center> [Table + LM] [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://www.aclweb.org/anthology/2020.acl-main.398.pdf)
	- Jonathan Herzig (Google) et al, in ACL 2020. [[official code]](https://github.com/google-research/tapas#how-to-cite-tapas) (tensorflow) [[video]](https://www.youtube.com/watch?v=cIUtRNhY6Rw&ab_channel=YannicKilcher)
- [KG + LM] [JAKET: Joint Pre-training of Knowledge Graph and Language Understanding](https://arxiv.org/abs/2010.00796)
	- Donghan Yu (Carnegie Mellon University) et al, On arXiv 2020.
## Citation

```
@article{yu2020survey,
  title={A Survey of Knowledge-Enhanced Text Generation},
  author={Yu, Wenhao and Zhu, Chenguang and Li, Zaitang and Hu, Zhiting and Wang, Qingyun and Ji, Heng and Jiang, Meng},
  journal={arXiv preprint arXiv:2010.04389},
  year={2020}
}
```

## Acknowledgement

<img src="images/ack.png" width="20" align=center> This page is contributed by [Wenhao Yu](https://wyu97.github.io/)(wyu1@nd.edu) and [Qingyun Wang](https://eaglew.github.io/cv/)(qingyun4@illinois.edu).
