# Knowledge-enriched Text Generation Reading-List

Here is a list of recent publications about **Knowledge-enhanced text generation**.
(Last Update on **Oct. 11th, 2020**)

## <img src="images/new.png" width="25"> Survey paper

[A Survey of knoweldge-enhanced Text Generation](https://arxiv.org/abs/2010.04389). Wenhao Yu (ND), Chenguang Zhu (Microsoft), Zaitang Li (CUHK), Zhiting Hu (UCSD), Qingyun Wang (UIUC), Heng Ji (UIUC), Meng Jiang (ND). arXiv. 2010.04389

## Basic NLG papers and codes
- [Seq2Seq] [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)
	- Ilya Sutskever (Google) et al, In NeurIPS 2014.
- [SeqAttn] [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
	- Dzmitry Bahdanau (Jacobs University) et al, In ICLR 2015.
- [CopyNet] [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393)
	- Jiatao Gu (The University of Hong Kong) et al, In ACL 2016.
- [PointerNet] [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
	- Abigail See (Stanford University) et al, In ACL 2017.
- [Transformer] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
	- Ashish Vaswani (Google) et al, In NeurIPS 2017.


## Topic-enhanced text generation
- [Dialogue System] [Topic Aware Neural Response Generation](https://arxiv.org/pdf/1606.08340.pdf)
	- Chen Xing (Nankai University) et al, In AAAI 2017.
- [Dialogue System] [A Neural TopicalExpansion Framework for Unstructured Persona-oriented Dialogue Generation](https://arxiv.org/abs/2002.02153)
	- Minghong Xu (Shandong University) et al, In ECAI 2020. \[[code](https://github.com/Minghong-Xu/Neural_Topical_Expansion_for_UPDS)\](tensorflow)
- [Summarization] [Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://arxiv.org/abs/1808.08745)
	- Shashi Narayan (University of Edinburgh) et al, In EMNLP 2018. \[[code](https://github.com/EdinburghNLP/XSum)\](pytorch)
- [Summarization] [Topic-Guided Variational Autoencoders for Text Generation](https://arxiv.org/abs/1903.07137),
	- Wenlin Wang (Duke University) et al, In NAACL 2019.
- [Summarization] [Document Summarization with VHTM: Variational Hierarchical Topic-Aware Mechanism](https://aaai.org/ojs/index.php/AAAI/article/view/6277),
	- Xiyan Fu (Nankai Univeristy) et al, In AAAI 2020.

## Keyword-enhanced text generation
- [Dialogue System] [Sequence to Backward and Forward Sequences: A Content-Introducing Approach to Generative Short-Text Conversation](https://arxiv.org/pdf/1607.00970.pdf)
	- Lili Mou (Peking University) et al, In COLING 2016. \[[code](https://github.com/MaZhiyuanBUAA/Seq2BFforDialogueGeneration)\](tensorflow)
- [Dialogue System] [Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory](https://arxiv.org/abs/1704.01074)
	- Hao Zhou (Tsinghua University) et al, In AAAI 2018. \[[code](https://github.com/loadder/ECM-tf)\](tensorflow)
- [Dialogue System] [Generating Responses with a Specific Emotion in Dialog](https://www.aclweb.org/anthology/P19-1359.pdf)
	- Zhenqiao Song (Fudan University) et al, In ACL 2019.
- [Summarization] [Guiding Generation for Abstractive Text Summarization based on Key Information Guide Network](https://www.aclweb.org/anthology/N18-2009.pdf)
	- Chenliang Li (Beijing University of Posts and Telecommunications) et al, In NAACL 2018.
- [Summarization] [Coherent Comment Generation for Chinese Articles with a Graph-to-Sequence Model](https://arxiv.org/pdf/1906.01231.pdf)
	- Wei Li (Peking University) et al, In ACL 2019. \[[code](https://github.com/lancopku/Graph-to-seq-comment-generation)\](pytorch)
- [Summarization] [Keywords-Guided Abstractive Sentence Summarization](https://pdfs.semanticscholar.org/4e8d/103703fda8875f13c7593d80bc7428f05ded.pdf?_ga=2.124808970.1605688764.1602448023-651806684.1580066755&_gac=1.89511017.1599265589.Cj0KCQjwy8f6BRC7ARIsAPIXOjiX6Icl-pa3DjJSds7dc1teH9asRBgylw-EJHrcZfi8qlL0U9nJky4aAiD4EALw_wcB)
	- Haoran Li (JD AI Research) et al, In AAAI 2020.

## Knowledge base-enhanced text generation
- [Question Answering] [Generating Natural Answers by Incorporating Copying and Retrieving Mechanisms in Sequence-to-Sequence Learning](https://www.aclweb.org/anthology/P17-1019.pdf)
	- Shizhu He (Chinese Academy of Sciences) et al, In ACL 2017.
- [Dialogue System] [Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems](https://arxiv.org/abs/1804.08217)
	- Andrea Madotto (Hong Kong University of Science and Technology) et al, In ACL 2019. \[[code](https://github.com/HLTCHKUST/Mem2Seq)\](pytorch)
- [Dialogue System] [Global-to-local Memory Pointer Networks for Task-Oriented Dialogue](https://arxiv.org/abs/1901.04713)
	- Chien-Sheng Wu (Hong Kong University of Science and Technology) et al, In ICLR 2019. \[[code](https://github.com/jasonwu0731/GLMP)\](pytorch)
- [Dialogue System] [Improving Knowledge-aware Dialogue Generation via Knowledge Base Question Answering](https://arxiv.org/abs/1912.07491)
	- Jian Wang (South China University of Technology) et al, In AAAI 2020. \[[code](https://github.com/siat-nlp/TransDG)\](pytorch)
- [Dialogue System] [Learning to Select Knowledge for Response Generation in Dialog Systems](https://arxiv.org/abs/1902.04911)
	- Rongzhong Lian (Baidu) et al, In IJCAI 2019.
- [Dialogue System] [TopicKA: Generating Commonsense Knowledge-Aware Dialogue Responses Towards the Recommended Topic Fact](https://www.ijcai.org/Proceedings/2020/0521.pdf)
	- Sixing Wu (Peking University) et al, In IJCAI 2020. \[[code](https://github.com/pku-orangecat/IJCAI2020-TopicKA)\]
- [Content Manipulation] [Learning to Select Bi-Aspect Information for Document-Scale Text Content Manipulation](https://arxiv.org/abs/2002.10210)
	- Xiaocheng Feng (Harbin Institute of Technology) et al, In AAAI 2020. \[[code](https://github.com/syw1996/SCIR-TG-Data2text-Bi-Aspect)\]
- [Content Manipulation] [Fact-based Text Editing](https://www.aclweb.org/anthology/2020.acl-main.17.pdf)
	- Hayate Iso (Nara Institute of Science and Technology) et al, In ACL 2020. \[[code](https://github.com/isomap/factedit)\]

## Knowledge graph-enhanced text generation
- [Dialogue System] [Commonsense Knowledge Aware Conversation Generation with Graph Attention](https://www.ijcai.org/Proceedings/2018/0643.pdf)
	- Hao Zhou (Tsinghua University) et al, In IJCAI 2018. \[[code](https://github.com/thu-coai/ccm)\](tensorflow) \[[code](https://github.com/Lyusungwon/CCM-pytorch)\](pytorch)
- [Dialogue System] [Knowledge Aware Conversation Generation with Explainable Reasoning over Augmented Graphs](https://arxiv.org/abs/1903.10245)
	- Zhibin Liu,  (Baidu) et al, In EMNLP 2019.
- [Dialogue System] [DyKgChat: Benchmarking Dialogue Generation Grounding on Dynamic Knowledge Graphs](https://arxiv.org/abs/1910.00610)
	- Yi-Lin Tuan (National Taiwan University) et al, In EMNLP 2019. \[[code](https://github.com/Pascalson/DyKGChat)\](tensorflow)
- [Dialogue System] [Grounded Conversation Generation as Guided Traverses in Commonsense Knowledge Graphs](https://arxiv.org/abs/1911.02707)
	- Houyu Zhang (Brown University) et al, In ACL 2020. \[[code](https://github.com/thunlp/ConceptFlow)\](pytorch)
- [Scientific Writing] [Text Generation from Knowledge Graphs with Graph Transformers](https://arxiv.org/pdf/1904.02342.pdf)
	- Rik Koncel-Kedziorski (University of Washington) et al, In NAACL 2018. \[[code](https://github.com/rikdz/GraphWriter)\](pytorch)
- [Scientific Writing] [PaperRobot: Incremental Draft Generation of Scientific Ideas](https://arxiv.org/pdf/1905.07870.pdf)
	- Qingyun Wang (Rensselaer Polytechnic Institute) et al, In ACL 2019. \[[code](https://github.com/EagleW/PaperRobot)\](pytorch)
- [Story Generation] [Story Ending Generation with Incremental Encoding and Commonsense Knowledge](https://arxiv.org/abs/1808.10113)
	- Jian Guan (Tsinghua University) et al, In AAAI 2019. \[[code](https://github.com/JianGuanTHU/StoryEndGen)\](tensorflow)
- [Story Generation] [Language Generation with Multi-Hop Reasoning on Commonsense Knowledge Graph](https://arxiv.org/abs/2009.11692)
	- Haozhe Ji (Tsinghua University) et al, In EMNLP 2020. \[[code](https://github.com/cdjhz/multigen)\]

## Open knowledge graph-enhanced text generation (Kg constructed by OpenIE)
- [Summarization] [Using Local Knowledge Graph Construction to Scale Seq2Seq Models to Multi-Document Inputs](https://arxiv.org/abs/1910.08435)
	- Angela Fan (Facebook AI Research) et al, In EMNLP 2019.
- [Summarization] [Knowledge Graph-Augmented Abstractive Summarization with Semantic-Driven Cloze Reward](https://arxiv.org/abs/2005.01159)
	- Luyang Huang (Northeastern University) et al, In ACL 2020. \[[code](https://github.com/luyang-huang96/GraphAugmentedSum)\](pytorch)
- [Summarization] [Boosting Factual Correctness of Abstractive Summarization with Knowledge Graph](https://arxiv.org/abs/2003.08612)
	- Chenguang Zhu (Microsoft Research) et al, On arXiv.

## Grounded text-enhanced text generation
- [Dialogue System] [A Knowledge-Grounded Neural Conversation Model](https://arxiv.org/abs/1702.01932)
	- Marjan Ghazvininejad (University of Southern California) et al, In AAAI 2018. \[[data](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling)\]
- [Dialogue System] [Wizard of Wikipedia: Knowledge-Powered Conversational agents](https://arxiv.org/pdf/1811.01241.pdf)
	- Emily Dinan (Facebook AI Research) et al, In ICLR 2019. \[[code](https://github.com/facebookresearch/ParlAI/tree/master/projects/wizard_of_wikipedia)\](pytorch)
- [Dialogue System] [Sequential Latent Knowledge Selection for Knowledge-Grounded Dialogue](https://arxiv.org/abs/2002.07510)
	- Byeongchang Kim (Seoul National University) et al, In ICLR 2020. \[[code](https://github.com/bckim92/sequential-knowledge-transformer)\](tensorflow)
- [Dialogue System] [Conversing by Reading: Contentful Neural Conversation with On-demand Machine Reading](https://arxiv.org/abs/1906.02738)
	- Lianhui Qin (University of Washington) et al, In ACL 2019. \[[code](https://github.com/qkaren/converse_reading_cmr)\](pytorch)
- [Question Answering] [Generating Well-Formed Answers by Machine Reading with Stochastic Selector Networks](https://aaai.org/ojs/index.php/AAAI/article/view/6238)
	- Bin Bi (Alibaba) et al, In AAAI 2020.
- [Summarization] [Retrieve, Rerank and Rewrite: Soft Template Based Neural Summarization](https://www.aclweb.org/anthology/P18-1015.pdf)
	- Ziqiang Cao (The Hong Kong Polytechnic University) et al, In ACL 2018. \[[data](https://github.com/harvardnlp/sent-summary)\]
- [Summarization] [BiSET: Bi-directional Selective Encoding with Template for Abstractive Summarization](https://arxiv.org/abs/1906.05012)
	- Kai Wang (Sun Yat-sen University) et al, In ACL 2019. \[[code](https://github.com/InitialBug/BiSET)\](pytorch)

## Knowledge-enhanced pretraining
- [Story Generation] [A Knowledge-Enhanced Pretraining Model for Commonsense Story Generation](https://arxiv.org/abs/1906.05012)
	- Jian Guan (Tsinghua University) et al, In TACL 2020. \[[code](https://github.com/JianGuanTHU/CommonsenseStoryGen)\](tensorflow)
- [Abductive Reasoning] [Abductive Commonsense Reasoning](https://arxiv.org/abs/1908.05739)
	- Chandra Bhagavatula (Allen Institute for AI) et al, In ICLR 2020. \[[code](https://github.com/allenai/abductive-commonsense-reasoning)\](pytorch)

## Citation

```
@article{yu2020survey,
  title={A Survey of knoweldge-enhanced Text Generation},
  author={Yu, Wenhao and Zhu, Chenguang and Li, Zaitang and Hu, Zhiting, and Wang, Qingyun and Ji, Heng and Jiang, Meng},
  journal={arXiv preprint arXiv:2010.04389},
  year={2020}
}
```

## Acknowledgement

<img src="images/ack.png" width="20"> This page is contributed by [Wenhao Yu](https://wyu97.github.io/)(wyu1@nd.edu) and [Qingyun Wang](https://eaglew.github.io/cv/)(qingyun4@illinois.edu).