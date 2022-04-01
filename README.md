## Back-Training excels Self-Training at Unsupervised Domain Adaptation of Question Generation and Passage Retrieval
by *Devang Kulshreshtha, Robert Belfer, Iulian Vlad Serban and Siva Reddy*

This repository contains code and data for reproducing the results of our [paper](https://arxiv.org/abs/2104.08801) accepted to appear at **EMNLP 2021**.

Website hosting dataset coming soon!

### Data description
The MLQuestions dataset can be found in data/ subdirectory.

### Downloading pre-trained source domain NaturalQuestions checkpoint
1. Download DPR IR model checkpoint pre-trained on NaturalQuestions. Instructions can be found in DPR official [repository](https://github.com/facebookresearch/DPR). Store it in IR/NQ-checkpoint subdirectory. The model file name will be bert-base-encoder.cp
2.The BART QG model trained on NaturalQuestions can be found on [huggingface](https://huggingface.co/McGill-NLP/bart-qg-nq-checkpoint). You don't need to download the model, it will be downloaded automatically when you run the main code as described below.

### Instructions for running the code
1. Run eval-no-adaptation.sh to run source to target domain without adaptation model (section 3 of paper).
2. Run UDA-SelfTraining.sh and UDA-BackTraining.sh to run self-training and back-training experiments on MLQuestions data as described in section 4 of the paper.
3. (Optional) You can also run consistency experiments by running any/all of UDA-SelfTraining-SC.sh, UDA-SelfTraining-CC.sh, UDA-BackTraining-SC.sh, UDA-BackTraining-CC.sh. SC denotes self-consistency and CC denotes cross-consistency.

### Python Libraries and Dependencies
Please install python modules from requirements.txt file.
Note: For METEOR score to be computed, Java must be installed. Otherwise only BLEU 1-4 will be computed for QG metrics.

## Citation

If you find this useful in your research, please consider citing:

    @inproceedings{kulshreshtha-etal-2021-back,
        title = "Back-Training excels Self-Training at Unsupervised Domain Adaptation of Question Generation and Passage Retrieval",
        author = "Kulshreshtha, Devang  and
          Belfer, Robert  and
          Serban, Iulian Vlad  and
          Reddy, Siva",
        booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
        month = nov,
        year = "2021",
        address = "Online and Punta Cana, Dominican Republic",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.emnlp-main.566",
        pages = "7064--7078",
        abstract = "In this work, we introduce back-training, an alternative to self-training for unsupervised domain adaptation (UDA). While self-training generates synthetic training data where natural inputs are aligned with noisy outputs, back-training results in natural outputs aligned with noisy inputs. This significantly reduces the gap between target domain and synthetic data distribution, and reduces model overfitting to source domain. We run UDA experiments on question generation and passage retrieval from the Natural Questions domain to machine learning and biomedical domains. We find that back-training vastly outperforms self-training by a mean improvement of 7.8 BLEU-4 points on generation, and 17.6{\%} top-20 retrieval accuracy across both domains. We further propose consistency filters to remove low-quality synthetic data before training. We also release a new domain-adaptation dataset - MLQuestions containing 35K unaligned questions, 50K unaligned passages, and 3K aligned question-passage pairs.",
    }
