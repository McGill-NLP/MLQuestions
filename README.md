## Back-Training excels Self-Training at Unsupervised Domain Adaptationof Question Generation and Passage Retrieval
by *Devang Kulshreshtha, Robert Belfer, Iulian Vlad Serban and Siva Reddy*

Code and data for reproducing the results of our paper available on [arXiv](https://arxiv.org/abs/2104.08801).

### Data description
The MLQuestions dataset can be found in data/ subdirectory.

### Instructions for running the code

1. Download DPR IR model checkpoint pre-trained on NaturalQuestions. Instructions can be found in DPR official [repository](https://github.com/facebookresearch/DPR). Store it in IR/NQ-checkpoint subdirectory.
2. Download BART QG model trained on NaturalQuestions from [here](https://drive.google.com/drive/folders/1TyvdAdP57_uWPoqzg0iZfNABin4GAHfw?usp=sharing). Place the downloaded files config.json and pytorch_model.bin in QG/NQ-checkpoint subdirectory.
3. Run eval-no-adaptation.sh to run source to target domain without adaptation model (section 3 of paper).
4. Run UDA-SelfTraining.sh and UDA-BackTraining.sh to run self-training and back-training experiments on MLQuestions data as described in section 4 of the paper.

### Python Libraries and Dependencies