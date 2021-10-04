## Back-Training excels Self-Training at Unsupervised Domain Adaptation of Question Generation and Passage Retrieval
by *Devang Kulshreshtha, Robert Belfer, Iulian Vlad Serban and Siva Reddy*

Code and data for reproducing the results of our [paper](https://arxiv.org/abs/2104.08801) accepted to appear at *EMNLP 2021* 

### Data description
The MLQuestions dataset can be found in data/ subdirectory.

### Downloading pre-trained source domain NaturalQuestions checkpoint
1. Download DPR IR model checkpoint pre-trained on NaturalQuestions. Instructions can be found in DPR official [repository](https://github.com/facebookresearch/DPR). Store it in IR/NQ-checkpoint subdirectory. The model file name will be bert-base-encoder.cp
2. Download BART QG model trained on NaturalQuestions from [here](https://drive.google.com/drive/folders/1TyvdAdP57_uWPoqzg0iZfNABin4GAHfw?usp=sharing). Place the downloaded files config.json and pytorch_model.bin in QG/NQ-checkpoint subdirectory.

### Instructions for running the code
1. Run eval-no-adaptation.sh to run source to target domain without adaptation model (section 3 of paper).
2. Run UDA-SelfTraining.sh and UDA-BackTraining.sh to run self-training and back-training experiments on MLQuestions data as described in section 4 of the paper.
3. (Optional) You can also run consistency experiments by running any/all of UDA-SelfTraining-SC.sh, UDA-SelfTraining-CC.sh, UDA-BackTraining-SC.sh, UDA-BackTraining-CC.sh. SC denotes self-consistency and CC denotes cross-consistency.

### Python Libraries and Dependencies
Please install python modules from requirements.txt file.
Note: For METEOR score to be computed, Java must be installed. Otherwise only BLEU 1-4 will be computed for QG metrics.
