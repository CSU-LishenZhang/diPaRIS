# diPaRIS: Dynamic and Interpretable Protein-RNA Interactions Prediction with U-shaped Network and Novel Structure Coding
 ## Introduction
Dynamic protein-RNA interactions are fundamental to various biological processes and disease development. These interactions occur at specific RNA binding sites, often influenced by the structural patterns of nucleotides. While existing computational methods incorporate RNA structural data in vivo, they often fail to capture the full context of nucleotide interactions, limiting their accuracy.
﻿

We introduce **diPaRIS**, a deep learning framework designed to predict dynamic protein-RNA interactions with increased accuracy and interpretability. The framework features a novel coding scheme that encodes nucleotide correlations, providing a detailed representation of RNA structural dynamics. Utilizing a U-shaped network architecture combined with attention mechanisms, diPaRIS enhances interpretability through sequence motif learning and the generation of attribution maps.
﻿

This repository includes the source code for both model training and prediction, along with an example dataset for demonstration purposes.
 
 ### diPaRIS Overview
<img src="https://github.com/CSU-LishenZhang/diPaRIS/blob/main/Overview.jpg" alt="diPaRIS Overview" width="1000"/>

## Getting Started
 ### Hardware Requirements
The `diPaRIS` package runs efficiently on standard computers with sufficient RAM for handling in-memory operations. However, for faster computations—especially during model training and large-scale inference—using a GPU is highly recommended.
﻿
### OS Requirements
The package is supported on *Linux* and has been tested on the following systems:
- CentOS Linux release 7.9.2009
﻿
### Software Requirements
- Python >= 3.8.0
- Keras == 2.10.0
- Scikit-learn >= 1.2.0
- SciPy >= 1.9.3
- NumPy >= 1.23.5
- TensorFlow-GPU == 2.10.1
- Keras-NLP == 0.4.1
- argparse >= 1.4.0
﻿
## Installation
1. **Clone the repository:**
```bash
git clone https://github.com/CSU-LishenZhang/diPaRIS.git
cd diPaRIS
```
﻿
2. *(Optional) Conda users:* Create and activate a conda environment:
```bash
conda create -n diPaRIS_env python=3.8
source activate diPaRIS_env
```
- Environment setup typically completes in about 30 seconds.
﻿
3. **Install dependencies:**
```bash
pip install -r requirements.txt
```
- Dependency installation may take 5 to 15 minutes, depending on your network speed.
﻿
## Usage
 ### Directory Structure:
```
- Overview.jpg -- Overview of the diPaRIS architecture.
- requirements.txt -- List of required packages and their versions.
- dataset/ -- Example dataset for training and evaluating models.
- dataset/AKAP1-HepG2/ -- Binding site data for AKAP1 in the HepG2 cell line.
- dataset/AKAP1-HepG2/negative_seq -- Sequences of negative samples from non-binding regions.
- dataset/AKAP1-HepG2/positive_seq -- Sequences of positive samples from binding sites.
- dataset/AKAP1-HepG2/negative_str -- Structures of negative samples from non-binding regions.
- dataset/AKAP1-HepG2/positive_str -- Structures of positive samples from binding sites.
- code/ -- Source code for both training and prediction tasks.
- code/ePooling.py -- Source code for global expectation pooling in diPaRIS.
- code/diPaRIS_train.py -- Example script for training a model using diPaRIS.
- code/diPaRIS_predict.py -- Example script for predicting binding sites using diPaRIS.
```
 ##  Steps required to train a model using diPaRIS
You can train the model and test it with 5-fold cross-validation in a very simple way by run `diPaRIS_train.py` under the `/code` directory.

**Example Command:**

```bash
python diPaRIS_train.py -d YourDataset
```
- `-d`: Specifies the dataset for training (optional). If the `-d` parameter is not provided, the program will automatically train on the default dataset (`AKAP1-HepG2`).

### Step-by-step description of full demo is as follows：
#### 1. **Environment Setup and Imports**
This step involves setting up the necessary environment and importing essential libraries like `Keras`, `sklearn`, and `numpy`. This section also includes GPU configuration settings to optimize the training process.

#### 2. **Data Preprocessing Functions**
These functions handle loading and processing the input sequences and structure information:
- **`coden(seq)`**: Converts RNA sequences to one-hot encoded vectors.
- **`chunks_two(seq, win)`**: Splits sequences into subsequences of length `win` (used later for generating k-mer probabilities).
- **`icshapeDS(seq, icshape)`**: Generates feature vectors using RNA sequence and icSHAPE data (probability-based and other structural features).

#### 3. **Loading and Preparing the Dataset**
The `dealwithdata(protein)` function is responsible for loading the positive and negative sequences along with the structural information. It preprocesses the data and returns train-test splits for both sequence and icSHAPE data.

#### 4. **Model Construction**
The `diPaRIS()` function defines the neural network architecture. This includes:
- Convolutional layers for feature extraction
- Bidirectional LSTM layers
- Transformer encoder layers
- Up-sampling and classification layers

#### 5. **Model Training and Cross-Validation**
The `main()` function handles the training process, performing K-Fold cross-validation, compiling the model, training it, and evaluating the performance. It also computes key metrics such as accuracy, precision, recall, F1-score, AUC, and AUPR.

## A Guide to Utilizing diPaRIS for the Prediction of Binding Sites

You can easily predict a binding site using diPaRIS by running the `diPaRIS_predict.py` script located in the `/code` directory with pre-trained diPaRIS models. The pre-trained diPaRIS models available for all 44 datasets described in our paper can be conveniently downloaded from Zenodo:  
[![Open in Zenodo](https://about.zenodo.org/static/img/logos/zenodo-gradient-square.svg)](https://doi.org/10.5281/zenodo.13895077)

**Example Command:**

```bash
python diPaRIS_predict.py --seq "Sequence of binding sites" --struc "Structure of binding sites" --protein "AKAP1-HepG2"
```

- `--seq`: The sequence of the binding sites you want to predict. Binding sites should be standardized to fragments of 101 nucleotides.
- `--struc`: The structure data corresponding to the sequence of the binding sites.
- `--protein`: The specific protein dataset for the prediction (optional).

**Note:**  
If the `--protein` parameter is not specified, the program will automatically traverse and make predictions across all available datasets.


## License

Copyright (C) 2020 Jianxin Wang(jxwang@mail.csu.edu.cn)

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, see <http://www.gnu.org/licenses/>.

Jianxin Wang(jxwang@mail.csu.edu.cn), School of Information Science and Engineering, Central South University, ChangSha, CHINA, 410083

## Contact

If any questions, please do not hesitate to contact us at:

Lishen Zhang, zls0424158@csu.edu.cn

Jianxin Wang, jxwang@csu.edu.cn
