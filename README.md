The structure and flow of your documentation are clear and well-organized. Here’s a refined version of the "diPaRIS" description for enhanced clarity and smooth flow:
﻿
---
﻿
# diPaRIS: Dynamic and Interpretable Protein-RNA Interactions Prediction with U-shaped Network and Novel Structure Coding
﻿
## Introduction
Dynamic protein-RNA interactions are fundamental to various biological processes and disease development. These interactions occur at specific RNA binding sites, often influenced by the structural patterns of nucleotides. While existing computational methods incorporate RNA structural data in vivo, they often fail to capture the full context of nucleotide interactions, limiting their accuracy.
﻿
We introduce **diPaRIS**, a deep learning framework designed to predict dynamic protein-RNA interactions with increased accuracy and interpretability. The framework features a novel coding scheme that encodes nucleotide correlations, providing a detailed representation of RNA structural dynamics. Utilizing a U-shaped network architecture combined with attention mechanisms, diPaRIS enhances interpretability through sequence motif learning and the generation of attribution maps.
﻿
This repository includes the source code for both model training and prediction, along with an example dataset for demonstration purposes.
﻿
### diPaRIS Overview
![alt text](https://github.com/CSU-LishenZhang/diPaRIS/blob/main/Overview.jpg)
﻿
## Getting Started
﻿
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
﻿
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
﻿  
##  1. How to train a model using diPaRIS
You can train the model and test it with 5-fold cross-validation in a very simple way by run `diPaRIS_train.py` under the `/code` directory.
### Step-by-step description of full demo is as follows：
1. Load libraries.
```python
import math
import keras_nlp
import numpy as np
...
```
2. Load Data in current environment.


## There are some parameters you can set in the diPaRIS.py file as needed:
In the function `dealwithdata`, you can set the *data path* for diPaRIS. The default path is located under `'../dataset/' + protein + '/'`. Adjust this path as needed for your data setup. 

In the function `main`, you can set the *storage path* for trained diPaRIS models. The default path is under the working directory. Adjust this path in the `model.save`, `os.path.exists`, and `model.load_weights` functions as needed for your data setup. This ensures that your model can be saved and loaded correctly.

In the function `main`, you can set the *dataset name* for diPaRIS. The default dataset is `AKAP1-HepG2`. Adjust this name as needed for your data setup. You can also prepare the dataset yourself. The data required for diPaRIS prediction is illustrated in `/AKAP1-HepG2`.

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
