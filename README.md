# License

Copyright (C) 2020 Jianxin Wang(jxwang@mail.csu.edu.cn), Chengqian Lu(chengqlu@csu.edu.cn)

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, see <http://www.gnu.org/licenses/>.

Jianxin Wang(jxwang@mail.csu.edu.cn), School of Information Science and Engineering, Central South University, ChangSha, CHINA, 410083


# diPaRIS
Dynamic and Interpretable Protein-RNA Interactions Prediction with U-shaped Network and Novel Structures Coding

## Requirements
- keras
- sklearn
- scipy
- numpy
- tensorflow
- keras_nlp

##  How to train diPaRIS
You can train the model of 5-fold cross-validation in a very simple way by run `diPaRIS.py` under the `/code` directory.

### There are some parameters you can set in the diPaRIS.py file as needed:
In the function `dealwithdata`, you can set the *data path* for diPaRIS. The default path is located under `'../dataset/' + protein + '/'`. Adjust this path as needed for your data setup. 

In the function `main`, you can set the *storage path* for trained diPaRIS models. The default path is under the working directory. Adjust this path in the `model.save`, `os.path.exists`, and `model.load_weights` functions as needed for your data setup. This ensures that your model can be saved and loaded correctly.

In the function `main`, you can set the *dataset name* for diPaRIS. The default dataset is `AKAP1-HepG2`. Adjust this name as needed for your data setup. You can also prepare the dataset yourself. The data required for diPaRIS prediction is illustrated in `/AKAP1-HepG2`.

Thank you and enjoy the tool!
