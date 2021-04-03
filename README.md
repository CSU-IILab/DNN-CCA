# DNN-CCA

Deep neural network aided canonical correlation analysis (DNN-CCA) in Tensorflow and Keras



This repository contains code for case study I in paper 

> Z. Chen, K. Liang, S. X. Ding, C. Yang, T. Peng and X. Yuan, "A comparative study of deep neural network aided canonical correlation analysis-based process monitoring and fault detection methods,"  *IEEE Transactions on Neural Networks and Learning Systems*, 2021.



## 1 Getting Started

### 1.1 Installation

Python3.6 and Tensorflow1.15 are required and should be installed on the host machine following the official guide. 

1. Clone this repository

```
git clone https://github.com/CSU-IILab/DNN-CCA
```

2. Install the required packages

``` 
pip install -r requirements.txt
```



## 2 Instructions

This repository provides the complete code for building deep neural network aided CCA, and a Jupyter Notebook for model training and testing.

### 2.1 Model definition

- Model structures are defined in *lib/model_xxx.py*, no hyperparameter included. 

- All the models can be run via *lib/run_dcca.py*
- The dataset is generated by simple numerical equation of random variables, consistent with the paper.

### 2.2 Deep neural network aided CCA Notebook

- It's a script to run the Linear CCA and deep neural network aided CCA, it controls the running of the model by executing *lib/lcca_detect.py* and *lib/run_dcca.py*. 
-  All the hyperparameters can be set by using this notebook, as well as train and test the model.



## 3 Citation

Please cite our paper if you use this code or any of the models.

```
@article{chen2021dnncca,
  title={A comparative study of deep neural network aided canonical correlation analysis-based process monitoring and fault detection methods},
  author={Chen, Zhiwen and Liang, Ketian and Ding, Steven X and Yang, Chao and Peng, Tao and Yuan, Xiaofeng},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={},
  number={},
  pages={},
  year={},
  publisher={IEEE},
  doi={}
}
```



## 4 License

MIT License



## 5 Related works

- Canonical correlation analysis-based fault detection and process monitoring ([Matlab source code](https://ww2.mathworks.cn/matlabcentral/fileexchange/66947-canonical-correlation-analysis-based-fault-detection-and-process-monitoring-algorithm))
> Z. Chen, S. X. Ding, T. Peng, C. Yang and W. Gui, "Fault detection for non-Gaussian process using generalized canonical correlation analysis and randomized algorithms," *IEEE Transactions on Industrial Electronics*, vol. 65, no. 2, pp. 1559-1567, 2018.



- Distributed CCA-based fault detection ([Matlab source code](https://www.mathworks.com/matlabcentral/fileexchange/89278-distributed-cca-based-fault-detection-method))
> Z. Chen, Y. Cao, S. X. Ding, K. Zhang, T. Koenings, T. Peng, C. Yang and W. Gui, "A Distributed Canonical Correlation Analysis-Based Fault Detection Method for Plant-Wide Process Monitoring," *IEEE Transactions on Industrial Informatics*, vol. 15, no. 5, pp. 2710-2720, 2019.



