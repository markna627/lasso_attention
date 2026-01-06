# Attention Module as a Solution to a LASSO Problem

This project replaces the typical Transformer's attention module as a solution to the corresponding LASSO problem with L1 regularizer. 

## Overview
The model was built from scratch with PyTorch. Unlike typical softmax attention from the first introduction, this project proves that attention score can also be understood as a solution to a linear regression problem the model solves at test time. Furthermore this project:
* builds the attention module as a solution to the corresponding lasso problem with L1 regularizer.
* proves the sparsity of the solution at the cost of computational efficiency.
* uses a synthetic data for associative memory recall task between keys and values
* compares against softmax attention for its sparsity and better mechanistic interpretebility. 
### How to Run
```
pip install -r requirements.txt
git clone https://github.com/markna627/lasso_attention.git
cd lasso_attention
python3 train.py
```

#### Model Performance
![Training/Validatation Loss](/assets/val_train.png)

#### Sparsity of the Solution
![Sparsity](/assets/sparsity.png)

### Notes 
Colab demo is available:
[Here](https://colab.research.google.com/drive/1Id82Qo1EfvXNXwdgcQs-TemPUhNi5B0h?usp=sharing)

### Related Works
 - Test-time Regression (Agarwal et al., 2023) unifies attention, fast-weight, and state-based sequence models by interpreting them as performing closed-form regression at inference time.
