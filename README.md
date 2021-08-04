# Self-Supervised Learning for OOD Detection

A Simplified Pytorch implementation of *Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty(NeurIPS 2019)*

**The code supports only Multi-class OOD Detection experiment(in-dist: CIFAR-10, Out-of-dist: CIFAR-100/SVHN)** 

- Command
```
python3 test.py
```

- Result
  - Metric : AUROC

| Dataset \ Method | RotNet | MSP |
|:---:|:---:|:---:|
| CIFAR100 | 0.8310 | 0.7695 |
| SVHN | 0.9755 | 0.8747 |

- Reference
  - full code(by authors): https://github.com/hendrycks/ss-ood
  - Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty(NeurIPS 2019): https://arxiv.org/abs/1906.12340
  - A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks(ICLR 2017): https://arxiv.org/abs/1610.02136

