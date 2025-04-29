# TarDiff: Target-Oriented Diffusion Guidance for Synthetic Electronic Health Record Time Series Generation

  

## Introduction
![TarDiff OverView](./images/overview.png)

Synthetic Electronic Health Record (EHR) time-series generation is crucial for advancing clinical machine learning models, as it helps address data scarcity by providing more training data. However, most existing approaches focus primarily on replicating statistical distributions and temporal dependencies of real-world data. We argue that fidelity to observed data alone does not guarantee better model performance, as common patterns may dominate, limiting the representation of rare but important conditions. This highlights the need for generate synthetic samples to improve performance of specific clinical models to fulfill their target outcomes. To address this, we propose TarDiff, a novel target-oriented diffusion framework that integrates task-specific influence guidance into the synthetic data generation process. Unlike conventional approaches that mimic training data distributions, TarDiff optimizes synthetic samples by quantifying their expected contribution to improving downstream model performance through influence functions. Specifically, we measure the reduction in task-specific loss induced by synthetic samples and embed this influence gradient into the reverse diffusion process, thereby steering the generation towards utility-optimized data. Evaluated on six publicly available EHR datasets, TarDiff achieves state-of-the-art performance, outperforming existing methods by up to 20.4% in AUPRC and 18.4% in AUROC. Our results demonstrate that TarDiff not only preserves temporal fidelity but also enhances downstream model performance, offering a robust solution to data scarcity and class imbalance in healthcare analytics.


## Preparation

### Env
Prepare TarDiff's environment.
```
conda env create -f environment.yaml
conda activate tardiff
```

Prepare TS downstream task environment depands on the repo you used for the specific task.

### Data Preprocess
You can download eICU dataset and MIMIC-III dataset from the link below:


**Note**: We only use the time series recorded in both two EHR dataset. We prepared our script for extracting ts data on these dataset for reference, you can check them in directory **data_preprocess**

### Model Train
Train the base diffusion model on **eICU** dataset with task **mortality** prediction
**Note**:Before generation, you need to prepare the weights and architecture of a model for guidance under **eval_model**. This model should be trained on the original data and will provide task-based guidance during the sampling process.
```
bash train.sh
```
  
### Synt Data Generation Guidanced By Target Task Influence
```
bash generation.sh
```
You can check your synt data on ./test_eICU_mortality directory.

### Evaluation Your Data on Downstream Model!
Finally, you can use the generated data to train your own downstream task model!