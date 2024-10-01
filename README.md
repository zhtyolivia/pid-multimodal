# Quantifying multimodal interactions in multimodal medical data

There has been an increasing interest in combining different modalities such as radiology, pathology, genomic, and clinical data to improve the accuracy and robustness of diagnosis and prognosis in medicine. However, most existing works choose their datasets and modeling approaches empirically and in an ad hoc manner. Four partial information decomposition (PID)-based metrics have been shown to provide a theoretical and systematic understanding of multimodal data interactions in machine learning settings: redundancy between two modalities, uniqueness of each modality, and synergy that emerges when the fused modalities induce new information. However, these metrics have only been evaluated in a limited range of biomedical data, and the existing work did not elucidate the effect of different parameterizations in calculating the PID. In this work, we further assess the PID metrics using four different multimodal radiology cohorts in lung, prostate, and brain cancers. We found that, though promising, the PID metrics were not always consistent with the machine learning model performance and only one dataset had perfect consistency. We identified potential sources of inconsistency and provided suggestions for future works.

![pipeline](figures/pipeline.png)

This repository contains the four multimodal medical datasets and code we used in this work. Users can use this repository to calculate PID metrics and run survival anslysis models (linear/nonlinear, single-/multi-modality) with which we assess the consistency between PID metrics and downstream model performance. 

## Data preparation 

For all four datasets, the data splits used for repeated cross-validation are provided in the JSON files under the `datasets` directory. Moreover, the prostate T2W-ADC and lung radiopathomic datasets are also provided in CSV format. 

The lung cancer radiogenomic data was obtained from a study by Grossmann et al. The brain radiogenomic dataset was obtained from the Cancer Imaging Archive and the Cancer Genome Atlas. Please download the features and outcomes using the respective links. Note that for the brain radiogenomic dataset, we did one-hot encoding on categorical radiomic traits, resulting in 20 binary variables.

The resulting file structure is shown below. The following PID calculation and modeling scripts will assume this file structure: 

```
pid-multimodal/
  ├── datasets/               
  │   ├── lung_radiopathomic/ 
  │   ├── prostate_t2w_adc/   
  │   ├── ...   
  ├── linear_models 
  │   ├── ...   
  ├── nonlinear_models 
  │   ├── ...
  ├── ...         
```

## Calculate PID-based metrics 

We adapted the implementation of PID-based metrics for quantifying multimodal interactions from Liang et al. See [this repository](https://github.com/pliang279/PID/tree/1f6e9d09598754f0dcf7d4ce7e7ffe1c377b0035) for further details. 

### Step 1: Installing required packages 

To calculate PID metrics, first create a pytorch docker container with the following command:
```
docker run  --shm-size=2g --gpus all -it --rm -v /:/workspace -v /etc/localtime:/etc/localtime:ro nvcr.io/nvidia/pytorch:21.12-py3
```

See [this link](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) for additional information on this docker image. 

Next, navigate to `pid_calculation` and install required packages with this command:
```
pip install -r pid_requirement.txt
```

### Step 2: Calculating PID metrics

Now, you can calculate PID metrics using the ```calculate_pid.py``` script by specifying the dataset name, number of PCA components, and the number of clusters. For example, to calculate PID metrics on the lung_radiopathomic dataset, 

```
python calculate_pid.py --dataset lung_radiopathomic --pca-components 2 --k 3
```

## Linear models 

The first type of survival analysis model we have in this study is the linear cox proportional hazards model. This section describes how to run the linear models that we implemented in R. 

### Step 1: Installing R and RStudio 

To run those models: 

* Install R: https://www.r-project.org/

* Also, install RStudio: https://posit.co/download/rstudio-desktop/

### Step 2: Running linear models 

After installing R and RStudio, launch RStudio. 

In RStudio, open `concat_lung_radiopathomic.R` in the `linear_models` directory. Then, run the script in this file to run the concatenation-fused model on the lung radiopathomic dataset. The script will install necessary packages, run the model with repeated cross-validation, and print the concordance-index (C-index). 

Similarly, you can run the canonical correlation analysis (CCA)-fused model, unimodal models, and the ensemble model on this dataset with the other three R scripts in the `linear_models` directory. 

## Nonlinear models 

Aside from linear cox models, we also implemented multi-layer perceptron (MLP) models with with DeepSurv loss as our nonlinear survival analysis model. 

### Step 1: Installing required  packages

Similar to calculating PID metrics, create a docker container:

```
docker run  --shm-size=2g --gpus all -it --rm -v /:/workspace -v /etc/localtime:/etc/localtime:ro nvcr.io/nvidia/pytorch:21.12-py3
```

Then, intall the [lifelines package](https://lifelines.readthedocs.io/en/latest/): 

```
pip install lifelines 
```

### Step 2: Running nonlinear models 

Next, to perform early fusion on the lung radiopathomic dataset and obtain the cross-validation C-index of this model, navigate to `nonlinear_models` and run: 
```
CUDA_VISIBLE_DEVICES=0 python early_fusion.py --dataset lung_radiopathomic 
```

Similarly, for tensor fusion: 
```
CUDA_VISIBLE_DEVICES=0 python tensor_fusion.py --dataset lung_radiopathomic 
```

Also, to run the ensemble model: 
```
CUDA_VISIBLE_DEVICES=0 python ensemble.py --dataset lung_radiopathomic 
```

Finally, to run unimodal models (e.g., the radiomic-only model): 
```
CUDA_VISIBLE_DEVICES=0 python unimodal.py --dataset lung_radiopathomic --modality radiomic
```

The same scripts can be used to run the nonlinear models on the other three datasets by specifying dataset name and model hyperparameters. 