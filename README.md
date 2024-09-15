# Quantifying multimodal interactions in multimodal medical data


## Data preparation 

You can download the four multimodal medical datasets used in this study [here](https://drive.google.com/drive/folders/13aZ5mFqh6dB-SVbxolOGTLOcxzYssZmx?usp=sharing). To compute PID metrics and run the models, place the datasets in the `datasets` directory. The CSV files include the features and outcomes, and the JSON files contain the data splits used for repeated cross-validation. 

## Calculate PID-based metrics 
We adapted the implementation of PID-based metrics for quantifying multimodal interactions from Liang et al. See the [original repository](https://github.com/pliang279/PID/tree/1f6e9d09598754f0dcf7d4ce7e7ffe1c377b0035) for further details. 

### Installing required packages 

First, create a pytorch docker container with the following command:
```
docker run  --shm-size=2g --gpus all -it --rm -v /:/workspace -v /etc/localtime:/etc/localtime:ro nvcr.io/nvidia/pytorch:21.12-py3
```

See [this link](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) for additional information on this docker image. 

Next, navigate to `pid-multimodal/pid_calculation` and install required packages by running the following commands:
```
pip install -r pid_requirement.txt
```

### Calculating PID matrics

Then, calculate PID metrics using the ```calculate_pid.py``` script. For example, to calculate PID metrics on the lung_radiopathomic dataset, 

```
python calculate_pid.py --dataset lung_radiopathomic --pca-components 2 --k 3
```

## Linear models 

We implemented linear cox proportional hazards model using R. The R scripts are under ```./linear_models```. To run those scripts, 

## Non-linear models 

### Installing required  packages
Same as for calculating PID metrics, create a docker container:
```
docker run  --shm-size=2g --gpus all -it --rm -v /:/workspace -v /etc/localtime:/etc/localtime:ro nvcr.io/nvidia/pytorch:21.12-py3
```

Then, navigate to `pid-multimodal/nonlinear_models` and run the following commands to install necessary packages: 
```
chmod +x install_pkgs.sh
./install_pkgs.sh
```

### Running nonlinear models 

Next, to perform early fusion on the lung radiopathomic dataset, run: 
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
CUDA_VISIBLE_DEVICES=0 unimodal.py -- dataset lung_radiopathomic --modality radiomic
```