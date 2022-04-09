# EEG Classification

## <div align="center">About</div>
This repository is about the assignment from DLP Lab4-1

## <div align="center">Usage</div>
### Train
{model} - Choose the model: (0) EEGNet, (1) DeepConvNet
```
python main.py -m {model}
```

### Inference
{model} - Choose the model: (0) EEGNet, (1) DeepConvNet \
{activation} - Choose the activation func: (0) ELU, (1) ReLU, (2) LeakyReLU
```
python inference.py -m {model} -acti {activation}
```
