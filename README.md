# Deep Q-Networks for DOOM (1993)

Teaching an agent to play DOOM (1993) using deep Q-networks (DQNs). 


## Requirements

To begin, follow all installation instructions for the [OpenAI gym Vizdoom wrapper](https://github.com/shakenes/vizdoomgym.git) for environment setup. 

To install requirements for our particular project, run:

```setup
pip install -r requirements.txt
```

## Training

To train the model, run 

```train
python train.py
```
Selectable hyperparameters can be read about by running `python train.py --help`

## Evaluation

To evaluate the trained model on a given environment, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth), trained for the different possible environments used in this project.

## Results

Here are a few videos documenting the trained agent in different available environments:
