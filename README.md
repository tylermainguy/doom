# Deep Q-Networks for DOOM (1993)

Teaching an agent to play DOOM (1993) using deep Q-networks (DQNs). 


## Requirements

To begin, follow all installation instructions for the [OpenAI gym Vizdoom wrapper](https://github.com/shakenes/vizdoomgym.git) for environment setup. Some installations (MacOS) require that you first follow the [build instructions for Vizdoom](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md) before trying the installation for VizdoomGym.

To install requirements particular to our project, run:

```setup
pip install -r requirements.txt
```

## Training

To train the model, run 

```train
python main.py
```

Selectable hyperparameters can be read about by running `python main.py --help`

## Evaluation

To evaluate the trained model on a given environment, run:

```eval
python main.py --eval --game=<gametype>
```
where `gametype` is either `"defend"`, or `"health"`.

## Pre-trained Models

You can download pretrained models here:

- [Pretrained models](https://queensuca-my.sharepoint.com/:f:/g/personal/16tsm_queensu_ca/Ela7AaJUtvZMugjjwWp66T8BvF0FR78FSwccNQNeUJzNUg?e=G5ynLT), each folder corresponds to pretrained models on a given task. Download and place the folder of interest in `models/` for usage.

## Results

Here are a few videos documenting the trained agent in different available environments:
