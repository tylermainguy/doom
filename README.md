# Deep Q-Networks for DOOM (1993)

Teaching an agent to play DOOM (1993) using deep Q-networks (DQNs). 


## Requirements

To begin, follow all installation instructions for the [OpenAI gym Vizdoom wrapper](https://github.com/shakenes/vizdoomgym.git) for environment setup. Some installations (MacOS) require that you first follow the [build instructions for Vizdoom](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md) before trying the installation for VizdoomGym.


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

**Defend the Line**

https://user-images.githubusercontent.com/29045168/115936978-b0831e80-a464-11eb-8105-1cc24441444e.mp4

**Defend the Center**

https://user-images.githubusercontent.com/29045168/115937832-df01f900-a466-11eb-8a9f-725ffb67c6b0.mp4



