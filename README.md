# 2022 March madness predictions

Based on the Kaggle competition [Men's March Mania 2022](https://www.kaggle.com/c/mens-march-mania-2022).

## Goal

After Selection Sunday and before any post-season games are played, give a prediction (winning
probability) for each potential match-up of the NCAA tournament. 

There are 64 to 68 college basket-ball teams at the beginning of the NCAA tournament, depending
on the number of play-in games. 

Out of the 2000+ winning probabilities you give, only about 30 of them will be used to score
your predictions, which are the probabilities for the match-ups that actually occurred in the
tournament. The score is the logarithmic loss.

## Stages

The goal above is the ultimate goal and corresponds to stage 2. 

Stage 1 lets you try out your algorithms against the past 5 NCAA tournaments. 

## Data

The competition provides the following data for the 1985-2021 range:
- Division 1 teams information;
- Regular season games results for all division 1 teams;
- Tournament seeds;
- Tournament games results.

In addition to that, the competition gives:
- Game statistics for 2003-2021;
- Game locations for 2010-2021;
- Team rankings for 2003-2021.

## Model

### Stage 1

We will train on the 1985-2015 data and for each year from 2016 to 2021 (minus 2020), we will
use the regular season and the seeds to predict winning probabilities. We will score our
predictions using the actual tournament results. 

Given the training dataset above, we need an API which accepts as arguments: a year and two
teams and outputs the winning probability of the team with the lowest ID.

First approach: feed the regular season games record to a Multi-layer Perceptron (MLP)
Classifier.

### Stage 2

We will train on the 1985-2021 data.