# NBA - Bank Marketing

## Goal of the project:
This project implements a Next Best Action (NBA) system prototype which process a direct marketing dataset of a bank.

The goal is to estimate which customer will subscribe to their term deposit before the current campaign using the information available, therefore the model tries to support a decision of which customer should be targeted during the campaign.

## Dataset justification:

The project uses the [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing).

This dataset contains historical data of bank customers which can be used for future predictions and marketing decisions. This aligns with the NBA use case: based on the information available before a campaign, the model estimates which customers are likely to subscribe to a term deposit.

We have customer context, historical interactions. Therefore we can estimate whether a customer willing to subscribe or not for a term deposit in the next campaing.

Additionally, this dataset is suitable to show multiple ML concepts and strategies, such as:
- Class imbalance
- Feature engineering
- Feature scaling
- Model comparison
- Cross validation

The limitaion is that this dataset will produce a model which only capable to classify customer willingness to subscribe term deposit, therefore who is the potential customer for the next campaign. NBA would be to target this customer or not in the future campaign.

## Quick start:
The project is managed by `uv`. Install it [here](https://docs.astral.sh/uv/getting-started/installation/).

1. Just shoot `./run.sh` and this calls the dataset preparation and the deterministic training pipeline.

Manually, these steps needed:
1. Create a .venv: `uv venv .venv`
2. `source .venv/bin/activate`
3. `uv sync`
4. Download dataset: `cd ./data` and run `./download.sh`
5. `python src/train.py`

## Repo design and help for the review:
There are two directories in this repository:
`data` and `src`.

`data`: Contains the dataset which need to be downloaded and put here (`download.sh` downloads it for you).
`data/notebooks`: Contains my exploratory data analysis and my experiments of the data where I used logistic regression, a decision tree and random forest.

`src`: Contains my final training and evaluation pipeline in the `train.py` file. The pipeline based on the scikit-learn library tools.

## System evaluation:
The repo contains my final results of the Bank Marketing dataset. See details [here](./src/README.md).

## AI USAGE STATEMENT:
I used a generative AI-based tool to complete the homework.

- I used AI for brainstorming.
- I used AI for evaluation of my understandings.
- I used AI to generate helper tools and functions for my system in `train.py`.

## Thank you
Thank you for reading this readme and taking the time to review this project. Any feedback is really appreciated!
