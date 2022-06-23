This repository prsents the solution team LIPI developed while participating in FinSim-4-ESG (shared task of FinNLP workshop colocated with IJCAI-ECAI 2022).

The starter codes, data, meta-data has been provided by the organizers of [FinSim 2022](https://sites.google.com/nlg.csie.ntu.edu.tw/finnlp-2022/shared-task-finsim4-esg). More details about this shared task is availabe [here](https://sites.google.com/nlg.csie.ntu.edu.tw/finnlp-2022/shared-task-finsim4-esg).


The raw dataset are released under the CC BY-NC-ND license. More details are available [here](https://drive.google.com/file/d/1uLt-2WlCz9YE43DLXoQQOowhomBEnSTs/view). You can obtain the raw datasets by reaching out to the organizers if you fulfill the terms and conditions.

Our work can be cited as follows:

```bibtex 
@inproceedings{ghosh-2022-finsim-esg,
    title = "Ranking Environment, Social And Governance Related Concepts And Assessing Sustainability Aspect Of Financial Texts",
    author={Ghosh, Sohom and Naskar, Sudip Kumar},
    booktitle = "Proceedings of the Fourth Workshop on Financial Technology and Natural Language Processing (FinNLP@IJCAI-ECAI 2022)",
    month = "July" ,
    year = "2022",
    address = "Vienna, Austria",
    publisher = "-",
    url = "https://mx.nthu.edu.tw/~chungchichen/FinNLP2022_IJCAI/14.pdf",
    pages = "87--92",
}

```
Anything below this has been authored by the FinSim-4 organizers.

# FinSim 2022 official toolkit and datasets for ESG related tasks

Any issue using this repository can be reported to the organizers at *fin.sim.task@gmail.com*.

## FinSim 2022 Goal
The goal of the FinSim 2022 is to evaluate word representations for the ESG (Environment, Social, Governance) insights of the financial companies.

We propose to achieve this by analyzing how word representations automatically learnt from financial corpora can be used to classify ESG terms and sustainable sentences.

This edition focuses on ESG taxonomy elaborated by Fortia and sustainability of different activites of the financial companies.

# 1. Datasets
Under `data/`, two **main** files used for FinSim-ESG 2022 can be found. 

## 1.1. ESG Taxonomy 
### 1.1.1. Training set format
The list of terms and their ESG concepts is available under `data/terms` in a json file using the following format:
```json
{"term": " low-carbon", "concept": "Carbon factor"},...
"term": "Greenhouse gas emissions", "concept":"Emissions"}
```
## 1.2. Sustainable sentences 
### 1.1.2. Training set format
The list of sentences and their categories is available under `data/sentences` in a json file using the following format:
```json
{"sentence": "At Vauban Infrastructure Partners, we integrate in our daily work practices to avoid, reduce or offset our carbon emissions.", "label": "Sustainable"},...
"sentence": "Scope 3 is currently not subject to reporting, therefore it is not applicable.", "label":"Unsustainable"}
```
### 1.3. Test set format
The test set will contain a list of terms/sentences without their concepts/labels in the same format but with the value of `concept` & `label` set to null:
```json
{"term": "low-carbon", "concept": null}
```
### 1.4. Prediction set format
The expected prediction format is the following:
```json
{"term": " low-carbon", "label": null, "predicted_concepts":["Carbon factor", "Emissions","Waste management", "Biodiversity", "Employee development", "Community", "Audit Oversight"] }
```
### 1.5 File naming
Each run file needs to be named using the participant team's name and the run separated by `_`, with the extension `_predictions.json` .
For instance the team name `FinSimHeroes` will be allowed to send 2 following term files:
`FinSimHeroes_1_predictions.json`, `FinSimHeroes_2_predictions.json`


## 1.6. Tagset and ESG Taxonomy
### 1.6.1.Tagset
We have worked hard to elaborate ESG Taxonomy.
These labels refer to the most important and most frequently used types of ESG related topics.
In FinSim-ESG 2022, we propose a tagset of 25 concepts under `data/tagset`.


## 1.7. English ESG reports  data

We provide a set of ESG, Sustaintability, Annual reports in English of some financial companies to be used for training embeddings for this task available under `data/English_reports`.
Those have been downloaded from various websites and should NOT be re-distributed.

A number of script utilities are available to process PDF format, extract, and train embeddings, although it is not mandatory to use these scripts to submit to the task.
Please see relevant sections below.

# 2. Scripts
All scripts need to be run from the root of this repository using commands such as:
```bash
python -m baselines.baseline_1
```

## Baselines
Two baselines for this task can be improved:
 * a Logistisc Regression classifier based on custom embeddings trained on the corpus provided.
 * a distance-based classifier using custom embeddings.

To run these baselines, you would need to first train word embeddings and save the model under the `models/` folder.
If for some reason you cannot train embeddings, please contact the organizers at *fin.sim.task@gmail.com*.

## Scorer : Average Rank and Mean Accuracy
The official FinSim scores are average rank and mean accuracy. Those are simple measures and are implemented in the following script

```
python -m utils.scorer --predicted_data_path=<path_to_prediction_json>
```
