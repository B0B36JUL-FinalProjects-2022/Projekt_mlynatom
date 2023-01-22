<p align="center">
 <img src="fig/DALL·E 2023-01-05 23.35.01 - titanic disaster.png" alt="DALLE generated image"/>
</p>

# Titanic - Machine Learning from Disaster
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_mlynatom/blob/main/LICENSE)

Projekt_mlynatom package provides data processing of Titanic - Machine Learning from Disaster data and prediction of survival
by ridge logistic regression and neural networks.

## Installation

The package is not available from official repositories and can be installed with the following command.
```julia
(@v1.8) pkg> add https://github.com/B0B36JUL-FinalProjects-2022/Projekt_mlynatom
```

## Description of data from kaggle.com
### Data Dictionary
| Variable | Definition                                 | Key                                            |
| -------- | ------------------------------------------ | ---------------------------------------------- |
| survival | Survival                                   | 0 = No, 1 = Yes                                |
| pclass   | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| sex      | Sex                                        |                                                |
| Age      | Age in years                               |                                                |
| sibsp    | # of siblings / spouses aboard the Titanic |                                                |
| parch    | # of parents / children aboard the Titanic |                                                |
| ticket   | Ticket number                              |                                                |
| fare     | Passenger fare                             |                                                |
| cabin    | Cabin number                               |                                                |
| embarked | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |
### Variable Notes
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

## Usage
This package focuses on two main parts: Data processing and prediction of survival.
All functions are implemented in files in [src](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_mlynatom/tree/main/src) folder. Example usage
is shown in [classify.jl](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_mlynatom/blob/main/examples/classify.jl).
### Data processing
In this part (mainly in src/data_preparation.jl) are data processing functions provided.
Main focus is on processing categorical values, filling missing values and creating new
data from other.
- categorical values are converted to dummy encoding
- missing age, fare and embark are filled with data, see [classify.jl](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_mlynatom/blob/main/examples/classify.jl).
- from name column titles are separated and then used as new data

### Prediction of survival
<p align="center">
 <img src="fig/DALL·E 2023-01-05 23.40.19 - titanic passengers in photorealistic version.png" alt="DALLE generated image"/>
</p>

In this part are implemented 2 options. First is ridge logistic regression and second are neural networks. 
- Neural networks are trained and defined using [Flux](https://fluxml.ai/Flux.jl/stable/) library with custom training loop.
- Ridge logistic regression is regularization method used to avoid overfitting.