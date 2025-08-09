# Inflow Generation for Computational Wind Engineering🚀

[![Python Version](https://img.shields.io/badge/python-3.x-blue)](https://www.python.org/)
[![License: TBD](https://img.shields.io/badge/license-TBD-lightgrey)](LICENSE)

---

## ​ Table of Contents

- [Introduction](#introduction)  
- [Features](#features)
- [Project Structure](#project-structure)
- [GPR Model](#usage)  

---

## ​ Introduction

This repository contains tools for **generating target inflow profiles** for **computational wind engineering** purposes.

---

## ​ Features

- Python-based framework—compatible with Python 3  
- Includes scripts for:
  - General-purpose functions used by most repo scripts (`modelDefinition.py`) 
  - Generating an ABL from ASCE 49-21 (`ASCEMattia.py`)
  - Gaussian process regression models hyperparameters tuning (`fitModel.py`,`resultsToDatabase.py`)
  - Optimizing inflow generator inputs to achieve a target ABL (`optimizeParameters.py`)
  - Generate CharLES domain for an ABL simulation  (`resultsToDatabase.py`, `paperPlots.py`)

---

## ​ Project Structure
├── GPRModels
├── RegressionPlots
├── InflowGeneration/
├──── codeABLs.py
├──── formatSetup.py
├──── generateInflow.py
├──── modelDefinition.py
├──── optimizeParameters.py
├──── paperPlots.py
├──── resultsToDatabase.py
├──── verifyInflow.py
├──── ASCEMattia.py
├──── GPRDatabase/
├──── Predictions/
├──── TestCases/
└── *.mat

---

## ​ GPR Model

Scripts `fitModel.py` and `resultsToDatabase.py` can be used to perform the GPR model hyperparameters tuning. You need to run the script at least two times. One to fit the GPR model on the upstream database (inflow generator inputs), and one to fit the downstream database (target ABLs). The syntax to run the scripts is 

`python3.9 fitModel.py '[x1,x2,x3]' setToFit`

where x1,x2,x3 are the x locations at which you want to fit the model (read the paper for more info), while setToFit can be Gridsearch, Inflow, or Plot. Gridsearch fits the downstream database for y in [0;1m], while Inflow fits the upstream database for y in [0;1.5m]. I'll focus on the Plot option later.

To properly run the script, you first need to define the QoIs for the fit (QoIs = ['u','Iu','Iv','Iw'] for Gridsearch, and QoIs = ['u','uu','vv','ww','uv'] for Inflow), yMax (1.0 for Gridsearch and 1.5 for Inflow) and then the TestID, which is the name of the output folder containing the GPR models fitted during the hyperparameter search. Each model is saved as GPRModels/xxx_TestID_QoI/yyy.pkl, where xxx is the x location at which the hyperparameter tuning is performed, QoI is the specific quantity of interest (e.g. uu or Iw), and yyy is the specific model identifier.

Once the gridsearch terminates, the `resultsToDatabase.py` scripts save the models that yields the best predictions on the dev set. The command to run this script is 

`python3.9 resultsToDatabase.py`

To properly run the script, you need to modify testID to reflect the name of the model, xList to define the x location at which you preformed the gridsearch, and the QoIs (quantities). For each QoI and x location, the code will sort the predictions of each model fitted during the hyperpameter tuning phase and save the best one based on the RMSE of the predictions on the dev set. The model will be saved in the GPRModels folder.

At this point, you can plot the predictions of the best models on the train, and either the dev or the test set by running the command

`python3.9 fitModel.py '[x1,x2,x3]' Plot`

To visualize either the Test set or the Dev set predictions, you need to define setToPlot = 'Dev' or setToPlot = 'Test', respectively. 
