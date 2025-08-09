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

