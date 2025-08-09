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

Scripts `fitModel.py` and `resultsToDatabase.py` can be used to perform the GPR model hyperparameters tuning. You need to run the scripts two times. One to fit the GPR model on the upstream database (inflow generator inputs), and one to fit the downstream database (target ABLs). The syntax to run the script is

You can get started quickly:

```bash
git clone https://github.com/mattiafc/InflowGeneration.git
cd InflowGeneration
pip install -r requirements.txt  # (Assuming there's a requirements file)```

where
