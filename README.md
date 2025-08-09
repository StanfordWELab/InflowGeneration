# Inflow Generation for Computational Wind Engineering🚀

[![Python Version](https://img.shields.io/badge/python-3.x-blue)](https://www.python.org/)
[![License: TBD](https://img.shields.io/badge/license-TBD-lightgrey)](LICENSE)

---

## ​ Table of Contents

- [Introduction](#introduction)  
- [Features](#features)
- [Usage](#usage)  
- [Project Structure](#project-structure)

---

## ​ Introduction

This repository contains tools for **generating target inflow profiles** for **computational wind engineering** purposes.

---

## ​ Features

- Python-based framework—compatible with Python 3  
- Includes scripts for:
  - Generate an ABL from ASCE 49-21 provisons (`ASCEMattia.py`)
  - Gaussian process regression models hyperparameters tuning (`fitModel.py`,`resultsToDatabase.py`)
  - Optimizing inflow generator inputs to achieve a target ABL (`optimizeParameters.py`)
  - Generate CharLES domain for an ABL simulation  (`resultsToDatabase.py`, `paperPlots.py`)

---
