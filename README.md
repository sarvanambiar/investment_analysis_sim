# Investment Analysis Project

*A comprehensive analysis of stock returns and portfolio optimization using CAPM, Fama-French models, and tangent portfolio theory.*

---

## Table of Contents
- [Introduction](#introduction)
- [Project Objectives](#project-objectives)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Instructions](#instructions)
- [Results and Discussion](#results-and-discussion)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction
This project involves an investment analysis and portfolio optimization using financial data from Yahoo Finance and factor models like CAPM and Fama-French. It includes calculating stock alphas, betas, Sharpe ratios, and tangent portfolios while adjusting for various margin constraints.

---

## Project Objectives
- Perform CAPM analysis to calculate stock alphas and betas, and compare them with Yahoo Finance values.
- Extend the analysis by incorporating SMB, HML, and momentum (MOM) factors using Fama-French models.
- Compute Sharpe ratios of a Single Index Model (SIM) portfolio.
- Optimize tangent portfolios and evaluate the impact of different margin constraints on Sharpe ratios.

---

## Features
- **Data Processing:**
  - Download monthly stock and market index data.
  - Preprocess Fama-French factors and momentum factor datasets.

- **Analysis:**
  - Compute alphas and betas using CAPM.
  - Incorporate SMB, HML, and MOM factors to refine the analysis.
  - Compare computed betas with Yahoo Finance betas.

- **Portfolio Optimization:**
  - Compute tangent portfolio weights and Sharpe ratios.
  - Adjust portfolio weights for different margin constraints:
    - Uniform 50% initial margin.
    - Margin loan only for short positions.
    - Uniform 75% initial margin.

---

## Repository Structure
- `data/`: Contains input datasets for analysis (e.g., factor files).
- `scripts/`: Python script implementing the analysis.
- `results/`: Outputs like tables, summaries, and weights.
- `README.md`: Project overview and instructions.

---

## Requirements
### Libraries:
- `numpy`
- `pandas`
- `statsmodels`
- `yfinance`

Install the libraries using:
```bash
pip install numpy pandas statsmodels yfinance
