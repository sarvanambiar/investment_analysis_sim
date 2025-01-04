
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
- [License](#license)

---

## Introduction

This project performs an in-depth investment analysis and portfolio optimization using financial data and factor models. It incorporates **CAPM**, **Fama-French models**, and **tangent portfolio theory**, exploring methods to calculate alphas, betas, Sharpe ratios, and optimize portfolios under various constraints.

You can access the project [here](https://github.com/sarvanambiar/investment_analysis_sim/tree/main).

---

## Project Objectives

- Perform CAPM analysis to calculate stock alphas and betas and compare them with Yahoo Finance values.
- Extend the analysis by incorporating SMB, HML, and momentum (MOM) factors using Fama-French models.
- Solve Kuhn-Tucker optimization for constrained portfolio weights.
- Compute Sharpe ratios for a Single Index Model (SIM) portfolio.
- Optimize tangent portfolios and evaluate the impact of different margin constraints on Sharpe ratios.

---

## Features

### **Data Processing**
- Download and preprocess stock, market index, and Fama-French factor datasets.

### **Analysis**
- Compute alphas and betas using CAPM.
- Refine the analysis by incorporating SMB, HML, and MOM factors.
- Solve Kuhn-Tucker optimization for portfolio constraints.

### **Portfolio Optimization**
- Compute tangent portfolio weights and Sharpe ratios.
- Adjust portfolio weights for various margin requirements:
  - Uniform 50% initial margin.
  - Margin loan for short positions only.
  - Uniform 75% initial margin.

---

## Repository Structure

```plaintext
investment_analysis_sim/
├── data/          # Input datasets for analysis (e.g., stock prices, factors)
├── scripts/       # Python scripts for data analysis and portfolio optimization
├── excel/         # Kuhn-Tucker optimization implementation and results
├── results/       # Summaries and outputs of the analysis
├── LICENSE        # License information
└── README.md      # Project overview and instructions
```

---

## Requirements

### Libraries
The following Python libraries are required:

- `numpy`
- `pandas`
- `statsmodels`
- `yfinance`

Install them using the following command:

```bash
pip install numpy pandas statsmodels yfinance
```

---

## Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sarvanambiar/investment_analysis_sim.git
   cd investment_analysis_sim
   ```

2. **Prepare the Data**:
   - Place stock and market index data files in the `data/` directory.
   - Ensure Fama-French factor files are formatted correctly.

3. **Run the Analysis**:
   - Execute the main script:
     ```bash
     python scripts/analysis.py
     ```
   - Outputs, including portfolio weights and Sharpe ratios, will be saved in the `results/` folder.

4. **View Kuhn-Tucker Implementation**:
   - Navigate to the `excel/` folder to access the Excel file containing the Kuhn-Tucker implementation.

---

## Results and Discussion

- Results are summarized in the [`results/`](https://github.com/sarvanambiar/investment_analysis_sim/tree/main/results) folder, including:
  - Portfolio weights for tangent and SIM portfolios.
  - Sharpe ratios under different margin constraints.
  - Documentation on Kuhn-Tucker optimization outcomes.

---

## License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project, provided proper attribution is given to the original author(s).

For detailed terms, see the [LICENSE](https://github.com/sarvanambiar/investment_analysis_sim/blob/main/LICENSE) file.

---
