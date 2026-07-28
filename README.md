# ML Stock Analysis Project

## Overview
This project compares Logistic Regression and SVM models on S&P 500 and Nasdaq indices to identify predictive patterns.

## Methodology
We utilized historical daily returns as features to predict the direction of the next day's price. The comparison focuses on model selection and generalization capabilities.

## Comparison and Discussion
The results show the performance of both linear and non-linear models. 
- Logistic Regression provides a baseline for linear separability.
- SVM with an RBF kernel allows for non-linear decision boundaries.

The similarity in accuracy across models suggests that market returns are close to a random walk, aligning with the Efficient Market Hypothesis. This implies that neither model captures a strong predictive signal from past daily returns alone.

## Conclusion
Model selection is critical in financial data. While non-linear models like SVM are theoretically more powerful, the lack of significant accuracy gains indicates that the complexity of the data requires more than simple technical indicators for robust prediction.
