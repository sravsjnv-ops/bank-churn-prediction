# Bank Churn Prediction — Neural Network Classifier

Predicting which bank customers are likely to churn in the next six months, using a **neural network classifier** built with TensorFlow/Keras.

## Business context
Acquiring a new customer costs 5–7× more than retaining one. This model gives the bank a ranked list of at-risk customers so retention teams can intervene early.

## Approach
1. **EDA** — target is imbalanced (~80/20); churners skew older, carry higher balances, and are far more likely to be inactive members; German customers churn at ~2× the rate of France/Spain
2. **Preprocessing** — encoding, scaling, stratified train/validation/test split
3. **Modeling** — neural network classifier with class-weighting to handle imbalance; multiple architectures and optimizers (SGD vs Adam) compared
4. **Evaluation** — recall-focused metric selection (missing a churner costs more than a false alarm), threshold tuning, confusion-matrix analysis

## Files
- `bank_churn_prediction.ipynb` — full notebook with outputs

## Tech stack
Python · pandas · TensorFlow/Keras · scikit-learn · seaborn

## How to run
```bash
pip install -r requirements.txt
jupyter notebook bank_churn_prediction.ipynb
```

*Built as part of the PGP in AI & Machine Learning (Great Learning) — Introduction to Neural Networks module.*
