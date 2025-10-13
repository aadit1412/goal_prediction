# Goal Prediction Model

This repository presents a **data-driven approach to predicting goals** in football matches, combining event-level statistics, player positioning, and match context to estimate the likelihood of goal-scoring opportunities. The analysis leverages **machine learning techniques** to model goal probability and understand key performance drivers.

---

## Project Context

The project was designed to study **which factors contribute most to goal-scoring chances** and how event-level match data can be translated into predictive insights.  

Instead of focusing solely on historical outcomes, the model estimates **probabilistic likelihoods** for goals, helping highlight patterns in **team attacks, player positioning, and passing sequences**.

Key aspects of the analysis include:

- Using **XGBoost and Logistic Regression models** to predict goal occurrences  
- Processing **event-level match data** to engineer features for passes, shots, and possession chains  
- Identifying **key contributors to goal probability**, such as number of shots, xG, final-third ball receipts, and player position  
- Comparing models based on metrics like ROC-AUC, Brier Score, and Log-loss

The goal is to translate raw event data into a **quantitative understanding** of goal-scoring potential, supporting tactical and performance analysis.

---

## What's Included

- `goal_prediction.ipynb`: Full notebook with code, feature engineering, model training, and visualizations  
- `goal_prediction.py`: Clean Python script version of the workflow
- `app.py`: Interactive Streamlit app
- `xgb_model.pkl`: xgboost model
- `goal_prediction_logreg`: Logistic regression
- `README.md`: Project overview and context

---

## Tools Used

- **Python** (Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, XGBoost)  
- **Event-level football data** (e.g., passes, shots, carries, dribbles)  

---

## Future Work

Future updates will explore **expected threat (xT) integration, advanced spatial metrics, and more advanced model tuning** to tackle dataset imbalance, refine goal prediction accuracy, and link predictions to tactical decision-making.

---

## Contact

If you’d like to discuss football analytics, predictive modeling, or collaboration opportunities:

- [LinkedIn](https://www.linkedin.com/in/aaditpahuja)  
- 📧 aaditpahuja@gmail.com
