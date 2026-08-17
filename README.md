# ML Assignment 2

Phishing website classification using 5 models and a Streamlit app.

Student ID: 2025AC05412

This project classifies websites as phishing or legitimate using URL, domain, webpage, security, traffic and related website features.

## a. Problem statement

I used a phishing websites dataset and trained 5 classification models to check if a site is phishing or legitimate. The target column is `Result` where -1 means phishing and 1 means legitimate. After training I saved the models as pickle files and built a Streamlit page so the test csv can be uploaded and the metrics can be compared.

## Dataset

**Phishing Websites Data**

Kaggle: [https://www.kaggle.com/datasets/sai10py/phishing-websites-data](https://www.kaggle.com/datasets/sai10py/phishing-websites-data)

The dataset contains 11,055 instances, 30 input features, and the target variable `Result`.

- `Result = 1` : legitimate website
- `Result = -1` : phishing website

## b. Dataset description

Dataset: Phishing Websites Data — Kaggle

I downloaded it from a public source. Original file had 11055 rows and 30 features plus the Result column. All features were already numeric (-1, 0, 1). No missing values.

There were a lot of duplicate rows (around 5206). I dropped exact duplicates first. Some rows had the same features but different Result labels, so I removed those also, otherwise the same sample can go to both train and test.

After cleaning I had 5721 rows. Train test split is 80-20 with stratify and random_state=42. Test set has 1145 rows and is saved as test_data.csv.

This is more than 12 features and more than 500 instances so it matches the assignment limit.

## c. Github Repository Link

https://github.com/2025ac05412-SUHAS/2025AC05412_ML

## d. Models used

Same dataset for all models. Metrics are on the held out test data. Precision / Recall / F1 are for class 1 (legitimate).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9214 | 0.9783 | 0.9028 | 0.9386 | 0.9204 | 0.8434 |
| Decision Tree | 0.9467 | 0.9469 | 0.9378 | 0.9531 | 0.9454 | 0.8935 |
| kNN | 0.9345 | 0.9778 | 0.9378 | 0.9260 | 0.9319 | 0.8689 |
| Naive Bayes | 0.6629 | 0.9727 | 1.0000 | 0.3032 | 0.4654 | 0.4283 |
| Random Forest (Ensemble) | 0.9677 | 0.9947 | 0.9543 | 0.9801 | 0.9671 | 0.9357 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Pretty good baseline. I scaled the features because LR is sensitive to that. Accuracy is about 92% and AUC is high. A bit behind the tree models. |
| Decision Tree | Accuracy is higher than LR and kNN (94.67%) but AUC is a little lower. I think the tree is fitting the test data well but RF is more stable. |
| kNN | Also used scaling. 93.45% accuracy, similar to LR. Not bad, but RF is clearly better. |
| Naive Bayes | This one did not work well here. Accuracy only 66% and recall is very low. Features are like -1/0/1, not continuous, so GaussianNB is a bad match. AUC is still high which is confusing, but the actual predictions are not useful. |
| Random Forest (Ensemble) | Best numbers on almost everything. Accuracy 96.77%, AUC 0.9947, MCC 0.9357. I used 200 trees. |
| Overall Winner for your dataset? | Random Forest. It gave the best accuracy and MCC on this dataset. |

## How to run

Install packages first (this also happens automatically on Streamlit Cloud from requirements.txt):

```
pip install -r requirements.txt
```

Train models (only needed if you want to retrain):

```
python train_models.py
```

This saves pickle files in model/, metrics in results/, and test_data.csv.

Run the app:

```
streamlit run streamlit_app.py
```

Then upload test_data.csv, pick a model, click Evaluate.

The app only loads the saved .pkl files. It does not train again when it starts.
