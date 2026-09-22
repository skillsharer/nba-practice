import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, make_scorer

SEED = 42

def plot_pr_curve(precision, recall, model_name):
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} - Cross-Validated Precision-Recall Curve")
    plt.grid(alpha=0.3)
    plt.show()

class DataPreprocessor:
    def __init__(self):
        self.preprocessor = self.build_preprocessor()

    def replace_minus_one(self, feature):
        return np.where(feature == -1, 0, feature)

    def build_preprocessor(self):
        standard_features = ["age"]
        robust_features = ["balance"]
        log_features = ["previous"]
        categorical_features = ["job", "marital", "education", "default", "housing", "loan", "poutcome"]

        pdays_preprocessor = Pipeline(
            steps=[("replace_minus_one", FunctionTransformer(self.replace_minus_one)),
                   ("log_transform", FunctionTransformer(np.log1p))
                   ])

        return ColumnTransformer(
            transformers=[
                (
                    "standard",
                    StandardScaler(),
                    standard_features
                ),
                (
                    "robust",
                    RobustScaler(),
                    robust_features
                ),
                (
                    "log",
                    FunctionTransformer(np.log1p),
                    log_features
                ),
                (
                    "pdays",
                    pdays_preprocessor,
                    ["pdays"]
                ),
                (
                    "categorical",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first"),
                    categorical_features
                )
            ]
        )

    def build_pipeline(self, model):
        return Pipeline(steps=[("preprocessor", self.preprocessor), ("model", model)])

class ModelTrainer:
    def __init__(self):
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        self.preprocessor = DataPreprocessor()
        self.models = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
            "decision_tree": DecisionTreeClassifier(random_state=SEED),
            "random_forest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
        }
        self.param_grids = {
            "logistic_regression": {},
            "decision_tree": {"model__max_depth": [3, 5, 7, 10, 15],  "model__min_samples_leaf": [5, 10, 20, 50, 100]},
            "random_forest": {"model__n_estimators": [100, 200], "model__max_depth": [3, 5, 7, 10, 15], "model__min_samples_leaf": [5, 10, 20, 50, 100]},
        }

    def compare_models(self, X_train, y_train):
        results = {}
        fitted_models = {}

        for name, model in self.models.items():
            print(f"Training: {name}")
            pipeline = self.preprocessor.build_pipeline(model)
            ap_scorer = make_scorer(average_precision_score, response_method="predict_proba", pos_label="yes")

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=self.param_grids[name],
                scoring=ap_scorer,
                cv=self.cv,
                n_jobs=-1,
                refit=True,
            )

            grid.fit(X_train, y_train)

            results[name] = grid.best_score_
            fitted_models[name] = grid.best_estimator_

            print(f"Best CV AP: {grid.best_score_}")
            print(f"Best parameters: {grid.best_params_}")

        best_name = max(results, key=results.get)
        best_model = fitted_models[best_name]

        print("Model comparison: ")

        for name, score in results.items():
            print(f"{name}: {score} mean CV AP.")

        print("OOF threshold selection:")
        print(f"Selected model: {best_name}")

        return best_name, best_model, results

    def evaluate(self, model, X_test, y_test, threshold=0.5):
        classes = model.classes_
        yes_index = np.where(classes == "yes")[0][0]
        probabilities = model.predict_proba(X_test)[:, yes_index]
        predictions = np.where(probabilities >= threshold, "yes", "no")
        ap = average_precision_score(y_test == "yes", probabilities)
        precision = precision_score(y_test, predictions, pos_label="yes")
        recall = recall_score(y_test, predictions, pos_label="yes")
        f1 = f1_score(y_test, predictions, pos_label="yes")
        cm = confusion_matrix(y_test, predictions, labels=["no", "yes"])

        print(f"Threshold: {threshold}")
        print(f"Average Precision: {ap}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        print(f"Confusion matrix: {cm}")

        return {
            "average_precision": ap,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
        }

def maximise_pr(precision, recall, thresholds):
    best_f1 = 0
    best_threshold = 0
    for i, threshold in enumerate(thresholds):
        p = precision[i]
        r = recall[i]
        if p + r == 0:
            continue
        f1 = 2 * (p * r) / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = p
            best_recall = r
    print(f"Threshold: {best_threshold}")
    print(f"Precision: {best_precision}")
    print(f"Recall: {best_recall}")
    print(f"F1: {best_f1}")

    return best_threshold

def predict(model, data, threshold):
    yes_index = np.where(model.classes_ == "yes")[0][0]
    probabilities = model.predict_proba(data)[:, yes_index]
    predictions = np.where(probabilities >= threshold, "yes", "no")
    return predictions, probabilities

def main():
    test_size = 0.2
    shuffle = True
    trainer = ModelTrainer()
    dataset = pd.read_csv("../data/bank-full.csv", sep=';')

    for feature in ["duration", "contact", "day", "month", "campaign"]:
        dataset = dataset.drop(columns=[feature])
    independent_variables = dataset.drop(columns=["y"])
    dependent_variable = dataset["y"]

    X_train, X_test, y_train, y_test = train_test_split(independent_variables, dependent_variable, random_state=SEED, test_size=test_size, shuffle=shuffle, stratify=dependent_variable)
    best_name, best_model, cv_results = trainer.compare_models(X_train, y_train)
    oof_probabilities = cross_val_predict(best_model, X_train, y_train, cv=trainer.cv, method="predict_proba", n_jobs=-1)
    yes_index = np.where(best_model.classes_ == "yes")[0][0]
    precision, recall, thresholds = precision_recall_curve(y_train, oof_probabilities[:, yes_index], pos_label="yes")
    threshold = maximise_pr(precision, recall, thresholds)
    print("Final test set evaluation")
    trainer.evaluate(best_model, X_test, y_test, threshold)
    plot_pr_curve(precision, recall, best_name)

if __name__ == "__main__":
    main()