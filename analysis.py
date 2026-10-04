import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv("online_shoppers_intention.csv")

# Display the first 5 rows
print(df.head())

# Display the number of rows and columns
print("Dataset shape:", df.shape)

# Display information about the dataset
print(df.info())


# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Check the target variable
print("\nRevenue value counts:")
print(df["Revenue"].value_counts())

# Check the percentage of each class
print("\nRevenue percentage:")
print(df["Revenue"].value_counts(normalize=True) * 100)




# Descriptive statistics for numerical variables
print("\nDescriptive Statistics:")
print(df.describe())


# Detect outliers using the IQR method
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

print("\nOutlier analysis:")

for column in numerical_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(f"{column}: {len(outliers)} outliers")


# Skewness analysis
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

print("\nSkewness analysis:")
print(df[numerical_columns].skew().sort_values(ascending=False))


# Correlation analysis
correlation_matrix = df[numerical_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Analyse categorical variables

categorical_columns = df.select_dtypes(include=["str", "bool"]).columns

print("\nCategorical columns:")
print(categorical_columns)

for column in categorical_columns:
    print(f"\n{column} value counts:")
    print(df[column].value_counts())


# Save a copy of the original dataset
df_original = df.copy()

print("\nOriginal dataset saved.")
print("Rows:", len(df_original))

# Remove duplicate rows
df_clean = df.drop_duplicates()

print("\nAfter removing duplicates:")
print("Original rows:", len(df))
print("Rows after removing duplicates:", len(df_clean))
print("Duplicates removed:", len(df) - len(df_clean))

# Make a copy for preprocessing
df_preprocessed = df_clean.copy()

# Convert boolean columns to integers
df_preprocessed["Weekend"] = df_preprocessed["Weekend"].astype(int)
df_preprocessed["Revenue"] = df_preprocessed["Revenue"].astype(int)

# One-hot encode categorical variables
df_preprocessed = pd.get_dummies(
    df_preprocessed,
    columns=["Month", "VisitorType"],
    drop_first=True
)

print("\nAfter categorical encoding:")
print("Rows:", df_preprocessed.shape[0])
print("Columns:", df_preprocessed.shape[1])

print("\nFirst 5 rows after encoding:")
print(df_preprocessed.head())


# Check feature ranges before scaling
print("\nFeature ranges before scaling:")

numerical_features = df_preprocessed.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numerical_features:
    print(
        f"{column}: "
        f"min={df_preprocessed[column].min()}, "
        f"max={df_preprocessed[column].max()}"
    )

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Numerical features suitable for scaling
scaling_features = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

# StandardScaler
standard_scaler = StandardScaler()

df_standard = df_preprocessed.copy()

df_standard[scaling_features] = standard_scaler.fit_transform(
    df_standard[scaling_features]
)

print("\nAfter StandardScaler:")
print(df_standard[scaling_features].describe().loc[["mean", "std"]])

# MinMaxScaler
minmax_scaler = MinMaxScaler()

df_minmax = df_preprocessed.copy()

df_minmax[scaling_features] = minmax_scaler.fit_transform(
    df_minmax[scaling_features]
)

print("\nAfter MinMaxScaler:")
print(df_minmax[scaling_features].describe().loc[["min", "max"]])

# Log transformation for highly skewed variables

skewed_features = [
    "Informational_Duration",
    "ProductRelated_Duration",
    "PageValues",
    "Administrative_Duration"
]

df_log = df_preprocessed.copy()

for column in skewed_features:
    df_log[column] = np.log1p(df_log[column])

print("\nSkewness before log transformation:")
print(df_preprocessed[skewed_features].skew())

print("\nSkewness after log transformation:")
print(df_log[skewed_features].skew())


# Class distribution after removing duplicates

print("\nClass distribution after removing duplicates:")
print(df_clean["Revenue"].value_counts())

print("\nClass distribution percentage:")
print(df_clean["Revenue"].value_counts(normalize=True) * 100)


from sklearn.model_selection import train_test_split

# Separate features and target
X = df_preprocessed.drop(columns=["Revenue"]).copy()

bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)

y = df_preprocessed["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining set:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting set:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)

X_train_over, y_train_over = ros.fit_resample(X_train, y_train)

print("\nAfter Random Oversampling:")
print("X_train:", X_train_over.shape)
print("y_train:", y_train_over.shape)

print("\nClass distribution:")
print(y_train_over.value_counts())

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)

X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

print("\nAfter Random Undersampling:")
print("X_train:", X_train_under.shape)
print("y_train:", y_train_under.shape)

print("\nClass distribution:")
print(y_train_under.value_counts())


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print("X_train:", X_train_smote.shape)
print("y_train:", y_train_smote.shape)

print("\nClass distribution:")
print(y_train_smote.value_counts())


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=1000))
])

model_scaled.fit(X_train, y_train)

y_pred = model_scaled.predict(X_test)

print("\nScaled Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))


model_smote = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=1000))
])

model_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = model_smote.predict(X_test)

print("\nSMOTE + Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_smote))
print("Precision:", precision_score(y_test, y_pred_smote))
print("Recall:", recall_score(y_test, y_pred_smote))
print("F1 Score:", f1_score(y_test, y_pred_smote))


model_over = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=1000))
])

model_over.fit(X_train_over, y_train_over)

y_pred_over = model_over.predict(X_test)

print("\nRandom Oversampling + Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_over))
print("Precision:", precision_score(y_test, y_pred_over))
print("Recall:", recall_score(y_test, y_pred_over))
print("F1 Score:", f1_score(y_test, y_pred_over))

model_under = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=1000))
])

model_under.fit(X_train_under, y_train_under)

y_pred_under = model_under.predict(X_test)

print("\nRandom Undersampling + Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_under))
print("Precision:", precision_score(y_test, y_pred_under))
print("Recall:", recall_score(y_test, y_pred_under))
print("F1 Score:", f1_score(y_test, y_pred_under))

from sklearn.metrics import ConfusionMatrixDisplay

models = {
    "Baseline": (model_scaled, y_pred),
    "SMOTE": (model_smote, y_pred_smote),
    "Oversampling": (model_over, y_pred_over),
    "Undersampling": (model_under, y_pred_under)
}

for name, (model, predictions) in models.items():
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()

from sklearn.metrics import roc_auc_score

auc_baseline = roc_auc_score(
    y_test,
    model_scaled.predict_proba(X_test)[:, 1]
)

auc_smote = roc_auc_score(
    y_test,
    model_smote.predict_proba(X_test)[:, 1]
)

auc_over = roc_auc_score(
    y_test,
    model_over.predict_proba(X_test)[:, 1]
)

auc_under = roc_auc_score(
    y_test,
    model_under.predict_proba(X_test)[:, 1]
)

print("\nROC-AUC Results:")
print("Baseline:", auc_baseline)
print("SMOTE:", auc_smote)
print("Random Oversampling:", auc_over)
print("Random Undersampling:", auc_under)


results = pd.DataFrame({
    "Method": [
        "Scaled Logistic Regression",
        "SMOTE",
        "Random Oversampling",
        "Random Undersampling"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_smote),
        accuracy_score(y_test, y_pred_over),
        accuracy_score(y_test, y_pred_under)
    ],
    "Precision": [
        precision_score(y_test, y_pred),
        precision_score(y_test, y_pred_smote),
        precision_score(y_test, y_pred_over),
        precision_score(y_test, y_pred_under)
    ],
    "Recall": [
        recall_score(y_test, y_pred),
        recall_score(y_test, y_pred_smote),
        recall_score(y_test, y_pred_over),
        recall_score(y_test, y_pred_under)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred),
        f1_score(y_test, y_pred_smote),
        f1_score(y_test, y_pred_over),
        f1_score(y_test, y_pred_under)
    ],
    "ROC-AUC": [
        auc_baseline,
        auc_smote,
        auc_over,
        auc_under
    ]
})

print("\nFinal Model Comparison:")
print(results.round(4))


# Separate features and target
X_raw = df_clean.drop(columns=["Revenue"])
y_raw = df_clean["Revenue"].astype(int)

# Split into training and testing data
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw,
    y_raw,
    test_size=0.2,
    random_state=42,
    stratify=y_raw
)

print("\nFinal train/test split:")
print("Training data:", X_train_raw.shape)
print("Testing data:", X_test_raw.shape)

print("\nTraining target distribution:")
print(y_train_raw.value_counts())

print("\nTesting target distribution:")
print(y_test_raw.value_counts())


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

# Numerical features that need log transformation
log_features = [
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated_Duration",
    "PageValues"
]

# Other numerical features
numeric_features = [
    "Administrative",
    "Informational",
    "ProductRelated",
    "BounceRates",
    "ExitRates",
    "SpecialDay"
]

# Categorical features
categorical_features = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

print("\nLog-transformed features:")
print(log_features)

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Pipeline for skewed numerical features
log_pipeline = Pipeline([
    ("log", FunctionTransformer(np.log1p)),
    ("scaler", StandardScaler())
])

# Pipeline for normal numerical features
numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

# Pipeline for categorical features
categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine all preprocessing steps
preprocessor = ColumnTransformer([
    ("log", log_pipeline, log_features),
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("\nPreprocessing pipeline created successfully.")


# Fit preprocessing using training data only
X_train_processed = preprocessor.fit_transform(X_train_raw)

# Apply the same preprocessing to testing data
X_test_processed = preprocessor.transform(X_test_raw)

print("\nAfter preprocessing:")
print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)


# Create baseline Logistic Regression model
baseline_model = Pipeline([
    ("preprocessor", preprocessor),
    ("logistic", LogisticRegression(max_iter=2000))
])

# Train the model
baseline_model.fit(X_train_raw, y_train_raw)

# Make predictions
y_pred_baseline = baseline_model.predict(X_test_raw)

# Calculate evaluation metrics
baseline_accuracy = accuracy_score(y_test_raw, y_pred_baseline)
baseline_precision = precision_score(y_test_raw, y_pred_baseline)
baseline_recall = recall_score(y_test_raw, y_pred_baseline)
baseline_f1 = f1_score(y_test_raw, y_pred_baseline)

print("\nBaseline Logistic Regression:")
print("Accuracy:", baseline_accuracy)
print("Precision:", baseline_precision)
print("Recall:", baseline_recall)
print("F1 Score:", baseline_f1)

# SMOTE pipeline
smote_model = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("logistic", LogisticRegression(max_iter=2000))
])

# Train the SMOTE model
smote_model.fit(X_train_raw, y_train_raw)

# Make predictions
y_pred_smote = smote_model.predict(X_test_raw)

# Calculate evaluation metrics
smote_accuracy = accuracy_score(y_test_raw, y_pred_smote)
smote_precision = precision_score(y_test_raw, y_pred_smote)
smote_recall = recall_score(y_test_raw, y_pred_smote)
smote_f1 = f1_score(y_test_raw, y_pred_smote)

print("\nSMOTE + Logistic Regression:")
print("Accuracy:", smote_accuracy)
print("Precision:", smote_precision)
print("Recall:", smote_recall)
print("F1 Score:", smote_f1)

from imblearn.over_sampling import RandomOverSampler

# Random Oversampling pipeline
oversampling_model = ImbPipeline([
    ("preprocessor", preprocessor),
    ("oversampler", RandomOverSampler(random_state=42)),
    ("logistic", LogisticRegression(max_iter=2000))
])

# Train the model
oversampling_model.fit(X_train_raw, y_train_raw)

# Make predictions
y_pred_over = oversampling_model.predict(X_test_raw)

# Calculate evaluation metrics
over_accuracy = accuracy_score(y_test_raw, y_pred_over)
over_precision = precision_score(y_test_raw, y_pred_over)
over_recall = recall_score(y_test_raw, y_pred_over)
over_f1 = f1_score(y_test_raw, y_pred_over)

print("\nRandom Oversampling + Logistic Regression:")
print("Accuracy:", over_accuracy)
print("Precision:", over_precision)
print("Recall:", over_recall)
print("F1 Score:", over_f1)

from imblearn.under_sampling import RandomUnderSampler

# Random Undersampling pipeline
undersampling_model = ImbPipeline([
    ("preprocessor", preprocessor),
    ("undersampler", RandomUnderSampler(random_state=42)),
    ("logistic", LogisticRegression(max_iter=2000))
])

# Train the model
undersampling_model.fit(X_train_raw, y_train_raw)

# Make predictions
y_pred_under = undersampling_model.predict(X_test_raw)

# Calculate evaluation metrics
under_accuracy = accuracy_score(y_test_raw, y_pred_under)
under_precision = precision_score(y_test_raw, y_pred_under)
under_recall = recall_score(y_test_raw, y_pred_under)
under_f1 = f1_score(y_test_raw, y_pred_under)

print("\nRandom Undersampling + Logistic Regression:")
print("Accuracy:", under_accuracy)
print("Precision:", under_precision)
print("Recall:", under_recall)
print("F1 Score:", under_f1)



# Create final model comparison table
results = pd.DataFrame({
    "Method": [
        "Baseline Logistic Regression",
        "SMOTE",
        "Random Oversampling",
        "Random Undersampling"
    ],
    "Accuracy": [
        baseline_accuracy,
        smote_accuracy,
        over_accuracy,
        under_accuracy
    ],
    "Precision": [
        baseline_precision,
        smote_precision,
        over_precision,
        under_precision
    ],
    "Recall": [
        baseline_recall,
        smote_recall,
        over_recall,
        under_recall
    ],
    "F1 Score": [
        baseline_f1,
        smote_f1,
        over_f1,
        under_f1
    ]
})

print("\nFinal Model Comparison:")
print(results.round(4))


from sklearn.metrics import roc_auc_score

# Calculate ROC-AUC
baseline_auc = roc_auc_score(
    y_test_raw,
    baseline_model.predict_proba(X_test_raw)[:, 1]
)

smote_auc = roc_auc_score(
    y_test_raw,
    smote_model.predict_proba(X_test_raw)[:, 1]
)

over_auc = roc_auc_score(
    y_test_raw,
    oversampling_model.predict_proba(X_test_raw)[:, 1]
)

under_auc = roc_auc_score(
    y_test_raw,
    undersampling_model.predict_proba(X_test_raw)[:, 1]
)

print("\nROC-AUC Results:")
print("Baseline:", baseline_auc)
print("SMOTE:", smote_auc)
print("Random Oversampling:", over_auc)
print("Random Undersampling:", under_auc)


from sklearn.metrics import RocCurveDisplay

plt.figure(figsize=(8, 6))

RocCurveDisplay.from_predictions(
    y_test_raw,
    baseline_model.predict_proba(X_test_raw)[:, 1],
    name="Baseline"
)

RocCurveDisplay.from_predictions(
    y_test_raw,
    smote_model.predict_proba(X_test_raw)[:, 1],
    name="SMOTE"
)

RocCurveDisplay.from_predictions(
    y_test_raw,
    oversampling_model.predict_proba(X_test_raw)[:, 1],
    name="Random Oversampling"
)

RocCurveDisplay.from_predictions(
    y_test_raw,
    undersampling_model.predict_proba(X_test_raw)[:, 1],
    name="Random Undersampling"
)

plt.title("ROC Curves for Logistic Regression Models")
plt.tight_layout()
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

models = {
    "Baseline": (baseline_model, y_pred_baseline),
    "SMOTE": (smote_model, y_pred_smote),
    "Random Oversampling": (oversampling_model, y_pred_over),
    "Random Undersampling": (undersampling_model, y_pred_under)
}

for name, (model, predictions) in models.items():
    ConfusionMatrixDisplay.from_predictions(
        y_test_raw,
        predictions
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()


# Add ROC-AUC to the results table
results["ROC-AUC"] = [
    baseline_auc,
    smote_auc,
    over_auc,
    under_auc
]

print("\nFinal Model Comparison with ROC-AUC:")
print(results.round(4))

# Plot and save model performance comparison
results_plot = results.set_index("Method")

results_plot[
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Comparison of Logistic Regression Models")
plt.ylabel("Score")
plt.xlabel("Method")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.legend()
plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

results.round(4).to_csv(
    "final_model_comparison.csv",
    index=False
)

print("\nFinal model comparison saved as final_model_comparison.csv")

# Check the saved final results
final_results = pd.read_csv("final_model_comparison.csv")

print("\nSaved CSV contents:")
print(final_results)


# Task 3 - Numerical data quality summary

print("\n=== Numerical Data Quality Summary ===")

quality_summary = pd.DataFrame({
    "Missing Values": df[numerical_columns].isnull().sum(),
    "Mean": df[numerical_columns].mean(),
    "Median": df[numerical_columns].median(),
    "Standard Deviation": df[numerical_columns].std(),
    "Skewness": df[numerical_columns].skew()
})

print(quality_summary.round(3))

# Task 3 - Outlier summary using IQR

outlier_features = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

outlier_summary = []

for column in outlier_features:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_count = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ).sum()

    outlier_percentage = (outlier_count / len(df)) * 100

    outlier_summary.append([
        column,
        outlier_count,
        outlier_percentage
    ])

outlier_summary = pd.DataFrame(
    outlier_summary,
    columns=[
        "Feature",
        "Outlier Count",
        "Outlier Percentage"
    ]
)

print("\n=== Outlier Summary ===")
print(outlier_summary.round(2))

correlation_matrix = df[outlier_features].corr()

strong_correlations = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        correlation = correlation_matrix.iloc[i, j]

        if abs(correlation) >= 0.5:
            strong_correlations.append([
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation
            ])

strong_correlations = pd.DataFrame(
    strong_correlations,
    columns=["Feature 1", "Feature 2", "Correlation"]
)

print("\n=== Strong Correlations (|r| >= 0.5) ===")
print(strong_correlations.round(3))

class_summary = df_clean["Revenue"].value_counts().reset_index()

class_summary.columns = ["Revenue", "Count"]

class_summary["Percentage"] = (
    class_summary["Count"] / len(df_clean) * 100
)

print("\n=== Class Distribution ===")
print(class_summary.round(2))


categorical_check = {
    "Month": df_clean["Month"].unique(),
    "VisitorType": df_clean["VisitorType"].unique(),
    "Weekend": df_clean["Weekend"].unique(),
    "Revenue": df_clean["Revenue"].unique()
}

print("\n=== Categorical Values ===")

for column, values in categorical_check.items():
    print(column, ":", values)

print("\n=== Final Preprocessed Data ===")
print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)

print("\nFirst 5 rows of processed training data:")
print(X_train_processed[:5])