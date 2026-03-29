import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# loading dataset
file_path = "house_dataset.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types and missing values:")
print(df.info())
print("\nBasic statistics:")
print(df.describe())


# doing some data analysis and checking for some missing values if there
if df.isnull().sum().sum() > 0:
    print("\nMissing values found. Handling them...")
    # filling numeric columns with median
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
    print("Missing values handled.")
else:
    print("\nNo missing values found.")

# checking correlation with target variable 'price'
corr_matrix = df.corr()
print("\nCorrelation with price:")
print(corr_matrix['price'].sort_values(ascending=False))

# visualizing correlations 
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlations")
plt.show()

# splitting features and target
X = df.drop('price', axis=1)
y = df['price']

# train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# feature scaling (useful for linear models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model training and evaluation
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
best_model = None
best_r2 = -np.inf

for name, model in models.items():
    # using scaled data for linear models, original for tree-based
    if name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
        X_tr = X_train_scaled
        X_te = X_test_scaled
    else:
        X_tr = X_train
        X_te = X_test

    # training model
    model.fit(X_tr, y_train)

    # predicting on test set
    y_pred = model.predict(X_te)

    # evaluate
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # cross-validation score
    if name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    results[name] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "CV_R2_mean": cv_scores.mean(),
        "CV_R2_std": cv_scores.std()
    }

    print(f"\n{name}:")
    print(f"  MAE  = {mae:.2f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  R²   = {r2:.4f}")
    print(f"  CV R² (5-fold) = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # tracking best model based on test R^2
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name
        # if best model is linear, also save the scaler
        if name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
            best_scaler = scaler
        else:
            best_scaler = None

print(f"\n{'='*50}")
print(f"Best model: {best_model_name} with test R² = {best_r2:.4f}")

# feature importance (for random forest)
if best_model_name == "Random Forest":
    importances = best_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    print("\nFeature importances (Random Forest):")
    print(importance_df)

    # plot feature importance
    plt.figure(figsize=(8,5))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()

# saving the best model
joblib.dump(best_model, "best_house_price_model.pkl")
if best_scaler is not None:
    joblib.dump(best_scaler, "scaler.pkl")
    print("\nBest model and scaler saved to 'best_house_price_model.pkl' and 'scaler.pkl'")
else:
    print("\nBest model saved to 'best_house_price_model.pkl'")

# creating a sample input
sample_input = X_test.iloc[0].values.reshape(1, -1)
if best_scaler is not None:
    sample_input_scaled = best_scaler.transform(sample_input)
    predicted_price = best_model.predict(sample_input_scaled)
else:
    predicted_price = best_model.predict(sample_input)

actual_price = y_test.iloc[0]
print(f"\nSample prediction:")
print(f"  Input features: {dict(zip(X.columns, X_test.iloc[0]))}")
print(f"  Predicted price: {predicted_price[0]:.2f}")
print(f"  Actual price: {actual_price:.2f}")
print(f"  Difference: {abs(predicted_price[0] - actual_price):.2f}")