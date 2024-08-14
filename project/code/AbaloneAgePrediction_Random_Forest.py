import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Separate features and target from training data
X_train = train_data.drop(['id', 'Rings'], axis=1)
y_train = train_data['Rings'] + 1  # Adding 1 to avoid zero which is not valid in RMSLE calculation
X_test = test_data.drop('id', axis=1)

# Define categorical and numerical features
categorical_features = ['Sex']
numerical_features = [col for col in X_train.columns if col != 'Sex']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create a pipeline that includes preprocessing and the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data for validation (optional, can use cross-validation instead)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train_split, y_train_split)

# Evaluate the model using RMSLE
y_pred = model.predict(X_val_split)
rmsle = np.sqrt(mean_squared_log_error(y_val_split, y_pred))
print(f"Validation RMSLE: {rmsle}")

# Using cross-validation to evaluate the model
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')
cross_val_rmsle = np.sqrt(-scores.mean())
print(f"Cross-validated RMSLE: {cross_val_rmsle}")

# Predict on test data
test_predictions = model.predict(X_test)

# Create a DataFrame for submission
submission = pd.DataFrame({
    'id': test_data['id'],
    'Rings': (test_predictions.round() - 1).astype(int)  # Subtracting 1 to reverse the earlier shift
})

# Save submission to a CSV file
submission.to_csv('submission_random_forest.csv', index=False)
