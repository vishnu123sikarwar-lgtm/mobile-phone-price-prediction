"""
Feature Extraction and Price Prediction for Mobile Phones
Executable End-to-End Pipeline
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 1. Load Data
FILE_PATH = r"C:\Users\user\Downloads\Processed_Flipdata.xlsx"
df = pd.read_excel(FILE_PATH)
print(f"[+] Loaded dataset shape: {df.shape}")

# 2. Data Cleaning & Outlier Correction
# Correct Motorola G62 5G height outlier (41.94 cm -> 16.64 cm)
df.loc[df['Mobile Height'] > 30.0, 'Mobile Height'] = 16.64

# Extract Megapixels from Camera Strings
def parse_mp(value):
    if pd.isna(value):
        return 0
    match = re.search(r'(\d+)\s*MP', str(value), re.IGNORECASE)
    return int(match.group(1)) if match else 0

df['Rear_Camera_MP'] = df['Rear Camera'].apply(parse_mp)
df['Front_Camera_MP'] = df['Front Camera'].apply(parse_mp)

# Brand Extraction
def extract_brand(model_str):
    name = str(model_str).strip().split()[0].upper()
    if 'MICROMAX' in name:
        return 'MICROMAX'
    if name == 'I':
        return 'ITEL'
    return name

df['Brand'] = df['Model'].apply(extract_brand)

# Group low-frequency brands to prevent one-hot sparsity
brand_counts = df['Brand'].value_counts()
top_brands = brand_counts[brand_counts >= 5].index.tolist()
df['Brand_Grouped'] = df['Brand'].apply(lambda b: b if b in top_brands else 'OTHER')

# 5G Capability Flag
df['Is_5G'] = (
    df['Model'].str.contains('5G', case=False, na=False) | 
    df['Processor_'].str.contains('5G', case=False, na=False)
).astype(int)

# Processor Family Simplification
def categorize_processor(proc):
    proc = str(proc).upper()
    if 'SNAPDRAGON' in proc or 'QUALCOMM' in proc:
        return 'SNAPDRAGON'
    elif 'DIMENSITY' in proc or 'HELIO' in proc or 'MEDIATEK' in proc or 'MT' in proc:
        return 'MEDIATEK'
    elif 'UNISOC' in proc or 'SPREADTRUM' in proc or 'SC' in proc:
        return 'UNISOC'
    elif 'EXYNOS' in proc or 'SEC' in proc:
        return 'EXYNOS'
    elif 'BIONIC' in proc or 'A1' in proc:
        return 'APPLE'
    elif 'TENSOR' in proc or 'GOOGLE' in proc:
        return 'GOOGLE'
    else:
        return 'OTHER'

df['Processor_Family'] = df['Processor_'].apply(categorize_processor)

# 3. Train-Test Split
target = 'Prize'
numerical_features = [
    'Memory', 'RAM', 'Battery_', 'AI Lens', 
    'Mobile Height', 'Rear_Camera_MP', 'Front_Camera_MP', 'Is_5G'
]
categorical_features = ['Brand_Grouped', 'Processor_Family']

X = df[numerical_features + categorical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Pipeline Definition
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 5. Model Evaluation Suite
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
}

results = {}
for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'Pipeline': pipe}
    print(f"[{name}] Test R²: {r2:.3f} | MAE: ₹{mae:,.0f} | RMSE: ₹{rmse:,.0f}")

# 6. Feature Importance from Champion Model
rf_pipeline = results['Random Forest']['Pipeline']
feature_names = (
    numerical_features + 
    list(rf_pipeline.named_steps['preprocessor']
         .named_transformers_['cat']
         .get_feature_names_out(categorical_features))
)
rf_importances = rf_pipeline.named_steps['regressor'].feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Top 10 Most Important Features (Random Forest) ---")
print(importance_df.head(10).to_string(index=False))

# 7. Visualization Plot Generation
plt.figure(figsize=(10, 6))
top_importances = importance_df.head(10)
sns.barplot(x='Importance', y='Feature', data=top_importances, hue='Feature', palette='mako', legend=False)
plt.title('Top 10 Feature Importances - Random Forest Regressor')
plt.xlabel('Relative Gini Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
best_preds = results['Random Forest']['Pipeline'].predict(X_test)
plt.scatter(y_test, best_preds, alpha=0.6, color='#1f77b4')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Price (₹)')
plt.ylabel('Predicted Price (₹)')
plt.title('Random Forest: Actual vs. Predicted Handset Prices')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300)
plt.close()

print("[+] Visualizations saved as 'feature_importance.png' and 'actual_vs_predicted.png'")