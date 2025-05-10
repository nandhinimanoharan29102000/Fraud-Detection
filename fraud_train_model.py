import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Step 1: Load dataset
df = pd.read_csv("AISquad-Dataset.csv")

# Step 2: Select features and target
features = ['cc_num', 'merchant', 'category', 'amt', 'city_pop', 'state']
target = 'is_fraud'
X = df[features]
y = df[target]

# Step 3: Label encoding for SMOTE compatibility
X_encoded = X.copy()
le_dict = {}
for col in ['merchant', 'category', 'state']:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    le_dict[col] = le  # Save for debugging or inverse transform if needed

# Step 4: Apply SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

# Step 5: Train-test split AFTER SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Step 6: Preprocessing pipeline (OneHotEncoding)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['merchant', 'category', 'state']),
    ],
    remainder='passthrough'
)

# Step 7: Define model pipeline
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Step 8: Train model
pipeline.fit(X_train, y_train)

# Step 9: Evaluate model
y_pred = pipeline.predict(X_test)
print("🔍 Model Performance on Test Data:")
print(classification_report(y_test, y_pred))

# Step 10: Save trained model
joblib.dump(pipeline, "fraud_model.joblib")
print("✅ Model training completed and saved as 'fraud_model.joblib'")
