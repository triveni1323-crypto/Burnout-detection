# train_model.py
# This file handles all ML model training, evaluation,
# comparison and saving of the best model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("   BURNOUT DETECTION - MACHINE LEARNING TRAINING")
print("=" * 55)

# ─────────────────────────────────────────
# STEP 1: LOAD PROCESSED DATA
# ─────────────────────────────────────────
print("\n📂 STEP 1: Loading processed dataset...")

df = pd.read_csv('processed_data.csv')
print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────
# STEP 2: PREPARE FEATURES AND LABELS
# ─────────────────────────────────────────
print("\n🎯 STEP 2: Preparing Features and Labels...")

feature_cols = [
    'typing_speed',
    'study_hours',
    'break_count',
    'sleep_hours',
    'mood_score',
    'device_usage_hours',
    'task_completion',
    'distraction_count'
]

X = df[feature_cols]
y = df['burnout_label']

print(f"   Features: {feature_cols}")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   Classes: 0=Low, 1=Medium, 2=High")

# ─────────────────────────────────────────
# STEP 3: SPLIT DATA
# ─────────────────────────────────────────
print("\n✂️  STEP 3: Splitting Data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # Same split every time
    stratify=y           # Keep same class ratio in both splits
)

print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing  samples: {X_test.shape[0]}")
print("✅ Data split complete!")

# ─────────────────────────────────────────
# STEP 4: TRAIN LOGISTIC REGRESSION
# ─────────────────────────────────────────
print("\n🤖 STEP 4: Training Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Calculate metrics
lr_accuracy  = accuracy_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predictions, average='weighted')
lr_recall    = recall_score(y_test, lr_predictions, average='weighted')
lr_f1        = f1_score(y_test, lr_predictions, average='weighted')

# Cross validation score
lr_cv = cross_val_score(lr_model, X, y, cv=5, scoring='accuracy')

print(f"   ✅ Logistic Regression Results:")
print(f"      Accuracy  : {lr_accuracy:.4f}  ({lr_accuracy*100:.2f}%)")
print(f"      Precision : {lr_precision:.4f}")
print(f"      Recall    : {lr_recall:.4f}")
print(f"      F1-Score  : {lr_f1:.4f}")
print(f"      CV Score  : {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

# ─────────────────────────────────────────
# STEP 5: TRAIN RANDOM FOREST
# ─────────────────────────────────────────
print("\n🌲 STEP 5: Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,     # Number of trees in the forest
    max_depth=10,         # Maximum depth of each tree
    random_state=42,
    min_samples_split=5,  # Min samples to split a node
    min_samples_leaf=2    # Min samples at leaf node
)

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Calculate metrics
rf_accuracy  = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions, average='weighted')
rf_recall    = recall_score(y_test, rf_predictions, average='weighted')
rf_f1        = f1_score(y_test, rf_predictions, average='weighted')

# Cross validation score
rf_cv = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')

print(f"   ✅ Random Forest Results:")
print(f"      Accuracy  : {rf_accuracy:.4f}  ({rf_accuracy*100:.2f}%)")
print(f"      Precision : {rf_precision:.4f}")
print(f"      Recall    : {rf_recall:.4f}")
print(f"      F1-Score  : {rf_f1:.4f}")
print(f"      CV Score  : {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

# ─────────────────────────────────────────
# STEP 6: MODEL COMPARISON
# ─────────────────────────────────────────
print("\n📊 STEP 6: Model Comparison...")
print()
print(f"{'Metric':<15} {'Logistic Reg':>15} {'Random Forest':>15}")
print("-" * 47)
print(f"{'Accuracy':<15} {lr_accuracy*100:>14.2f}% {rf_accuracy*100:>14.2f}%")
print(f"{'Precision':<15} {lr_precision:>15.4f} {rf_precision:>15.4f}")
print(f"{'Recall':<15} {lr_recall:>15.4f} {rf_recall:>15.4f}")
print(f"{'F1-Score':<15} {lr_f1:>15.4f} {rf_f1:>15.4f}")
print(f"{'CV Score':<15} {lr_cv.mean():>15.4f} {rf_cv.mean():>15.4f}")
print("-" * 47)

# Determine best model
if rf_accuracy >= lr_accuracy:
    best_model = rf_model
    best_name  = "Random Forest"
    best_acc   = rf_accuracy
else:
    best_model = lr_model
    best_name  = "Logistic Regression"
    best_acc   = lr_accuracy

print(f"\n🏆 Best Model: {best_name} ({best_acc*100:.2f}% accuracy)")

# ─────────────────────────────────────────
# STEP 7: CLASSIFICATION REPORT
# ─────────────────────────────────────────
print("\n📋 STEP 7: Detailed Classification Report (Random Forest):")
print()
target_names = ['Low Risk', 'Medium Risk', 'High Risk']
print(classification_report(y_test, rf_predictions, target_names=target_names))

# ─────────────────────────────────────────
# STEP 8: SAVE CHARTS
# ─────────────────────────────────────────
print("\n📈 STEP 8: Generating ML Charts...")
os.makedirs('static/images', exist_ok=True)
os.makedirs('models', exist_ok=True)
sns.set_style("whitegrid")

# --- Chart 1: Confusion Matrix - Logistic Regression ---
plt.figure(figsize=(8, 6))
cm_lr = confusion_matrix(y_test, lr_predictions)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title('Confusion Matrix - Logistic Regression',
          fontsize=14, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('static/images/cm_logistic.png', dpi=100)
plt.close()
print("   ✅ Saved: cm_logistic.png")

# --- Chart 2: Confusion Matrix - Random Forest ---
plt.figure(figsize=(8, 6))
cm_rf = confusion_matrix(y_test, rf_predictions)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title('Confusion Matrix - Random Forest',
          fontsize=14, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('static/images/cm_random_forest.png', dpi=100)
plt.close()
print("   ✅ Saved: cm_random_forest.png")

# --- Chart 3: Model Comparison Bar Chart ---
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
lr_scores = [lr_accuracy, lr_precision, lr_recall, lr_f1]
rf_scores = [rf_accuracy, rf_precision, rf_recall, rf_f1]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width/2, lr_scores, width,
                label='Logistic Regression', color='#3498db', edgecolor='black')
bars2 = plt.bar(x + width/2, rf_scores, width,
                label='Random Forest', color='#2ecc71', edgecolor='black')

plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Comparison: Logistic Regression vs Random Forest',
          fontsize=14, fontweight='bold')
plt.xticks(x, metrics)
plt.ylim(0, 1.1)
plt.legend()

# Add value labels on bars
for bar in bars1:
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('static/images/model_comparison.png', dpi=100)
plt.close()
print("   ✅ Saved: model_comparison.png")

# --- Chart 4: Feature Importance (Random Forest) ---
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=feature_cols
).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feature_importance)))
feature_importance.plot(kind='barh', color=colors, edgecolor='black')
plt.title('Feature Importance - Random Forest',
          fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig('static/images/feature_importance.png', dpi=100)
plt.close()
print("   ✅ Saved: feature_importance.png")

# ─────────────────────────────────────────
# STEP 9: SAVE MODELS
# ─────────────────────────────────────────
print("\n💾 STEP 9: Saving Trained Models...")

joblib.dump(lr_model, 'models/logistic_model.pkl')
print("   ✅ Saved: models/logistic_model.pkl")

joblib.dump(rf_model, 'models/random_forest_model.pkl')
print("   ✅ Saved: models/random_forest_model.pkl")

joblib.dump(best_model, 'models/best_model.pkl')
print(f"   ✅ Saved: models/best_model.pkl ({best_name})")

# Save scaler for use in Flask app
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
joblib.dump(scaler, 'models/scaler.pkl')
print("   ✅ Saved: models/scaler.pkl")

# Save feature columns list
import json
with open('models/features.json', 'w') as f:
    json.dump(feature_cols, f)
print("   ✅ Saved: models/features.json")

print("\n" + "=" * 55)
print("✅ MACHINE LEARNING TRAINING COMPLETE!")
print("=" * 55)
print(f"\n🏆 Best Model  : {best_name}")
print(f"📈 Accuracy    : {best_acc*100:.2f}%")
print(f"📁 Models saved in: models/")
print(f"📊 Charts saved in: static/images/")
print("\n▶️  Ready for Phase 6: Web Application!")