# preprocess.py
# This file handles all data cleaning, normalization,
# feature engineering and exploratory data analysis (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

print("=" * 50)
print("   BURNOUT DETECTION - DATA PREPROCESSING")
print("=" * 50)

# ─────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────
print("\n📂 STEP 1: Loading dataset...")

df = pd.read_csv('student_data.csv')

print(f"✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")

# ─────────────────────────────────────────
# STEP 2: BASIC EXPLORATION
# ─────────────────────────────────────────
print("\n📊 STEP 2: Basic Exploration...")
print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nBasic Statistics:")
print(df.describe())

# ─────────────────────────────────────────
# STEP 3: CHECK MISSING VALUES
# ─────────────────────────────────────────
print("\n🔍 STEP 3: Checking Missing Values...")
missing = df.isnull().sum()
print(missing)

if missing.sum() == 0:
    print("✅ No missing values found!")
else:
    print("⚠️ Missing values found - filling with median...")
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    print("✅ Missing values handled!")

# ─────────────────────────────────────────
# STEP 4: CHECK DUPLICATES
# ─────────────────────────────────────────
print("\n🔍 STEP 4: Checking Duplicates...")
duplicates = df.duplicated().sum()
print(f"   Duplicate rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print(f"✅ Removed {duplicates} duplicate rows!")
else:
    print("✅ No duplicates found!")

# ─────────────────────────────────────────
# STEP 5: ENCODE TARGET LABEL
# ─────────────────────────────────────────
print("\n🔄 STEP 5: Encoding Target Labels...")

# Convert Low/Medium/High to numbers
# Low=0, Medium=1, High=2
label_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['burnout_label'] = df['burnout_risk'].map(label_map)

print("   Label Mapping:")
print("   Low    → 0")
print("   Medium → 1")
print("   High   → 2")
print(f"\n   Distribution:")
print(df['burnout_risk'].value_counts())
print("✅ Labels encoded!")

# ─────────────────────────────────────────
# STEP 6: FEATURE SELECTION
# ─────────────────────────────────────────
print("\n🎯 STEP 6: Selecting Features...")

# These are our input features (X)
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

print(f"   Input Features (X): {feature_cols}")
print(f"   Target Label  (y): burnout_label")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print("✅ Features selected!")

# ─────────────────────────────────────────
# STEP 7: NORMALIZE FEATURES
# ─────────────────────────────────────────
print("\n📏 STEP 7: Normalizing Features (0 to 1 scale)...")

# MinMaxScaler converts all values to range 0-1
# This prevents features with large values (like typing_speed)
# from dominating features with small values (like break_count)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

print("   Before Normalization (typing_speed):")
print(f"   Min: {X['typing_speed'].min():.2f}  Max: {X['typing_speed'].max():.2f}")
print("   After Normalization (typing_speed):")
print(f"   Min: {X_scaled['typing_speed'].min():.2f}  Max: {X_scaled['typing_speed'].max():.2f}")
print("✅ Normalization complete!")

# ─────────────────────────────────────────
# STEP 8: SAVE PROCESSED DATA
# ─────────────────────────────────────────
print("\n💾 STEP 8: Saving Processed Data...")

# Combine scaled features with label
processed_df = X_scaled.copy()
processed_df['burnout_label'] = y.values
processed_df['burnout_risk'] = df['burnout_risk'].values

# Save to CSV
processed_df.to_csv('processed_data.csv', index=False)
print("✅ Processed data saved as: processed_data.csv")

# ─────────────────────────────────────────
# STEP 9: VISUALIZATIONS
# ─────────────────────────────────────────
print("\n📈 STEP 9: Generating Visualizations...")

# Create output folder for charts
os.makedirs('static/images', exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# --- Chart 1: Burnout Risk Distribution ---
plt.figure(figsize=(8, 5))
colors = ['#2ecc71', '#f39c12', '#e74c3c']
df['burnout_risk'].value_counts().plot(
    kind='bar',
    color=colors,
    edgecolor='black'
)
plt.title('Burnout Risk Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Risk Level', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('static/images/risk_distribution.png', dpi=100)
plt.close()
print("   ✅ Chart 1 saved: risk_distribution.png")

# --- Chart 2: Feature Correlation Heatmap ---
plt.figure(figsize=(10, 8))
correlation = df[feature_cols + ['burnout_label']].corr()
sns.heatmap(
    correlation,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    center=0,
    square=True
)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('static/images/correlation_heatmap.png', dpi=100)
plt.close()
print("   ✅ Chart 2 saved: correlation_heatmap.png")

# --- Chart 3: Study Hours vs Burnout Risk ---
plt.figure(figsize=(8, 5))
sns.boxplot(
    x='burnout_risk',
    y='study_hours',
    data=df,
    palette={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
    order=['Low', 'Medium', 'High']
)
plt.title('Study Hours vs Burnout Risk', fontsize=16, fontweight='bold')
plt.xlabel('Burnout Risk Level', fontsize=12)
plt.ylabel('Study Hours per Day', fontsize=12)
plt.tight_layout()
plt.savefig('static/images/study_hours_boxplot.png', dpi=100)
plt.close()
print("   ✅ Chart 3 saved: study_hours_boxplot.png")

# --- Chart 4: Sleep Hours vs Burnout Risk ---
plt.figure(figsize=(8, 5))
sns.boxplot(
    x='burnout_risk',
    y='sleep_hours',
    data=df,
    palette={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
    order=['Low', 'Medium', 'High']
)
plt.title('Sleep Hours vs Burnout Risk', fontsize=16, fontweight='bold')
plt.xlabel('Burnout Risk Level', fontsize=12)
plt.ylabel('Sleep Hours per Night', fontsize=12)
plt.tight_layout()
plt.savefig('static/images/sleep_hours_boxplot.png', dpi=100)
plt.close()
print("   ✅ Chart 4 saved: sleep_hours_boxplot.png")

# --- Chart 5: Mood Score Distribution ---
plt.figure(figsize=(10, 5))
for risk, color in zip(['Low', 'Medium', 'High'],
                       ['#2ecc71', '#f39c12', '#e74c3c']):
    subset = df[df['burnout_risk'] == risk]['mood_score']
    plt.hist(subset, alpha=0.6, label=risk, color=color, bins=10, edgecolor='black')

plt.title('Mood Score Distribution by Risk Level', fontsize=16, fontweight='bold')
plt.xlabel('Mood Score (1-10)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(title='Risk Level')
plt.tight_layout()
plt.savefig('static/images/mood_distribution.png', dpi=100)
plt.close()
print("   ✅ Chart 5 saved: mood_distribution.png")

print("\n" + "=" * 50)
print("✅ ALL PREPROCESSING STEPS COMPLETE!")
print("=" * 50)
print("\nFiles Generated:")
print("  📄 processed_data.csv   - Clean normalized data")
print("  📊 static/images/risk_distribution.png")
print("  📊 static/images/correlation_heatmap.png")
print("  📊 static/images/study_hours_boxplot.png")
print("  📊 static/images/sleep_hours_boxplot.png")
print("  📊 static/images/mood_distribution.png")
print("\n▶️  Ready for Phase 5: Machine Learning!")