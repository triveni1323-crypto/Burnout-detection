# data/generate_data.py
# This script generates synthetic student behavioral data
# for training our burnout detection ML model

import pandas as pd
import numpy as np
import os

# Set random seed so we get same data every time we run
np.random.seed(42)

# Number of student records to generate
NUM_SAMPLES = 1000

def generate_student_data():
    """
    Generate synthetic student behavioral data.
    Each row = one student's data for one day.
    """

    data = []

    # Generate 1000 records: 400 Low, 350 Medium, 250 High
    # Real burnout data is imbalanced - more low risk students
    distributions = [
        ('Low',    400),
        ('Medium', 350),
        ('High',   250)
    ]

    for risk_level, count in distributions:

        for _ in range(count):

            if risk_level == 'Low':
                record = {
                    'typing_speed':      np.random.uniform(60, 85),
                    'study_hours':       np.random.uniform(3, 6),
                    'break_count':       np.random.randint(4, 8),
                    'sleep_hours':       np.random.uniform(7, 9),
                    'mood_score':        np.random.randint(7, 11),
                    'device_usage_hours':np.random.uniform(3, 6),
                    'task_completion':   np.random.uniform(75, 100),
                    'distraction_count': np.random.randint(0, 4),
                    'burnout_risk':      'Low'
                }

            elif risk_level == 'Medium':
                record = {
                    'typing_speed':      np.random.uniform(40, 62),
                    'study_hours':       np.random.uniform(6, 9),
                    'break_count':       np.random.randint(2, 5),
                    'sleep_hours':       np.random.uniform(5, 7),
                    'mood_score':        np.random.randint(4, 8),
                    'device_usage_hours':np.random.uniform(6, 9),
                    'task_completion':   np.random.uniform(50, 76),
                    'distraction_count': np.random.randint(3, 8),
                    'burnout_risk':      'Medium'
                }

            else:  # High
                record = {
                    'typing_speed':      np.random.uniform(10, 42),
                    'study_hours':       np.random.uniform(9, 14),
                    'break_count':       np.random.randint(0, 3),
                    'sleep_hours':       np.random.uniform(2, 5),
                    'mood_score':        np.random.randint(1, 5),
                    'device_usage_hours':np.random.uniform(9, 14),
                    'task_completion':   np.random.uniform(10, 51),
                    'distraction_count': np.random.randint(7, 16),
                    'burnout_risk':      'High'
                }

            data.append(record)

    # Convert list to DataFrame
    df = pd.DataFrame(data)

    # Round float values to 2 decimal places
    float_cols = ['typing_speed', 'study_hours', 'sleep_hours',
                  'device_usage_hours', 'task_completion']
    df[float_cols] = df[float_cols].round(2)

    # Shuffle the data randomly
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add student ID column
    df.insert(0, 'student_id', range(1, len(df) + 1))

    return df


def save_data(df):
    """Save generated data to CSV files"""

    # Save raw data
    raw_path = 'student_data.csv'
    
    df.to_csv(raw_path, index=False)
    print(f"✅ Raw data saved: {raw_path}")
    print(f"   Total records: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print()

    # Show distribution
    print("📊 Burnout Risk Distribution:")
    print(df['burnout_risk'].value_counts())
    print()

    # Show first 5 rows
    print("👀 Sample Data (first 5 rows):")
    print(df.head())


if __name__ == '__main__':
    print("🔄 Generating student burnout dataset...")
    print()
    df = generate_student_data()
    save_data(df)
    print()
    print("✅ Dataset generation complete!")