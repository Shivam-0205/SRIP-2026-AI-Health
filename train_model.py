#!/usr/bin/env python3
"""
Sleep Apnea Detection - Model Training Script
SRIP 2026 - AI for Health Internship
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse

def load_and_prepare_data(dataset_path):
    """Load dataset and prepare features"""
    df = pd.read_csv(dataset_path)
    
    # Focus on AP04 (only participant with events)
    df_ap04 = df[df['participant'] == 'AP04'].copy()
    
    # Features
    feature_cols = ['avg_spo2', 'min_spo2', 'max_spo2', 'std_spo2']
    X = df_ap04[feature_cols].values
    y = df_ap04['binary_label'].values  # 0=Normal, 1=Abnormal
    
    return X, y, df_ap04

def train_model(X, y, class_weight=None):
    """Train Random Forest model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    return model, X_test_scaled, y_test, scaler

def find_best_threshold(model, X_test, y_test):
    """Find optimal threshold for F1-score"""
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred_temp = (y_pred_prob >= thresh).astype(int)
        f1_temp = f1_score(y_test, y_pred_temp, zero_division=0)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_thresh = thresh
    
    return best_thresh, best_f1

def evaluate_model(model, X_test, y_test, threshold):
    """Evaluate model and return metrics"""
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_prob)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, 
                                   target_names=['Normal', 'Abnormal'],
                                   zero_division=0)
    
    return metrics, cm, report, y_pred_prob

def plot_results(cm, y_test, y_pred_prob, output_dir='models'):
    """Plot confusion matrix and ROC curve"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - AP04')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300)
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - AP04')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train sleep apnea detection model')
    parser.add_argument('-dataset', type=str, default='Dataset/breathing_dataset.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('-output', type=str, default='models',
                       help='Output directory for results')
    args = parser.parse_args()
    
    print("Loading data...")
    X, y, df = load_and_prepare_data(args.dataset)
    
    print(f"Total samples: {len(y)}")
    print(f"Class distribution: Normal={np.sum(y==0)}, Abnormal={np.sum(y==1)}")
    
    # Calculate class weight
    class_weight = {0: 1.0, 1: np.sum(y==0)/np.sum(y==1)}
    print(f"Class weights: Normal=1.0, Abnormal={class_weight[1]:.2f}")
    
    print("\nTraining Random Forest...")
    model, X_test, y_test, scaler = train_model(X, y, class_weight)
    
    print("\nFinding optimal threshold...")
    threshold, best_f1 = find_best_threshold(model, X_test, y_test)
    print(f"Best threshold: {threshold:.2f} (F1={best_f1:.3f})")
    
    print("\nEvaluating model...")
    metrics, cm, report, y_pred_prob = evaluate_model(model, X_test, y_test, threshold)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("\n" + report)
    print(f"Confusion Matrix:\n{cm}")
    
    # Save results
    print("\nSaving results...")
    results = {
        'participant': 'AP04',
        'metrics': metrics,
        'threshold': threshold,
        'test_samples': len(y_test),
        'abnormal_in_test': int(np.sum(y_test==1))
    }
    
    with open(f'{args.output}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    plot_results(cm, y_test, y_pred_prob, args.output)
    
    print(f"\nResults saved to {args.output}/")
    print("   - results.json")
    print("   - confusion_matrix.png")
    print("   - roc_curve.png")

if __name__ == "__main__":
    main()
