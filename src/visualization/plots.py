"""
Visualization utilities for fraud detection
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 30,
    save_path: str = None
):
    """Plot feature importance"""
    plt.figure(figsize=(12, 10))

    top_features = importance_df.head(top_n)

    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, save_path: str = None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path: str = None):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y_true, y_pred_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, save_path: str = None):
    """Plot correlation heatmap"""
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
    plt.title('Feature Correlation Matrix')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_distribution(df: pd.DataFrame, column: str, hue: str = None, save_path: str = None):
    """Plot distribution"""
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x=column, hue=hue, bins=50, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_fraud_rate_by_category(
    df: pd.DataFrame,
    category_col: str,
    target_col: str = 'is_fraud',
    save_path: str = None
):
    """Plot fraud rate by category"""
    fraud_rate = df.groupby(category_col)[target_col].agg(['mean', 'count'])
    fraud_rate = fraud_rate.sort_values('mean', ascending=False).head(20)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    x = range(len(fraud_rate))
    ax1.bar(x, fraud_rate['mean'], alpha=0.7, label='Fraud Rate')
    ax1.set_xlabel(category_col)
    ax1.set_ylabel('Fraud Rate', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    plt.xticks(x, fraud_rate.index, rotation=45, ha='right')

    ax2 = ax1.twinx()
    ax2.plot(x, fraud_rate['count'], 'r-o', label='Count')
    ax2.set_ylabel('Count', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title(f'Fraud Rate by {category_col}')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_eda_plots(df: pd.DataFrame, output_dir: str = "plots/eda"):
    """Create comprehensive EDA plots"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Amount distribution
    plot_distribution(
        df.sample(n=min(100000, len(df))),
        'trx_amt',
        hue='is_fraud' if 'is_fraud' in df.columns else None,
        save_path=f"{output_dir}/amount_distribution.png"
    )

    # Fraud rate by channel
    if 'trx_channel' in df.columns and 'is_fraud' in df.columns:
        plot_fraud_rate_by_category(
            df,
            'trx_channel',
            save_path=f"{output_dir}/fraud_rate_by_channel.png"
        )

    # Fraud rate by transaction type
    if 'trx_type' in df.columns and 'is_fraud' in df.columns:
        plot_fraud_rate_by_category(
            df,
            'trx_type',
            save_path=f"{output_dir}/fraud_rate_by_type.png"
        )
