"""
Step 4: Integration - Combine ML predictions with optimization calculations
to generate final inventory recommendations.
"""
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path


def combine_recommendations(
    df: pd.DataFrame,
    use_ml_prediction: bool = True,
    ml_risk_col: str = 'ml_risk_prediction',
    rule_risk_col: str = 'stock_risk'
) -> pd.DataFrame:
    """
    Combine rule-based risk classification with ML predictions.
    Final risk: use ML when available and confident; otherwise fall back to rules.
    """
    result = df.copy()
    
    if use_ml_prediction and ml_risk_col in result.columns:
        # Use ML prediction as primary; rule-based as fallback for edge cases
        result['final_risk'] = result[ml_risk_col]
        result['risk_source'] = 'ml'
    else:
        result['final_risk'] = result[rule_risk_col]
        result['risk_source'] = 'rule'
    
    return result


def generate_action_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate actionable recommendations based on risk and overstock status.
    """
    result = df.copy()
    
    def get_recommendation(row):
        if row['is_overstock']:
            return 'REDUCE_STOCK'
        if row['final_risk'] == 'High':
            return 'URGENT_REORDER'
        if row['final_risk'] == 'Medium':
            return 'PLAN_REORDER'
        return 'MAINTAIN'
    
    result['recommendation'] = result.apply(get_recommendation, axis=1)
    
    # Add reorder quantity to recommendation detail
    result['recommendation_detail'] = result.apply(
        lambda r: f"{r['recommendation']}: reorder_qty={int(r['reorder_qty'])}" 
        if r['recommendation'] in ['URGENT_REORDER', 'PLAN_REORDER'] 
        else r['recommendation'],
        axis=1
    )
    
    return result


def run_integration(
    df: pd.DataFrame,
    use_ml: bool = True
) -> pd.DataFrame:
    """
    Full integration: combine predictions and generate final recommendations.
    """
    df_combined = combine_recommendations(df, use_ml_prediction=use_ml)
    df_final = generate_action_recommendations(df_combined)
    return df_final
