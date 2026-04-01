"""
Step 1: Data Exploration for Inventory Optimization
Load and explore the dataset to understand demand patterns, variability, and influencing factors.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# Column mapping for retail_store_inventory.csv and similar formats
RETAIL_COLUMN_MAP = {
    'Date': 'date',
    'Store ID': 'store_id',
    'Product ID': 'product_id',
    'Region': 'region',
    'Inventory Level': 'inventory_level',
    'Units Sold': 'units_sold',
    'Demand Forecast': 'demand_forecast',
    'Price': 'price',
    'Discount': 'discount_pct',  # 0-20 → convert to 0-0.2
    'Holiday/Promotion': 'is_promotion',
    'Seasonality': 'seasonality',  # categorical → factor
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map retail dataset columns to expected format."""
    df = df.copy()
    for old_name, new_name in RETAIL_COLUMN_MAP.items():
        if old_name in df.columns:
            df[new_name] = df[old_name]
    # Convert discount (0-20) to discount_pct (0-1)
    if 'discount_pct' in df.columns and df['discount_pct'].max() > 1:
        df['discount_pct'] = df['discount_pct'] / 100
    # Convert Seasonality to numeric factor
    if 'seasonality' in df.columns:
        season_map = {'Autumn': 1.0, 'Winter': 0.9, 'Spring': 1.1, 'Summer': 1.05}
        df['seasonality_factor'] = df['seasonality'].map(season_map).fillna(1.0)
    return df


def load_data(filepath: str, standardize: bool = True) -> pd.DataFrame:
    """Load inventory dataset from CSV. Auto-standardizes retail format columns."""
    df = pd.read_csv(filepath)
    if standardize and any(c in df.columns for c in RETAIL_COLUMN_MAP):
        df = standardize_columns(df)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df


def explore_dataset(df: pd.DataFrame) -> dict:
    """
    Explore the dataset and return summary statistics.
    Covers: inventory levels, units sold, demand forecast, pricing, seasonality, promotions, store/region variations.
    """
    summary = {}
    
    # Basic info
    summary['shape'] = df.shape
    summary['columns'] = list(df.columns)
    summary['dtypes'] = df.dtypes.astype(str).to_dict()
    summary['missing_values'] = df.isnull().sum().to_dict()
    
    # Inventory levels
    summary['inventory'] = {
        'mean': float(df['inventory_level'].mean()),
        'std': float(df['inventory_level'].std()),
        'min': int(df['inventory_level'].min()),
        'max': int(df['inventory_level'].max()),
        'median': float(df['inventory_level'].median()),
    }
    
    # Units sold (demand)
    summary['units_sold'] = {
        'mean': float(df['units_sold'].mean()),
        'std': float(df['units_sold'].std()),
        'min': int(df['units_sold'].min()),
        'max': int(df['units_sold'].max()),
        'median': float(df['units_sold'].median()),
    }
    
    # Demand forecast
    summary['demand_forecast'] = {
        'mean': float(df['demand_forecast'].mean()),
        'std': float(df['demand_forecast'].std()),
        'min': int(df['demand_forecast'].min()),
        'max': int(df['demand_forecast'].max()),
    }
    
    # Pricing & discounts
    if 'price' in df.columns:
        summary['price'] = {
            'mean': float(df['price'].mean()),
            'std': float(df['price'].std()),
            'min': float(df['price'].min()),
            'max': float(df['price'].max()),
        }
    if 'discount_pct' in df.columns:
        summary['discount'] = {
            'mean': float(df['discount_pct'].mean()),
            'max': float(df['discount_pct'].max()),
            'promo_rate': float(df['discount_pct'].gt(0).mean()),
        }
    
    # Seasonality & promotions
    if 'seasonality_factor' in df.columns:
        summary['seasonality'] = {
            'mean': float(df['seasonality_factor'].mean()),
            'std': float(df['seasonality_factor'].std()),
            'min': float(df['seasonality_factor'].min()),
            'max': float(df['seasonality_factor'].max()),
        }
    if 'is_promotion' in df.columns:
        summary['promotions'] = {
            'promo_pct': float(df['is_promotion'].mean() * 100),
            'promo_count': int(df['is_promotion'].sum()),
        }
    
    # Store & region variations
    if 'store_id' in df.columns:
        summary['stores'] = {
            'n_stores': int(df['store_id'].nunique()),
            'top_stores_by_volume': df.groupby('store_id')['units_sold'].sum().nlargest(5).to_dict(),
        }
    if 'region' in df.columns:
        summary['regions'] = {
            'n_regions': int(df['region'].nunique()),
            'region_demand': df.groupby('region')['units_sold'].mean().to_dict(),
        }
    
    # Demand variability (coefficient of variation)
    cv = df['units_sold'].std() / df['units_sold'].mean() if df['units_sold'].mean() > 0 else 0
    summary['demand_variability'] = {'cv': float(cv)}
    
    return summary


def get_demand_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze demand patterns by product, store, and time."""
    patterns = []
    
    if 'date' in df.columns:
        df_temp = df.copy()
        df_temp['month'] = df_temp['date'].dt.month
        monthly = df_temp.groupby('month')['units_sold'].agg(['mean', 'std', 'count'])
        patterns.append(('monthly', monthly))
    
    if 'product_id' in df.columns:
        product_demand = df.groupby('product_id')['units_sold'].agg(['mean', 'std', 'count'])
        patterns.append(('by_product', product_demand))
    
    return patterns


def run_exploration(filepath: str) -> tuple[pd.DataFrame, dict]:
    """
    Full exploration pipeline.
    Returns (dataframe, exploration_summary).
    """
    df = load_data(filepath)
    summary = explore_dataset(df)
    return df, summary
