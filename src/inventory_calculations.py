"""
Step 2: Core Inventory Optimization Calculations
Implements safety stock, reorder point, reorder quantity, risk classification, and overstock detection.
"""
import pandas as pd
import numpy as np
from typing import Optional

# Default parameters
Z_SCORE = 1.65  # ~90% service level
LEAD_TIME_DAYS = 7


def compute_average_demand(units_sold: pd.Series, groupby: Optional[list] = None) -> pd.Series:
    """Average demand = mean(units_sold). Can be grouped by product/store."""
    if groupby:
        return units_sold.groupby(groupby).transform('mean')
    return pd.Series([units_sold.mean()] * len(units_sold), index=units_sold.index)


def compute_demand_std(units_sold: pd.Series, groupby: Optional[list] = None) -> pd.Series:
    """Demand variability = standard deviation of units_sold."""
    if groupby:
        return units_sold.groupby(groupby).transform('std').fillna(units_sold.std())
    return pd.Series([units_sold.std()] * len(units_sold), index=units_sold.index)


def compute_safety_stock(
    demand_std: pd.Series,
    z: float = Z_SCORE,
    lead_time: int = LEAD_TIME_DAYS
) -> pd.Series:
    """Safety stock = Z * demand_std * sqrt(lead_time)"""
    return z * demand_std * np.sqrt(lead_time)


def compute_reorder_point(
    average_demand: pd.Series,
    safety_stock: pd.Series,
    lead_time: int = LEAD_TIME_DAYS
) -> pd.Series:
    """Reorder point (ROP) = (average_demand * lead_time) + safety_stock"""
    return (average_demand * lead_time) + safety_stock


def compute_reorder_quantity(
    forecast_demand: pd.Series,
    current_inventory: pd.Series
) -> pd.Series:
    """Reorder quantity = forecast_demand - current_inventory. If negative → 0."""
    roq = forecast_demand - current_inventory
    return roq.clip(lower=0)


def classify_stock_risk(
    inventory: pd.Series,
    forecast_demand: pd.Series,
    reorder_point: pd.Series
) -> pd.Series:
    """
    Stock risk classification:
    - High: inventory < forecast demand
    - Medium: inventory < reorder point (but >= forecast)
    - Low: otherwise
    """
    risk = np.where(
        inventory < forecast_demand,
        'High',
        np.where(inventory < reorder_point, 'Medium', 'Low')
    )
    return pd.Series(risk, index=inventory.index)


def detect_overstock(
    inventory: pd.Series,
    forecast_demand: pd.Series,
    threshold: float = 1.5
) -> pd.Series:
    """Overstock if inventory > threshold * forecast_demand (default 1.5x)."""
    return inventory > (threshold * forecast_demand)


def run_inventory_calculations(
    df: pd.DataFrame,
    z: float = Z_SCORE,
    lead_time: int = LEAD_TIME_DAYS,
    groupby: Optional[list] = None
) -> pd.DataFrame:
    """
    Run all inventory optimization calculations on the dataset.
    Returns dataframe with added columns.
    """
    result = df.copy()
    
    # Groupby for product-level or store-product level stats (optional)
    group_cols = groupby if groupby else []
    
    # 1. Average demand
    result['average_demand'] = compute_average_demand(
        result['units_sold'], group_cols if group_cols else None
    )
    
    # 2. Demand variability
    result['demand_std'] = compute_demand_std(
        result['units_sold'], group_cols if group_cols else None
    )
    
    # 3. Safety stock
    result['safety_stock'] = compute_safety_stock(
        result['demand_std'], z=z, lead_time=lead_time
    )
    
    # 4. Reorder point
    result['reorder_point'] = compute_reorder_point(
        result['average_demand'], result['safety_stock'], lead_time
    )
    
    # 5. Reorder quantity
    result['reorder_qty'] = compute_reorder_quantity(
        result['demand_forecast'], result['inventory_level']
    )
    
    # 6. Stock risk classification
    result['stock_risk'] = classify_stock_risk(
        result['inventory_level'],
        result['demand_forecast'],
        result['reorder_point']
    )
    
    # 7. Overstock detection
    result['is_overstock'] = detect_overstock(
        result['inventory_level'], result['demand_forecast']
    )
    
    return result
