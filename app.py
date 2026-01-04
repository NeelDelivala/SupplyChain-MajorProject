from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import os
import pandas as pd
from werkzeug.utils import secure_filename
import traceback
from functools import wraps
import random
from datetime import datetime, timedelta

# Import user database only
from models.user_db import create_user, verify_user, get_user_by_username, init_db

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supply-chain-secret-key-change-in-production'

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure required directories exist
for folder in ['uploads', 'results', 'results/forecasts', 
               'results/inventory_policies', 'results/optimized_routes',
               'templates', 'static/css', 'static/js', 'models']:
    os.makedirs(folder, exist_ok=True)

# Initialize database
init_db()

def create_default_demo_user():
    """Create a default demo user for testing"""
    try:
        success, message = create_user(
            username="demo",
            email="demo@supply-chain.com",
            password="demo123",
            full_name="Demo User",
            organization="Supply Chain Demo"
        )
        if success:
            print("    Default demo user created")
            print("    Username: demo | Password: demo123")
    except:
        pass

# Call it on startup
create_default_demo_user()


ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        success, user_data = verify_user(username, password)

        if success:
            session['user'] = user_data
            flash(f'Welcome back, {user_data["username"]}!', 'success')
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')

    if 'user' in session:
        return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        full_name = request.form.get('full_name')
        organization = request.form.get('organization')

        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')

        if len(password) < 6:
            return render_template('register.html', error='Password must be at least 6 characters')

        success, message = create_user(username, email, password, full_name, organization)

        if success:
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            return render_template('register.html', error=message)

    if 'user' in session:
        return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

# ============================================================================
# WEB PAGES (Routes) - Protected
# ============================================================================

@app.route('/')
@login_required
def index():
    return render_template('index.html', user=session.get('user'))

@app.route('/demand-forecasting')
@login_required
def demand_forecasting():
    return render_template('demand_forecasting.html', user=session.get('user'))

@app.route('/inventory-optimization')
@login_required
def inventory_optimization_page():
    return render_template('inventory_optimization.html', user=session.get('user'))

@app.route('/logistics-optimization')
@login_required
def logistics_optimization_page():
    return render_template('logistics_optimization.html', user=session.get('user'))

# ============================================================================
# MVP - MOCK RESULTS FUNCTIONS
# ============================================================================

def generate_mock_demand_forecast(df, forecast_horizon=30):
    """Generate mock demand forecasting results"""

    # Basic stats from data
    if 'Units_Sold' in df.columns:
        mean_demand = df['Units_Sold'].mean()
        std_demand = df['Units_Sold'].std()
    else:
        mean_demand = 100
        std_demand = 15

    # Generate forecast dates
    today = datetime.now()
    forecast_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') 
                      for i in range(1, forecast_horizon + 1)]

    # Generate realistic forecast values
    base_values = [mean_demand + random.gauss(0, std_demand * 0.3) 
                   for _ in range(forecast_horizon)]
    base_values = [max(0, v) for v in base_values]  # No negative values

    # Generate confidence intervals
    lower_bound = [v * 0.85 for v in base_values]
    upper_bound = [v * 1.15 for v in base_values]

    # Historical data (last 30 days)
    if 'Date' in df.columns and len(df) > 30:
        recent_df = df.tail(30)
        historical_dates = recent_df['Date'].astype(str).tolist()
        historical_values = recent_df['Units_Sold'].tolist() if 'Units_Sold' in df.columns else []
    else:
        historical_dates = [(today - timedelta(days=30-i)).strftime('%Y-%m-%d') 
                           for i in range(30)]
        historical_values = [mean_demand + random.gauss(0, std_demand) for _ in range(30)]

    return {
        'status': 'success',
        'model_type': 'ARIMA',
        'arima_order': (1, 1, 1),  # Mock order
        'data_info': {
            'rows': len(df),
            'columns': list(df.columns),
            'train_size': int(len(df) * 0.8),
            'test_size': int(len(df) * 0.2),
        },
        'stationarity': {
            'adf_statistic': -3.45,
            'p_value': 0.008,
            'is_stationary': True
        },
        'model_info': {
            'aic': 245.67,
            'bic': 258.34,
            'order': (1, 1, 1)
        },
        'metrics': {
            'mae': round(abs(mean_demand * 0.08), 2),
            'rmse': round(abs(mean_demand * 0.12), 2),
            'mape': round(random.uniform(5, 12), 2),
            'r2': round(random.uniform(0.85, 0.95), 3)
        },
        'forecast': {
            'horizon': forecast_horizon,
            'confidence_level': 95,
            'dates': forecast_dates,
            'values': [round(v, 2) for v in base_values],
            'lower_bound': [round(v, 2) for v in lower_bound],
            'upper_bound': [round(v, 2) for v in upper_bound],
            'mean_forecast': round(sum(base_values) / len(base_values), 2),
            'total_forecast': round(sum(base_values), 2)
        },
        'historical': {
            'dates': historical_dates[-30:],
            'values': [round(v, 2) for v in historical_values[-30:]]
        }
    }

def generate_mock_inventory_results(df):
    """Generate mock inventory optimization results"""

    # Get unique products
    if 'Product_ID' in df.columns:
        products = df['Product_ID'].unique()[:5]  # First 5 products
    else:
        products = [f'Product_{i}' for i in range(1, 6)]

    product_results = []
    total_current = 0
    total_optimized = 0

    for product in products:
        mean_demand = random.uniform(20, 100)
        current_cost = random.uniform(10000, 50000)
        eoq = round(random.uniform(100, 500), 2)
        safety_stock = round(random.uniform(50, 150), 2)
        rop = round(random.uniform(150, 350), 2)
        optimized_cost = current_cost * random.uniform(0.7, 0.85)

        total_current += current_cost
        total_optimized += optimized_cost

        product_results.append({
            'product_id': product,
            'demand_stats': {
                'mean_daily': round(mean_demand, 2),
                'std_daily': round(mean_demand * 0.15, 2),
                'annual': round(mean_demand * 365, 2)
            },
            'current_policy': {
                'stock_level': round(mean_demand * 30, 2),
                'total_cost': round(current_cost, 2)
            },
            'optimized_policy': {
                'eoq': eoq,
                'safety_stock': safety_stock,
                'reorder_point': rop,
                'lead_time_demand': round(mean_demand * 7, 2),
                'order_frequency_days': round(eoq / mean_demand, 2),
                'total_cost': round(optimized_cost, 2)
            },
            'cost_breakdown': {
                'ordering_cost': round(optimized_cost * 0.3, 2),
                'holding_cost': round(optimized_cost * 0.7, 2),
                'total_cost': round(optimized_cost, 2)
            },
            'improvements': {
                'cost_savings': round(current_cost - optimized_cost, 2),
                'savings_percent': round((1 - optimized_cost/current_cost) * 100, 2)
            }
        })

    return {
        'status': 'success',
        'rows': len(df),
        'columns': list(df.columns),
        'preview': df.head(5).to_dict('records'),
        'parameters': {
            'service_level': 95,
            'holding_cost': 2.0,
            'ordering_cost': 100.0,
            'stockout_penalty': 50.0
        },
        'improvements': {
            'cost_reduction': round((1 - total_optimized/total_current) * 100, 2),
            'total_savings': round(total_current - total_optimized, 2),
            'stockout_reduction': round(random.uniform(30, 45), 2),
            'service_level_achieved': 95
        },
        'optimization_results': {
            'products_optimized': len(product_results),
            'summary': {
                'total_current_cost': round(total_current, 2),
                'total_optimized_cost': round(total_optimized, 2),
                'total_savings': round(total_current - total_optimized, 2),
                'avg_savings_percent': round((1 - total_optimized/total_current) * 100, 2)
            },
            'product_results': product_results
        }
    }

def generate_mock_logistics_results(df):
    """Generate mock logistics optimization results"""

    n_routes = random.randint(3, 8)
    total_distance = random.uniform(500, 1500)
    cost_per_km = 2.5

    routes_detail = []
    for i in range(n_routes):
        n_stops = random.randint(3, 8)
        route = ['Depot'] + [f'Location_{j}' for j in range(n_stops)] + ['Depot']
        routes_detail.append(route)

    # Carrier recommendations if available
    carrier_recs = {}
    if 'delivery_partner' in df.columns:
        carriers = df['delivery_partner'].unique()[:3]
        for carrier in carriers:
            carrier_recs[carrier] = {
                'delivery_cost': round(random.uniform(150, 400), 2),
                'distance_km': round(random.uniform(30, 120), 2),
                'delayed': random.randint(0, 5)
            }

    return {
        'status': 'success',
        'rows': len(df),
        'columns': list(df.columns),
        'preview': df.head(5).to_dict('records'),
        'parameters': {
            'optimization_goal': 'cost',
            'vehicle_capacity': 1000,
            'max_distance': 200,
            'n_iterations': 100
        },
        'improvements': {
            'cost_savings': round(random.uniform(18, 28), 2),
            'distance_reduction': round(random.uniform(15, 25), 2),
            'delay_reduction': round(random.uniform(30, 40), 2),
            'efficiency_improvement': round(random.uniform(20, 30), 2)
        },
        'routes': {
            'n_routes': n_routes,
            'total_distance': round(total_distance, 2),
            'avg_distance': round(total_distance / n_routes, 2),
            'total_cost': round(total_distance * cost_per_km, 2),
            'routes_detail': routes_detail[:5]  # First 5 routes
        },
        'carrier_recommendations': carrier_recs,
        'convergence': {
            'iterations': 100,
            'initial_distance': round(total_distance * 1.3, 2),
            'final_distance': round(total_distance, 2),
            'improvement': round(random.uniform(20, 30), 2)
        }
    }

# ============================================================================
# API ENDPOINTS - Protected (MVP with Mock Results)
# ============================================================================

@app.route('/api/forecast', methods=['POST'])
@login_required
def forecast():
    """Demand Forecasting - MVP Version (Mock Results)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Read data
            df = pd.read_csv(filepath)

            forecast_horizon = int(request.form.get('forecast_horizon', 30))

            print(f"\n{'='*60}")
            print(f" DEMAND FORECASTING (MVP) - User: {session['user']['username']}")
            print(f"{'='*60}")
            print(f"File: {filename}, Rows: {len(df)}")
            print(f"Columns: {', '.join(df.columns.tolist())}")
            print(f"Forecast Horizon: {forecast_horizon} days")

            # Generate mock results
            results = generate_mock_demand_forecast(df, forecast_horizon)

            # Save results
            results_file = os.path.join(app.config['RESULTS_FOLDER'], 
                                       'forecasts', 
                                       f'{session["user"]["username"]}_{filename}')

            forecast_df = pd.DataFrame({
                'Date': results['forecast']['dates'],
                'Forecast': results['forecast']['values'],
                'Lower_Bound': results['forecast']['lower_bound'],
                'Upper_Bound': results['forecast']['upper_bound']
            })
            forecast_df.to_csv(results_file, index=False)

            print(f" Mock forecast complete! Saved to {results_file}")
            print(f"   MAPE: {results['metrics']['mape']}%")
            print(f"   Mean Forecast: {results['forecast']['mean_forecast']}")
            print(f"{'='*60}\n")

            return jsonify(results)

        return jsonify({'error': 'Invalid file type. Please upload CSV'}), 400

    except Exception as e:
        print(f" ERROR: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize-inventory', methods=['POST'])
@login_required
def optimize_inventory_route():
    """Inventory Optimization - MVP Version (Mock Results)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            print(f"\n{'='*60}")
            print(f" INVENTORY OPTIMIZATION (MVP) - User: {session['user']['username']}")
            print(f"{'='*60}")
            print(f"File: {filename}, Rows: {len(df)}")
            print(f"Columns: {', '.join(df.columns.tolist())}")

            # Generate mock results
            results = generate_mock_inventory_results(df)

            # Save results
            results_file = os.path.join(app.config['RESULTS_FOLDER'], 
                                       'inventory_policies', 
                                       f'{session["user"]["username"]}_{filename}')

            pd.DataFrame(results['optimization_results']['product_results']).to_csv(
                results_file, index=False)

            print(f" Mock optimization complete! Saved to {results_file}")
            print(f"   Cost Reduction: {results['improvements']['cost_reduction']}%")
            print(f"   Total Savings: ${results['improvements']['total_savings']}")
            print(f"{'='*60}\n")

            return jsonify(results)

        return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        print(f" ERROR: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize-logistics', methods=['POST'])
@login_required
def optimize_logistics_route():
    """Logistics Optimization - MVP Version (Mock Results)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            print(f"\n{'='*60}")
            print(f" LOGISTICS OPTIMIZATION (MVP) - User: {session['user']['username']}")
            print(f"{'='*60}")
            print(f"File: {filename}, Rows: {len(df)}")
            print(f"Columns: {', '.join(df.columns.tolist())}")

            # Generate mock results
            results = generate_mock_logistics_results(df)

            # Save results
            results_file = os.path.join(app.config['RESULTS_FOLDER'], 
                                       'optimized_routes', 
                                       f'{session["user"]["username"]}_{filename}')

            pd.DataFrame(results['routes']['routes_detail']).to_csv(
                results_file, index=False)

            print(f" Mock optimization complete! Saved to {results_file}")
            print(f"   Routes Created: {results['routes']['n_routes']}")
            print(f"   Total Distance: {results['routes']['total_distance']} km")
            print(f"   Cost Savings: {results['improvements']['cost_savings']}%")
            print(f"{'='*60}\n")

            return jsonify(results)

        return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        print(f" ERROR: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print(" SUPPLY CHAIN OPTIMIZATION PLATFORM - MVP VERSION")
    print("="*70)
    print()
    print("  MVP MODE: Using simulated results for demonstration")
    print("   Real algorithms (ARIMA, EOQ/ROP, ACO) will be implemented later")
    print()
    print(" Authentication: ENABLED")
    print("   Login required for all modules")
    print()
    print("Available Modules:")
    print("    Demand Forecasting (Mock ARIMA results)")
    print("    Inventory Optimization (Mock EOQ/ROP results)")
    print("    Logistics Optimization (Mock ACO results)")
    print()
    print("Folders:")
    print(f"   Upload: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"   Results: {os.path.abspath(app.config['RESULTS_FOLDER'])}")
    print()
    print(" Server Information:")
    print("   URL: http://127.0.0.1:5000")
    print("   First-time users: Visit http://127.0.0.1:5000/register")
    print()
    print(" This is an MVP - results are simulated for demonstration")
    print("   Upload your CSV files and see the interface in action!")
    print()
    print(" Press Ctrl+C to stop the server")
    print("="*70)
    print()

    app.run(debug=True, host='127.0.0.1', port=5000)