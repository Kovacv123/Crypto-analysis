import math
import warnings
from flask import Flask, render_template, request, jsonify, send_file, Response
import os
import time
import datetime
from datetime import datetime
import datetime as dt
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from flask_sqlalchemy import SQLAlchemy
from binance.client import Client
from functools import wraps
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


#warnings.filterwarnings("ignore")
app = Flask(__name__)
#################################################OPTIONS FINDER#############################################

# Configure SQLite Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user_input.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize the Binance client
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret)

# Track the last email sent time
last_email_time = None

# Basic Authentication Configuration
USERNAME = 'dukihytien'
PASSWORD = 'password123'

def check_auth(username, password):
    """This function is called to check if a username/password combination is valid."""
    return username == USERNAME and password == PASSWORD

def authenticate():
    """Sends a 401 response that enables basic auth."""
    return Response(
        'Could not verify your login!\n'
        'Please provide valid credentials', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

def fetch_options(crypto):
    try:
        options = client.options_price()
        filtered_options = [opt for opt in options if opt['symbol'].startswith(crypto)]
        return filtered_options
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def extract_date_from_symbol(symbol):
    try:
        parts = symbol.split('-')
        date_str = parts[1]
        return datetime.strptime(date_str, '%y%m%d')
    except Exception as e:
        print(f"Error extracting date from symbol '{symbol}': {e}")
        return None

def filter_options(options, underlying, option_type, expiry_date, earliest_expiry_date, min_price, max_price, underlying_price, min_volume):
    filtered = []
    for option in options:
        strike_price = float(option['strikePrice'])
        expiration_date = extract_date_from_symbol(option['symbol'])
        if expiration_date is None or expiration_date < earliest_expiry_date or expiration_date > expiry_date:
            continue
        if option['symbol'].startswith(underlying) and option['symbol'].endswith(option_type):
            option['expiration_date'] = expiration_date
            if strike_price >= underlying_price - min_price and strike_price <= underlying_price + max_price:
                if float(option.get('volume', 0)) >= min_volume:
                    filtered.append(option)
    return filtered

def evaluate_options(options, option_type, underlying_price, percentage_threshold):
    results = []
    for option in options:
        strike_price = float(option['strikePrice'])
        ask_price = float(option.get('askPrice', 0))
        bid_price = float(option.get('bidPrice', 0))
        volume = float(option.get('volume', 0))

        if option_type == 'C':
            if underlying_price < strike_price:
                status = 'Out of the Money'
                true_price = None
                should_highlight = False
            else:
                status = 'In the Money'
                true_price = underlying_price - strike_price
                if ask_price != 0:
                    should_highlight = ask_price <= true_price * percentage_threshold
        else:
            if underlying_price > strike_price:
                status = 'Out of the Money'
                true_price = None
                should_highlight = False
            else:
                status = 'In the Money'
                true_price = strike_price - underlying_price
                if ask_price != 0:
                    should_highlight = ask_price <= true_price * percentage_threshold

        true_percentage = ask_price / true_price if should_highlight else None

        option_data = {
            'symbol': option['symbol'],
            'underlying_price': underlying_price,
            'strike_price': strike_price,
            'ask_price': ask_price,
            'bid_price': bid_price,
            'volume': volume,
            'status': status,
            'true_price': true_price,
            'highlight': should_highlight,
            'percentage_threshold': true_percentage,
        }
        results.append(option_data)

    return results

# Define the UserInput model
class UserInput(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    expiry_date = db.Column(db.String(20))
    crypto = db.Column(db.String(10))
    option_type = db.Column(db.String(1))
    percentage_threshold = db.Column(db.Float)
    min_price = db.Column(db.Float)
    max_price = db.Column(db.Float)
    min_volume = db.Column(db.Float)

# Create tables before the first request
@app.before_request
def create_tables():
    db.create_all()

@app.route('/')
@requires_auth
def home():
    return render_template('index.html')

@app.route('/page2')
def page2():
    return render_template('index_patterns.html')

@app.route('/evaluate', methods=['POST'])
@requires_auth
def evaluate():
    expiry_date_str = request.form['expiry_date']
    earliest_expiry_date_str = request.form['earliest_expiry_date']
    crypto = request.form['crypto']
    option_type = request.form['option_type']
    percentage_threshold = float(request.form['percentage_threshold'])
    min_price = float(request.form['min_price'])
    max_price = float(request.form['max_price'])
    min_volume = float(request.form['min_volume'])

    expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')
    earliest_expiry_date = datetime.strptime(earliest_expiry_date_str, '%Y-%m-%d')

    # Save user input to the database
    user_input = UserInput(
        expiry_date=expiry_date_str,
        crypto=crypto,
        option_type=option_type,
        percentage_threshold=percentage_threshold,
        min_price=min_price,
        max_price=max_price,
        min_volume=min_volume
    )
    db.session.add(user_input)
    db.session.commit()
    print("User input saved successfully.")

    options = fetch_options(crypto)
    underlying_price = float(client.futures_symbol_ticker(symbol=f'{crypto}USDT')['price'])
    filtered_options = filter_options(options, crypto, option_type, expiry_date, earliest_expiry_date, min_price, max_price, underlying_price, min_volume)
    options_data = evaluate_options(filtered_options, option_type, underlying_price, percentage_threshold)

    return jsonify(options_data)

#################################################OPTIONS FINDER#############################################

#################################################PATTERN FINDER#############################################

# Ensure the images are saved in a folder
UPLOAD_FOLDER = 'static/images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

day_counter = 0
total_days = 0
def get_dfs(freq_in_hours, days, cryptos):
    global day_counter, total_days
    day_counter = 0
    hour_counter = 0
    total_days = days
    url = 'https://api.binance.com/api/v3/klines'
    limit = 1000
    crypto_data = {}

    total_hours = int(days * 24 / freq_in_hours)  # Ensure total_hours is an integer

    for crypto in cryptos:
        all_data = []
        end_time = int(time.time() * 1000)  # Reset end_time for each crypto
        remaining_hours = total_hours  # Reset remaining_hours for each crypto

        while remaining_hours > 0:
            params = {
                'symbol': crypto,
                'interval': f'{freq_in_hours}h',
                'limit': min(limit, remaining_hours),  # Fetch remaining hours if less than limit
                'endTime': end_time
            }
            response = requests.get(url, params=params)
            if response.status_code != 200:
                raise Exception(f"Error fetching data for {crypto}: {response.text}")

            data = response.json()

            if not data:  # Stop if no data is returned
                break

            for price_point in data:
                timestamp = int(price_point[0]) / 1000
                readable_time = dt.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                price = float(price_point[4])
                all_data.append((readable_time, price))
                hour_counter += 1
                day_counter = hour_counter / 24

            end_time = int(data[0][0])  # Update end_time for next batch
            remaining_hours -= len(data)
            time.sleep(1)

        # Save the data for the crypto
        crypto_data[crypto] = pd.DataFrame(all_data, columns=['Date', 'Price']).sort_values(by='Date')

    eth = crypto_data.get('ETHUSDT', pd.DataFrame())
    btc = crypto_data.get('BTCUSDT', pd.DataFrame())

    return eth.reset_index(drop=True), btc.reset_index(drop=True)


def filter_indices(df, threshold=24):
    indices_to_keep = []
    b = True
    for ind in df.index:
        if not indices_to_keep:
            indices_to_keep.append(ind)
        else:
            b = True
            for itk in indices_to_keep:
                if abs(ind - itk) < threshold:
                    b = False
                    break
            if b:
                indices_to_keep.append(ind)
    return df.loc[indices_to_keep]


matches_analyzed = 0
total_matches = 0


# Function for pattern matching and scoring
def find_pattern_with_scaled_composite_score_with_probs(data_frame, timestamps, future_timestamps, metric,
                                                        plots_treshold, no_best_mathces, mse_ponder=150, mae_ponder=10,
                                                        corr_ponder=1, distance_ponder_w=12.5, corr_ponder_w=1):
    global matches_analyzed, total_matches
    matches_analyzed = 0
    total_matches = 0
    fs = 48
    sampling_period = 1 / fs
    wavelet = 'morl'
    central_frequency = 0.849
    min_scale = central_frequency / (fs / 2 * sampling_period)
    max_scale = central_frequency / (2 / fs * sampling_period)
    num_scales = 30
    scales = np.geomspace(min_scale, max_scale, num_scales)

    price = data_frame['Price'].values
    pattern = price[-timestamps:] - price[-timestamps]
    scaling_pattern = price[-timestamps]
    error_list = []
    gain_list = []
    up_down_list = []
    plot_images = []

    total_matches = len(price) - 2 * timestamps
    for i in range(0, len(price) - 2 * timestamps):
        matches_analyzed += 1
        possible_pattern = price[i:i + timestamps] - price[i]
        scaling_possible_pattern = price[i]
        scaled_possible_pattern = scaling_pattern / scaling_possible_pattern * possible_pattern

        pattern_flat = np.array(pattern).astype(float).flatten()
        scaled_possible_pattern_flat = np.array(scaled_possible_pattern).astype(float).flatten()

        mse_metric = mse(pattern_flat, scaled_possible_pattern_flat)
        mae_metric = mae(pattern_flat, scaled_possible_pattern_flat)

        try:
            correlation = np.corrcoef(pattern_flat, scaled_possible_pattern_flat)[0, 1]
        except Exception as e:
            continue

        cwtmatr1, _ = pywt.cwt(pattern_flat, scales, 'morl')
        cwtmatr2, _ = pywt.cwt(scaled_possible_pattern_flat, scales, 'morl')

        distance_w = euclidean(cwtmatr1.flatten(), cwtmatr2.flatten())
        correlation_w = np.corrcoef(cwtmatr1.flatten(), cwtmatr2.flatten())[0, 1]

        index_and_errors = [i, mse_metric, mae_metric, correlation, distance_w, correlation_w]
        error_list.append(index_and_errors)

    error_df = pd.DataFrame(error_list,
                            columns=['Index', 'MSE', 'MAE', 'Correlation', 'Distance Wavelet', 'Correlation Wavelet'])

    scaler = MinMaxScaler()
    error_df[['MSE', 'MAE', 'Distance Wavelet']] = scaler.fit_transform(error_df[['MSE', 'MAE', 'Distance Wavelet']])

    error_df['1 - Correlation'] = 1 - error_df['Correlation']
    error_df['1 - Correlation'] = scaler.fit_transform(error_df[['1 - Correlation']])

    error_df['1 - Correlation Wavelet'] = 1 - error_df['Correlation Wavelet']
    error_df['1 - Correlation Wavelet'] = scaler.fit_transform(error_df[['1 - Correlation Wavelet']])

    error_df['Composite Score'] = mse_ponder * error_df['MSE'] + mae_ponder * error_df['MAE'] + corr_ponder * error_df[
        '1 - Correlation'] + corr_ponder_w * error_df['1 - Correlation Wavelet'] + distance_ponder_w * error_df[
                                      'Distance Wavelet']

    error_df_sorted_repeating_indices = error_df.sort_values(by=metric, ascending=True)
    error_df_sorted = filter_indices(error_df_sorted_repeating_indices[:3 * no_best_mathces], 24)

    for j in range(0, len(error_df_sorted) - 1):
        y_identified = price[-timestamps] + (
                price[error_df_sorted.index[j]: error_df_sorted.index[j] + timestamps + future_timestamps] - price[
            error_df_sorted.index[j]]) * price[-timestamps] / price[error_df_sorted.index[j]]
        y_real = price[-timestamps:]
        if j < plots_treshold:
            plt.figure(figsize=(8, 4))
            plt.title(
                f'Moment of identified pattern: {data_frame.Date[error_df_sorted.index[j]]}, No. {j + 1} best match')
            plt.plot(y_identified, label='y_identified')
            plt.plot(y_real, label='y_real')
            plt.axvline(x=timestamps - 1, color='red', linestyle='--')
            plt.legend()

            # Save image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"plot_{j}.png")
            plt.savefig(image_path)
            plt.close()

            # Add image to the list
            plot_images.append(f"/static/images/plot_{j}.png")
        gain = y_identified[-1] - y_identified[timestamps - 1]
        gain_list.append(gain)
    for i in range(0, len(gain_list)):
        if gain_list[i] > 0:
            up_down_list.append(1)
        else:
            up_down_list.append(0)
    return error_df_sorted, plot_images, gain_list, up_down_list


def number_of_best_matces():
    n = 10
    return n


def stats(gains, uds, n):
    if n < 15:
        m = 30
    else:
        m = n

    # Calculate std_gain safely
    std_gain = np.std(gains[:m]) if len(gains) >= m else 0

    gains = gains[:n]
    uds = uds[:n]

    gains_mean = np.mean(gains) if len(gains) > 0 else 0
    uds_mean = np.mean(uds) if len(uds) > 0 else 0

    gains_out_1std = []
    gains_out_2std = []

    # Populate gains_out_1std and gains_out_2std
    for g in gains:
        if abs(g) > abs(std_gain):
            gains_out_1std.append(g)
        if abs(g) > 2 * abs(std_gain):
            gains_out_2std.append(g)

    # Initialize means to avoid UnboundLocalError
    gains_out_1std_mean = 0
    gains_out_2std_mean = 0

    # Compute means for filtered data
    if len(gains_out_1std) > 0:
        gains_out_1std_mean = np.mean(gains_out_1std)

    if len(gains_out_2std) > 0:
        gains_out_2std_mean = np.mean(gains_out_2std)

    # Handle NaN values
    if math.isnan(gains_mean):
        gains_mean = 0
    if math.isnan(uds_mean):
        uds_mean = 0
    if math.isnan(gains_out_1std_mean):
        gains_out_1std_mean = 0
    if math.isnan(gains_out_2std_mean):
        gains_out_2std_mean = 0

    return (gains_mean, uds_mean, gains_out_1std_mean, gains_out_2std_mean, len(gains_out_1std), len(gains_out_2std),)



@app.route('/get_progress')
def get_progress():
    # Example values for day_counter and total_days
    global day_counter, total_days, matches_analyzed, total_matches
    return jsonify({'day_counter': day_counter, 'total_days': total_days, 'matches_analyzed': matches_analyzed,
                    'total_matches': total_matches})


@app.route('/show_best_matches', methods=['POST'])
def show_best_matches_and_stats():
    global day_counter, total_days, matches_analyzed, total_matches

    day_counter = 0
    total_days = 0
    matches_analyzed = 0
    total_matches = 0
    data = request.get_json()
    selected_cryptos = data.get('cryptos', [])
    num_days = data.get('numDays', 10)
    timestamps = data.get('timestamps', 48)
    future_timestamps = data.get('futureTimestamps', 24)
    plots_threshold = data.get('plotsThreshold', 10)

    images = []
    stats_data = None

    FUNCTION_URL = "https://faas-fra1-afec6ce7.doserverless.co/api/v1/web/fn-c7362c8a-5a73-4f84-91f0-228fc247a5e6/default/show_best_options"

    # Prepare payload to call DigitalOcean Function
    def prepare_payload(data_frame):
        return {
            "data_frame": data_frame.to_json(orient='split'),
            "timestamps": timestamps,
            "future_timestamps": future_timestamps,
            "metric": "Composite Score",
            "plots_treshold": plots_threshold,
            "no_best_mathces": 100
        }

    if 'ETH' in selected_cryptos:
        eth_df, _ = get_dfs(freq_in_hours=1, days=num_days, cryptos=['ETHUSDT'])
        payload = prepare_payload(eth_df)

        try:
            response = requests.post(FUNCTION_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                images.extend(result['plot_images'])
                gains, uds = result['gain_list'], result['up_down_list']
                n = number_of_best_matces()
                stats_data = stats(gains, uds, n)
            else:
                return jsonify(
                    {'status': 'Error', 'message': f"Failed to get ETH data, status code: {response.status_code}"})
        except requests.exceptions.RequestException as e:
            return jsonify({'status': 'Error', 'message': f"Request to DigitalOcean failed: {e}"})

    if 'BTC' in selected_cryptos:
        _, btc_df = get_dfs(freq_in_hours=1, days=num_days, cryptos=['BTCUSDT'])
        payload = prepare_payload(btc_df)

        try:
            response = requests.post(FUNCTION_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                images.extend(result['plot_images'])
                gains, uds = result['gain_list'], result['up_down_list']
                n = number_of_best_matces()
                stats_data = stats(gains, uds, n)
            else:
                return jsonify(
                    {'status': 'Error', 'message': f"Failed to get BTC data, status code: {response.status_code}"})
        except requests.exceptions.RequestException as e:
            return jsonify({'status': 'Error', 'message': f"Request to DigitalOcean failed: {e}"})

    if stats_data is None:
        return jsonify({'status': 'No valid data found for any selected cryptos'})

    return jsonify({
        'status': 'Best matches shown',
        'images': images,
        'gains_mean': stats_data[0],
        'uds_mean': stats_data[1],
        'gains_out_1std_mean': stats_data[2],
        'gains_out_2std_mean': stats_data[3],
        'len_gains_out_1std_mean': stats_data[4],
        'len_gains_out_2std_mean': stats_data[5]
    })


#################################################PATTERN FINDER#############################################

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)