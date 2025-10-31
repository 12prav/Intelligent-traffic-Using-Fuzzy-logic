import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import joblib
import os
import time

# --- Utilities ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df


def build_fuzzy_system():
    # Define fuzzy variables
    density = ctrl.Antecedent(np.arange(0, 101, 1), 'density')
    queue = ctrl.Antecedent(np.arange(0, 51, 1), 'queue')
    wait = ctrl.Antecedent(np.arange(0, 101, 1), 'wait_time')
    congestion = ctrl.Consequent(np.arange(0, 101, 1), 'congestion')

    # Membership functions for density
    density['low'] = fuzz.trimf(density.universe, [0, 0, 35])
    density['medium'] = fuzz.trimf(density.universe, [20, 45, 70])
    density['high'] = fuzz.trimf(density.universe, [55, 100, 100])

    # Membership functions for queue
    queue['low'] = fuzz.trimf(queue.universe, [0, 0, 5])
    queue['medium'] = fuzz.trimf(queue.universe, [2, 10, 20])
    queue['high'] = fuzz.trimf(queue.universe, [10, 50, 50])

    # Membership functions for wait_time
    wait['low'] = fuzz.trimf(wait.universe, [0, 0, 8])
    wait['medium'] = fuzz.trimf(wait.universe, [5, 15, 30])
    wait['high'] = fuzz.trimf(wait.universe, [20, 100, 100])

    # Congestion output
    congestion['low'] = fuzz.trimf(congestion.universe, [0, 0, 40])
    congestion['medium'] = fuzz.trimf(congestion.universe, [30, 50, 70])
    congestion['high'] = fuzz.trimf(congestion.universe, [60, 100, 100])

    # Rules
    rules = []
    rules.append(ctrl.Rule(density['low'] & queue['low'] & wait['low'], congestion['low']))
    rules.append(ctrl.Rule(density['low'] & queue['medium'], congestion['low']))
    rules.append(ctrl.Rule(density['medium'] & wait['medium'], congestion['medium']))
    rules.append(ctrl.Rule(density['high'] | queue['high'] | wait['high'], congestion['high']))
    rules.append(ctrl.Rule(density['medium'] & queue['low'] & wait['low'], congestion['medium']))

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim


def fuzz_congestion(sim, dens, que, wt):
    sim.input['density'] = float(np.clip(dens, 0, 100))
    sim.input['queue'] = float(np.clip(que, 0, 50))
    sim.input['wait_time'] = float(np.clip(wt, 0, 100))
    try:
        sim.compute()
        val = sim.output['congestion']
    except Exception:
        # fallback if compute fails
        val = (dens / 100.0) * 100
    # Map to label
    if val < 40:
        label = 'Low'
    elif val < 65:
        label = 'Medium'
    else:
        label = 'High'
    return val, label


# --- Main App ---
st.set_page_config(page_title='Intelligent Traffic Signal Control', layout='wide')
st.title('Intelligent Traffic Signal Control — Fuzzy + ML (MVP)')

DATA_PATH = r"c:\Users\praveen jain\OneDrive\Desktop\ml project\traffic.csv"

with st.sidebar:
    st.header('Model / Data')
    st.markdown('Dataset: `traffic.csv`')
    model_type = st.selectbox('Model', ['RandomForest', 'LinearRegression'])
    test_size = st.slider('Test set fraction', 0.05, 0.4, 0.2)
    train_button = st.button('Train model')
    st.markdown('---')
    st.markdown("**Made by Praveen Jain**")

# Load data
with st.spinner('Loading data...'):
    df = load_data(DATA_PATH)

# (dataset sample table removed per user request)

# Preprocessing
features = ['hour', 'dayofweek', 'weekend', 'density', 'queue', 'wait_time', 'weather_flag', 'incident']
if not all(f in df.columns for f in features + ['green_time']):
    st.error('Dataset missing required columns. Please check CSV.')
else:
    X = df[features].copy()
    y = df['green_time'].copy()

    # Try to load saved model if available
    model = None
    model_path = os.path.join(os.path.dirname(DATA_PATH), 'model.pkl')
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.success('Loaded saved model from disk')
        except Exception:
            model = None

    # Sidebar option: auto-train if no model exists
    auto_train = st.sidebar.checkbox('Auto-train if no saved model', value=True)

    def train_and_save(X, y, model_type, test_size, model_path):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        if model_type == 'RandomForest':
            m = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            m = LinearRegression()
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        try:
            joblib.dump(m, model_path)
        except Exception:
            pass
        return m, mae

    # Train model when clicked (or auto-train if enabled and no model)
    if train_button:
        st.info('Training model — this may take a few seconds')
        model, mae = train_and_save(X, y, model_type, test_size, model_path)
        st.success(f'Training finished. MAE on test set: {mae:.2f} seconds')
        st.info(f'Model saved to {model_path}')
    elif model is None and auto_train:
        with st.spinner('No saved model found — auto-training a model now'):
            # small sleep to let UI render
            time.sleep(0.5)
            model, mae = train_and_save(X, y, model_type, test_size, model_path)
        st.success(f'Auto-training finished. MAE on test set: {mae:.2f} seconds')
        st.info(f'Model saved to {model_path}')
        # show feature importances for RF
        if model_type == 'RandomForest':
            importances = model.feature_importances_
            fi = pd.Series(importances, index=features).sort_values(ascending=False)
            st.write('Feature importances:')
            st.bar_chart(fi)
    else:
        st.info('Model not trained yet. Click "Train model" in the sidebar to train from the CSV')

    # Build fuzzy system
    sim = build_fuzzy_system()

    # UI for manual input
    st.header('Predict green light duration')
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider('Hour (0-23)', 0, 23, 12)
        dayofweek = st.slider('Day of week (0=Mon)', 0, 6, 4)
        weekend = st.selectbox('Weekend (0=no,1=yes)', [0, 1], index=0)
        weather_flag = st.selectbox('Weather flag (0=Clear,1=Rain,2=Heavy Rain)', [0, 1, 2], index=0)
        incident = st.selectbox('Incident (0=no,1=yes)', [0, 1], index=0)
    with col2:
        density = st.slider('Density (0-100)', 0.0, 100.0, 50.0)
        queue = st.slider('Queue length (vehicles)', 0, 30, 5)
        wait_time = st.slider('Wait time (s)', 0.0, 100.0, 10.0)

    # Predict using model if available
    input_df = pd.DataFrame([{
        'hour': hour,
        'dayofweek': dayofweek,
        'weekend': weekend,
        'density': density,
        'queue': queue,
        'wait_time': wait_time,
        'weather_flag': weather_flag,
        'incident': incident
    }])

    pred_ml = None
    if model is not None:
        pred_ml = model.predict(input_df)[0]
    else:
        # simple baseline using density and queue
        pred_ml = 20 + (density / 100.0) * 60 + queue * 1.5

    # Fuzzy congestion
    cong_val, cong_label = fuzz_congestion(sim, density, queue, wait_time)

    # Adjust ML prediction based on fuzzy output
    # If congestion is high, increase time by up to 25%; if low, decrease by up to 15%
    if cong_label == 'High':
        final = pred_ml * 1.25
    elif cong_label == 'Medium':
        final = pred_ml * 1.05
    else:
        final = pred_ml * 0.90

    final = float(np.clip(final, 5, 180))

    st.subheader('Results')
    st.metric('ML predicted green_time (s)', f"{pred_ml:.1f}")
    st.metric('Fuzzy congestion score', f"{cong_val:.1f} ({cong_label})")
    st.metric('Recommended green_time (s)', f"{final:.1f}")

    st.markdown('Traffic condition summary:')
    st.write(f'- Hour: {hour}, Day: {dayofweek}, Weekend: {weekend}')
    st.write(f'- Density: {density:.1f}, Queue: {queue}, Wait time: {wait_time:.1f}s')
    st.write(f'- Weather flag: {weather_flag}, Incident: {incident}')

    st.markdown('Notes: The ML model is trained from the provided CSV when you click "Train model". The fuzzy system categorizes congestion using density, queue and wait time, and the final recommendation adjusts the ML output.')
    st.markdown('---')
    st.markdown('<div style="text-align:center; font-size:14px;">Made by Praveen Jain</div>', unsafe_allow_html=True)
