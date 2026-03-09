# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:44:41 2023

@author: binata
"""
"""
Physics-Informed LSTM (PI LSTM) Surrogate Model (Exact MB-based) for Street-Scale Flood Forecasting in Norfolk, VA

This model is developed by incorporating the mass balance (MB) equation into the customized loss function of the LSTM surrogate model, where, the total loss (L_total) combines the data loss (L_data) and physics loss (L_physics): 
L_total= (α * L_data + β * L_phy)	

In this script, the exact MB equation is enforced, which states that the change of flood volume at the current timestep is equal to the net difference between the total inflow and the total outflow that occurred during the current and previous timestep:
∆volume_(t) = Qin_(t) * ∆t + Rain_vol_(t) * Area - Qout_(t) * ∆t - Qpipe_(t) * ∆t
                                                                             
Model Details:
--------------
- Forecast horizon: 4 hours (n_ahead = 4)
- Lookback window: 4 hours (n_back = 4)

- Input features:
-----------------
    - Data loss inputs: 
        - Elevation (ELV) 
        - Topographic Wetness Index (TWI) 
        - Depth-To-Water (DTW) 
        - Rainfall (RH)
        - Tide (TD)
    - Physics loss inputs: 
        - Qin (FI)
        - Rainfall Volume (RH_Vol)
        - Qout (FO)
        - Qpipe (Q)
        
- Output:
---------
        - Water Depth (w_depth_new)
        - Water Volume (w_depth_new_Vol)

Functionality:
--------------
- Loads `node_data`, `tide_data`, and `weather_data` from a relational database.
- Uses `lstm_data_tools.py` to preprocess data into 3D tensors for training, validation, and testing.
- Creates two types of tensors - 
        - scaled tensors for computing data loss and 
        - unscaled (raw) tensors for computing the physics loss
- Computes data loss and physics loss using a customized loss function
- Optimizes PI LSTM models based on Bayesian Optimization
- Selects the best hyperparameters based on 'val_rmse_wd'.
- Trains PI LSTM model (Exact) using the best hyperparameters, saves the model, and writes predictions to CSV files.
- Trains a baseline LSTM model with data λ = 1 and phy λ = 0, saves the model, and writes predictions to CSV files.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import math
import random
import lstm_data_tools as ldt
from sklearn.metrics import mean_squared_error
from math import sqrt
    
tf.keras.utils.set_random_seed(1) #set random seeds for all together - numpy, tf and python

os.getcwd()

db = ldt.SLF_Data_Builder(os.getcwd() + '/relational_database/')

'define run_name'
trial_model='LSTM_GPU_44_Exact'
trial_domain ='S_22'
trial_seed ='seed1_es1'
trial_lambda = 'lam_opt_ab_hp'
trial_data ='TUFLOW_max_pred'

trial_all = '{}_{}_{}_{}_{}'.format(trial_model, trial_domain, trial_seed, trial_lambda, trial_data)
print(trial_all)


'..............................................data............................................'

'train_df and test_df different'
#specify parameters
cols = ['FID_', 'Event', 'DateTime', 'ELV', 'DTW', 'TWI', 'Street_Area', 'RH', 'TD_HR', 'w_depth_new','w_depth_new_Vol', 
        'RH_Vol', 'FI', 'FO', 'Q']           
print("Data Columns: ", cols)
			

#specify events          
Events =   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24]   
train_Events=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
validation_Events=[21,22]
test_Events=[23, 24]

#set of streets data
path_FIDs="input_20/"
FID_selected=pd.read_csv(path_FIDs+"D0_R40_S22.csv") #change
FIDs=FID_selected['FID_']

train_nodes = FIDs
train_events = train_Events

validation_nodes = FIDs
validation_events = validation_Events

test_nodes = FIDs
test_events = test_Events

#full data
data_org = db.get_data(nodes = FIDs, events = Events, columns=cols)
train_data_org = db.get_data(nodes = train_nodes, events = train_events, columns=cols)
train_data_org.head()

validation_data_org = db.get_data(nodes = validation_nodes, events = validation_events, columns=cols)
validation_data_org.head()

test_data_org = db.get_data(nodes = test_nodes, events = test_events, columns=cols)
test_data_org.head()

'scaled'
cols2scale = ['RH', 'TD_HR', 'w_depth_new', 'w_depth_new_Vol', 'ELV', 'DTW', 'TWI'] #, 'Street_Area'

db.fit_scaler(train_data_org, columns_to_fit=cols2scale, scaler_type='Standard')

'later to be used inside physics_loss'
scaler_wdepth = db.scaler_dict['w_depth_new']
mean_wdepth = tf.constant(scaler_wdepth.mean_[0], dtype=tf.float32)
std_wdepth = tf.constant(scaler_wdepth.scale_[0], dtype=tf.float32)

scaler_wdepth_vol = db.scaler_dict['w_depth_new_Vol']
mean_wdepth_vol = tf.constant(scaler_wdepth_vol.mean_[0], dtype=tf.float32)
std_wdepth_vol = tf.constant(scaler_wdepth_vol.scale_[0], dtype=tf.float32)

train_data = db.scale_data(train_data_org, columns_to_scale=cols2scale)
train_data.head()

validation_data = db.scale_data(validation_data_org, columns_to_scale=cols2scale)
validation_data.head()

test_data = db.scale_data(test_data_org, columns_to_scale=cols2scale)
test_data.head()

print(len(train_events), len(validation_events), len(test_events))

lstm_train_data = ldt.SLF_LSTM_Data(train_data)
lstm_validation_data = ldt.SLF_LSTM_Data(validation_data)
lstm_test_data = ldt.SLF_LSTM_Data(test_data)


n_back = 4 
n_ahead = 4 


forecast_cols = ['RH', 'TD_HR']
x_cols = ['w_depth_new','w_depth_new_Vol','ELV', 'DTW', 'TWI'] 
y_cols = ['w_depth_new','w_depth_new_Vol',]

lstm_train_data.build_data(
n_back = n_back,
n_ahead = n_ahead,
forecast_cols = forecast_cols,
y_cols = y_cols,
x_cols = x_cols,
verbose = False
)

lstm_validation_data.build_data(
n_back = n_back,
n_ahead = n_ahead,
forecast_cols = forecast_cols,
y_cols = y_cols,
x_cols = x_cols,
verbose = False
)

lstm_test_data.build_data(
n_back = n_back,
n_ahead = n_ahead,
forecast_cols = forecast_cols,
y_cols = y_cols,
x_cols = x_cols,
verbose = False
)

train_x, train_y = lstm_train_data.get_lstm_data()
validation_x, validation_y = lstm_validation_data.get_lstm_data()
test_x, test_y = lstm_test_data.get_lstm_data()

train_x = np.asarray(train_x).astype(np.float32)
train_y = np.asarray(train_y).astype(np.float32)

validation_x = np.asarray(validation_x).astype(np.float32)
validation_y = np.asarray(validation_y).astype(np.float32)

test_x = np.asarray(test_x).astype(np.float32)
test_y = np.asarray(test_y).astype(np.float32)

print('Data Shapes')
print('Train x:', train_x.shape)
print('Train y:', train_y.shape)

print('Validation x:', validation_x.shape)
print('Validation y:', validation_y.shape)

print('Test x:', test_x.shape)
print('Test y:', test_y.shape)


'unscaled - raw'
forecast_cols = ['RH_Vol', 'FI', 'FO', 'Q']
x_cols = ['w_depth_new','w_depth_new_Vol', 'Street_Area']
y_cols = ['w_depth_new','w_depth_new_Vol']

# Use raw (unscaled) data to create a separate LSTM object
lstm_train_data_raw = ldt.SLF_LSTM_Data(train_data_org)
lstm_validation_data_raw = ldt.SLF_LSTM_Data(validation_data_org)

lstm_train_data_raw.build_data(
    n_back=n_back,
    n_ahead=n_ahead,
    forecast_cols=forecast_cols,
    y_cols=y_cols,
    x_cols=x_cols,
    verbose=False
)

lstm_validation_data_raw.build_data(
    n_back=n_back,
    n_ahead=n_ahead,
    forecast_cols=forecast_cols,
    y_cols=y_cols,
    x_cols=x_cols,
    verbose=False
)

train_x_raw, _ = lstm_train_data_raw.get_lstm_data()
validation_x_raw, _ = lstm_validation_data_raw.get_lstm_data()

# Ensure float32
train_x_raw = np.asarray(train_x_raw).astype(np.float32)
validation_x_raw = np.asarray(validation_x_raw).astype(np.float32)

print('Train x raw:', train_x_raw.shape)
print('Validation x raw:', validation_x_raw.shape)

'..............................................model............................................'

def physics_informed_loss(y_true, y_pred, train_x_raw, mean_wdepth, std_wdepth, mean_wdepth_vol, std_wdepth_vol, lambda_phy, lambda_data):
    
    'data loss'
    # Reshape y_pred to (batch, 4, 2)
    y_pred = tf.reshape(y_pred, (-1, 4, 2))  # [batch, time, feature]
    y_true = tf.reshape(y_true, (-1, 4, 2))

    pred_depth = y_pred[:, :, 0]
    pred_volume = y_pred[:, :, 1]
    true_depth = y_true[:, :, 0]
    true_volume = y_true[:, :, 1]

    # Water depth and volume loss (standard)
    data_loss_depth = tf.reduce_mean(tf.square(pred_depth - true_depth))
    data_loss_volume = tf.reduce_mean(tf.square((pred_volume - true_volume)))

    # Total data loss
    data_loss = data_loss_depth + data_loss_volume
    
    'physics loss'
    physics_losses = []

    # Extract volume component only: shape (batch, 4)
    y_pred_vol = y_pred[:, :, 1]
    y_true_vol = y_true[:, :, 1]

    y_pred_vol_unscaled = y_pred_vol * std_wdepth_vol + mean_wdepth_vol
    y_true_vol_unscaled = y_true_vol * std_wdepth_vol + mean_wdepth_vol

    for i in range(4):
        rain_vol_current = train_x_raw[:, 3 , 7+(4*i)]
        inflow_vol_current = train_x_raw[:, 3 , 7+(4*i)+1]
        outflow_vol_current = train_x_raw[:, 3 , 7+(4*i)+2]
        pipe_vol_current = train_x_raw[:, 3 , 7+(4*i)+3]
        
        depth_pred_vol_current = y_pred_vol_unscaled[:, i]
        
        if i == 0:
            depth_true_vol_prev = train_x_raw[:, 3, 1]
            depth_pred_vol_prev = depth_true_vol_prev
        else:
            depth_pred_vol_prev = y_pred_vol_unscaled[:, i - 1]

        area = train_x_raw[:, 0, 2]

        lhs = (depth_pred_vol_current - depth_pred_vol_prev)
        rhs = (rain_vol_current + inflow_vol_current - outflow_vol_current - pipe_vol_current)
        
        residual = ((lhs - rhs)/area)
        residual_norm = residual / std_wdepth
        physics_losses.append(tf.square(residual_norm))

    physics_loss = tf.reduce_mean(tf.add_n(physics_losses))
    total_loss = lambda_data * data_loss + lambda_phy * physics_loss
    return total_loss, data_loss, physics_loss


def run_model(hp, return_model=False):
    
    # # configure LSTM network hyperparameters
    n_back = 4
    n_ahead = 4
    
    """
    params = [lambda_phy, lambda_data, num_units, dropout, act, opt, lr]
    returns: val_rmse_wd (float)  -> lower is better
    """

    # # ---- hyperparameters ----
    lambda_phy = hp.get("lambda_phy")
    lambda_data = hp.get("lambda_data")
    
    act = hp.get("activation")
    optimizer = hp.get('optimizer')
    lr = hp.get("lr")
        
    # --- Decode units ---
    units_cfg = hp.get("units_pair")
    units_split = units_cfg.split('_')
    num_units1 = int(units_split[0])
    num_units2 = int(units_split[1])
    
    # --- Decode dropout ---
    drop_cfg = hp.get("dropout_pair")
    drop_split = drop_cfg.split('_')
    dp_rate1 = float(drop_split[0])
    dp_rate2 = float(drop_split[1])
    
    # Set the optimizer
    if optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
    else:
        raise
        
    # create the LSTM model
    model = Sequential()
    #1st hidden layer
    model.add(LSTM(units=num_units1, activation=act, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
    model.add(Dropout(rate=dp_rate1))
    #2nd hidden layer
    model.add(LSTM(units=num_units2, activation=act))
    model.add(Dropout(rate=dp_rate2))
    #output layer
    model.add(Dense(units=n_ahead*2, activation='linear')) #two outputs = w_depth and volume
    
    # Compile the model
    # model.compile(loss=physics_informed_loss, optimizer=optimizer, metrics=['mae']) 
    
    # fit model
    # === Datasets with dual inputs ===
    train_dataset = tf.data.Dataset.from_tensor_slices(((train_x, train_x_raw), train_y)).batch(512)
    val_dataset = tf.data.Dataset.from_tensor_slices(((validation_x, validation_x_raw), validation_y)).batch(512)
    
    #train
    train_loss_results = []
    train_loss_data_results = []
    train_loss_phy_results = []

    #val
    val_loss_results = []
    val_loss_data_results = []
    val_loss_phy_results = []
    
    # --- Early stopping config ---
    patience  = 15        # stop if no val improvement for these many epochs
    min_delta = 1e-4      # required improvement in val loss
    best_val = np.inf
    best_epoch = -1
    wait = 0
    best_weights = None
    
    for epoch in range(150):    
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_loss_data_avg = tf.keras.metrics.Mean()
        epoch_loss_phy_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()
        epoch_val_loss_data_avg = tf.keras.metrics.Mean()
        epoch_val_loss_phy_avg = tf.keras.metrics.Mean()
    
        # Training
        for (x_batch, x_batch_raw), y_batch in train_dataset:
            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                loss, data_loss, phy_loss = physics_informed_loss(y_batch, y_pred, x_batch_raw, mean_wdepth, std_wdepth, mean_wdepth_vol, std_wdepth_vol, lambda_phy, lambda_data)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg.update_state(loss)
            epoch_loss_data_avg.update_state(data_loss)
            epoch_loss_phy_avg.update_state(phy_loss)
    
        # Validation
        for (x_batch_val, x_batch_val_raw), y_batch_val in val_dataset:
            y_pred_val = model(x_batch_val, training=False)
            val_loss, val_data_loss, val_phy_loss = physics_informed_loss(y_batch_val, y_pred_val, x_batch_val_raw, mean_wdepth, std_wdepth, mean_wdepth_vol, std_wdepth_vol, lambda_phy, lambda_data)
            epoch_val_loss_avg.update_state(val_loss)
            epoch_val_loss_data_avg.update_state(val_data_loss)
            epoch_val_loss_phy_avg.update_state(val_phy_loss)
        
        # Log results
        train_loss_results.append(epoch_loss_avg.result().numpy())
        train_loss_data_results.append(epoch_loss_data_avg.result().numpy())
        train_loss_phy_results.append(epoch_loss_phy_avg.result().numpy())
        val_loss_results.append(epoch_val_loss_avg.result().numpy())
        val_loss_data_results.append(epoch_val_loss_data_avg.result().numpy())
        val_loss_phy_results.append(epoch_val_loss_phy_avg.result().numpy())  
        
        print(f"Epoch {epoch+1:03d}: Train Loss = {epoch_loss_avg.result():.4f}, Val Loss = {epoch_val_loss_avg.result():.4f}")
        
        # Early stopping logic (by validation loss)
        current_val = epoch_val_loss_avg.result().numpy()
        if current_val < best_val - min_delta:
            best_val   = current_val
            best_epoch = epoch
            wait = 0
            best_weights = model.get_weights()
            print(f"  ↳ val improved to {best_val:.6f}")
        else:
            wait += 1
            print(f"  ↳ no improvement ({wait}/{patience})")
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1} (val={best_val:.6f})")
                break
    
    print(f"Best epoch = {best_epoch+1} (val={best_val:.6f})")
    model.set_weights(best_weights)
     
    
    'val_rmse_wd'
    'val'
    preds = model.predict(validation_x)
    preds = preds.reshape((-1, 4, 2))  # shape: [samples, time, features]
    
    'water depth'
    preds_wd = preds[:, : , 0]
    validation_y_wd = validation_y[:, : , 0]
    val_rmse_wd_scaled = np.sqrt(np.mean((preds_wd - validation_y_wd)**2))
    print(f"validation_rmse_wd_scaled: ", val_rmse_wd_scaled)
    
    'volume'
    preds_vol = preds[:, : , 1]
    validation_y_vol = validation_y[:, : , 1]
    val_rmse_vol_scaled = np.sqrt(np.mean((preds_vol - validation_y_vol)**2))
    print(f"validation_rmse_vol_scaled: ", val_rmse_vol_scaled)
    
    '''unscaling'''
    validation_data_1=validation_data.copy()
    
    'water depth'
    'remap multi-ahead'
    for k in range(n_ahead):
        preds_col = pd.Series(preds_wd[:,k], index=lstm_validation_data.data_map)
        validation_data_1[f'preds_y{k+1}_s'] = preds_col
        validation_data_1[f'preds_y{k+1}'] = validation_data_1[f'preds_y{k+1}_s'].shift(k)
        del validation_data_1[f'preds_y{k+1}_s']
    
        real_col = pd.Series(validation_y_wd[:,k], index=lstm_validation_data.data_map)
        validation_data_1[f'real_y{k+1}_s'] = real_col
        validation_data_1[f'real_y{k+1}'] = validation_data_1[f'real_y{k+1}_s'].shift(k)
        del validation_data_1[f'real_y{k+1}_s']
    
    'volume'
    'remap multi-ahead'
    for k in range(n_ahead):
        preds_col = pd.Series(preds_vol[:,k], index=lstm_validation_data.data_map)
        validation_data_1[f'preds_vol{k+1}_s'] = preds_col
        validation_data_1[f'preds_vol{k+1}'] = validation_data_1[f'preds_vol{k+1}_s'].shift(k)
        del validation_data_1[f'preds_vol{k+1}_s']
    
        real_col = pd.Series(validation_y_vol[:,k], index=lstm_validation_data.data_map)
        validation_data_1[f'real_vol{k+1}_s'] = real_col
        validation_data_1[f'real_vol{k+1}'] = validation_data_1[f'real_vol{k+1}_s'].shift(k)
        del validation_data_1[f'real_vol{k+1}_s']
    
    validation_data_1_inv = validation_data_1.copy()
    validation_data_1_inv.head()
          
    cols2scale = ['RH','TD_HR','w_depth_new','w_depth_new_Vol', 
    'preds_y1', 'real_y1', 'preds_y2','real_y2', 
    'preds_y3', 'real_y3', 'preds_y4', 'real_y4', 
    'preds_vol1', 'real_vol1', 'preds_vol2','real_vol2', 
    'preds_vol3', 'real_vol3', 'preds_vol4', 'real_vol4']
    orig_cols = ['RH', 'TD_HR','w_depth_new', 'w_depth_new_Vol',
    'w_depth_new', 'w_depth_new', 'w_depth_new', 'w_depth_new', 
    'w_depth_new', 'w_depth_new', 'w_depth_new', 'w_depth_new',
    'w_depth_new_Vol','w_depth_new_Vol','w_depth_new_Vol','w_depth_new_Vol',
    'w_depth_new_Vol','w_depth_new_Vol','w_depth_new_Vol','w_depth_new_Vol']
    
    #inverse scale the test data
    validation_data_1_inv = db.inverse_scale_data(validation_data_1_inv, columns_to_scale=cols2scale, orig_col_names=orig_cols)
    validation_data_1_inv.head()  


    ''''unscaled metrics'''
    
    'w_depth'
    real_cols = [f'real_y{k}' for k in range(1, n_ahead + 1)]
    pred_cols = [f'preds_y{k}' for k in range(1, n_ahead + 1)]
    
    real_all = pd.Series(dtype='float64')
    preds_all = pd.Series(dtype='float64')
    
    for real_col, pred_col in zip(real_cols, pred_cols):
        if real_col in validation_data_1_inv.columns and pred_col in validation_data_1_inv.columns:
            df_co = validation_data_1_inv[[real_col, pred_col]].dropna()
            real_all = pd.concat([real_all, df_co[real_col]])
            preds_all = pd.concat([preds_all, df_co[pred_col]])
    
    if not real_all.empty:
        # RMSE
        val_rmse_wd = sqrt(mean_squared_error(real_all, preds_all))
        print(f"validation_rmse_wd: ", val_rmse_wd)
    
    if return_model:
        return model, train_loss_results, train_loss_data_results, train_loss_phy_results, val_loss_results, val_loss_data_results, val_loss_phy_results
    else:
        return float(val_rmse_wd)
    
    
'..............................................opt............................................'
'with class'
import kerastuner
from kerastuner.tuners import BayesianOptimization

def define_hp(hp):
    hp.Choice("lambda_phy", [0.5, 0.75, 1.0])
    hp.Choice("lambda_data", [0.5, 0.75, 1.0])
    hp.Choice("units_pair", ["128_64", "256_128", "512_256", "1024_512"])
    hp.Choice("dropout_pair", ["0.2_0.1", "0.15_0.1"])
    hp.Choice("activation", ["relu", "selu"])
    hp.Choice("optimizer", ["RMSprop","Adam", "Nadam"])
    hp.Choice("lr", [0.00001, 0.0001, 0.001, 0.01])

opt_results_log = []

class MyTuner(BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        # Get hyperparameters
        hp = trial.hyperparameters
        # Run model
        val_rmse_wd = run_model(hp)
        # Report result back to KerasTuner
        self.oracle.update_trial(trial.trial_id, {'val_rmse_wd': val_rmse_wd})
        # ---- Save to Python list for making a CSV later ----
        trial_result = hp.values.copy()          # all hyperparams
        trial_result["val_rmse_wd"] = val_rmse_wd
        trial_result["trial_id"] = trial.trial_id
        opt_results_log.append(trial_result)

tuner = MyTuner(
    hypermodel=define_hp, 
    objective=kerastuner.Objective("val_rmse_wd", "min"),
    max_trials=25,
    directory='1.Run',
    project_name='PI_LSTM_Tuning',
    overwrite=True
)

tuner.search()

results_df = pd.DataFrame(opt_results_log)
results_df.to_csv(f"1.Result_PI_LSTM_Opt_hp/{trial_all}_Bayesian_tuner_results.csv", index=False)

best_hp = tuner.get_best_hyperparameters(1)[0]

print("Best λ phy =", best_hp.get("lambda_phy"))
print("Best λ data =", best_hp.get("lambda_data"))
print("Best units =", best_hp.get("units_pair"))
print("Best dropout =", best_hp.get("dropout_pair"))
print("Best act =", best_hp.get("activation"))
print("Best opt =", best_hp.get("optimizer"))
print("Best lr =", best_hp.get("lr"))

'..............................................best PI LSTM Exact model............................................'
hp_model ='best'

best_model, train_loss_results, train_loss_data_results, train_loss_phy_results, val_loss_results, val_loss_data_results, val_loss_phy_results = run_model(best_hp, return_model=True)

# # Save the model
best_model.save(f"1.Result_PI_LSTM_Opt_hp/{hp_model}_{trial_all}.h5")
print(f"Saved {hp_model} to disk")


'..............................................best LSTM model............................................'
print("\nRunning model with best architecture but phy λ = 0 and data λ = 1 (data-driven baseline)...")

hp_model ='datadriven'

best_hp_zero = best_hp.copy()
best_hp_zero.values['lambda_phy'] = 0
best_hp_zero.values['lambda_data'] = 1

data_driven_model, train_loss_results, train_loss_data_results, train_loss_phy_results, val_loss_results, val_loss_data_results, val_loss_phy_results = run_model(best_hp_zero, return_model=True)

# # Save the model
data_driven_model.save(f"1.Result_PI_LSTM_Vol_TUFLOW_Opt_hp/{hp_model}_{trial_all}.h5")
print(f"Saved {hp_model} to disk")



