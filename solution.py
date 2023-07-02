import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sns
import os

def create_file(filename,predicted_data):
        try:
            with open(filename, 'w') as f:
                for d in predicted_data:
                    f.write(str(d)+",\n")
            print("File " + filename + " created successfully.")
        except IOError:
            print("Error: could not create file " + filename)
   
def read_file(filename):
    try:
        with open(filename, 'r') as f:
            contents = f.read()
            print(contents)
    except IOError:
        print("Error: could not read file " + filename)

#Load the data from the Excel file into a pandas dataframe
df = pd.read_excel('TEST.xlsx')
print(df.columns)

#Split the data into training, testing, and validation sets 60|20|20
X_train, X_test, y_train, y_test = train_test_split(df.drop('TEC', axis=1), df['TEC'], test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

#Variables
numeric_vars = ['TEC','HourUT', 'Ion Temperature(Ti/K)','Neutral Temperature(Tn/K)','Electron Temperature(Te/K)', 'TEC_TOP']

# Create correlation matrix
corr_matrix = df[numeric_vars].corr()

# Plot the correlation matrix using a heatmap
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f', square=True)
plt.savefig('correlation.png')
#plt.show()
'''
#Build an LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

#Compile the model
lstm_model.compile(optimizer='adam', loss='mae')
lstm_model.summary()

keras.utils.plot_model(lstm_model, to_file='plot_lstm.png', show_layer_names=True)
print("PLOT HAS BEEN GENERATED, PLEASE CHECK YOUR DIRECTORY")

#Fit the model to the training data
history_lstm = lstm_model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val))
'''

# Build an LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=128))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
lstm_model.summary()

keras.utils.plot_model(lstm_model, to_file='plot_lstm.png', show_layer_names=True)
print("PLOT HAS BEEN GENERATED, PLEASE CHECK YOUR DIRECTORY")

# For LSTM model
model_checkpoint_lstm1 = ModelCheckpoint('best_lstm_model1.h5', save_best_only=True, verbose=1)

# Fit the model to the training data
history_lstm = lstm_model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_val, y_val),callbacks = model_checkpoint_lstm1, verbose=1)
# Load the best saved model
lstm_model.load_weights('best_lstm_model1.h5')

'''
# Build an LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
lstm_model.add(Dropout(0.3))
lstm_model.add(LSTM(units=128))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
lstm_model.summary()

keras.utils.plot_model(lstm_model, to_file='plot_lstm.png', show_layer_names=True)
print("PLOT HAS BEEN GENERATED, PLEASE CHECK YOUR DIRECTORY")

# Fit the model to the training data
#history_lstm = lstm_model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_val, y_val), verbose=1)
'''

'''
# Define the LSTM model architecture
lstm_model = Sequential([
    LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(units=128, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

# Compile the model
lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
lstm_model.summary()

# Define early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
model_checkpoint = ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model
#history = lstm_model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint], verbose=1)

# Load the best saved model
lstm_model.load_weights('best_lstm_model.h5')
'''


# /
'''
#256, 0.4,64,200
# Define the hyperparameters to tune
param_grid = {
    'lstm_units': [128, 256, 512],
    'dropout_rate': [0.2, 0.3, 0.4],
    'batch_size': [32, 64, 128],
    'epochs': [150, 200, 250]
}

# Define the model to use in the GridSearchCV
def create_model(lstm_units=128, dropout_rate=0.2):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=lstm_units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    lstm_model.add(Dropout(dropout_rate))
    lstm_model.add(LSTM(units=lstm_units))
    lstm_model.add(Dropout(dropout_rate))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return lstm_model

# Create the GridSearchCV object
grid_search = GridSearchCV(
    estimator=KerasRegressor(build_fn=create_model, verbose=1),
    param_grid=param_grid,
    cv=5
)

# Fit the GridSearchCV to the training data
grid_search_results = grid_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding score
print(f"Best: {grid_search_results.best_score_:.3f} using {grid_search_results.best_params_}")
# /
'''
# Build an RNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(units=256, return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2])))
rnn_model.add(Dropout(0.3))
rnn_model.add(SimpleRNN(units=128, return_sequences = False))
rnn_model.add(Dropout(0.2))
rnn_model.add(Dense(units=1))

# Compile the model
rnn_model.compile(optimizer='adam', loss='mae')
rnn_model.summary()

keras.utils.plot_model(rnn_model, to_file='plot_rnn.png', show_layer_names=True)
print("PLOT HAS BEEN GENERATED, PLEASE CHECK YOUR DIRECTORY")

# Fit the model to the training data
#history_rnn = rnn_model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val))

# Load the best saved model
rnn_model.load_weights('best_rnn_model.h5')

#Build the Linear Regression model
linear_model = LinearRegression()

#Fit the model to the training data
linear_model.fit(X_train.reshape((X_train.shape[0], X_train.shape[2])), y_train)

#Build the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)

#Fit the model to the training data
gb_model.fit(X_train.reshape((X_train.shape[0], X_train.shape[2])), y_train)

#Build Train random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

#Fit the model to the training data
rf_model.fit(X_train.reshape((X_train.shape[0], X_train.shape[2])), y_train)


#Make predictions on the training, testing, and validation sets for the linear regression model
y_train_pred_lin = linear_model.predict(X_train.reshape((X_train.shape[0], X_train.shape[2])))
y_test_pred_lin = linear_model.predict(X_test.reshape((X_test.shape[0], X_test.shape[2])))
y_val_pred_lin = linear_model.predict(X_val.reshape((X_val.shape[0], X_val.shape[2])))

#Make predictions on the training, testing, and validation sets for the gradient boosting model
y_train_pred_gb = gb_model.predict(X_train.reshape((X_train.shape[0], X_train.shape[2])))
y_test_pred_gb = gb_model.predict(X_test.reshape((X_test.shape[0], X_test.shape[2])))
y_val_pred_gb = gb_model.predict(X_val.reshape((X_val.shape[0], X_val.shape[2])))

#Make predictions on the training, testing, and validation sets for the LSTM model
y_train_pred_lstm = lstm_model.predict(X_train)
y_test_pred_lstm = lstm_model.predict(X_test)
y_val_pred_lstm = lstm_model.predict(X_val)

# Make predictions on the training, testing, and validation sets for the RNN model
y_train_pred_rnn = rnn_model.predict(X_train)
y_test_pred_rnn = rnn_model.predict(X_test)
y_val_pred_rnn = rnn_model.predict(X_val)

# Make predictions on validation set Random Forest
y_train_pred_rf = rf_model.predict(X_train.reshape((X_train.shape[0], X_train.shape[2])))
y_test_pred_rf = rf_model.predict(X_test.reshape((X_test.shape[0], X_test.shape[2])))
y_val_pred_rf = rf_model.predict(X_val.reshape((X_val.shape[0], X_val.shape[2])))

#Print the MSE and R^2 score for the training, testing, and validation sets for the linear regression model
print("======================================")
print("Linear Regression Model:")
print("Training set:")
train_loss_lin = mean_squared_error(y_train, y_train_pred_lin)
print("MSE: ", train_loss_lin)
print("R^2 score: ", r2_score(y_train, y_train_pred_lin))

y_train_pred_lin_filename = "y_train_pred_lin.txt"
create_file(y_train_pred_lin_filename, y_train_pred_lin)
#print("Linear Regression Train Predictions: ", y_train_pred_lin)

print("\nTesting set:")
test_loss_lin = mean_squared_error(y_test, y_test_pred_lin)
print("MSE: ", test_loss_lin)
print("R^2 score: ", r2_score(y_test, y_test_pred_lin))

y_test_pred_lin_filename = "y_test_pred_lin.txt"
create_file(y_test_pred_lin_filename, y_test_pred_lin)
#print("Linear Regression Test Predictions: ", y_test_pred_lin)


print("\nValidation set:")
val_loss_lin = mean_squared_error(y_val, y_val_pred_lin)
print("MSE: ", val_loss_lin)
print("R^2 score: ", r2_score(y_val, y_val_pred_lin))

# Print the MSE and R^2 score for the Gradient Boosting model
print("======================================")
print("Gradient Boosting Model:")
print("Training set:")
train_loss_rmse_gb = np.sqrt(mean_squared_error(y_train, y_train_pred_gb))
train_loss_gb = mean_squared_error(y_train, y_train_pred_gb)
print("RMSE:", train_loss_rmse_gb)
print("MSE: ", train_loss_gb)
print("R^2 score: ", r2_score(y_train, y_train_pred_gb))

y_train_pred_gb_filename = "y_train_pred_gb.txt"
create_file(y_train_pred_gb_filename, y_train_pred_gb)

#print("Gradient Boosting Train Predictions: ", y_train_pred_gb)

print("Testing set:")
test_loss_rmse_gb = np.sqrt(mean_squared_error(y_test, y_test_pred_gb))
test_loss_gb = mean_squared_error(y_test, y_test_pred_gb)
print("RMSE:", test_loss_rmse_gb)
print("MSE: ", test_loss_gb)
print("R^2 score: ", r2_score(y_test, y_test_pred_gb))

y_test_pred_gb_filename = "y_test_pred_gb.txt"
create_file(y_test_pred_gb_filename, y_test_pred_gb)

#print("Gradient Boosting Test Predictions: ", y_test_pred_gb)

print("Validation set:")
val_loss_rmse_gb = np.sqrt(mean_squared_error(y_val, y_val_pred_gb))
val_loss_gb = mean_squared_error(y_val, y_val_pred_gb)
print("RMSE:", val_loss_rmse_gb)
print("MSE: ", val_loss_gb)
print("R^2 score: ", r2_score(y_val, y_val_pred_gb))

#Print the MSE and R^2 score for the training, testing, and validation sets for the Random Forest model
print("======================================")
print("Random Forest Model:")
print("Training set:")
train_loss_rf = mean_squared_error(y_train, y_train_pred_rf)
print("MSE: ", train_loss_rf)
print("R^2 score: ", r2_score(y_train, y_train_pred_rf))

y_train_pred_rf_filename = "y_train_pred_rf.txt"
create_file(y_train_pred_rf_filename, y_train_pred_rf)

#print("Random Forest Train Predictions: ", y_train_pred_rf)

print("\nTesting set:")
test_loss_rf = mean_squared_error(y_test, y_test_pred_rf)
print("MSE: ", test_loss_rf)
print("R^2 score: ", r2_score(y_test, y_test_pred_rf))

y_test_pred_rf_filename = "y_test_pred_rf.txt"
create_file(y_test_pred_rf_filename, y_test_pred_rf)
#print("Random Forest Test Predictions: ", y_test_pred_rf)

print("\nValidation set:")
val_loss_rf = mean_squared_error(y_val, y_val_pred_rf)
print("MSE: ", val_loss_rf)
print("R^2 score: ", r2_score(y_val, y_val_pred_rf))

#Print the MSE and R^2 score for the LSTM model
print("======================================")
print("LSTM Model:")
print("Training set:")


train_loss_lstm = mean_absolute_error(y_train, y_train_pred_lstm)
print("MAE: ", train_loss_lstm)
print("R^2 score: ", r2_score(y_train, y_train_pred_lstm))
print("\nTesting set:")

y_train_pred_lstm_filename = "y_train_pred_lstm.txt"
create_file(y_train_pred_lstm_filename, y_train_pred_lstm)
#print("LSTM Test Predictions: ", y_test_pred_lstm)

test_loss_lstm = mean_absolute_error(y_test, y_test_pred_lstm)
print("MAE: ", test_loss_lstm)
print("R^2 score: ", r2_score(y_test, y_test_pred_lstm))

y_test_pred_lstm_filename = "y_test_pred_lstm.txt"
create_file(y_test_pred_lstm_filename, y_test_pred_lstm)
#print("LSTM Test Predictions: ", y_test_pred_lstm)

print("\nValidation set:")
val_loss_lstm = mean_absolute_error(y_val, y_val_pred_lstm)
print("MAE: ", val_loss_lstm)
print("R^2 score: ", r2_score(y_val, y_val_pred_lstm))
# Print the predicted values
#print('LSTM TEST Predictions:', y_test_pred_lstm)
#print('LSTM TRAIN Predictions:', y_train_pred_lstm)

# Print the MSE and R^2 score for the training, testing, and validation sets for the RNN model
print("======================================")
print("SimpleRNN Model:")
print("Training set:")
train_loss_rnn = mean_absolute_error(y_train, y_train_pred_rnn)
print("MAE: ", train_loss_rnn)
print("R^2 score: ", r2_score(y_train, y_train_pred_rnn))

y_train_pred_rnn_filename = "y_train_pred_rnn.txt"
create_file(y_train_pred_rnn_filename, y_train_pred_rnn)
#print("LSTM Test Predictions: ", y_train_pred_rnn)

print("\nTesting set:")
test_loss_rnn = mean_absolute_error(y_test, y_test_pred_rnn)
print("MAE: ", test_loss_rnn)
print("R^2 score: ", r2_score(y_test, y_test_pred_rnn))

y_test_pred_rnn_filename = "y_test_pred_rnn.txt"
create_file(y_test_pred_rnn_filename, y_test_pred_rnn)
#print("LSTM Test Predictions: ", y_test_pred_rnn)

print("\nValidation set:")
val_loss_rnn = mean_absolute_error(y_val, y_val_pred_rnn)
print("MAE: ", val_loss_rnn)
print("R^2 score: ", r2_score(y_val, y_val_pred_rnn))
# Print the predicted values
#print('RNN TEST Predictions:', y_test_pred_rnn)
#print('RNN TRAIN Predictions:', y_train_pred_rnn)

# Plot predicted vs. actual values for Linear Regression model on testing set
#plt.scatter(y_test, y_test_pred_lin)
#plt.xlabel('Actual TEC')
#plt.ylabel('Predicted TEC')
#plt.title('Linear Regression Model: Predicted vs. Actual TEC (Testing Set)')
#plt.show()

# Plotting the results for Linear Regression
fig, ax = plt.subplots()
ax.scatter(y_test, y_test_pred_lin, color='blue')
ax.set_xlabel('Actual TEC')
ax.set_ylabel('Predicted TEC')
ax.set_title('Linear Regression Model: Predicted vs. Actual TEC (Testing Set)')
ax.plot(np.arange(0, 55, 1), np.arange(0, 55, 1), color='black', linestyle='-')
plt.savefig('fig1.png')

# Plotting the results for Gradient Boosting
fig, ax = plt.subplots()
ax.scatter(y_test, y_test_pred_gb, color='yellow')
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Gradient Boosting Model - Actual vs. Predicted Values (Testing Set)")
ax.plot(np.arange(0, 55, 1), np.arange(0, 55, 1), color='black', linestyle='-')
#plt.show()
plt.savefig('fig2.png')

# Plotting the results for Random Rainforest
fig, ax = plt.subplots()
ax.scatter(y_test, y_test_pred_rf, color='grey')
ax.set_xlabel('Actual TEC')
ax.set_ylabel('Predicted TEC')
ax.set_title('Random Forest Model: Predicted vs. Actual TEC (Testing Set)')
ax.plot(np.arange(0, 55, 1), np.arange(0, 55, 1), color='black', linestyle='-')
plt.savefig('fig3.png')

# Plotting the results for LSTM
fig, ax = plt.subplots()
ax.scatter(y_test, y_test_pred_lstm, color='red')
ax.set_xlabel('Actual TEC')
ax.set_ylabel('Predicted TEC')
ax.set_title('LSTM Model: Predicted vs. Actual TEC (Testing Set)')
ax.plot(np.arange(0, 55, 1), np.arange(0, 55, 1), color='black', linestyle='-')
plt.savefig('fig4.png')

# Plotting the results for RNN
fig, ax = plt.subplots()
ax.scatter(y_test, y_test_pred_rnn, color='green')
ax.set_xlabel('Actual TEC')
ax.set_ylabel('Predicted TEC')
ax.set_title('RNN Model: Predicted vs. Actual TEC (Testing Set)')
ax.plot(np.arange(0, 55, 1), np.arange(0, 55, 1), color='black', linestyle='-')
plt.savefig('fig5.png')

#plt.show()
'''
#Plot training and validation losses for LSTM and SimpleRNN
plt.figure(figsize=(10, 5))
plt.plot(history_lstm.history['loss'],color='Red', label='LSTM Training Loss')
plt.plot(history_lstm.history['val_loss'],color='Blue', linestyle='--', label='LSTM Validation Loss')
plt.plot(history_rnn.history['loss'],color='Brown', label='RNN train')
plt.plot(history_rnn.history['val_loss'],color='Black', linestyle='--', label='RNN val')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.savefig('fig6.png')
#plt.show()


#Plot training and validation losses for all models
plt.figure(figsize=(10, 5))
plt.plot(history_lstm.history['loss'],color='Red', label='LSTM Training Loss')
plt.plot(history_lstm.history['val_loss'],color='Blue', linestyle='--', label='LSTM Validation Loss')
plt.axhline(y=train_loss_lin, color='Orange', label='Linear Regression Training Loss')
plt.axhline(y=val_loss_lin, color='Green', linestyle='--', label='Linear Regression Validation Loss')
plt.axhline(y=train_loss_gb, color='Yellow', label='Gradient Boosting Training Loss')
plt.axhline(y=val_loss_gb, color='Pink', linestyle='--', label='Gradient Boosting Validation Loss')
plt.axhline(y=train_loss_rf, color='Purple', label='Random Forest Training Loss')
plt.axhline(y=val_loss_rf, color='Grey', linestyle='--', label='Random Forest Validation Loss')
plt.plot(history_rnn.history['loss'],color='Brown', label='RNN train')
plt.plot(history_rnn.history['val_loss'],color='Black', linestyle='--', label='RNN val')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Relative Error')
plt.legend()
plt.savefig('fig7.png')
#plt.show()
'''

# plot actual vs predicted values for linear regression and gradient boosting
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_test_pred_lin, color='blue', label='Linear Regression')
plt.scatter(y_test, y_test_pred_gb, color='red', label='Gradient Boosting')
plt.scatter(y_test, y_test_pred_rf, color='green', label='Random Forest')
plt.plot(np.arange(0, 55, 1), np.arange(0, 55, 1), color='black', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.savefig('fig8.png')
#plt.show()


'''
# Create a list of model names and MSE scores for the training, testing, and validation sets
models = ['Linear Regression', 'Gradient Boosting', 'LSTM']
train_mse = [train_loss_lin, train_loss_gb, train_loss_lstm]
test_mse = [test_loss_lin, test_loss_gb, test_loss_lstm]
val_mse = [val_loss_lin, val_loss_gb, val_loss_lstm]

# Plot the MSE scores for the three models
plt.figure(figsize=(10, 6))
plt.bar(models, train_mse, width=0.25, color='blue', alpha=0.8, label='Training set')
plt.bar([m + 0.25 for m in models], test_mse, width=0.25, color='red', alpha=0.8, label='Testing set')
plt.bar([m + 0.5 for m in models], val_mse, width=0.25, color='green', alpha=0.8, label='Validation set')
plt.ylim(0, 0.02)
plt.ylabel('MSE')
plt.title('Model comparison')
plt.legend()
plt.show()
'''
