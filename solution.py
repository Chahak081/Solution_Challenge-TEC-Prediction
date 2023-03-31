import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Conv1D, MaxPooling1D
import matplotlib.pyplot as plt

# Load the data from file
df = pd.read_excel('data1.xlsx')
print(df.columns)
# Extract the TEC_top data and convert to NumPy array
y = df['TEC_top'].to_numpy()

# Extract the input data and normalize using StandardScaler
scaler = StandardScaler()
X = df.iloc[:, 4:-1].values
X = scaler.fit_transform(X)

# Define input and output sequences
seq_len = 20
X_train, y_train = [], []
X_val, y_val = [], []
for i in range(seq_len, X.shape[0]):
    if i < int(0.8*X.shape[0]):
        X_train.append(X[i-seq_len:i, :])
        y_train.append(y[i])
    else:
        X_val.append(X[i-seq_len:i, :])
        y_val.append(y[i])
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)

# Define LSTM model
model = Sequential()
model.add(LSTM(units=128, input_shape=(seq_len, X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')

model.summary()

# Train LSTM model
history = model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_val, y_val))

# Evaluate LSTM model
train_loss = model.evaluate(X_train, y_train)
val_loss = model.evaluate(X_val, y_val)

print(f'LSTM training loss: {train_loss:.4f}')
print(f'LSTM validation loss: {val_loss:.4f}')

# Define Multi-Layer Perceptron model
mlp_model = Sequential()
mlp_model.add(Flatten(input_shape=(seq_len, X_train.shape[2])))
mlp_model.add(Dense(units=128, activation='relu'))
mlp_model.add(Dropout(0.2))
mlp_model.add(Dense(units=1))
mlp_model.compile(optimizer='adam', loss='mse')

mlp_model.summary()

# Train MLP model
history_mlp = mlp_model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_val, y_val))

# Evaluate MLP model
train_loss_mlp = mlp_model.evaluate(X_train, y_train)
val_loss_mlp = mlp_model.evaluate(X_val, y_val)

print(f'MLP training loss: {train_loss_mlp:.4f}')
print(f'MLP validation loss: {val_loss_mlp:.4f}')

# Define Convolutional Neural Network model
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(seq_len, X_train.shape[2])))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Dropout(0.2))
cnn_model.add(Flatten())
cnn_model.add(Dense(units=1))
cnn_model.compile(optimizer='adam', loss='mse')

#Train CNN model
history_cnn = cnn_model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_val, y_val))

#Evaluate CNN model
train_loss_cnn = cnn_model.evaluate(X_train, y_train)
val_loss_cnn = cnn_model.evaluate(X_val, y_val)

print(f'LSTM training loss: {train_loss:.4f}')
print(f'LSTM validation loss: {val_loss:.4f}')
print(f'MLP training loss: {train_loss_mlp:.4f}')
print(f'MLP validation loss: {val_loss_mlp:.4f}')

print(f'CNN training loss: {train_loss_cnn:.4f}')
print(f'CNN validation loss: {val_loss_cnn:.4f}')

# Define Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Evaluate Linear Regression model
train_loss_lr = np.mean((lr_model.predict(X_train.reshape(X_train.shape[0], -1)) - y_train) ** 2)
val_loss_lr = np.mean((lr_model.predict(X_val.reshape(X_val.shape[0], -1)) - y_val) ** 2)

print(f'Linear Regression training loss: {train_loss_lr:.4f}')
print(f'Linear Regression validation loss: {val_loss_lr:.4f}')

# Define Decision Tree model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Evaluate Decision Tree model
train_loss_dt = np.mean((dt_model.predict(X_train.reshape(X_train.shape[0], -1)) - y_train) ** 2)
val_loss_dt = np.mean((dt_model.predict(X_val.reshape(X_val.shape[0], -1)) - y_val) ** 2)

print(f'Decision Tree training loss: {train_loss_dt:.4f}')
print(f'Decision Tree validation loss: {val_loss_dt:.4f}')

# Define Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

#Evaluate Random Forest model
train_loss_rf = np.mean((rf_model.predict(X_train.reshape(X_train.shape[0], -1)) - y_train) ** 2)
val_loss_rf = np.mean((rf_model.predict(X_val.reshape(X_val.shape[0], -1)) - y_val) ** 2)

print(f'Random Forest training loss: {train_loss_rf:.4f}')
print(f'Random Forest validation loss: {val_loss_rf:.4f}')

# Define Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=10, random_state=42)
gb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Evaluate Gradient Boosting model
train_loss_gb = np.mean((gb_model.predict(X_train.reshape(X_train.shape[0], -1)) - y_train) ** 2)
val_loss_gb = np.mean((gb_model.predict(X_val.reshape(X_val.shape[0], -1)) - y_val) ** 2)

print(f'Gradient Boosting training loss: {train_loss_gb:.4f}')
print(f'Gradient Boosting validation loss: {val_loss_gb:.4f}')

#Plot training and validation losses for all models
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='LSTM Training Loss')
plt.plot(history.history['val_loss'], label='LSTM Validation Loss')
plt.axhline(y=train_loss_lr, color='r', linestyle='--', label='Linear Regression Training Loss')
plt.axhline(y=val_loss_lr, color='g', linestyle='--', label='Linear Regression Validation Loss')
plt.axhline(y=train_loss_dt, color='m', linestyle='--', label='Decision Tree Training Loss')
plt.axhline(y=val_loss_dt, color='c', linestyle='--', label='Decision Tree Validation Loss')
plt.axhline(y=train_loss_rf, color='y', linestyle='--', label='Random Forest Training Loss')
plt.axhline(y=val_loss_rf, color='k', linestyle='--', label='Random Forest Validation Loss')
plt.axhline(y=train_loss_gb, color='b', linestyle='--', label='Gradient Boosting Training Loss')
plt.axhline(y=val_loss_gb, color='magenta', linestyle='--', label='Gradient Boosting Validation Loss')
plt.plot(history_mlp.history['loss'], label='MLP train')
plt.plot(history_mlp.history['val_loss'], label='MLP val')
plt.plot(history_cnn.history['loss'], label='CNN train')
plt.plot(history_cnn.history['val_loss'], label='CNN val')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Make predictions using LSTM model
#y_pred_lstm = model.predict(X_val)
#y_pred_lstm = y_pred_lstm[:, 0, 0]  # reshape to (39,)
#print(y_pred_lstm)

# Plot predicted vs actual values
#plt.plot(y_val, label='Actual')
#plt.plot(y_pred_lstm, label='Predicted')
#plt.legend()
#plt.show()

