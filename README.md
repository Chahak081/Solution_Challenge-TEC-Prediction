# Solution_Challenge-TEC-Prediction
The TEC in the ionosphere is modified by changing solar Extreme UV radiation, geomagnetic storms, and the atmospheric waves that propagate up from the lower atmosphere. Therefore depend on season, geomagnetic conditions, solar cycle and activity which impact climate change.
Guidelines to run code
The code works in any python ide like VSCode, Spyder and etc.

1)Importing Libraries:

* numpy and pandas for data handling
* scikit-learn for preprocessing and modeling
* tensorflow.keras for building deep learning models
* matplotlib for visualization

2)Loading Data:
* The data is loaded from an Excel file named 'data1.xlsx' using pandas

3)Preprocessing:
* The target variable 'TEC_top' is extracted and converted to a numpy array
* The input data is extracted from the DataFrame and normalized using StandardScaler

4)Sequence Generation:
* The input and target sequences are generated from the preprocessed data using a sliding window approach.
* The sequence length is set to 20 and the sequences are split into training and validation sets

5)LSTM Model:
* A sequential model with an LSTM layer, a dropout layer and a dense layer is defined
* The model is compiled with an Adam optimizer and mean squared error loss
* The model is trained on the training data and evaluated on both the training and validation sets

6)MLP Model:
* A sequential model with a flatten layer, a dense layer, a dropout layer and a final dense layer is defined
* The model is compiled with an Adam optimizer and mean squared error loss
* The model is trained on the training data and evaluated on both the training and validation sets

7)CNN Model:
* A sequential model with a convolutional layer, a max pooling layer, a dropout layer, a flatten layer and a final dense layer is defined
* The model is compiled with an Adam optimizer and mean squared error loss
* The model is trained on the training data and evaluated on both the training and validation sets

8)Linear Regression Model:
* A linear regression model is defined using scikit-learn
* The model is trained on the training data and evaluated on both the training and validation sets

9)Decision Tree Model:
* A decision tree model is defined using scikit-learn
* The model is trained on the training data and evaluated on both the training and validation sets

10)Random Forest Model:
* A random forest model with 100 trees is defined using scikit-learn
* The model is trained on the training data and evaluated on both the training and validation sets

11)Printing Results:
* The training and validation losses for all the models are printed

12)Visualization:
* A graph of the training and validation losses for the LSTM, MLP and CNN models is plotted using matplotlib

Overall, the code is a demonstration of various models that can be used for time series forecasting, including deep learning models like LSTM and CNN, and traditional machine learning models like linear regression, decision trees and random forests. The models are trained and evaluated on a dataset of TEC_top values, which is a measure of the total electron content of the ionosphere. The performance of each model is compared and the results are printed and visualized.

Instructions to run the code:

1. Ensure that all the necessary libraries are installed on your system. The libraries required for this code are numpy, pandas, scikit-learn, tensorflow and matplotlib. If any of these libraries are not installed, you can install them using pip.

2. Download the dataset named "data1.xlsx" and ensure that it is saved in the same directory as your code file.

3. Open your preferred Python editor and create a new Python file.

4. Copy and paste the code into the new Python file.

5. Save the file with a suitable name and the ".py" extension.

6. Run the code by either clicking on the Run button in your Python editor or by typing "python filename.py" in the command prompt or terminal.

7. The code will load the data from the "data1.xlsx" file, preprocess the data and train five different models - LSTM, MLP, CNN, Linear Regression and Decision Tree. The trained models will be evaluated on the validation data and the training and validation losses for each model will be printed.

8. The code will also plot the training and validation losses for the LSTM, MLP and CNN models. The plots will be displayed on the screen.

9. Once the code finishes executing, you should be able to see the training and validation losses for each model printed on the screen. You should also be able to see the plots for the LSTM, MLP and CNN models.




