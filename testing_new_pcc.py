import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data into a DataFrame
data = pd.read_csv('/Users/anaselyoussef/Desktop/algo/outputfiles_ffnn_predictions.txt', header=None, names=['Peptide', 'Prediction', 'Target'])

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(data['Target'], data['Prediction'])
print(f'Mean Squared Error (MSE): {mse}')

# Calculate Correlation
correlation = data['Target'].corr(data['Prediction'])
print(f'Correlation between Predictions and Targets: {correlation}')

# Plotting Predictions vs Targets
plt.figure(figsize=(10, 6))
plt.scatter(data['Target'], data['Prediction'], alpha=0.5)
plt.xlabel('Actual Target')
plt.ylabel('Predicted Value')
plt.title('Predicted Values vs Actual Targets')
plt.show()
