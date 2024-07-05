import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df1 = pd.read_csv('/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output/train.csv')
df2 = pd.read_csv('/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output/valid.csv')

# Extracting training and validation loss columns
training_loss = df1['Value']
validation_loss = df2['Value']

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(training_loss, label='Training Loss')
# plt.plot(validation_loss, label='Validation Loss')
plt.title('GNN Training Loss, LR_init = 0.0001')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjust layout
plt.show()
