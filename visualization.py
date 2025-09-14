import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/output 3.csv')

def visualize_carbon_intensity(data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Datetime', y='carbon_intensity', data=data, color='green')
    plt.title('Carbon Intensity Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('Carbon Intensity')
    plt.show()

#visualize_carbon_intensity(data)

def visualize_renewable_energy(data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Datetime', y='Renewable energy percentage (RE%)', data=data, color='blue')
    plt.title('Renewable Energy Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('Renewable Energy')
    plt.show()

#visualize_renewable_energy(data)

data.to_csv('data/final.csv')