import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(df, column):
    plt.figure()
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of{column}")
    plt.show()
    
def plot_target_relationship(df, feature, target):
    plt.figure()
    sns.boxplot(x=target, y=feature, data=df)
    plt.show()