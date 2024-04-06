
import pandas as pd
import numpy as np

def plot_metrices(training_accuracy, training_loss, testing_accuracy, testing_loss):
    metrics_df = pd.DataFrame(np.array(training_accuracy), columns=["accuracy"])
    metrics_df["val_accuracy"] = np.array(testing_accuracy)
    metrics_df["loss"] = np.array(training_loss)
    metrics_df["val_loss"] = np.array(testing_loss)
    metrics_df[["loss", "val_loss"]].plot()
    metrics_df[["accuracy", "val_accuracy"]].plot()