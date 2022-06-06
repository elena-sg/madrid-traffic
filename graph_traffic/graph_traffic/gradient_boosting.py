import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_importances_old(ids_to_use, estimators, column_names):
    fig, axs = plt.subplots(1, len(ids_to_use), figsize=(8, 10), sharey=True)
    for j, i in enumerate(ids_to_use):
        reg = estimators[i]
        ax = axs[j]
        coefs = reg.feature_importances_
        pd.DataFrame(zip(coefs, column_names)).iloc[::-1].rename(columns={0: "importances", 1: "features"}).plot.barh(
            x=1, ax=ax, legend=False)
        ax.set_title(f"{i}")
    fig.suptitle("Feature importances")
    plt.show()

def get_column_score(coefs, column, column_dict):
    if column_dict[column] in coefs.keys():
        return coefs[column_dict[column]]
    else:
        return 0

def plot_feature_importances(ids_to_use, estimators, column_names, horizontal=True):
    column_dict = {col: f"f{i}" for i, col in enumerate(column_names)}
    fig, axs = plt.subplots(1, len(ids_to_use), figsize=(8, 2*len(ids_to_use)), sharey=True)
    if len(ids_to_use) == 1:
        axs = [axs]
    for j, i in enumerate(ids_to_use):
        reg = estimators[i]
        ax = axs[j]
        coefs = reg.get_booster().get_fscore()
        coefs = [coefs[column_dict[col]] for col in column_names]
        if horizontal:
            pd.DataFrame(zip(coefs, column_names)).sort_values(0, ascending=True).rename(columns={0: "importances", 1: "features"}).plot.barh(
                x=1, ax=ax, legend=False)
        else:
            pd.DataFrame(zip(coefs, column_names)).sort_values(0, ascending=False).rename(columns={0: "importances", 1: "features"}).plot.bar(
                x=1, ax=ax, legend=False)
        ax.set_title(f"{i}")
    fig.suptitle("Feature importances")
    plt.show()