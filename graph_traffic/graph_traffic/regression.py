from datetime import datetime, timedelta

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import clone
from graph_traffic.model_selection import timeseries_cv
from graph_traffic.custom_transformer import transform_df
from graph_traffic.config import project_path
from graph_traffic.merge_data import merge_data
from graph_traffic.get_data import get_mmagns
import itertools
from time import time
import pickle
import matplotlib as mpl
import numpy as np

mpl.rcParams['axes.grid'] = False



def get_combinations(dict_possible):
    keys, values = zip(*dict_possible.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def try_combinations(data_dict, meteo_combinations, temporal_combinations, pipeline):
    training_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_folder = f"{project_path}/training_history/regression"

    meteo_values = {}
    temporal_values = {}
    results = {}
    training_time = {}
    alpha = {}

    for i, meteo_dict in enumerate(meteo_combinations):
        print(f"\n{i}")
        meteo_values[i] = meteo_dict

        mmagns = get_mmagns(meteo_dict)

        df = merge_data(data_dict["ids_list"][0], data_dict["from_date"], data_dict["to_date"], data_dict["target"], mmagns)

        with open(f"{training_folder}/{training_datetime}_meteo_values.pkl", "wb") as f:
            pickle.dump(meteo_values, f)

        for j, temporal_dict in enumerate(temporal_combinations):
            df_t = transform_df(df, meteo_dict, temporal_dict, data_dict["interactions"], data_dict["target"])

            data_size = df_t.shape[0]

            train_x = df_t[:int(0.8 * data_size):11, 1:]
            train_y = df_t[:int(0.8 * data_size):11, 0].ravel()

            if np.linalg.matrix_rank(train_x) != train_x.shape[1]:
                continue

            temporal_values[j] = temporal_dict
            print(j, end="\r")

            start_time = time()
            _, _, results[(i, j)], alpha[(i, j)] = timeseries_cv(pipeline, train_x, train_y, with_previous_timesteps=False,
                                                                 with_alpha=True)
            training_time[(i, j)] = time() - start_time

        if i == 0:
            with open(f"{training_folder}/{training_datetime}_temporal_values.pkl", "wb") as f:
                pickle.dump(temporal_values, f)

        with open(f"{training_folder}/{training_datetime}_results.pkl", "wb") as f:
            pickle.dump(results, f)

        with open(f"{training_folder}/{training_datetime}_times.pkl", "wb") as f:
            pickle.dump(training_time, f)

        with open(f"{training_folder}/{training_datetime}_alphas.pkl", "wb") as f:
            pickle.dump(alpha, f)


def train_with_args(data_dict, meteo_dict, temporal_dict, pipeline_class, train_until=None):
    mmagns = get_mmagns(meteo_dict)
    #dates = pd.date_range(data_dict["from_date"], data_dict["to_date"], freq="15min")
    dfs_dict = {}
    ids_used = []
    train_sizes = {}
    test_sizes = {}
    for i in data_dict["ids_list"]:
        print(i, end="\r")
        dfs_dict[i] = merge_data(i, data_dict["from_date"], data_dict["to_date"], data_dict["target"], mmagns)
        if train_until is None:
            train_sizes[i] = int(0.8 * dfs_dict[i].shape[0])
            test_sizes[i] = int(0.2 * dfs_dict[i].shape[0])
        else:
            train_sizes[i] = len(dfs_dict[i][dfs_dict[i].date <= train_until])
            test_sizes[i] = len(dfs_dict[i][(dfs_dict[i].date > train_until) &
                                            (dfs_dict[i].date <= train_until + timedelta(days=30))])
        #if dates.intersection(dfs_dict[i].date).empty:
        #    continue
        #dates = dates.intersection(dfs_dict[i].date)
        #ids_used.append(i)

    for i in data_dict["ids_list"]:
        df = dfs_dict[i]
        #df = df[df.date.isin(dates)]
        dfs_dict[i] = transform_df(df, meteo_dict, temporal_dict, data_dict["interactions"], data_dict["target"])

    #data_size = dfs_dict[i].shape[0]

    #all_hours = dates.hour + dates.minute / 60

    #test_dates = all_hours.values[int(0.8 * data_size):]

    estimators = {}
    maes = {}
    mses = {}
    for sensor_id in data_dict["ids_list"]:
        print(sensor_id)
        train_x = dfs_dict[sensor_id][:train_sizes[sensor_id], 1:]
        train_y = dfs_dict[sensor_id][:train_sizes[sensor_id], 0].ravel()

        test_x = dfs_dict[sensor_id][train_sizes[sensor_id]:train_sizes[sensor_id]+test_sizes[sensor_id], 1:]
        test_y = dfs_dict[sensor_id][train_sizes[sensor_id]:train_sizes[sensor_id]+test_sizes[sensor_id], 0].ravel()
        pipeline = clone(pipeline_class)
        print("Shape of train predictors and labels:", train_x.shape, train_y.shape)
        pipeline.fit(train_x, train_y)

        estimators[sensor_id] = pipeline

        test_pred = pipeline.predict(test_x)
        maes[sensor_id] = mean_absolute_error(test_y, test_pred)
        mses[sensor_id] = mean_squared_error(test_y, test_pred)
        print("MAE:", maes[sensor_id])
        print("MSE:", mses[sensor_id])

    return ids_used, estimators, dfs_dict, maes, mses


def coefs_plot(ids_used, estimators, column_names, title="Model coefficients"):
    fig, axs = plt.subplots(1, len(ids_used), figsize=(8, 10), sharey=True)
    for j, i in enumerate(ids_used):
        ax = axs[j]
        coefs = estimators[i][-1].coef_
        pd.DataFrame(zip(coefs, column_names)).iloc[::-1].rename(columns={0: "importances", 1: "features"}).plot.barh(
            x=1, ax=ax, legend=False)
        ax.set_title(f"{i}")
    fig.suptitle(title)
    plt.show()

