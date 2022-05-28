from datetime import datetime
from graph_traffic.model_selection import timeseries_cv
from graph_traffic.custom_transformer import transform_df
from graph_traffic.config import project_path
from graph_traffic.merge_data import merge_data
import itertools
from time import time
import pickle
import matplotlib as mpl

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
        # if i<=40:
        #    continue
        print(f"\n{i}")
        meteo_values[i] = meteo_dict

        mmagns = []
        if meteo_dict["rain"] != "drop":
            mmagns.append("precipitacion")
        if meteo_dict["wind"] != "drop":
            mmagns += ["dir_viento", "velocidad_viento"]
        if meteo_dict["temperature"] != "drop":
            mmagns.append("temperatura")
        if meteo_dict["humidity"] != "drop":
            mmagns.append("humedad_relativa")
        if meteo_dict["pressure"] != "drop":
            mmagns.append("presion_barometrica")
        if meteo_dict["radiation"] != "drop":
            mmagns.append("radiacion_solar")

        df = merge_data(data_dict["ids_list"][0], data_dict["from_date"], data_dict["to_date"], data_dict["target"], mmagns)

        with open(f"{training_folder}/{training_datetime}_meteo_values.pkl", "wb") as f:
            pickle.dump(meteo_values, f)

        for j, temporal_dict in enumerate(temporal_combinations):
            df_t = transform_df(df, meteo_dict, temporal_dict, data_dict["interactions"], data_dict["target"])

            data_size = df_t.shape[0]

            train_x = df_t[:int(0.8 * data_size):11, 3:]
            train_y = df_t[:int(0.8 * data_size):11, 0].ravel()

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
