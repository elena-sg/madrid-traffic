from matplotlib import pyplot as plt
from graph_traffic.config import figures_path, data_path
from graph_traffic.merge_data import merge_data
from matplotlib.dates import YearLocator
import os
import pandas as pd
import matplotlib as mpl

mpl.rcParams['axes.grid'] = False
mpl.rcParams.update({'font.size': 12})

figures_path_expl = f"{figures_path}/explorative"

titles = {
    "intensidad": "Cars / hour",
    "ocupacion": "% time busy road",
    "vmed": "Average speed (Km/h)",
    "carga": "Uso de la vía (0-100)"
}

sup_title = dict(
    intensidad="Intensidad",
    ocupacion="Ocupación",
    vmed="Velocidad media"
)

sup_title_meteo = dict(
    velocidad_viento="Velocidad del viento",
    dir_viento="Dirección del viento",
    temperatura="Temperatura",
    humedad_relativa="Humedad relativa",
    presion_barometrica="Presión barómetrica",
    radiacion_solar="Radiación solar",
    precipitacion="Precipitación"
)

targets = ["intensidad", "ocupacion", "vmed"]
mmagns = ["velocidad_viento", "dir_viento", "temperatura", "humedad_relativa", "presion_barometrica",
          "radiacion_solar", "precipitacion"]


def plot_date_hist(df, ax):
    ax.hist(df.date, bins=52*3)
    #df.date.hist(ax=ax, bins=52 * 3)
    # _ = ax.set(
    #     xticks=["2019-01-01", "2020-01-01", "2021-01-01", "202-01-01"],
    #     xticklabels=["2019", "2020", "2021", "2022"]
    # )
    ax.xaxis.set_major_locator(YearLocator())
    return ax


def plot_week(df, target, ax):
    df_year = df.groupby(["year", "weekday", "hour"]).mean()
    for year in df_year.index.unique(level="year"):
        df_year.loc[year][target].plot(ax=ax, label=year)
    _ = ax.set(
        xticks=[i * 4* 24 for i in range(7)],
        xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        xlabel="Día de la semana",
        ylabel=titles[target],
    )
    return ax


def describe_magnitude(subfig, id, target):
    df = merge_data(id, mmagns=[], target=target)
    #fig, ax = plt.subplots(1, 2, figsize=(20, 2))
    ax = subfig.subplots()#, figsize=(20, 2))
    ax = plot_week(df, target, ax)
    #ax[1] = plot_date_hist(df, ax[1])
    ax.legend()

    #subfig.tight_layout()
    subfig.suptitle(sup_title[target], y=0.99)
    return subfig


def describe_all_magnitudes(id):
    fig = plt.figure(figsize=(15, len(targets)*2))
    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=len(targets), ncols=1)
    for i, target in enumerate(targets):
        subfigs[i] = describe_magnitude(subfigs[i], id, target)
    fig.tight_layout()
    plt.savefig(f"{figures_path_expl}/magnitudes.png")
    plt.show()


def get_plot_n_observations():
    files_path = data_path + "/03-by-location/traffic"
    dates = dict.fromkeys(targets, pd.Series([]))
    ids_no_data = dict.fromkeys(targets, [])
    files = os.listdir(files_path)
    n_files = len(files)
    fig, ax = plt.subplots(len(targets), 1, figsize=(15, len(targets)*3))
    for i, file in enumerate(files):
        print(f"{i}/{n_files}", end="\r")
        id = int(file.split(".")[0])
        for target in targets:
            try:
                df = merge_data(id, target=target)
                dates[target] = pd.concat([dates[target], df.date], ignore_index=True)
            except Exception:
                ids_no_data[target].append(id)
    for i, target in enumerate(targets):
        ax[i].hist(dates[target], bins=52*3)
        ax[i].set_title(sup_title[target])
        ax[i].xaxis.set_major_locator(YearLocator())
    fig.tight_layout()
    plt.savefig(f"{figures_path_expl}/n_observations.png")
    plt.show()
    return ids_no_data


def get_plot_n_observations_meteo():
    files_path = data_path + "/03-by-location/meteo"
    dates = dict.fromkeys(mmagns, pd.Series([]))
    ids_no_data = dict.fromkeys(mmagns, [])
    files = os.listdir(files_path)
    n_files = len(files)
    fig, ax = plt.subplots(len(mmagns), 1, figsize=(15, len(mmagns)*3))
    for i, file in enumerate(files):
        print(f"{i}/{n_files}", end="\r")
        df = pd.read_csv(files_path+"/"+file, parse_dates=["fecha"])
        for magn in mmagns:
            dates[magn] = pd.concat([dates[magn], df[["fecha", magn]].dropna().fecha], ignore_index=True)
    for i, magn in enumerate(mmagns):
        ax[i].hist(dates[magn], bins=52*3)
        ax[i].set_title(sup_title_meteo[magn])
        ax[i].xaxis.set_major_locator(YearLocator())
    fig.tight_layout()
    plt.savefig(f"{figures_path_expl}/n_observations_meteo.png")
    plt.show()
    return ids_no_data


def get_rel(feature1, feature2, alpha=0.01):
    files_path = data_path + "/03-by-location/traffic"
    list1 = []
    list2 = []
    files = os.listdir(files_path)
    n_files = len(files)

    for i, file in enumerate(files):
        print(f"{i}/{n_files}", end="\r")
        df = pd.read_csv(files_path + "/" + file, usecols=[feature1, feature2]).dropna()
        df = df[(df[feature1] != 0) & (df[feature2] != 0)]
        if not df.empty:
            list1 += df[feature1].tolist()
            list2 += df[feature2].tolist()

    # fig, ax = plt.subplots()
    # ax.scatter(list1, list2, alpha=alpha)
    # ax.set_title(f"Relación entre {sup_title[feature1]} and {sup_title[feature2]}")
    # ax.set_xlabel(f"{sup_title[feature1]}: {titles[feature1]}")
    # ax.set_ylabel(f"{sup_title[feature2]}: {titles[feature2]}")
    # fig.tight_layout()
    # plt.savefig(f"{figures_path_expl}/rel_{feature1}_{feature2}.png")
    # plt.savefig(f"{figures_path_expl}/rel_{feature1}_{feature2}.svg")
    # plt.show()

    return list1, list2