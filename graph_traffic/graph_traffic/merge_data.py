import numpy as np
import os
import pandas as pd
from graph_traffic.custom_funcs import  make_stable_values_null, rows_no_change
from graph_traffic.custom_transformer import get_season, day_type, bank_holidays, school_holidays
from graph_traffic.config import data_path

traffic_path = os.path.join(data_path, "03-by-location", "traffic")
meteo_path = os.path.join(data_path, "03-by-location", "meteo")


tmagns = ['intensidad', 'ocupacion', 'vmed']
mmagns = ['temperatura', 'humedad_relativa', 'presion_barometrica', 'radiacion_solar',
          'precipitacion', 'dir_viento', 'velocidad_viento']

mapping = pd.read_csv(os.path.join(data_path, '03-by-location', 'id_mapping.csv'))

def merge_data(id_t):
    ids_m = mapping[mapping.id_t == id_t].iloc[0][[f'id_{magn}' for magn in mmagns]].astype(int)

    # read traffic data
    dft = pd.read_csv(f'{traffic_path}/{id_t:0d}.csv', parse_dates=['fecha'], index_col='fecha')
    if dft.empty:
        raise ValueError("No data for the provided id")
    # read meteorological data
    dfm = {estacion: pd.read_csv(f'{meteo_path}/estacion-{estacion:.0f}.csv', parse_dates=['fecha'], index_col='fecha') for estacion in ids_m.unique()}

    # Si hay más de 4 filas sin cambio, damos el valor por nulo
    dft[tmagns] = dft[tmagns].apply(make_stable_values_null, nrows=4)
    for estacion, dfmi in dfm.items():
        nm = dfmi[mmagns].apply(rows_no_change)
        for m in mmagns:
            if m in ['precipitacion', 'radiacion_solar', 'presion_barometrica']:
                continue
            dfmi[m] = np.where((nm[m]>4) & (dfmi[m]!=0), np.nan, dfmi[m])
        dfm[estacion] = dfmi

    # Hacer el merge de todas las variables meteorológicas con el tráfico
    df = dft
    for m in mmagns:
        df = df.merge(dfm[ids_m[f"id_{m}"]][[m]],
                     left_index=True, right_index=True,
                    how='left')

    del dft, dfm
    df = df.sort_index()
    df[mmagns] = df[mmagns].interpolate(method="linear", limit=4)

    # fill gaps in the date
    dates = pd.date_range("2019-01-01", "2020-12-31", freq="15min")
    df = df.reindex(dates).reset_index().rename(columns={"index": "date"})

    # new features based on the calendar
    df["year"] = df.date.dt.year
    df["season"] = get_season(df.date)
    df["month"] = df.date.dt.month
    df["week"] = df.date.dt.week
    df["day_of_month"] = df.date.dt.day
    df["day_type"] = day_type(df.date)
    df["week_day"] = df.date.dt.weekday
    df["bank_holiday"] = df.date.dt.date.isin(bank_holidays)
    df["working_day"] = (df.date.dt.weekday <= 4) & (~df.bank_holiday)
    df["school_holiday"] = school_holidays(df.date) | (~df.working_day)
    df["hour"] = df.date.dt.hour + df.date.dt.minute / 60
    df["minute"] = df.date.dt.minute

    # amount of wind to east and north
    wv = df["velocidad_viento"]
    wd_rad = df['dir_viento'] * np.pi / 180
    df['windx'] = wv * np.cos(wd_rad)
    df['windy'] = wv * np.sin(wd_rad)

    return df