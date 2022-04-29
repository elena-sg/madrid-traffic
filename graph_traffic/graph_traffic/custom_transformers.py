import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from paths import path
from datetime import datetime


cal = pd.read_csv(path+'calendario.csv', sep=';')
cal['fecha'] = pd.to_datetime(cal['Día'], dayfirst=True)
cal['laborable / festivo / domingo festivo'] = cal['laborable / festivo / domingo festivo'].str.lower()
festivos = cal[cal['laborable / festivo / domingo festivo']=='festivo'].fecha.dt.date.unique()

def rain_categories(rain):
    rain_intervals = [
    rain == 0,
    rain < 2.5,
    rain >= 2.5
    ]

    rain_categories = [
        "no_rain", "moderate_rain", "strong_rain"
    ]

    return np.select(rain_intervals, rain_categories, default="no_rain")

rain_categories_transformer = FunctionTransformer(rain_categories)

def day_type(dia_semana):
    conditions = [
        dia_semana == 6,
        dia_semana < 5,
        dia_semana == 5
    ]
    
    day_types = [
        "domingo",
        "lunes-viernes",
        "sabado"
    ]
    
    return np.select(conditions, day_types, default=np.nan)

def season(fecha):
    dt = fecha.dt
    conditions = [
        (dt.month <= 3) & (dt.day <= 21),
        (dt.month <= 6) & (dt.day <= 21),
        (dt.month <= 9) & (dt.day <= 22),
        (dt.month <= 12) & (dt.day <= 21),
        (dt.month <= 12) & (dt.day <= 31)
    ]
    seasons = [
        "invierno",
        "primavera",
        "verano",
        "otono",
        "invierno"
    ]
    return np.select(conditions, seasons)


def school_holidays(fecha):
    condition = (fecha <= datetime(2019, 7, 1)) |\
                (fecha.between(
                    datetime(2019, 3, 1), datetime(2019, 3, 4)
                )) |\
                (fecha.between(
                    datetime(2019, 4, 12), datetime(2019, 4, 22)
                )) |\
                (fecha.between(
                    datetime(2019, 5, 1), datetime(2019, 5, 5)
                )) |\
                (fecha.between(
                    datetime(2019, 6, 21), datetime(2019, 9, 10)
                )) |\
                (fecha.between(
                    datetime(2019, 10, 31), datetime(2019, 11, 3)
                )) |\
                (fecha.between(
                    datetime(2019, 12, 7), datetime(2019, 12, 9)
                )) |\
                (fecha.between(
                    datetime(2019, 12, 21), datetime(2020, 1, 7)
                )) |\
                (fecha.between(
                    datetime(2020, 2, 28), datetime(2020, 3, 2)
                )) |\
                (fecha.between(
                    datetime(2020, 4, 3), datetime(2020, 4, 13)
                )) |\
                (fecha.between(
                    datetime(2020, 5, 1), datetime(2020, 5, 1)
                )) |\
                (fecha.between(
                    datetime(2020, 6, 23), datetime(2020, 9, 9)
                )) |\
                (fecha.between(
                    datetime(2020, 10, 12), datetime(2020, 10, 12)
                )) |\
                (fecha.between(
                    datetime(2020, 11, 2), datetime(2020, 11, 2)
                )) |\
                (fecha.between(
                    datetime(2020, 11, 9), datetime(2020, 11, 9)
                )) |\
                (fecha.between(
                    datetime(2020, 12, 7), datetime(2020, 12, 8)
                )) |\
                (fecha.between(
                    datetime(2020, 12, 23), datetime(2021, 1, 10)
                )) |\
                (fecha.between(
                    datetime(2021, 2, 19), datetime(2021, 2, 22)
                )) |\
                (fecha.between(
                    datetime(2021, 3, 19), datetime(2021, 3, 19)
                )) |\
                (fecha.between(
                    datetime(2021, 3, 26), datetime(2021, 4, 5)
                )) |\
                (fecha.between(
                    datetime(2021, 5, 1), datetime(2021, 5, 4)
                )) |\
                (fecha.between(
                    datetime(2021, 6, 23), datetime(2021, 9, 8)
                )) |\
                (fecha.between(
                    datetime(2021, 10, 9), datetime(2021, 10, 12)
                )) |\
                (fecha.between(
                    datetime(2021, 10, 30), datetime(2021, 11, 1)
                )) |\
                (fecha.between(
                    datetime(2021, 12, 4), datetime(2021, 12, 8)
                )) |\
                (fecha.between(
                    datetime(2021, 12, 23), datetime(2022, 1, 9)
                ))
    return condition
                
                
def rows_no_change(col):
    return col.groupby(
        ((col != col.shift())).cumsum()
    ).transform('size')

def make_stable_values_null(col, nrows=4):
    no_change = rows_no_change(col)
    return np.where(no_change > nrows, np.nan, col)

interpolate_transformer = FunctionTransformer(lambda x: x.interpolate(method="linear", limit=4))