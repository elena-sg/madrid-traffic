from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import Nystroem

from graph_traffic.config import data_path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, PowerTransformer, \
    QuantileTransformer, SplineTransformer, KBinsDiscretizer, PolynomialFeatures
from math import floor

# read calendar data to add calendar-related features
cal = pd.read_csv(f"{data_path}/03-by-location/calendario.csv", sep=';')
cal['fecha'] = pd.to_datetime(cal['Día'], dayfirst=True)
cal['laborable / festivo / domingo festivo'] = cal['laborable / festivo / domingo festivo'].str.lower()
bank_holidays = cal[cal['laborable / festivo / domingo festivo'] == 'festivo'].fecha.dt.date.unique()


def day_type(date):
    weekday = date.dt.weekday
    conditions = [
        weekday == 6,
        weekday < 5,
        weekday == 5
    ]

    day_types = [
        "sun",
        "mon-fri",
        "sat"
    ]

    return np.select(conditions, day_types, default=np.nan)


def get_season(date):
    dt = date.dt
    conditions = [
        (dt.month <= 3) & (dt.day <= 21),
        (dt.month <= 6) & (dt.day <= 21),
        (dt.month <= 9) & (dt.day <= 22),
        (dt.month <= 12) & (dt.day <= 21),
        (dt.month <= 12) & (dt.day <= 31)
    ]
    seasons = [
        "winter",
        "spring",
        "summer",
        "fall",
        "winter"
    ]
    return np.select(conditions, seasons)


def school_holidays(date):
    condition = (date <= datetime(2019, 1, 7)) | \
                (date.between(
                    datetime(2019, 3, 1), datetime(2019, 3, 4)
                )) | \
                (date.between(
                    datetime(2019, 4, 12), datetime(2019, 4, 22)
                )) | \
                (date.between(
                    datetime(2019, 5, 1), datetime(2019, 5, 5)
                )) | \
                (date.between(
                    datetime(2019, 6, 21), datetime(2019, 9, 10)
                )) | \
                (date.between(
                    datetime(2019, 10, 31), datetime(2019, 11, 3)
                )) | \
                (date.between(
                    datetime(2019, 12, 7), datetime(2019, 12, 9)
                )) | \
                (date.between(
                    datetime(2019, 12, 21), datetime(2020, 1, 7)
                )) | \
                (date.between(
                    datetime(2020, 2, 28), datetime(2020, 3, 2)
                )) | \
                (date.between(
                    datetime(2020, 4, 3), datetime(2020, 4, 13)
                )) | \
                (date.between(
                    datetime(2020, 5, 1), datetime(2020, 5, 1)
                )) | \
                (date.between(
                    datetime(2020, 6, 23), datetime(2020, 9, 9)
                )) | \
                (date.between(
                    datetime(2020, 10, 12), datetime(2020, 10, 12)
                )) | \
                (date.between(
                    datetime(2020, 11, 2), datetime(2020, 11, 2)
                )) | \
                (date.between(
                    datetime(2020, 11, 9), datetime(2020, 11, 9)
                )) | \
                (date.between(
                    datetime(2020, 12, 7), datetime(2020, 12, 8)
                )) | \
                (date.between(
                    datetime(2020, 12, 23), datetime(2021, 1, 10)
                )) | \
                (date.between(
                    datetime(2021, 2, 19), datetime(2021, 2, 22)
                )) | \
                (date.between(
                    datetime(2021, 3, 19), datetime(2021, 3, 19)
                )) | \
                (date.between(
                    datetime(2021, 3, 26), datetime(2021, 4, 5)
                )) | \
                (date.between(
                    datetime(2021, 5, 1), datetime(2021, 5, 4)
                )) | \
                (date.between(
                    datetime(2021, 6, 23), datetime(2021, 9, 8)
                )) | \
                (date.between(
                    datetime(2021, 10, 9), datetime(2021, 10, 12)
                )) | \
                (date.between(
                    datetime(2021, 10, 30), datetime(2021, 11, 1)
                )) | \
                (date.between(
                    datetime(2021, 12, 4), datetime(2021, 12, 8)
                )) | \
                (date.between(
                    datetime(2021, 12, 23), datetime(2022, 1, 9)
                ))
    return condition


# Precipitacion - rain
def get_rain_categories(rain):
    rain_intervals = [
        rain == 0,
        rain < 2.5,
        rain >= 2.5
    ]

    rain_categories = [
        "no_rain", "moderate_rain", "strong_rain"
    ]

    return np.select(rain_intervals, rain_categories, default="no_rain")


rain_categories_transformer = FunctionTransformer(get_rain_categories)

rain_one_hot = make_pipeline(
    rain_categories_transformer,
    OneHotEncoder(handle_unknown="ignore", sparse=False)
)

rain_ordinal = make_pipeline(
    rain_categories_transformer,
    OrdinalEncoder(categories=[["no_rain", "moderate_rain", "strong_rain"]]),
)

rain_transformer = dict(
    one_hot=rain_one_hot,
    ordinal=rain_ordinal,
    numerico_power=PowerTransformer(method='yeo-johnson'),
    numerico_quantile_uniform=QuantileTransformer(output_distribution="uniform"),
    numerico_quantile_normal=QuantileTransformer(output_distribution="normal"),
    passthrough="passthrough"
)

# Season
season_transformer = dict(
    one_hot=OneHotEncoder(handle_unknown="ignore", sparse=False),
    ordinal=OrdinalEncoder(categories=[["summer", "spring", "fall", "winter"]]),
    drop="drop"
)

# Boolean columns
all_bool_columns = ["bank_holiday", "working_day", "school_holiday", "state_of_alarm"]
# bool_categories = [[False, True]] * len(bool_columns)

# Temporal columns: month, day, hour, minute
# Possibilities: numeric, one_hot, trigonometric, spline

period_dict = dict(
    month=12,
    day_of_month=31,
    hour=24,
    minute=4,
    season=4,
    weekday=7
)


def sin_transformer(period, n):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * n * np.pi))


def cos_transformer(period, n):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * n * np.pi))


def sincos(period, n_variables=1):
    features = []
    for i in range(1, n_variables+1):
        features += [(f"sin_{i}", sin_transformer(period, n=i)),
                     (f"cos_{i}", cos_transformer(period, n=i))]
    return FeatureUnion(features)


def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )

def get_temp_categories(dim):
    if dim == "weekday":
        return list(range(1, 8))
    elif dim == "minute":
        return [0, 15, 30, 45]
    elif dim == "hour":
        return list(range(24))
    elif dim == "month":
        return list(range(1, 13))
    elif dim == "day_of_month":
        return list(range(1, 32))
    elif dim == "year":
        return [2019, 2020, 2021]

def temp_transformer(approach, period, dim):
    if approach == "passthrough":
        return "passthrough"
    elif approach == "one_hot":
        return OneHotEncoder(handle_unknown="ignore", sparse=False, categories=[get_temp_categories(dim)])
    elif approach.startswith("fourier"):
        if "_" in approach:
            n_variables = int(approach.split("_")[-1])
        else:
            n_variables = 1
        return sincos(period, n_variables=n_variables)
    elif approach.startswith("spline"):
        if "_" in approach:
            n_splines = int(approach.split("_")[-1])
        else:
            n_splines = period // 2
        return periodic_spline_transformer(period, n_splines=n_splines)
    elif approach == "drop":
        return approach


def hour_transformer(approach):
    if approach == "one_hot":
        return make_pipeline(
            #KBinsDiscretizer(n_bins=24, encode="ordinal"),
            FunctionTransformer(np.floor),
            temp_transformer(approach, period_dict["hour"], "hour")
        )
    else:
        return temp_transformer(approach, period_dict["hour"], "hour")



def temp_categorical(dim, approach):
    if approach in ["passthrough", "drop"]:
        return approach
    elif approach == "one_hot":
        return OneHotEncoder(handle_unknown="ignore", sparse=False,
                          categories=[get_temp_categories(dim)])


def hour_workday_interaction(hora):
    return make_pipeline(
        ColumnTransformer([
            ("hour", hour_transformer(hora), ["hour"]),
            ("is_workingday", FunctionTransformer(lambda x: x == True), ["working_day"])
        ]),
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
        ColumnTransformer([
            ("marginal", "drop", list(range(period_dict["hour"]+1))),
            # select only hour*working_day (not the marginal columns, since we already have them)
        ], remainder="passthrough")
    )


columns_viento = dict(xy=["windx", "windy"], wind_speed=["velocidad_viento"])

rain_columns = dict(
    one_hot=[f"rain_{i + 1}" for i in range(3)],
    ordinal=["rain"],
    numerico_power=["rain"],
    numerico_quantile_uniform=["rain"],
    numerico_quantile_normal=["rain"],
    drop=[],
    passthrough=["rain"]
)

wind_columns = dict(
    xy=["windx", "windy"],
    wind_speed=["wind_speed"],
    drop=[]
)


def get_temp_column_names(dimension, approach):
    if approach in ["passthrough", "ordinal"]:
        return [dimension]
    elif approach == "one_hot":
        if dimension == "year":
            return [f"{dimension}_{i + (dimension != 'hour') * 1}" for i in range(3)]
        return [f"{dimension}_{i + (dimension!='hour')*1}" for i in range(period_dict[dimension])]
    elif approach.startswith("spline"):
        if "_" in approach:
            n_splines = int(approach.split("_")[-1])
        else:
            n_splines = period_dict[dimension] // 2
        return [f"{dimension}_{i + 1}" for i in range(n_splines)]
    elif approach == "drop":
        return []
    elif approach.startswith("fourier"):
        if "_" in approach:
            n_variables = int(approach.split("_")[-1])
        else:
            n_variables = 1
        names = []
        for i in range(n_variables):
            names += [f"sin_{dimension}_{i+1}", f"cos_{dimension}_{i+1}"]
        return names

def get_interactions_columns(interactions, hour_approach):
    if interactions == "drop":
        return []
    elif interactions == "poly":
        return [f"interaction_{i}" for i in range(len(get_temp_column_names("hour", hour_approach)))]


def get_column_names(meteo_dict, temporal_dict, interactions, target):
    if interactions == "kernel":
        return [target, "hour", "working_day"] + [f"c{i + 1}" for i in range(300)]
    column_names = [target, "hour", "working_day"]
    column_names += get_temp_column_names("hour", temporal_dict["hour"])
    column_names += [c for c in all_bool_columns if temporal_dict[c] == "passthrough"]
    column_names += get_temp_column_names("year", temporal_dict["year"])
    column_names += get_temp_column_names("season", temporal_dict["season"])
    column_names += get_temp_column_names("month", temporal_dict["month"])
    column_names += get_temp_column_names("day_of_month", temporal_dict["day_of_month"])
    column_names += get_temp_column_names("weekday", temporal_dict["weekday"])
    column_names += get_temp_column_names("minute", temporal_dict["minute"])
    column_names += rain_columns[meteo_dict["rain"]]
    column_names += wind_columns[meteo_dict["wind"]]
    column_names += [dim for dim in ["temperature", "humidity", "pressure", "radiation"] if
                     meteo_dict[dim] == "passthrough"]
    column_names += get_interactions_columns(interactions, temporal_dict["hour"])
    return column_names


# hour: passthrough, one_hot, fourier, spline, drop
# bank_holiday: passthrough, drop
# working_day: passthrough, drop
# school_holiday: passthrough, drop
# state of alarm: passthrough, drop
# year: passthrough, drop, one_hot
# season: one_hot, ordinal, drop
# month: passthrough, one_hot, fourier, spline, drop
# day_of_month: passthrough, one_hot, fourier, spline, drop
# weekday: passthrough, drop, one_hot
# minute: passthrough, drop, one_hot
# rain: one_hot, ordinal, numerico_power, numerico_quantile_uniform, numerico_quantile_normal, drop, passthrough
# wind: xy, wind_speed, drop
# temperature: passthrough, drop
# humidity: passthrough, drop
# pressure: passthrough, drop
# radiation: passthrough, drop
# interactions: poly, kernel, drop


def preprocessing_transformer(meteo_dict, temporal_dict, interactions, target):
    # bool_columns = [k for (k, v) in temporal_dict.items() if k in all_bool_columns and v!="drop"]
    bool_columns = [c for c in all_bool_columns if temporal_dict[c] == "passthrough"]
    transformers = [("target", "passthrough", [target])]
    n_columns_before_interactions = len(get_column_names(meteo_dict, temporal_dict, "drop", target))
    transformers += [
        ("hour", "passthrough", ["hour"]),
        ("workingday", OrdinalEncoder(categories=[[False, True]]), ["working_day"]),
        ("hour_transformed", hour_transformer(temporal_dict["hour"]), ["hour"]),
        ("bool", OrdinalEncoder(categories=[[False, True]] * len(bool_columns)), bool_columns),
        ("year", temp_categorical("year", temporal_dict["year"]), ["year"]),
        ("season", season_transformer.get(temporal_dict["season"]), ["season"]),
        ("month", temp_transformer(temporal_dict["month"], period_dict["month"], "month"), ["month"]),
        ("day_of_month", temp_transformer(temporal_dict["day_of_month"], period_dict["day_of_month"], "day_of_month"), ["day_of_month"]),
        ("weekday", temp_categorical("weekday", temporal_dict["weekday"]), ["weekday"]),
        ("minute", temp_categorical("minute", temporal_dict["minute"]), ["minute"]),
    ]

    if meteo_dict["rain"] != "drop":
        transformers.append(("rain", rain_transformer.get(meteo_dict["rain"]), ["precipitacion"]))
    if meteo_dict["wind"] != "drop":
        transformers.append(("wind", "passthrough", columns_viento.get(meteo_dict["wind"])))
    if meteo_dict["temperature"] != "drop":
        transformers.append(("temperature", meteo_dict["temperature"], ["temperatura"]))
    if meteo_dict["humidity"] != "drop":
        transformers.append(("humidity", meteo_dict["humidity"], ["humedad_relativa"]))
    if meteo_dict["pressure"] != "drop":
        transformers.append(("pressure", meteo_dict["pressure"], ["presion_barometrica"]))
    if meteo_dict["radiation"] != "drop":
        transformers.append(("radiation", meteo_dict["radiation"], ["radiacion_solar"]))

    step1 = ColumnTransformer(transformers=transformers)

    kernel_transformer = ColumnTransformer([
        ("target", "passthrough", [0, 1]),
        ("interactions", Nystroem(kernel="poly", degree=2, n_components=300, random_state=0),
         list(range(1, n_columns_before_interactions)))  # all columns different to the target
    ])

    if interactions == "poly":
        return FeatureUnion([
            ("marginal", step1),
            ("interactions", hour_workday_interaction(temporal_dict["hour"]))
        ])
    elif interactions == "kernel":
        return make_pipeline(
            step1,
            kernel_transformer
        )
    else:
        return step1


def transform_df(df, meteo_dict, temporal_dict, interactions, target):
    transformer = preprocessing_transformer(meteo_dict, temporal_dict, interactions, target)
    df = transformer.fit_transform(df)
    # df = np.nan_to_num(df)
    return df
