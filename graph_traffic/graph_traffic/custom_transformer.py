from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import Nystroem

from graph_traffic.config import data_path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, PowerTransformer, \
    QuantileTransformer, SplineTransformer, KBinsDiscretizer, PolynomialFeatures

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
)

# Season
season_transformer = dict(
    one_hot=OneHotEncoder(handle_unknown="ignore", sparse=False),
    ordinal=OrdinalEncoder(categories=[["summer", "spring", "fall", "winter"]])
)

# Boolean columns
bool_columns = ["bank_holiday", "working_day", "school_holiday"]
bool_categories = [[False, True]] * len(bool_columns)

# Temporal columns: month, day, hour, minute
# Possibilities: numeric, one_hot, trigonometric, spline

period_dict = dict(
    month=12,
    day_of_month=31,
    hour=24,
    minute=60,
)


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def sincos(period):
    return FeatureUnion([
        ("sin", sin_transformer(period)),
        ("cos", cos_transformer(period))
    ])


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


def temp_transformer(approach, period):
    if approach == "numeric":
        return "passthrough"
    elif approach == "one_hot":
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    elif approach == "trigonometric":
        return sincos(period)
    elif approach == "spline":
        return periodic_spline_transformer(period, n_splines=period // 2)


def hour_transformer(approach):
    if approach == "one_hot":
        return make_pipeline(
            KBinsDiscretizer(n_bins=24, encode="ordinal"),
            temp_transformer(approach, period_dict["hour"])
        )
    else:
        return temp_transformer(approach, period_dict["hour"])


def hour_workday_interaction(hora):
    return make_pipeline(
        ColumnTransformer([
            ("marginal", hour_transformer(hora), ["hora"]),
            ("workingday", FunctionTransformer(lambda x: x == True), ["dia_laborable"])
        ]),
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
    )


columns_viento = dict(xy=["windx", "windy"], wind_speed=["velocidad_viento"])

passthrough_columns = ["intensidad", "temperatura", "humedad_relativa", "presion_barometrica", "radiacion_solar"]


# feature engineering possibilities for each feature
# intensidad: passthrough
# temperatura: passthrough
# humedad_relativa: passtrough
# presion_barometrica: passthrough
# radiacion_solar: passthrough
# precipitacion/rain: one_hot, ordinal, numeric_power, numeric_quantile_uniform, numeric_quantile_normal
# viento/wind: xy, wind_speed
# year: one_hot
# season: one_hot, ordinal
# month: numeric, one_hot, trigonometric, spline
# day_of_month: numeric, one_hot, trigonometric, spline
# bank_holiday: bool
# working_day: bool
# school_holidays: bool
# hour: numeric, one_hot, trigonometric, spline
# minute: numeric, one_hot, trigonometric, spline
# interactions: poly, kernel, False


def preprocessing_transformer(rain, wind, season, month, day_of_month, hour, interactions):
    step1 = ColumnTransformer(transformers=[
        ("passthrough", "passthrough", passthrough_columns),
        ("rain", rain_transformer.get(rain), ["precipitacion"]),
        ("wind", "passthrough", columns_viento.get(wind)),
        ("year", OneHotEncoder(handle_unknown="ignore", sparse=False), ["year"]),
        ("season", season_transformer.get(season), ["season"]),
        ("month", temp_transformer(month, period_dict["month"]), ["month"]),
        ("day_of_month", temp_transformer(day_of_month, period_dict["day_of_month"]), ["day_of_month"]),
        ("hour", hour_transformer(hour), ["hour"]),
        ("minute", temp_transformer("one_hot", None), ["minute"]),
        ("bool", OrdinalEncoder(categories=bool_categories), bool_columns),
    ])
    if interactions == "poly":
        return FeatureUnion([
            ("marginal", step1),
            ("interactions", hour_workday_interaction(hour))
        ])
    elif interactions == "kernel":
        return make_pipeline(
            step1,
            Nystroem(kernel="poly", degree=2, n_components=300, random_state=0)
        )
    else:
        return step1


def transform_df(df: pd.DataFrame, rain, wind, season, month, day_of_month, hour, interactions):
    transformer = preprocessing_transformer(rain, wind, season, month, day_of_month, hour, interactions)
    df = transformer.fit_transform(df)
    df = np.nan_to_num(df)
    return df
