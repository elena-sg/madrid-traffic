from datetime import datetime

from sklearn.compose import ColumnTransformer

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
        "winter",
        "spring",
        "summer",
        "fall",
        "winter"
    ]
    return np.select(conditions, seasons)


def school_holidays(fecha):
    condition = (fecha <= datetime(2019, 1, 7)) | \
                (fecha.between(
                    datetime(2019, 3, 1), datetime(2019, 3, 4)
                )) | \
                (fecha.between(
                    datetime(2019, 4, 12), datetime(2019, 4, 22)
                )) | \
                (fecha.between(
                    datetime(2019, 5, 1), datetime(2019, 5, 5)
                )) | \
                (fecha.between(
                    datetime(2019, 6, 21), datetime(2019, 9, 10)
                )) | \
                (fecha.between(
                    datetime(2019, 10, 31), datetime(2019, 11, 3)
                )) | \
                (fecha.between(
                    datetime(2019, 12, 7), datetime(2019, 12, 9)
                )) | \
                (fecha.between(
                    datetime(2019, 12, 21), datetime(2020, 1, 7)
                )) | \
                (fecha.between(
                    datetime(2020, 2, 28), datetime(2020, 3, 2)
                )) | \
                (fecha.between(
                    datetime(2020, 4, 3), datetime(2020, 4, 13)
                )) | \
                (fecha.between(
                    datetime(2020, 5, 1), datetime(2020, 5, 1)
                )) | \
                (fecha.between(
                    datetime(2020, 6, 23), datetime(2020, 9, 9)
                )) | \
                (fecha.between(
                    datetime(2020, 10, 12), datetime(2020, 10, 12)
                )) | \
                (fecha.between(
                    datetime(2020, 11, 2), datetime(2020, 11, 2)
                )) | \
                (fecha.between(
                    datetime(2020, 11, 9), datetime(2020, 11, 9)
                )) | \
                (fecha.between(
                    datetime(2020, 12, 7), datetime(2020, 12, 8)
                )) | \
                (fecha.between(
                    datetime(2020, 12, 23), datetime(2021, 1, 10)
                )) | \
                (fecha.between(
                    datetime(2021, 2, 19), datetime(2021, 2, 22)
                )) | \
                (fecha.between(
                    datetime(2021, 3, 19), datetime(2021, 3, 19)
                )) | \
                (fecha.between(
                    datetime(2021, 3, 26), datetime(2021, 4, 5)
                )) | \
                (fecha.between(
                    datetime(2021, 5, 1), datetime(2021, 5, 4)
                )) | \
                (fecha.between(
                    datetime(2021, 6, 23), datetime(2021, 9, 8)
                )) | \
                (fecha.between(
                    datetime(2021, 10, 9), datetime(2021, 10, 12)
                )) | \
                (fecha.between(
                    datetime(2021, 10, 30), datetime(2021, 11, 1)
                )) | \
                (fecha.between(
                    datetime(2021, 12, 4), datetime(2021, 12, 8)
                )) | \
                (fecha.between(
                    datetime(2021, 12, 23), datetime(2022, 1, 9)
                ))
    return condition


def transform_df(df: pd.DataFrame):
    df["year"] = df.fecha.dt.year
    df["season"] = season(df.fecha)
    df["month"] = df.fecha.dt.month
    df["day_of_month"] = df.fecha.dt.day
    df["day_of_year"] = df.fecha.dt.day_of_year + df.fecha.dt.hour / 24 + df.fecha.dt.minute / 24 / 60
    df["day_type"] = day_type(df.fecha)
    df["bank_holiday"] = df.fecha.dt.date.isin(bank_holidays)
    df["working_day"] = (df.fecha.dt.weekday <= 4) & (~df.bank_holiday)
    df["school_holidays"] = school_holidays(df.fecha) | (~df.working_day)
    df["hour"] = df.fecha.dt.hour + df.fecha.dt.minute / 60
    df["minute"] = df.fecha.dt.minute

    del df["fecha"]

    return df


# feature engineering possibilities for each feature
# precipitacion: one_hot, ordinal, numeric_power, numeric_quantile_uniform, numeric_quantile_normal
# estacion: one_hot, ordinal
# bank_holiday: bool
# working_day: bool
# school_holidays: bool
# year: one_hot
# month: numeric, one_hot, trigonometric, spline
# day: numeric, one_hot, trigonometric, spline
# hour: numeric, one_hot, trigonometric, spline
# minute: numeric, one_hot, trigonometric, spline

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

precipitacion_one_hot = make_pipeline(
    rain_categories_transformer,
    OneHotEncoder(handle_unknown="ignore", sparse=False)
)

precipitacion_ordinal = make_pipeline(
    rain_categories_transformer,
    OrdinalEncoder(categories=[["no_rain", "moderate_rain", "strong_rain"]]),
)

precipitacion_transformer = dict(
    one_hot=precipitacion_one_hot,
    ordinal=precipitacion_ordinal,
    numerico_power=PowerTransformer(method='yeo-johnson'),
    numerico_quantile_uniform=QuantileTransformer(output_distribution="uniform"),
    numerico_quantile_normal=QuantileTransformer(output_distribution="normal"),
)

# Estacion - season
estacion_transformer = dict(
    one_hot=OneHotEncoder(handle_unknown="ignore", sparse=False),
    ordinal=OrdinalEncoder(categories=[["verano", "primavera", "otono", "invierno"]])
)

# Boolean columns
bool_columns = ["bank_holiday", "working_day", "school_holiday"]
bool_categories = [[False, True]] * len(bool_columns)

# Temporal columns: month, day, hour, minute
# Possibilities: numeric, one_hot, trigonometric, spline

period_dict = dict(
    month=12,
    day=31,
    hour=24,
    minute=60
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


def transformer_temporal(approach, period):
    if approach == "numeric":
        return "passthrough"
    elif approach == "one_hot":
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    elif approach == "trigonometric":
        return sincos(period)
    elif approach == "spline":
        return periodic_spline_transformer(period, n_splines=period // 2)


def transformer_hora(approach):
    if approach == "one_hot":
        return make_pipeline(
            KBinsDiscretizer(n_bins=24, encode="ordinal"),
            transformer_temporal(approach, period_dict["hora"])
        )
    else:
        return transformer_temporal(approach, period_dict["hora"])


def hour_workday_interaction(hora):
    return make_pipeline(
        ColumnTransformer([
            ("marginal", transformer_hora(hora), ["hora"]),
            ("workingday", FunctionTransformer(lambda x: x == True), ["dia_laborable"])
        ]),
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
    )