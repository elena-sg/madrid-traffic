import pandas as pd
from graph_traffic.config import data_path
from graph_traffic.custom_transformer import transform_df, preprocessing_transformer

# transformations in columns
rain = "ordinal"
wind = "wind_speed"
season = "ordinal"
month = "spline"
day_of_month = "trigonometrix"
hour = "spline"
interactions = None

df = pd.read_csv(f"{data_path}/04-traffic-meteo-merged/10011.csv", parse_dates=["fecha"])
dft = transform_df(df, rain, wind, season, month, day_of_month, hour, interactions)
