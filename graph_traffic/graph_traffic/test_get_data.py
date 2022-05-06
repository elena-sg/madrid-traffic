from graph_traffic.get_data import get_data, plot_graph

ids_list = [3954, 3973]
rain = "ordinal"
wind = "wind_speed"
season = "ordinal"
month = "spline"
day_of_month = "trigonometric"
hour = "spline"
interactions = None
seq_len = 1
with_graph = True
from_date = "2020-01-01"
to_date = "2020-01-02"
dataset_name = "small"

x, y, g = get_data(ids_list, seq_len, rain, wind, season, month, day_of_month, hour, interactions, with_graph,
                   from_date, to_date, dataset_name)
plot_graph(g, ids_list)
