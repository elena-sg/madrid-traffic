from graph_traffic.get_data import get_data, plot_graph

data_dict = dict(
    ids_list=[3954, 3973, 3978],
    seq_len=1,
    with_graph=True,
    from_date="2020-01-01",
    to_date="2020-01-02",
    dataset_name="small",
    target="intensidad",  # 1
    interactions=None
)

meteo_dict = dict(
    rain="numerico_power",  # 1
    wind="wind_speed",  # 1
    temperature="drop",  # 0
    humidity="passthrough",  # 1
    pressure="drop",  # 0
    radiation="drop"  # 0
)

temporal_dict = dict(
    season="ordinal",  # 1
    month="spline",  # 6
    day_of_month="trigonometric",  # 2
    hour="spline",  # 12,
    minute="drop",
    bank_holiday="passthrough",
    school_holiday="passthrough",
    working_day="passthrough"
)
# 5*1 + 6 + 2 + 12 + 3 = 28

x, y, g = get_data(data_dict, meteo_dict, temporal_dict)
plot_graph(g, data_dict["ids_list"])

print(x.shape[1:4] == (1, 3, 28))

meteo_dict = dict(
    rain="drop",
    wind="drop",
    temperature="drop",
    humidity="drop",
    pressure="drop",
    radiation="drop"
)

x, y, g = get_data(data_dict, meteo_dict, temporal_dict)
plot_graph(g, data_dict["ids_list"])

print(x.shape[1:4] == (1, 3, 25))

temporal_dict = dict(
    season="drop",  # 0
    month="drop",  # 0
    day_of_month="drop",  # 0
    hour="drop",  # 0
    minute="drop",
    bank_holiday="drop",
    school_holiday="drop",
    working_day="drop"
)

x, y, g = get_data(data_dict, meteo_dict, temporal_dict)
plot_graph(g, data_dict["ids_list"])

print(x.shape[1:4] == (1, 3, 1))

data_dict = dict(
    ids_list=[3954, 3973, 3978],
    seq_len=12,
    with_graph=True,
    from_date="2020-01-01",
    to_date="2020-01-02",
    dataset_name="small",
    target="intensidad",  # 1
    interactions=None
)

temporal_dict = dict(
    season="drop",  # 0
    month="drop",  # 0
    day_of_month="drop",  # 0
    hour="drop",  # 0
    minute="one_hot",
)

x, y, g = get_data(data_dict, meteo_dict, temporal_dict)
plot_graph(g, data_dict["ids_list"])

import numpy as np
import matplotlib.pyplot as plt

plt.plot(np.concatenate((x[0, :, 0, 0], y[0, :, 0, 0])))
plt.show()

plt.plot(np.concatenate((x[0, :, 0, 1], y[0, :, 0, 1])))
plt.show()

plt.plot(np.concatenate((x[0, :, 0, 2], y[0, :, 0, 2])))
plt.show()

plt.plot(np.concatenate((x[0, :, 0, 3], y[0, :, 0, 3])))
plt.show()

plt.plot(np.concatenate((x[0, :, 0, 4], y[0, :, 0, 4])))
plt.show()

print(x.shape[1:4] == (12, 3, 5))

# test if the order of the ids_list changes the result

data_dict = dict(
    ids_list=[3954, 3973, 3978],
    seq_len=12,
    with_graph=True,
    from_date="2020-01-01",
    to_date="2020-01-02",
    dataset_name="small",
    target="intensidad",  # 1
    interactions=None
)

x1, y1, g1 = get_data(data_dict, meteo_dict, temporal_dict)

data_dict = dict(
    ids_list=[3978, 3954, 3973, ],
    seq_len=12,
    with_graph=True,
    from_date="2020-01-01",
    to_date="2020-01-02",
    dataset_name="small",
    target="intensidad",  # 1
    interactions=None
)
x2, y2, g2 = get_data(data_dict, meteo_dict, temporal_dict)

print(np.all(x1 == x2))
print(np.all(y1 == y2))