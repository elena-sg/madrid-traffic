from graph_traffic.merge_data import merge_data

id_t = 1001
from_date = "2020-02-01"
to_date = "2020-02-20"

df = merge_data(id_t, from_date=from_date, to_date=to_date, target="intensidad", mmagns=[], seq_len=1)

df = merge_data(id_t, from_date=from_date, to_date=to_date, target="intensidad", mmagns=["precipitacion"], seq_len=12)
print(1)