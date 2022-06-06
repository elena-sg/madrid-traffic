import networkx as nx
from graph_traffic.model_selection import get_node_ids
from graph_traffic.get_data import get_graph, ubs_index, plot_graph
from graph_traffic.config import figures_path

ids = ids_to_use = get_node_ids(longitud_lims=(-3.7443, -3.7308),
                          latitud_lims=(40.3830, 40.3904))
ubs, ubs_dict = ubs_index(ids)

weight_threshold = 0.5
graph = get_graph(ubs, ubs_dict, weight_threshold=weight_threshold)[0]
#plot_graph(graph, ids, save_dir=f"{figures_path}/test", graph_name="spring", layout=nx.spring_layout)
#plot_graph(graph, ids, save_dir=f"{figures_path}/test", graph_name="circular", layout=nx.circular_layout)
plot_graph(graph, ids, save_dir=f"{figures_path}/test", graph_name="kamada-kawai", layout=nx.kamada_kawai_layout)
#plot_graph(graph, ids, save_dir=f"{figures_path}/test", graph_name="random", layout=nx.random_layout)
#plot_graph(graph, ids, save_dir=f"{figures_path}/test", graph_name="shell", layout=nx.shell_layout)
#plot_graph(graph, ids, save_dir=f"{figures_path}/test", graph_name="spectral", layout=nx.spectral_layout)

weight_threshold = 0.6
graph = get_graph(ubs, ubs_dict, weight_threshold=weight_threshold)[0]
plot_graph(graph, ids, save_dir=f"{figures_path}/test", graph_name="kamada-kawai-6", layout=nx.kamada_kawai_layout)

weight_threshold = 0.65
graph = get_graph(ubs, ubs_dict, weight_threshold=weight_threshold)[0]
plot_graph(graph, ids, save_dir=f"{figures_path}/test", graph_name="kamada-kawai-65", layout=nx.kamada_kawai_layout)