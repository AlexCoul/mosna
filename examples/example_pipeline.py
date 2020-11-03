
class_names = ['a', 'b', 'c']
mapping = {'a':0,
           'b':1,
           'c':2}

nodes, edges = make_P_net()
coords = nodes.loc[:, ['x', 'y']].values
pairs = edges.loc[:, ['source', 'target']].values
# one way to color nodes
att = flatten_categories(nodes, att=class_names)
col_att = categorical_to_integer(att)
plot_network(coords, pairs,  disp_id=True, 
             col_nodes=col_att, cmap_nodes='viridis')
# color nodes with manual colors
att_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
att = flatten_categories(nodes, att=class_names)
color_mapper = dict(zip(att.unique(), att_colors))
col_nodes = [color_mapper[x] for x in att]
plot_network(coords, pairs, disp_id=True, col_nodes=col_nodes)

G_nx = to_NetworkX(nodes, edges, attributes=class_names)
################ /!\ conversion probably didn't work
mixmat_nx  = nx.attribute_mixing_matrix(G_nx, 'nodes_class', mapping=mapping)
mixmat_us= mixing_matrix(nodes, edges, class_names)
np.allclose(mixmat_nx, mixmat_us)
ac_nx = nx.attribute_assortativity_coefficient(G_nx,'nodes_class')
ac_us = attribute_ac(mixmat_us)
print(ac_nx == ac_us)
################


nodes, edges = make_triangonal_net()
coords = nodes.loc[:, ['x', 'y']].values
pairs = edges.loc[:, ['source', 'target']].values
plot_network(coords, pairs,  disp_id=True)

nodes, edges = make_trigonal_net()
coords = nodes.loc[:, ['x', 'y']].values
pairs = edges.loc[:, ['source', 'target']].values
plot_network(coords, pairs,  disp_id=True)

nodes, edges, G = make_random_graph_2libs()
coords = nodes.loc[:, ['x', 'y']].values
pairs = edges.loc[:, ['source', 'target']].values
plot_network(coords, pairs,  disp_id=True)


nodes, edges = make_high_assort_net()
coords = nodes.loc[:, ['x', 'y']].values
pairs = edges.loc[:, ['source', 'target']].values
# color nodes with manual colors
att_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
att = flatten_categories(nodes, att=['a','b','c'])
color_mapper = dict(zip(att.unique(), att_colors))
col_nodes = [color_mapper[x] for x in att]
plot_network(coords, pairs, disp_id=True, col_nodes=col_nodes)

nodes, edges = make_high_disassort_net()
coords = nodes.loc[:, ['x', 'y']].values
pairs = edges.loc[:, ['source', 'target']].values
# color nodes with manual colors
att_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
att = flatten_categories(nodes, att=['a','b','c'])
color_mapper = dict(zip(att.unique(), att_colors))
col_nodes = [color_mapper[x] for x in att]
plot_network(coords, pairs, disp_id=True, col_nodes=col_nodes)



nodes, edges, G_nx = make_random_graph_2libs(nb_nodes=300)

mixmat_nx  = nx.attribute_mixing_matrix(G_nx, 'nodes_class', mapping=mapping)
mixmat_us= mixing_matrix(nodes, edges, class_names)
np.allclose(mixmat_nx, mixmat_us)

ac_nx = nx.attribute_assortativity_coefficient(G_nx,'nodes_class')
ac_us = attribute_ac(mixmat_us)
print(ac_nx == ac_us)

