"""
================================================================================
Network Graph Visualization for Misinformation Spread
================================================================================

Purpose:
  Visualize a directed network showing how information spreads from one node
  to others. Example 4.4 demonstrates a simple network with 4 nodes and
  directed edges representing information flow paths.

Network Structure:
  Nodes: 1, 2, 3, 4 (represent entities or individuals)
  Edges: Directed connections showing possible transmission routes
    Node 1 can reach nodes 2 and 3
    Node 2 can reach node 3
    Node 3 can reach node 4

Graph Type:
  Directed Acyclic Graph (DAG) - edges have direction, no cycles exist
  Useful for modeling information propagation where data flows one direction

================================================================================
"""

import networkx as nx
import matplotlib.pyplot as plt

# Create directed graph object
# DiGraph ensures edges have direction (node A to node B is different from B to A)
G = nx.DiGraph()

# Define nodes representing entities in the network
nodes = [1, 2, 3, 4]
G.add_nodes_from(nodes)

# Define directed edges representing information flow paths
# Format: (source, target) where source node can transmit to target node
edges = [(1, 3), (1, 2), (2, 3), (3, 4)]
G.add_edges_from(edges)

# Set node positions for consistent layout visualization
# Position format: {node_id: (x_coordinate, y_coordinate)}
pos = {1: (0, 1), 2: (1, 2), 3: (1, 0), 4: (2, 1)}

# Create figure and draw network
plt.figure(figsize=(6, 4))

# Draw graph with specified styling
# with_labels equals True displays node numbers
# node_color equals lightblue sets node appearance
# edge_color equals black sets edge color
# node_size equals 2000 sets node size in points
# font_size equals 12 sets label text size
nx.draw(
    G, 
    pos, 
    with_labels=True, 
    node_color='lightblue', 
    edge_color='black', 
    node_size=2000, 
    font_size=12, 
    font_weight='bold'
)

# Add labels on edges to show connection direction
# Format: {(source, target): "label_text"}
edge_labels = {(1, 3): "1→3", (1, 2): "1→2", (2, 3): "2→3", (3, 4): "3→4"}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

# Set title and display
plt.title("Simple Network for Misinformation Spread (Example 4.4)")
plt.show()