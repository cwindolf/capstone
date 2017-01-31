import networkx as nx


def pairwise_tree_distance(T, root=0):
    # need to make the graph directed to get lcas
    Td = nx.DiGraph()
    Td.add_nodes_from(T.nodes_iter())
    Td.add_edges_from(T.edges())
    # get dijkstra distances from source
    h = nx.single_source_dijkstra_path_length(T, root)
    for (u, v), lca in tree_all_pairs_lca(Td, root):
        # yielding (i, j), dist(i, j)
        yield (u, v), (h[u] + h[v] - 2 * h[lca])


def tree_all_pairs_lca(G, root):
    """
    modified from https://github.com/networkx/networkx/pull/869/
    uses Tarjan's offline lca algorithm
    """
    uf = nx.utils.union_find.UnionFind()
    ancestors = { node : uf[node] for node in G.nodes_iter() }
    marked = set()
    for node in nx.depth_first_search.dfs_postorder_nodes(G, root):
        marked.add(node)
        for v in G.nodes_iter():
            if v in marked:
                yield (node, v), ancestors[uf[v]]
        if node != root:
            parent = G.predecessors(node)[0]
            uf.union(parent, node)
            ancestors[uf[parent]] = parent
