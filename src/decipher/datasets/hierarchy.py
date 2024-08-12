import ndex2
import json
import networkx as nx
import pandas as pd
import numpy as np


class Hierarchy(object):
    def __init__(self, network, node_table=None, edge_table=None, membership_map=None):
        self.network = network
        self.node_table = node_table
        self.edge_table = edge_table
        self.membership_map = membership_map

    @classmethod
    def from_ndex(cls, uuid, host=None, username=None, password=None):
        client = ndex2.client.Ndex2(host=host, username=username, password=password)
        client_resp = client.get_network_as_cx_stream(uuid)
        net_cx = ndex2.create_nice_cx_from_raw_cx(json.loads(client_resp.content))

        node_table = cls.parse_cx_node_table(net_cx)

        node_id_namemap = (
            node_table["node-id"].reset_index().set_index("node-id")["index"].to_dict()
        )
        node_genes_map = {
            name: values.split() for name, values in node_table["Genes"].items()
        }
        namemap = node_table["Annotation"].to_dict()

        edge_table = cls.parse_cx_edge_table(net_cx)
        edge_table = edge_table.assign(
            source_name=edge_table["source"].map(node_id_namemap),
            target_name=edge_table["target"].map(node_id_namemap),
        )

        dG = nx.DiGraph()

        for _, row in edge_table.iterrows():
            dG.add_edge(
                row["target_name"], row["source_name"]
            )  # The edge_table goes the opposite

        hierarchy = cls(
            dG,
            node_table=node_table,
            edge_table=edge_table,
            membership_map=node_genes_map,
        )

        hierarchy.node_id_map = node_id_namemap
        hierarchy.namemap = namemap

        return hierarchy

    @staticmethod
    def parse_cx_node_table(net_cx):
        """Convert node attributes into a node table from a cx object"""

        node_table = []
        for node_id, node_obj in net_cx.get_nodes():
            attributes = net_cx.get_node_attributes(node_id)
            values = [attr["v"] for attr in attributes]
            values += [node_id]
            names = [attr["n"] for attr in attributes]
            names += ["node-id"]

            values = pd.Series(values, index=names)
            values.name = node_obj["n"]

            node_table.append(values)

        node_table = pd.concat(node_table, axis=1).T

        return node_table

    @staticmethod
    def parse_cx_edge_table(net_cx):
        """Convert edge attributes into edge table from a cx object"""

        edge_table = []
        for edge_id, edge_obj in net_cx.get_edges():
            edge_table.append([edge_id, edge_obj["s"], edge_obj["t"], edge_obj["i"]])

        edge_table = pd.DataFrame(
            edge_table, columns=["edge_id", "source", "target", "relation"]
        ).set_index("edge_id")

        return edge_table

    def filter_nodes(self, members, min_fraction=0, min_overlap=0, min_size=0):
        """Filter nodes in hierarchy based on overlap with a set of members

        Parameters
        ----------
        min_fraction: float
            The ratio of overlap to the size of all node members
        min_overlap: float
            Minimum number of overlap between members and node members
        min_size: float
            Minimum number of node members
        """

        nodes = []
        members_selected = []

        for node_, members_ in self.membership_map.items():
            if len(members_) < min_size:
                continue

            overlap = np.intersect1d(members_, members)

            if (len(overlap) / len(members_) >= min_fraction) & (
                len(overlap) >= min_overlap
            ):
                nodes.append(node_)
                members_selected.append(overlap)

        if members_selected:
            members_selected = np.unique(np.hstack(members_selected))
        else:
            members_selected = np.array([])

        return nodes, members_selected

    def subgraph(self, members, root, **kwargs):
        nodes, members_selected = self.filter_nodes(members, **kwargs)

        all_nodes = []
        for key in nodes:
            try:
                paths = nx.shortest_path(self.network, source=key, target=root)
            except nx.NetworkXNoPath:
                continue

            all_nodes.append(paths)

        all_nodes = np.unique(np.hstack(all_nodes))

        subnetwork = self.network.subgraph(all_nodes)
        node_table = self.node_table.loc[all_nodes]
        edges_in_network = (
            self.edge_table[["source_name", "target_name"]].isin(all_nodes).all(axis=1)
        )

        edge_table = self.edge_table.loc[edges_in_network]
        membership_map = {
            k: v for k, v in self.membership_map.items() if k in all_nodes
        }

        subgraph = self.__class__(subnetwork, node_table, edge_table, membership_map)

        return subgraph, members_selected


def get_nest(genes, **kwargs):
    # TODO: Not sure if we really need this...
    nest, _, _ = filter_nest(genes, **kwargs)
    nest.nest_membership_map = {
        nest.namemap[k]: v for k, v in nest.membership_map.items()
    }

    return nest


def filter_nest(genes, min_fraction=0.1, min_overlap=3, min_size=5):
    """Getting the main decipher_nest"""

    filter_kwargs = dict(
        min_fraction=min_fraction,
        min_overlap=min_overlap,
        min_size=min_size,
    )

    nest = Hierarchy.from_ndex("9a8f5326-aa6e-11ea-aaef-0ac135e8bacf")
    decipher_nest, selected_decipher_genes = nest.subgraph(
        genes, "NEST", **filter_kwargs
    )

    return nest, decipher_nest, selected_decipher_genes
