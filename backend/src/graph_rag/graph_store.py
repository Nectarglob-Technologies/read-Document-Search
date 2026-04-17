import networkx as nx


class GraphStore:

    def __init__(self):

        self.graph = nx.DiGraph()

    def add_entity(self, name, entity_type):

        self.graph.add_node(name, type=entity_type)

    def add_relation(self, source, relation, target):

        relation = relation.upper()

        self.graph.add_edge(source, target, relation=relation)

    def get_projects_using_material(self, material):

        results = []

        for u, v, data in self.graph.edges(data=True):

            if data.get("relation") == "USES" and v.lower() == material.lower():

                results.append(u)

        return results

    def get_projects_by_location(self, location):

        results = []

        for u, v, data in self.graph.edges(data=True):

            if data.get("relation") == "LOCATED_IN" and v.lower() == location.lower():

                results.append(u)

        return results

    def get_projects_by_contractor(self, contractor):

        results = []

        for u, v, data in self.graph.edges(data=True):

            if data.get("relation") == "BUILT_BY" and v.lower() == contractor.lower():

                results.append(u)

        return results
