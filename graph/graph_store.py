import pickle
from pathlib import Path

import networkx as nx


class GraphStore:
    def __init__(self):
        self.G = nx.DiGraph()

    def add_doc(self, doc_id: str, title: str, url: str):
        self.G.add_node(("Doc", doc_id), title=title, url=url)

    def add_chunk(self, chunk_id: str, text: str, doc_id: str):
        self.G.add_node(("Chunk", chunk_id), text=text)
        self.G.add_edge(("Doc", doc_id), ("Chunk", chunk_id), type="HAS_CHUNK")

    def add_concept(self, name: str):
        self.G.add_node(("Concept", name))

    def link_mentions(self, chunk_id: str, concept: str):
        self.G.add_edge(("Chunk", chunk_id), ("Concept", concept), type="MENTIONS")

    def compute_doc_pagerank(self):
        # project to doc graph (docs connected via shared concepts)
        doc_nodes = [n for n in self.G.nodes if n[0] == "Doc"]
        # naive: connect docs that share a concept via any chunk
        for _t1, id1 in doc_nodes:
            for _t2, id2 in doc_nodes:
                if id1 == id2:
                    continue
                # check shared concepts
                c1 = set()
                for _, ch, d in self.G.out_edges(("Doc", id1), data=True):
                    if d.get("type") == "HAS_CHUNK":
                        c1 |= set(
                            [
                                c[1]
                                for _, c, dd in self.G.out_edges(ch, data=True)
                                if dd.get("type") == "MENTIONS"
                            ]
                        )
                c2 = set()
                for _, ch, d in self.G.out_edges(("Doc", id2), data=True):
                    if d.get("type") == "HAS_CHUNK":
                        c2 |= set(
                            [
                                c[1]
                                for _, c, dd in self.G.out_edges(ch, data=True)
                                if dd.get("type") == "MENTIONS"
                            ]
                        )
                if c1.intersection(c2):
                    self.G.add_edge(("Doc", id1), ("Doc", id2), type="RELATED_DOC")
        pr = nx.pagerank(self.G.to_undirected())
        # save as attribute on docs
        for n, score in pr.items():
            if n[0] == "Doc":
                self.G.nodes[n]["pagerank"] = score

    def neighbor_chunks_by_concepts(self, chunk_id: str, max_neighbors: int = 8):
        # chunks that mention the same concepts
        concepts = [
            c
            for _, c, d in self.G.out_edges(("Chunk", chunk_id), data=True)
            if d.get("type") == "MENTIONS"
        ]
        neigh = set()
        for _, concept in concepts:
            # inbound chunks to concept
            for ch, _, _d in self.G.in_edges(("Concept", concept), data=True):
                if ch[0] == "Chunk" and ch[1] != chunk_id:
                    neigh.add(ch[1])
        return sorted(neigh)[:max_neighbors]

    def get_doc_info(self, doc_id: str):
        n = ("Doc", doc_id)
        if n not in self.G:
            return {"title": None, "url": None, "pagerank": 0.0}
        return {
            "title": self.G.nodes[n].get("title"),
            "url": self.G.nodes[n].get("url"),
            "pagerank": self.G.nodes[n].get("pagerank", 0.0),
        }

    def get_chunk_doc(self, chunk_id: str):
        for d, _ch, data in self.G.in_edges(("Chunk", chunk_id), data=True):
            if data.get("type") == "HAS_CHUNK":
                return d[1]
        return None

    def explain_paths(self, chunk_ids):
        # very simple: for each chunk, list its concepts and owning doc
        paths = []
        for cid in chunk_ids:
            doc_id = self.get_chunk_doc(cid)
            concepts = sorted(
                [
                    c[1]
                    for _, c, d in self.G.out_edges(("Chunk", cid), data=True)
                    if d.get("type") == "MENTIONS"
                ]
            )
            doc = self.get_doc_info(doc_id) if doc_id else {}
            paths.append(
                {
                    "chunk_id": cid,
                    "doc_id": doc_id,
                    "doc_title": doc.get("title"),
                    "url": doc.get("url"),
                    "concepts": concepts,
                }
            )
        return paths

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.G, f)

    @classmethod
    def load(cls, path: Path):
        self = cls()
        import pickle

        with open(path, "rb") as f:
            self.G = pickle.load(f)
        return self
