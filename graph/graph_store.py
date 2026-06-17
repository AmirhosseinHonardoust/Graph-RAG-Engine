from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import networkx as nx

DocNode = tuple[str, str]


class GraphStore:
    """A lightweight knowledge graph over documents, chunks, and concepts.

    Nodes are typed tuples: ``("Doc", id)``, ``("Chunk", id)``, and
    ``("Concept", name)``. Edges carry a ``type`` of ``HAS_CHUNK`` (Doc -> Chunk),
    ``MENTIONS`` (Chunk -> Concept), or ``RELATED_DOC`` (Doc -> Doc, derived from
    shared concepts during PageRank computation).
    """

    def __init__(self) -> None:
        self.G: nx.DiGraph = nx.DiGraph()

    def add_doc(self, doc_id: str, title: str, url: str) -> None:
        self.G.add_node(("Doc", doc_id), title=title, url=url)

    def add_chunk(self, chunk_id: str, text: str, doc_id: str) -> None:
        self.G.add_node(("Chunk", chunk_id), text=text)
        self.G.add_edge(("Doc", doc_id), ("Chunk", chunk_id), type="HAS_CHUNK")

    def add_concept(self, name: str) -> None:
        self.G.add_node(("Concept", name))

    def link_mentions(self, chunk_id: str, concept: str) -> None:
        self.G.add_edge(("Chunk", chunk_id), ("Concept", concept), type="MENTIONS")

    def _chunk_concepts(self, chunk_id: str) -> list[str]:
        """Return the concept names a chunk mentions."""
        return [
            target[1]
            for _, target, data in self.G.out_edges(("Chunk", chunk_id), data=True)
            if data.get("type") == "MENTIONS"
        ]

    def _doc_concepts(self) -> dict[str, set[str]]:
        """Map each document id to the set of concepts mentioned by its chunks.

        Built in a single pass over the graph's edges, replacing the previous
        all-pairs document comparison that re-walked the graph for every pair.
        """
        chunk_to_doc: dict[str, str] = {}
        chunk_to_concepts: dict[str, set[str]] = {}
        for source, target, data in self.G.edges(data=True):
            edge_type = data.get("type")
            if edge_type == "HAS_CHUNK":
                chunk_to_doc[target[1]] = source[1]
            elif edge_type == "MENTIONS":
                chunk_to_concepts.setdefault(source[1], set()).add(target[1])

        doc_concepts: dict[str, set[str]] = {}
        for chunk_id, concepts in chunk_to_concepts.items():
            doc_id = chunk_to_doc.get(chunk_id)
            if doc_id is not None:
                doc_concepts.setdefault(doc_id, set()).update(concepts)
        return doc_concepts

    def compute_doc_pagerank(self) -> None:
        """Connect documents that share concepts, then store PageRank per document.

        Uses a concept -> documents inverted index so each shared-concept link is
        discovered once, rather than comparing every document pair.
        """
        doc_concepts = self._doc_concepts()

        concept_to_docs: dict[str, set[str]] = {}
        for doc_id, concepts in doc_concepts.items():
            for concept in concepts:
                concept_to_docs.setdefault(concept, set()).add(doc_id)

        for docs_sharing_concept in concept_to_docs.values():
            related = sorted(docs_sharing_concept)
            for i, doc_a in enumerate(related):
                for doc_b in related[i + 1 :]:
                    self.G.add_edge(("Doc", doc_a), ("Doc", doc_b), type="RELATED_DOC")
                    self.G.add_edge(("Doc", doc_b), ("Doc", doc_a), type="RELATED_DOC")

        pagerank = nx.pagerank(self.G.to_undirected())
        for node, score in pagerank.items():
            if node[0] == "Doc":
                self.G.nodes[node]["pagerank"] = score

    def neighbor_chunks_by_concepts(self, chunk_id: str, max_neighbors: int = 8) -> list[str]:
        """Return chunks that mention at least one concept in common with ``chunk_id``."""
        neighbors: set[str] = set()
        for concept in self._chunk_concepts(chunk_id):
            for source, _, _ in self.G.in_edges(("Concept", concept), data=True):
                if source[0] == "Chunk" and source[1] != chunk_id:
                    neighbors.add(source[1])
        return sorted(neighbors)[:max_neighbors]

    def get_doc_info(self, doc_id: str) -> dict[str, Any]:
        node = ("Doc", doc_id)
        if node not in self.G:
            return {"title": None, "url": None, "pagerank": 0.0}
        return {
            "title": self.G.nodes[node].get("title"),
            "url": self.G.nodes[node].get("url"),
            "pagerank": self.G.nodes[node].get("pagerank", 0.0),
        }

    def get_chunk_doc(self, chunk_id: str) -> str | None:
        for source, _, data in self.G.in_edges(("Chunk", chunk_id), data=True):
            if data.get("type") == "HAS_CHUNK":
                return source[1]
        return None

    def explain_paths(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """For each chunk, return its owning document and the concepts it mentions."""
        paths: list[dict[str, Any]] = []
        for chunk_id in chunk_ids:
            doc_id = self.get_chunk_doc(chunk_id)
            doc = self.get_doc_info(doc_id) if doc_id else {}
            paths.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_title": doc.get("title"),
                    "url": doc.get("url"),
                    "concepts": sorted(self._chunk_concepts(chunk_id)),
                }
            )
        return paths

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.G, f)

    @classmethod
    def load(cls, path: Path) -> GraphStore:
        store = cls()
        with open(path, "rb") as f:
            store.G = pickle.load(f)
        return store
