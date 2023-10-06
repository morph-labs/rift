# for tests:
# pip install spacy
# python -m spacy download en_core_web_md

import asyncio
import math
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import openai
import tiktoken

import rift.ir.IR as IR
from rift.ir.IR import SymbolKindName, Vector
from rift.ir.parser import parse_files_in_paths


@dataclass
class Node(ABC):
    """Helper class to represent nodes in a boolean query tree."""

    @abstractmethod
    def node_similarity(self, symbol: IR.Symbol) -> float:
        """
        Computes the similarity between the node and a vector.
        """
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, a: Vector, b: Vector) -> float:
        """
        Computes the cosine similarity between two vectors.
        """
        dot_product = a.dot(b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        result = dot_product / (norm_a * norm_b)
        if math.isnan(result):
            return 0.0
        return result


class Text(Node):
    """Helper class to represent text nodes in a boolean query tree."""

    text: str
    vector: Vector

    def __init__(self, text: str) -> None:
        self.text = text
        self.vector = embed_fun(text)

    def node_similarity(self, symbol: IR.Symbol) -> float:
        if symbol.embedding is None:
            return 0.0
        return self.cosine_similarity(symbol.embedding, self.vector)


@dataclass
class Not(Node):
    """Helper class to represent not nodes in a boolean query tree."""

    node: Node

    @classmethod
    def op(cls, x: float) -> float:
        """Not operation."""
        return 1 - x

    def node_similarity(self, symbol: IR.Symbol) -> float:
        return self.op(self.node.node_similarity(symbol))


class And(Node):
    """Helper class to represent and nodes in a boolean query tree."""

    arguments: List[Node]

    def __init__(self, left: Node | List[Node], right: Optional[Node] = None) -> None:
        if isinstance(left, list):
            self.arguments = left
        else:
            assert right is not None
            self.arguments = [left, right]

    @classmethod
    def op(cls, args: List[float]) -> float:
        """And operation for a list of parameters."""
        return min(args)

    def node_similarity(self, symbol: IR.Symbol) -> float:
        return self.op([x.node_similarity(symbol) for x in self.arguments])


class Or(Node):
    """Helper class to represent or nodes in a boolean query tree."""

    arguments: List[Node]

    def __init__(self, left: Node | List[Node], right: Optional[Node] = None) -> None:
        if isinstance(left, list):
            self.arguments = left
        else:
            assert right is not None
            self.arguments = [left, right]

    @classmethod
    def op(cls, args: List[float]) -> float:
        """Or operation for a list of parameters."""
        return max(args)

    def node_similarity(self, symbol: IR.Symbol) -> float:
        return self.op([x.node_similarity(symbol) for x in self.arguments])


@dataclass
class Function(Node):
    """
    A class representing a function node in the IR index.

    Attributes:
        function (Callable[[IR.Symbol], float]): The function to calculate node similarity.
    """

    function: Callable[[IR.Symbol], float]

    def node_similarity(self, symbol: IR.Symbol) -> float:
        return self.function(symbol)


class Query:
    node: Node
    kinds: List[IR.SymbolKindName]
    num_results: int = 5

    def __init__(
        self,
        query: Union[str, Node],
        num_results: int = 5,
        kinds: List[IR.SymbolKindName] = ["Function"],
    ):
        if isinstance(query, str):
            query = Text(text=query)
        self.node = query
        self.num_results = num_results
        self.kinds = kinds


@dataclass
class Embedding:
    """
    Represents a vector embedding for a symbol.

    Attributes:
    - symbol: The primary symbol for which the embedding was generated.
    - aggregate_symbols: Symbols used for aggregating the embedding. For non-aggregate symbols
      (e.g., functions), this list contains only the primary symbol.
    """

    symbol: IR.Symbol
    aggregate_symbols: List[IR.Symbol]

    def similarity(self, query: Query) -> float:
        """
        Compute the maximum cosine similarity between the vectors of this embedding
        and the vectors of the provided query embedding.

        Args:
        - query: The query embedding against which similarity needs to be computed.

        Returns:
        - A float value representing the maximum cosine similarity.
        """

        similarities = [query.node.node_similarity(symbol) for symbol in self.aggregate_symbols]

        return max(similarities)


version = "0.0.3"

PathWithId = Tuple[str, IR.QualifiedId]


EmbeddingFunction = Callable[[str], Vector]


def openai_embedding(document: str) -> Vector:
    print("openai embedding for", document[:20], "...")
    model = "text-embedding-ada-002"
    MAX_TOKENS = 8192
    if token_length(document) >= MAX_TOKENS:
        print("Truncating document to 8192 tokens")
        tokens = Encoder.encode(document)
        tokens = tokens[
            : MAX_TOKENS - 1
        ]  # less than max tokens otherwise the embedding is full of nan
        document = Encoder.decode(tokens)
    vector = openai.Embedding.create(input=[document], model=model)["data"][0]["embedding"]  # type: ignore
    vector: Vector = np.array(vector)  # type: ignore
    return vector


embed_fun: EmbeddingFunction = openai_embedding


def set_embedding_function(openai: bool) -> None:
    global embed_fun
    if openai:
        embed_fun = openai_embedding
    else:
        import spacy

        nlp = spacy.load("en_core_web_md")

        def spacy_embedding(document: str) -> Vector:
            # print("spacy embedding for", document[:20], "...")
            return np.array(nlp(document).vector)

        embed_fun = spacy_embedding


@dataclass
class Index:
    embeddings: Dict[PathWithId, Embedding]  # (file_path, id) -> embedding
    project: IR.Project
    version: str = version

    def search(self, query: Query) -> List[Tuple[PathWithId, float, IR.Range]]:
        scores: List[Tuple[PathWithId, float, IR.Range]] = [
            (path_with_id, e.similarity(query=query), e.symbol.range)
            for path_with_id, e in self.embeddings.items()
            if e.symbol.symbol_kind.name() in query.kinds
        ]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[: query.num_results]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Index":
        with open(path, "rb") as f:
            index = pickle.load(f)
        # check version
        if index.version != version:
            raise ValueError(f"Index version {index.version} is not supported.")
        return index

    @dataclass
    class EmbeddingItem(ABC):
        """abstract class for embedding items: either a document or a reference"""

        symbol: IR.Symbol

    @dataclass
    class DocumentItem(EmbeddingItem):
        """
        A  document to be embedded in an IR.Symbol object.
        """

        document: str

    @dataclass
    class ReferenceItem(EmbeddingItem):
        """
        A reference to the embedding of a symbol. Used for aggregating embeddings of nested symbols.
        """

        pass

    @dataclass
    class SymbolEmbedding:
        items: List["Index.EmbeddingItem"]
        path_with_id: PathWithId
        symbol: IR.Symbol

    @classmethod
    def symbol_fits_length(cls, symbol: IR.Symbol, max_len: int) -> bool:
        sub = symbol.substring
        return sub[1] - sub[0] <= max_len

    @classmethod
    def symbol_needs_indexing(
        cls, symbol: IR.Symbol, kinds: List[SymbolKindName], max_len: int
    ) -> bool:
        kind = symbol.symbol_kind
        if kind.name() in kinds:
            return True
        elif isinstance(kind, IR.MetaSymbolKind):
            if symbol.parent and not cls.symbol_fits_length(symbol.parent, max_len):
                return True
        return False

    @classmethod
    def gather_nested_symbols(
        cls, symbol: IR.Symbol, kinds: List[SymbolKindName], max_len: int
    ) -> List["Index.EmbeddingItem"]:
        """Return references to documents in nested symbols"""
        items: List[Index.EmbeddingItem] = []
        for s in symbol.body:
            if cls.symbol_needs_indexing(s, kinds, max_len):
                items.append(Index.ReferenceItem(s))
                items.extend(cls.gather_nested_symbols(s, kinds, max_len))
        return items

    @staticmethod
    def symbol_is_leaf(symbol: IR.Symbol) -> bool:
        """Return True if the symbol is a leaf symbol in terms of embedding: it does not contain any nested symbols"""
        if isinstance(symbol.symbol_kind, IR.FunctionKind):
            return True
        if isinstance(symbol.symbol_kind, IR.MetaSymbolKind):
            return True
        return False

    @classmethod
    def documents_for_symbol(
        cls, file_path: str, symbol: IR.Symbol, kinds: List[SymbolKindName], max_len: int
    ) -> List["Index.EmbeddingItem"]:
        """Return documents for a symbol used for embedding"""
        if not cls.symbol_needs_indexing(symbol, kinds, max_len):
            return []
        if cls.symbol_is_leaf(symbol):
            if cls.symbol_fits_length(symbol, max_len):
                # treat leaf symbols as documents
                return [Index.DocumentItem(symbol, symbol.get_substring().decode())]
            else:
                items = cls.gather_nested_symbols(symbol, kinds, max_len)

                print(
                    f"Warning: symbol {symbol.get_qualified_id()} is too long ({len(symbol.get_substring().decode())} > {max_len}) {symbol.range} "
                )
                print(f"WWW Added {len(items)} items for {file_path}:{symbol.get_qualified_id()}")

                return items
        else:
            return cls.gather_nested_symbols(symbol, kinds, max_len)

    @classmethod
    async def create(
        cls,
        project: IR.Project,
        kinds: List[SymbolKindName] = [
            "Function",
            "Class",
            "File",
            "Def",
            "Section",
            "Structure",
            "Theorem",
        ],
        max_len: int = 100,
    ) -> "Index":
        """
        Creates an Index object from a given project and a function that returns embeddings for a given symbol.

        Args:
            project: The project to index.
            kinds: The kinds of symbols to index.

        Returns:
            An Index object containing the embeddings for the symbols in the project.
        """
        documents_to_embed: List["Index.DocumentItem"] = []
        symbol_embeddings: List["Index.SymbolEmbedding"] = []

        for file in project.get_files():
            all_symbols = file.search_symbol(lambda _: True)
            file_path = file.path
            for symbol in all_symbols:
                path_with_id = (file_path, symbol.get_qualified_id())
                if cls.symbol_needs_indexing(symbol, kinds, max_len):
                    items = cls.documents_for_symbol(file_path, symbol, kinds, max_len)
                    if len(items) > 0:
                        documents_to_embed.extend(
                            [i for i in items if isinstance(i, Index.DocumentItem)]
                        )
                        symbol_embeddings.append(
                            Index.SymbolEmbedding(
                                items=items,
                                path_with_id=path_with_id,
                                symbol=symbol,
                            )
                        )

        async def async_get_embedding(
            document_to_embed: Index.DocumentItem,
        ) -> Awaitable[Vector]:
            loop = asyncio.get_running_loop()
            return loop.run_in_executor(None, embed_fun, document_to_embed.document)

        # Parallel server requests
        embedded_results: List[Awaitable[Vector]] = await asyncio.gather(
            *(async_get_embedding(x) for x in documents_to_embed)
        )

        # Assign embeddings
        for n, res in enumerate(embedded_results):
            documents_to_embed[n].symbol.embedding = await res

        embeddings = {
            symbol_embedding.path_with_id: Embedding(
                symbol=symbol_embedding.symbol,
                aggregate_symbols=[doc.symbol for doc in symbol_embedding.items],
            )
            for symbol_embedding in symbol_embeddings
        }
        return cls(embeddings=embeddings, project=project)


Encoder = tiktoken.get_encoding("cl100k_base")


def token_length(string: str) -> int:
    return len(Encoder.encode(string))


import pytest


@pytest.mark.asyncio
async def test_index() -> None:
    this_dir = os.path.dirname(__file__)
    project_root = __file__  # this file only

    # project_root = os.path.dirname(
    #     os.path.dirname(os.path.dirname(this_dir))
    # )  # the whole rift project

    openai = False

    index_file = os.path.join(this_dir, "index.rci")
    set_embedding_function(openai=openai)
    if os.path.exists(index_file):
        start = time.time()
        print(f"Loading index from file... {index_file}")
        index = Index.load(index_file)
        print(f"Loaded index in {time.time() - start:.2f} seconds")
    else:
        project = parse_files_in_paths([project_root], metasymbols=True)
        print("Creating index...")
        start = time.time()
        index = await Index.create(
            project=project,
            kinds=["Function", "Class"],
        )

        print(f"Created index in {time.time() - start:.2f} seconds")
        print(f"Saving index to file... {index_file}")
        start = time.time()
        index.save(index_file)
        print(f"Saved index in {time.time() - start:.2f} seconds")
        # print(f"{project.dump_map()}")

    def test_search(
        node: Node, kinds: List[SymbolKindName] = ["Function"], num_results: int = 5
    ) -> None:
        start = time.time()
        query = Query(node, num_results=num_results, kinds=kinds)  #  ["Class", "File"]
        scores = index.search(query)
        print("\nSemantic Search Results:")
        for n, x, range in scores:
            print(f"{n}  {x:.3f}  {range}")
        elapsed = time.time() - start
        print(f"\nSearched in {elapsed:.2f} seconds")

    # in_query = Function(
    #     lambda symbol: 1.0 if symbol.get_qualified_id().startswith("Query.") else 0.0
    # )

    # def in_class_function(symbol: IR.Symbol) -> float:
    #     if isinstance(symbol.symbol_kind, IR.FunctionKind):
    #         if symbol.parent and isinstance(symbol.parent.symbol_kind, IR.ClassKind):
    #             return 1.0
    #         else:
    #             return 0.0
    #     else:
    #         return 1.0

    # in_class = Function(in_class_function)

    test_search(
        Text("Warning: symbol {symbol.get_qualified_id()} is too long ({len(symbol.get_substring().decode())} > {max_len}"),
        ["Function", "Expression", "If", "Call", "Guard", "Body"],
        10,
    )
    # test_search(And([Text("load")]), ["File"])
    # test_search(And([Text("load"), Not(in_query), Not(in_class)]), ["Function", "File"])
    # test_search(And([Text("load"), in_query]), ["Function", "File"])
