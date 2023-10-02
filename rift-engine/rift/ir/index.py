# for tests:
# pip install spacy
# python -m spacy download en_core_web_md

import asyncio
import math
import os
import pickle
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import openai
import tiktoken

import rift.ir.IR as IR
from rift.ir.parser import parse_files_in_paths

Vector = npt.NDArray[np.float32]


@dataclass
class Embedding:
    """
    A class representing a vector embedding.

    Attributes:
        vectors (List[Vector]): A list of vectors representing the embedding.

    Methods:
        cosine_similarity(a: Vector, b: Vector) -> float:
            Computes the cosine similarity between two vectors.
        similarity(query: "Embedding") -> float:
            Computes the maximum cosine similarity between the vectors in this embedding and the vectors in the given query embedding.
    """

    symbol: IR.Symbol
    vectors: List[Vector]

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

    def similarity(self, query: Vector) -> float:
        """
        Computes the maximum cosine similarity between the vectors in this embedding and the vectors in the given query embedding.
        """
        similarities = [self.cosine_similarity(v, query) for v in self.vectors]
        return max(similarities)


version = "0.0.2"

PathWithId = Tuple[str, IR.QualifiedId]


EmbeddingFunction = Callable[[str], Vector]


@dataclass
class Index:
    embeddings: Dict[PathWithId, Embedding]  # (file_path, id) -> embedding
    project: IR.Project
    version: str = version

    def search(
        self, query: Vector, num_results: int = 5, kinds: List[IR.SymbolKindName] = ["Function"]
    ) -> List[Tuple[PathWithId, float]]:
        scores: List[Tuple[PathWithId, float]] = [
            (path_with_id, e.similarity(query=query))
            for path_with_id, e in self.embeddings.items()
            if e.symbol.symbol_kind.name() in kinds
        ]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:num_results]

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

    @classmethod
    async def create(
        cls,
        embed_fun: EmbeddingFunction,
        project: IR.Project,
        documents_for_symbol: Callable[[IR.Symbol], List[str]],
    ) -> "Index":
        """
        Creates an instance of the Index class.

        Parameters:
        - embed_fun: A funcion that computes the embedding of a string.
        - project: The project containing files and function declarations.
        - document_for_symbol: An optional callable that returns a document for a given symbol.
            The document is used for semantic search.

        Returns:
        - An instance of the Index class.
        """

        @dataclass
        class DocumentWithEmbedding:
            document: str
            vector: Optional[Vector]

        @dataclass
        class SymbolEmbedding:
            documents_with_embeddings: List[DocumentWithEmbedding]
            symbol: IR.Symbol
            path_with_id: PathWithId

        all_documents_with_embeddings: List[DocumentWithEmbedding] = []
        symbol_embeddings: List[SymbolEmbedding] = []

        for file in project.get_files():
            all_symbols = file.search_symbol(lambda _: True)
            file_path = file.path
            for symbol in all_symbols:
                path_with_id = (file_path, symbol.get_qualified_id())
                documents = documents_for_symbol(symbol)
                if len(documents) > 0:
                    documents_with_embeddings = [
                        DocumentWithEmbedding(document, None) for document in documents
                    ]
                    all_documents_with_embeddings.extend(documents_with_embeddings)
                    symbol_embeddings.append(
                        SymbolEmbedding(
                            documents_with_embeddings=documents_with_embeddings,
                            symbol=symbol,
                            path_with_id=path_with_id,
                        )
                    )

        async def async_get_embedding(
            document_with_embedding: DocumentWithEmbedding,
        ) -> Awaitable[Vector]:
            loop = asyncio.get_running_loop()
            return loop.run_in_executor(None, embed_fun, document_with_embedding.document)

        # Parallel server requests
        embedded_results: List[Awaitable[Vector]] = await asyncio.gather(
            *(async_get_embedding(x) for x in all_documents_with_embeddings)
        )

        # Assign embeddings
        for n, res in enumerate(embedded_results):
            all_documents_with_embeddings[n].vector = await res

        embeddings = {
            symbol_embedding.path_with_id: Embedding(
                symbol=symbol_embedding.symbol,
                vectors=[
                    x.vector
                    for x in symbol_embedding.documents_with_embeddings
                    if x.vector is not None
                ],
            )
            for symbol_embedding in symbol_embeddings
        }
        return cls(embeddings=embeddings, project=project)


Encoder = tiktoken.get_encoding("cl100k_base")


def token_length(string: str) -> int:
    return len(Encoder.encode(string))


MAX_TOKENS = 8192


def openai_embedding(document: str) -> Vector:
    print("openai embedding for", document[:20], "...")
    if token_length(document) >= MAX_TOKENS:
        print("Truncating document to 8192 tokens")
        tokens = Encoder.encode(document)
        tokens = tokens[
            : MAX_TOKENS - 1
        ]  # less than max tokens otherwise the embedding is full of nan
        document = Encoder.decode(tokens)
    model = "text-embedding-ada-002"
    vector = openai.Embedding.create(input=[document], model=model)["data"][0]["embedding"]  # type: ignore
    vector: Vector = np.array(vector)  # type: ignore
    return vector


def get_embedding_function(openai: bool) -> EmbeddingFunction:
    if openai:
        return openai_embedding
    else:
        import spacy

        nlp = spacy.load("en_core_web_md")

        def spacy_embedding(document: str) -> Vector:
            print("spacy embedding for", document[:20], "...")
            return np.array(nlp(document).vector)

        return spacy_embedding


import pytest


@pytest.mark.asyncio
async def test_index() -> None:
    this_dir = os.path.dirname(__file__)
    project_root = __file__  # this file only
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(this_dir))
    )  # the whole rift project
    openai = False

    index_file = os.path.join(this_dir, "index.rci")
    embed_fun = get_embedding_function(openai=openai)
    if os.path.exists(index_file):
        start = time.time()
        print(f"Loading index from file... {index_file}")
        index = Index.load(index_file)
        print(f"Loaded index in {time.time() - start:.2f} seconds")
    else:
        project = parse_files_in_paths([project_root])
        print("Creating index...")
        start = time.time()

        def documents_for_symbol(symbol: IR.Symbol) -> List[str]:
            documents: List[str] = []
            if isinstance(symbol.symbol_kind, IR.FunctionKind):
                documents = [symbol.get_substring().decode()]
            elif isinstance(symbol.symbol_kind, IR.ClassKind):
                for s in symbol.body:
                    documents.extend(documents_for_symbol(s))
                return documents
            elif isinstance(symbol.symbol_kind, IR.FileKind):
                for s in symbol.body:
                    documents.extend(documents_for_symbol(s))
                return documents
            return documents

        index = await Index.create(
            documents_for_symbol=documents_for_symbol,
            embed_fun=embed_fun,
            project=project,
        )

        print(f"Created index in {time.time() - start:.2f} seconds")
        print(f"Saving index to file... {index_file}")
        start = time.time()
        index.save(index_file)
        print(f"Saved index in {time.time() - start:.2f} seconds")

    test_sentence = "Creates an instance of the Index class"
    start = time.time()
    query = embed_fun(test_sentence)
    scores = index.search(query, kinds=["Function"])  #  ["Class", "File"]

    print("\nSemantic Search Results:")
    for n, x in scores:
        print(f"{n}  {x:.3f}")
    elapsed = time.time() - start
    print(f"\nSearched in {elapsed:.2f} seconds")

