# for tests:
# pip install spacy
# python -m spacy download en_core_web_md

import asyncio
from dataclasses import dataclass
import math
import openai
import os
import pickle
import time
import numpy as np
import numpy.typing as npt
import tiktoken
from typing import Awaitable, Callable, Dict, List, Optional, Tuple
import rift.ir.IR as IR
from rift.ir.parser import parse_files_in_paths


@dataclass
class Embedding:
    embedding: npt.NDArray[np.float32]

    def similarity(self, other: "Embedding") -> float:
        dot_product = self.embedding.dot(other.embedding)
        norm_self = np.linalg.norm(self.embedding)
        norm_other = np.linalg.norm(other.embedding)
        if norm_self == 0 or norm_other == 0:
            return 0.0
        result = dot_product / (norm_self * norm_other)
        if math.isnan(result):
            return 0.0
        return result


version = "0.0.1"

PathWithId = Tuple[str, IR.QualifiedId]


EmbeddingFunction = Callable[[str], Embedding]


@dataclass
class Index:
    embeddings: Dict[PathWithId, Embedding]  # (file_path, id) -> embedding
    project: IR.Project
    version: str = version

    def search(self, query: Embedding, num_results: int = 5) -> List[Tuple[PathWithId, float]]:
        scores: List[Tuple[PathWithId, float]] = [
            (pathWithId, query.similarity(e)) for pathWithId, e in self.embeddings.items()
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
        document_for_symbol: Optional[Callable[[IR.Symbol], str]] = None,
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
        embeddings: Dict[PathWithId, Embedding] = {}

        # Collect all document first
        documents: List[str] = []
        paths_with_ids: List[PathWithId] = []
        for file in project.get_files():
            for f in file.get_function_declarations():
                file_path = file.path
                path_with_id = (file_path, f.get_qualified_id())
                if document_for_symbol:
                    document = document_for_symbol(f)
                else:
                    document = f.get_substring().decode()
                documents.append(document)
                paths_with_ids.append(path_with_id)

        async def async_get_embedding(document: str) -> Awaitable[Embedding]:
            loop = asyncio.get_running_loop()
            return loop.run_in_executor(None, embed_fun, document)

        # Parallel server requests
        embedded_results: List[Awaitable[Embedding]] = await asyncio.gather(
            *(async_get_embedding(document) for document in documents)
        )

        # Assign embeddings
        for path_with_id, embedding in zip(paths_with_ids, embedded_results):
            embeddings[path_with_id] = await embedding

        return cls(embeddings=embeddings, project=project)


Encoder = tiktoken.get_encoding("cl100k_base")


def token_length(string: str) -> int:
    return len(Encoder.encode(string))


MAX_TOKENS = 8192


def openai_embedding(document: str) -> Embedding:
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
    embedding = Embedding(np.array(vector))  # type: ignore
    return embedding


def get_embedding_function(openai: bool) -> EmbeddingFunction:
    if openai:
        return openai_embedding
    else:
        import spacy

        nlp = spacy.load("en_core_web_md")

        def nlp_embedding(document: str) -> Embedding:
            return Embedding(np.array(nlp(document).vector))

        return nlp_embedding


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

        def document_for_symbol(symbol: IR.Symbol) -> str:
            return symbol.get_substring().decode()

        index = await Index.create(
            document_for_symbol=document_for_symbol,
            embed_fun=embed_fun,
            project=project,
        )

        print(f"Created index in {time.time() - start:.2f} seconds")
        print(f"Saving index to file... {index_file}")
        start = time.time()
        index.save(index_file)
        print(f"Saved index in {time.time() - start:.2f} seconds")

    test_sentence = "Creates an instance of the Index class"
    query = embed_fun(test_sentence)
    scores = index.search(query)

    print("\nSemantic Search Results:")
    for n, x in scores:
        print(f"{n}  {x:.3f}")

    # bench search
    repetitions = 100
    start = time.time()
    for _ in range(repetitions):
        index.search(query)
    elapsed = time.time() - start
    print(f"\nSearched {repetitions} times in {elapsed:.2f} seconds")
