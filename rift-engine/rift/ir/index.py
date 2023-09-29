# for tests:
# pip install spacy
# python -m spacy download en_core_web_md

from dataclasses import dataclass
import os
import pickle
import time
import numpy as np
import numpy.typing as npt
from typing import Callable, Dict, List, Tuple
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
        return dot_product / (norm_self * norm_other)


version = "0.0.1"

PathWithId = Tuple[str, IR.QualifiedId]


@dataclass
class Index:
    embeddings: Dict[PathWithId, Embedding]  # (file_path, id) -> embedding
    project: IR.Project
    version: str = version

    def search(self, embedding: Embedding, num_results: int = 5) -> List[Tuple[PathWithId, float]]:
        scores = [(name, embedding.similarity(y)) for name, y in self.embeddings.items()]
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
    def create(cls, embed: Callable[[str], Embedding], project: IR.Project) -> "Index":
        """
        Creates an instance of the Index class.

        Parameters:
        - embed: A callable that embeds source code strings.
        - project: The project containing files and function declarations.

        Returns:
        - An instance of the Index class.
        """
        embeddings: Dict[PathWithId, Embedding] = {}
        for file in project.get_files():
            for f in file.get_function_declarations():
                file_path = file.path
                path_with_id = (file_path, f.get_qualified_id())
                code: str = f.get_substring().decode()
                embeddings[path_with_id] = embed(code)
        return cls(embeddings=embeddings, project=project)


def test_index() -> None:
    import spacy

    # Load the spaCy model
    nlp = spacy.load("en_core_web_md")

    def nlp_embedding(x: str) -> Embedding:
        return Embedding(np.array(nlp(x).vector))

    this_dir = os.path.dirname(__file__)
    index_file = os.path.join(this_dir, "index.rci")

    if os.path.exists(index_file):
        start = time.time()
        print(f"Loading index from file... {index_file}")
        index = Index.load(index_file)
        print(f"Loaded index in {time.time() - start:.2f} seconds")
    else:
        project = parse_files_in_paths([__file__])
        # project = parse_files_in_paths([os.path.dirname(os.path.dirname(this_dir))])
        print("Creating index...")
        start = time.time()
        index = Index.create(embed=nlp_embedding, project=project)
        print(f"Created index in {time.time() - start:.2f} seconds")
        print(f"Saving index to file... {index_file}")
        start = time.time()
        index.save(index_file)
        print(f"Saved index in {time.time() - start:.2f} seconds")

    test_sentence = "Creates an instance of the Index class"

    scores = index.search(nlp_embedding(test_sentence))

    print("\nSemantic Search Results:")
    for n, x in scores:
        print(f"{n}  {x:.3f}")

    # bench search
    repetitions = 100
    start = time.time()
    for _ in range(repetitions):
        index.search(nlp_embedding(test_sentence))
    elapsed = time.time() - start
    print(f"\nSearched {repetitions} times in {elapsed:.2f} seconds")
