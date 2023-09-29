# for tests:
# pip install spacy
# python -m spacy download en_core_web_md

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple


@dataclass
class Embedding:
    embedding: npt.NDArray[np.float32]

    def similarity(self, other: "Embedding") -> float:
        dot_product = self.embedding.dot(other.embedding)
        norm_self = np.linalg.norm(self.embedding)
        norm_other = np.linalg.norm(other.embedding)
        return dot_product / (norm_self * norm_other)


@dataclass
class Index:
    index: Dict[str, Embedding]

    def search(self, x: Embedding) -> List[Tuple[str, float]]:
        scores = [(name, x.similarity(y)) for name, y in self.index.items()]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores


def test_index() -> None:
    @dataclass
    class Entry:
        name: str
        code: str
        comment: str

        @property
        def text(self) -> str:
            return self.code + " " + self.comment

    entries: List[Entry] = [
        Entry(
            "foo",
            "def foo(x): return x+1",
            """
        This function adds 1 to the input parameter 'x'.
        
        :param x: The input value.
        :return: The result of adding 1 to 'x'.
        """,
        ),
        Entry(
            "bar",
            "def bar(x): return x*2",
            """
        This function multiplies the input parameter 'x' by 2.
        
        :param x: The input value.
        :return: The result of multiplying 'x' by 2.
        """,
        ),
        Entry(
            "baz",
            "def baz(x): return x-3",
            """
        This function subtracts 3 from the input parameter 'x'.
        
        :param x: The input value.
        :return: The result of subtracting 3 from 'x'.
        """,
        ),
        Entry(
            "qux",
            "def qux(x): return x/4",
            """
        This function divides the input parameter 'x' by 4.
        
        :param x: The input value.
        :return: The result of dividing 'x' by 4.
        """,
        ),
        Entry(
            "quux",
            "def quux(y): return y**2",
            """
        This function calculates the square of the input parameter 'y'.
        
        :param y: The input value.
        :return: The square of 'y'.
        """,
        ),
    ]

    import spacy

    # Load the spaCy model
    nlp = spacy.load("en_core_web_md")

    def nlp_embedding(x: str) -> Embedding:
        return Embedding(np.array(nlp(x).vector))

    def index_from_entries(entries: List[Entry]) -> Index:
        return Index({x.name: nlp_embedding(x.text) for x in entries})

    index: Index = index_from_entries(entries)

    test_sentence = "I am looking for a function that exponentiates a number."

    scores = index.search(nlp_embedding(test_sentence))

    print("\nSemantic Search Results:")
    for n, x in scores:
        print(f"{n}  {x:.3f}")
