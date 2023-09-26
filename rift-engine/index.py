# pip install spacy
# python -m spacy download en_core_web_md

from dataclasses import dataclass
import spacy
from typing import Dict, List, Tuple

# Load the spaCy model
nlp = spacy.load("en_core_web_md")

from spacy.tokens import Doc


@dataclass
class Entry:
    name: str
    code: str
    comment: str

    @property
    def text(self) -> str:
        return self.code + " " + self.comment


entities: List[Entry] = [
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


# Create the index
index: Dict[str, Entry] = {x.name: x for x in entities}


# Embed the functions
function_embeddings: list[tuple[str, Doc]] = [
    (name, nlp(entry.text)) for name, entry in index.items()
]


def search_index_semantic(sentence: str) -> List[Tuple[str, float]]:
    sentence_embedding: Doc = nlp(sentence)

    print("Sentence Embedding:", sentence_embedding)

    scores = [(name, sentence_embedding.similarity(x)) for name, x in function_embeddings]

    # Rank based on scores
    ranked_functions = sorted(scores, key=lambda x: x[1], reverse=True)

    return ranked_functions


# Test the search function
test_sentence = "I am looking for a function that exponentiates a number."

search_results_semantic = search_index_semantic(test_sentence)
print("\nSemantic Search Results:")
for n, x in search_results_semantic:
    print(f"{index[n].name}  {x:.3f}")
        
