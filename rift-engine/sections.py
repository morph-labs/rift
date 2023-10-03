from pdfminer.high_level import extract_text
import re
from typing import Dict, Tuple, TypedDict
import sys


class SectionStructure(TypedDict):
    span: Tuple[int, int]
    subsections: "DocumentStructure"


DocumentStructure = Dict[str, SectionStructure]


def extract_structure(pdf_path: str) -> DocumentStructure:
    def build_pattern(preceding_chars: str, title_start: str, title_continuation: str) -> str:
        """
        Helper function to build a regex pattern for section, subsection, or sub-sub-section titles.
        """
        return f"{preceding_chars}{title_start}{title_continuation}"

    # Common elements of the section pattern
    title_continuation = r"[A-Z][^\n]+"
    preceding_patterns = [r"\n\n", r"\f\n", r"\x0c"]

    def generate_depth_pattern(depth: int) -> str:
        """Generate regex pattern for a given depth."""
        return r"\d" + (r"\.\d" * depth) + r" "

    def compile_pattern_for_depth(depth: int) -> re.Pattern[str]:
        """Compile regex pattern for a given depth."""
        pattern = generate_depth_pattern(depth)
        return re.compile(
            "|".join(build_pattern(pre, pattern, title_continuation) for pre in preceding_patterns),
            re.MULTILINE,
        )

    def extract_for_depth(
        text: str, start_byte: int, end_byte: int, depth: int
    ) -> DocumentStructure:
        pattern = compile_pattern_for_depth(depth)
        matches = list(pattern.finditer(text, start_byte, end_byte))

        # Base case: If no matches are found for the current depth, return an empty dict
        if not matches:
            return {}

        doc: DocumentStructure = {}
        for i, match in enumerate(matches):
            title = match.group().strip()
            title_start_byte = match.start()
            title_end_byte = matches[i + 1].start() if i + 1 < len(matches) else end_byte

            doc[title] = {
                "span": (title_start_byte, title_end_byte),
                "subsections": extract_for_depth(text, title_start_byte, title_end_byte, depth + 1),
            }

        return doc

    text = extract_text(pdf_path)
    return extract_for_depth(text, start_byte=0, end_byte=len(text), depth=0)


def print_hierarchical_structure_with_spans(doc: DocumentStructure, indent: int = 0) -> None:
    """
    Print the hierarchical structure with proper indentations and associated byte spans.

    Args:
    - doc (DocumentStructure): Hierarchical structure of sections and subsections with byte spans.
    - indent (int): Current level of indentation.
    """
    indentation = "  " * indent
    for section, section_details in doc.items():
        print(f"{indentation}{section} {section_details['span']}")
        print_hierarchical_structure_with_spans(section_details["subsections"], indent + 1)


if __name__ == "__main__":
    pdf_path = sys.argv[1]
    structure = extract_structure(pdf_path)
    print_hierarchical_structure_with_spans(structure)
