from pdfminer.high_level import extract_text
import re
from typing import Dict, Tuple, TypedDict
import sys


class Section(TypedDict):
    span: Tuple[int, int]
    subsections: "Subsections"


Subsections = Dict[str, Section]


def extract_sections(pdf_path: str) -> Subsections:
    def build_pattern(preceding_chars: str, title_start: str, title_continuation: str) -> str:
        """
        Helper function to build a regex pattern for section, subsection, or sub-sub-section titles.
        """
        return f"{preceding_chars}{title_start}{title_continuation}"

    # The title_continuation pattern seeks titles that:
    # - Start with a capital letter (indicated by [A-Z]).
    # - Continue with any characters except for a newline (indicated by [^\n]+).
    # - End either with two newline characters (indicating a new section)
    #   or at the end of the document (indicated by \Z).
    title_continuation = r"[A-Z][^\n]+(?:\n{2}|\Z)"

    # preceding_patterns contains common patterns that might precede a section title:
    # - r"\n\n": Two newline characters, indicating a new paragraph or section.
    # - r"\f\n": A form feed character followed by a newline, often seen in PDF text extraction.
    # - r"\x0c": A form feed character (alternative representation), again common in PDFs.
    preceding_patterns = [r"\n\n", r"\f\n", r"\x0c"]

    def section_number_regexp(depth: int) -> str:
        """
        Returns a regular expression string that matches a section number at the given depth.
        """
        return r"\d" + (r"\.\d" * depth) + r"\.? "

    def title_pattern(depth: int) -> re.Pattern[str]:
        """
        Returns a compiled regular expression pattern for matching section titles at the given depth.
        """
        pattern = section_number_regexp(depth)
        return re.compile(
            "|".join(build_pattern(pre, pattern, title_continuation) for pre in preceding_patterns),
            re.MULTILINE,
        )

    def extract_subsections(text: str, start_byte: int, end_byte: int, depth: int) -> Subsections:
        pattern = title_pattern(depth)
        matches = list(pattern.finditer(text, start_byte, end_byte))

        # Base case: If no matches are found for the current depth, return an empty dict
        if not matches:
            return {}

        subsections: Subsections = {}
        for i, match in enumerate(matches):
            title = match.group().strip()
            title_start_byte = match.start()
            title_end_byte = matches[i + 1].start() if i + 1 < len(matches) else end_byte

            subsection: Section = {
                "span": (title_start_byte, title_end_byte),
                "subsections": extract_subsections(
                    text, title_start_byte, title_end_byte, depth + 1
                ),
            }
            subsections[title] = subsection

        return subsections

    text = extract_text(pdf_path)
    # print(f"XXX\n{text}\nXXX")
    return extract_subsections(text, start_byte=0, end_byte=len(text), depth=0)


def print_subsections(subsections: Subsections, indent: int = 0) -> None:
    """
    Print the hierarchical structure with proper indentations and associated byte spans.

    Args:
    - doc (DocumentStructure): Hierarchical structure of sections and subsections with byte spans.
    - indent (int): Current level of indentation.
    """
    indentation = "  " * indent
    for title, section in subsections.items():
        print(f"{indentation}{title} {section['span']}")
        print_subsections(section["subsections"], indent + 1)


if __name__ == "__main__":
    pdf_path = sys.argv[1]
    subsections = extract_sections(pdf_path)
    print_subsections(subsections)
