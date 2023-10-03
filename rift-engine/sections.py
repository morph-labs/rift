from pdfminer.high_level import extract_text
import re
from typing import Dict, Tuple, TypedDict
import sys


# Define the sub-sub-section structure
class SubSubSectionDict(TypedDict):
    span: Tuple[int, int]


# Define the subsection structure
class SubSectionDictStructure(TypedDict):
    span: Tuple[int, int]
    sub_subsections: Dict[str, SubSubSectionDict]


# Define the section structure
class SectionDict(TypedDict):
    span: Tuple[int, int]
    subsections: Dict[str, SubSectionDictStructure]


# The structure for the entire document
DocumentStructure = Dict[str, SectionDict]


def extract_structure(pdf_path: str) -> DocumentStructure:
    def build_pattern(preceding_chars: str, title_start: str, title_continuation: str) -> str:
        """
        Helper function to build a regex pattern for section, subsection, or sub-sub-section titles.

        Args:
        - preceding_chars (str): Characters that typically precede a title (e.g., "\n\n", "\f\n", "\x0c").
        - title_start (str): The expected starting pattern of the title (e.g., "i ", "i.j ", "i.j.k ").
        - title_continuation (str): The expected continuation of the title (e.g., "[A-Z][^\n]+").

        Returns:
        - A regex pattern string.
        """
        return f"{preceding_chars}{title_start}{title_continuation}"

    # Common elements of the section pattern
    title_start_section = r"\d "  # Matches titles like "1 Introduction"
    title_start_subsection = r"\d\.\d "  # Matches titles like "3.1 Code generation"
    title_start_sub_subsection = (
        r"\d\.\d\.\d "  # Matches titles like "3.1.1 Python code generation"
    )
    title_continuation = r"[A-Z][^\n]+"  # Continuation of the title

    # Constructing the patterns
    preceding_patterns = [r"\n\n", r"\f\n", r"\x0c"]

    section_patterns = [
        build_pattern(pre, title_start_section, title_continuation) for pre in preceding_patterns
    ]
    subsection_patterns = [
        build_pattern(pre, title_start_subsection, title_continuation) for pre in preceding_patterns
    ]
    sub_subsection_patterns = [
        build_pattern(pre, title_start_sub_subsection, title_continuation)
        for pre in preceding_patterns
    ]

    # Joining the patterns with the OR operator to form the final regex patterns
    section_pattern = re.compile("|".join(section_patterns), re.MULTILINE)
    subsection_pattern = re.compile("|".join(subsection_patterns), re.MULTILINE)
    sub_subsection_pattern = re.compile("|".join(sub_subsection_patterns), re.MULTILINE)

    def extract_structure_with_spans_corrected(pdf_path: str) -> DocumentStructure:
        """
        Corrected function to extract the hierarchical structure (sections, subsections, and sub-sub-sections)
        from a given PDF with associated byte spans. This function addresses the nesting issue.

        Args:
        - pdf_path (str): Path to the PDF file.

        Returns:
        - Nested structure of sections, their respective subsections, sub-sub-sections,
        and their associated byte spans.
        """
        # Extract text from the PDF
        text = extract_text(pdf_path)

        # Gather all matches for sections, subsections, and sub-sub-sections
        section_matches = list(section_pattern.finditer(text))
        subsection_matches = list(subsection_pattern.finditer(text))
        sub_subsection_matches = list(sub_subsection_pattern.finditer(text))

        # Create nested structure with byte spans
        structure: DocumentStructure = {}

        # Iterate through sections
        for i, section_match in enumerate(section_matches):
            section_title = section_match.group().strip()
            section_start_byte = section_match.start()
            section_end_byte = (
                section_matches[i + 1].start() if i + 1 < len(section_matches) else len(text)
            )

            structure[section_title] = {
                "span": (section_start_byte, section_end_byte),
                "subsections": {},
            }

            # Filter subsections that belong to the current section
            relevant_subsections = [
                sub
                for sub in subsection_matches
                if section_start_byte <= sub.start() < section_end_byte
            ]

            # Iterate through the filtered subsections
            for j, subsection_match in enumerate(relevant_subsections):
                subsection_title = subsection_match.group().strip()
                subsection_start_byte = subsection_match.start()
                subsection_end_byte = (
                    relevant_subsections[j + 1].start()
                    if j + 1 < len(relevant_subsections)
                    else section_end_byte
                )

                structure[section_title]["subsections"][subsection_title] = {
                    "span": (subsection_start_byte, subsection_end_byte),
                    "sub_subsections": {},
                }

                # Filter sub-sub-sections that belong to the current subsection
                relevant_sub_subsections = [
                    sub_sub
                    for sub_sub in sub_subsection_matches
                    if subsection_start_byte <= sub_sub.start() < subsection_end_byte
                ]

                # Iterate through the filtered sub-sub-sections
                for k, sub_subsection_match in enumerate(relevant_sub_subsections):
                    sub_subsection_title = sub_subsection_match.group().strip()
                    sub_subsection_start_byte = sub_subsection_match.start()
                    sub_subsection_end_byte = (
                        relevant_sub_subsections[k + 1].start()
                        if k + 1 < len(relevant_sub_subsections)
                        else subsection_end_byte
                    )

                    structure[section_title]["subsections"][subsection_title]["sub_subsections"][
                        sub_subsection_title
                    ] = {"span": (sub_subsection_start_byte, sub_subsection_end_byte)}

        return structure

    return extract_structure_with_spans_corrected(pdf_path)


def print_hierarchical_structure_with_spans(structure: DocumentStructure) -> None:
    """
    Print the hierarchical structure with proper indentations and associated byte spans.

    Args:
    - structure (dict): Hierarchical structure of sections and subsections with byte spans.
    """
    for section, section_details in structure.items():
        print(f"{section} {section_details['span']}")
        for subsection, subsection_details in section_details["subsections"].items():
            print(f"  {subsection} {subsection_details['span']}")
            for sub_subsection, sub_subsection_details in subsection_details[
                "sub_subsections"
            ].items():
                print(f"    {sub_subsection} {sub_subsection_details['span']}")


if __name__ == "__main__":
    pdf_path = sys.argv[1]
    structure = extract_structure(pdf_path)
    print_hierarchical_structure_with_spans(structure)
