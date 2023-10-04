#!/usr/bin/env python3

import argparse
import asyncio
import os
import time
from typing import List

from index import Index, Query, Text
import rift.ir.IR as IR
from rift.ir.parser import parse_files_in_paths

import git

repo_root = git.Repo('.', search_parent_directories=True).git.rev_parse("--show-toplevel")
morph_dir = os.path.join(repo_root, ".morph")
os.makedirs(morph_dir, exist_ok=True)
index_file = os.path.join(morph_dir, "index.rci")


def search(args):
    # Load the index
    index = Index.load(index_file)

    # Create a query from the command line arguments
    query = Query(Text(' '.join(args)))

    # Perform the search
    start = time.time()
    scores = index.search(query)

    # Print the results
    print("\nSemantic Search Results:")
    for n, x in scores:
        print(f"{n}  {x:.3f}")
    elapsed = time.time() - start
    print(f"\nSearched in {elapsed:.2f} seconds")


async def index_repo(args):
    # Create the index
    project_root = repo_root  # the root of the git repo

    # set_embedding_function(openai=true)
    print(f"Reading $project_root...")
    project = parse_files_in_paths([project_root])
    print(f"Creating index...")
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
        project=project,
    )

    print(f"Created index in {time.time() - start:.2f} seconds")
    print(f"Saving index to file... {index_file}")
    start = time.time()
    index.save(index_file)
    print(f"Saved index in {time.time() - start:.2f} seconds")

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Perform a search in the index or index a repository.')

    # Add the arguments
    parser.add_argument('command', choices=['search', 'index'], help='the command to execute')
    parser.add_argument('arguments', metavar='N', type=str, nargs='*',
                        help='the arguments for the command')

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the appropriate function
    if args.command == 'search':
        search(args.arguments)
    elif args.command == 'index':
        asyncio.run(index_repo(args.arguments))
