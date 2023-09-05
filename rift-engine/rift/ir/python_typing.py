
def generate_typing_types_set():
    # from PEP 484 – Type Hints: https://peps.python.org/pep-0484/
        
    # Fundamental Building Blocks
    fundamental_building_blocks = {
        'Any',
        'Union',
        'Callable',
        'Tuple',
        'TypeVar',
        'Generic',
        'Type'
    }

    # Generic Variants of Builtin Collections
    generic_variants_builtin = {
        'Dict',
        'DefaultDict',
        'List',
        'Set',
        'FrozenSet'
    }

    # Generic Variants of Container ABCs
    generic_variants_container_abcs = {
        'Awaitable',
        'AsyncIterable',
        'AsyncIterator',
        'ByteString',
        'Collection',
        'Container',
        'ContextManager',
        'Coroutine',
        'Generator',
        'Hashable',
        'ItemsView',
        'Iterable',
        'Iterator',
        'KeysView',
        'Mapping',
        'MappingView',
        'MutableMapping',
        'MutableSequence',
        'MutableSet',
        'Sequence',
        'AbstractSet',  # formerly Set
        'Sized'
    }

    # One-off Types
    one_off_types = {
        'Reversible',
        'SupportsAbs',
        'SupportsComplex',
        'SupportsFloat',
        'SupportsInt',
        'SupportsRound',
        'SupportsBytes'
    }

    # Convenience Definitions
    convenience_definitions = {
        'Optional',
        'Text',
        'AnyStr',
        'NamedTuple',
        'NewType',
    }

    # I/O Related Types
    io_related_types = {
        'IO',
        'BinaryIO',
        'TextIO'
    }

    # Combine all sets into a single set
    all_types = (
        fundamental_building_blocks
        | generic_variants_builtin
        | generic_variants_container_abcs
        | one_off_types
        | convenience_definitions
        | io_related_types
    )

    return all_types

typing_set = generate_typing_types_set()

# Add a function to check if a name is in the typing types set
def is_typing_type(name: str) -> bool:
    return name in typing_set