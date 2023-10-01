from rich.console import Console
from rich.logging import RichHandler
import logging

def configure_logger():
    FORMAT = "%(message)s"
    console = Console(stderr=True)
    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(console=console)],
    )