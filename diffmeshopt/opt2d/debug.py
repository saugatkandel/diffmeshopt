# debug.py
import logging
from contextlib import contextmanager

import torch

DEBUG = True

_generator: torch.Generator | None = None


def get_generator(
    device: torch.device | str = "cpu", manual_seed: int = 1234, reset: bool = False
) -> torch.Generator:
    global _generator
    if reset or _generator is None:
        if reset:
            logging.info("Resetting generator.")
        logging.info(f"Creating new generator with seed {manual_seed} on device {device}.")
        _generator = torch.Generator(device=device)
        _generator.manual_seed(manual_seed)
    return _generator


@contextmanager
def debug_warning(context_message: str):
    # Everything before 'yield' runs when entering the 'with' block
    print("THIS IS A DEBUG SETTING AND NEEDS TO BE CHANGED. Context:")
    print(context_message)
    try:
        yield
    finally:
        # Everything after 'yield' runs when exiting the block (even if an error happens)
        pass
