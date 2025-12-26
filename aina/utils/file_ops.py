# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
from importlib import import_module


def get_repo_root():
    """Returns the absolute path to the root directory of the RUKA repository.

    Returns:
        str: Absolute path to the repository root directory
    """
    # Get the directory containing this file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up 2 levels from ruka_hand/utils to reach repo root
    repo_root = os.path.dirname(os.path.dirname(current_dir))

    return repo_root


def load_function(func_name: str):
    """Load a function dynamically from a module."""
    module_name, function_name = func_name.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, function_name)


class suppress:
    def __init__(self, stdout=True, stderr=True):
        self.suppress_stdout = stdout
        self.suppress_stderr = stderr
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        """Enter the context and suppress stdout/stderr."""
        if self.suppress_stdout:
            self.original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        if self.suppress_stderr:
            self.original_stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context and restore stdout/stderr."""
        if self.suppress_stdout:
            sys.stdout.close()
            sys.stdout = self.original_stdout
        if self.suppress_stderr:
            sys.stderr.close()
            sys.stderr = self.original_stderr

    def __call__(self, func):
        """Allow this class to be used as a decorator."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:  # Use the context manager behavior
                return func(*args, **kwargs)

        return wrapper
