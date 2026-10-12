"""Markers: tests that need the MolProbity image are skipped without it"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import IMAGE  # noqa: E402


def pytest_collection_modifyitems(config, items):
    for item in items:
        needs = {'container', 'oracle', 'heavy'} & set(item.keywords)
        if needs and not IMAGE:
            item.add_marker(pytest.mark.skip(reason='REMOLPROBITY_IMAGE is not set'))
