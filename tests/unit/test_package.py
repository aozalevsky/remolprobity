import os
import re

import remolprobity


def test_version_is_the_projects():
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'pyproject.toml')) as f:
        project = re.search(r'^version = "([^"]+)"$', f.read(), re.M).group(1)
    assert remolprobity.__version__ == project
