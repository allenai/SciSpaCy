import os
import re
import unittest

from packaging.requirements import Requirement

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_project_dependencies():
    """
    Extract the `dependencies` list from pyproject.toml without a toml parser
    (keeps the test runnable on python 3.9, which has no stdlib tomllib).
    """
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), encoding="utf-8") as f:
        text = f.read()
    match = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.DOTALL)
    assert match is not None, "could not find dependencies list in pyproject.toml"
    return [Requirement(dep) for dep in re.findall(r'"([^"]+)"', match.group(1))]


def get_requirement(dependencies, name):
    for dep in dependencies:
        if dep.name == name:
            return dep
    raise AssertionError(f"no {name} dependency found in pyproject.toml")


class TestDependencyCompatibility(unittest.TestCase):
    """
    Regression tests for allenai/scispacy#536 ("Incompatibility with numpy v2").

    The old pin set (spacy>=3.7.0,<3.9.0 with numpy<2.0) forced numpy 1.x while
    allowing the numpy-2-era spacy 3.8 line, and made scispacy unresolvable in
    environments that already had numpy v2 (e.g. via pandas), producing
    `ValueError: numpy.dtype size changed, may indicate binary incompatibility`.
    The fix requires the numpy-2-compatible spacy line and drops the numpy<2 cap.
    """

    @classmethod
    def setUpClass(cls):
        cls.dependencies = read_project_dependencies()

    def test_spacy_requires_numpy2_compatible_line(self):
        spacy_req = get_requirement(self.dependencies, "spacy")
        # spacy 3.8 is the first line with numpy v2 support; 3.7.x must be excluded
        self.assertTrue(
            spacy_req.specifier.contains("3.8.0", prereleases=True),
            f"spacy pin {spacy_req} does not allow the numpy-2-compatible 3.8 line",
        )
        self.assertFalse(
            spacy_req.specifier.contains("3.7.9", prereleases=True),
            f"spacy pin {spacy_req} still allows the pre-numpy-2 3.7 line",
        )

    def test_numpy_cap_allows_v2(self):
        numpy_req = get_requirement(self.dependencies, "numpy")
        for version in ("2.0.0", "2.2.6"):
            self.assertTrue(
                numpy_req.specifier.contains(version, prereleases=True),
                f"numpy pin {numpy_req} excludes numpy v2 ({version})",
            )

    def test_numpy_has_sane_floor(self):
        # floor mirrors spacy 3.8's own numpy requirement on python >= 3.9
        numpy_req = get_requirement(self.dependencies, "numpy")
        self.assertTrue(
            numpy_req.specifier.contains("1.19.0", prereleases=True),
            f"numpy pin {numpy_req} unexpectedly excludes 1.19.0",
        )
