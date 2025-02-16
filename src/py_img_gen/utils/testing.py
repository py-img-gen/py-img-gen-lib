import logging
import pathlib
import shutil
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

TEST_DIR = tempfile.mkdtemp(prefix="py-img-gen-test_")


class PyImgGenTestCase(object):
    r"""A custom testing class that disables some of the more verbose `py-img-gen` lib
    logging and that creates and destroys a temp directory as a test fixture.
    """

    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()
    MODULE_ROOT = PROJECT_ROOT / "src" / "py_img_gen"
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"

    _TEST_DIR: Optional[pathlib.Path] = None

    @property
    def TEST_DIR(self) -> pathlib.Path:
        assert self._TEST_DIR is not None
        return self._TEST_DIR

    def setup_method(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=logging.DEBUG,
        )

        self._TEST_DIR = pathlib.Path(TEST_DIR)
        self._TEST_DIR.mkdir(exist_ok=True)
        logger.debug(f"Made temp dir to {TEST_DIR}")

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)
