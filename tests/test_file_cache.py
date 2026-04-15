import os
import pathlib
import json
import unittest
import shutil

import pytest

from scispacy.file_cache import filename_to_url, url_to_filename

class TestFileUtils(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.TEST_DIR = "/tmp/scispacy"
        os.makedirs(self.TEST_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR)

    def test_url_to_filename(self):
        for url in ['http://allenai.org', 'http://cool.org',
                    'https://www.google.com', 'http://pytorch.org',
                    'https://s3-us-west-2.amazonaws.com/cool' + '/long' * 20 + '/url']:
            filename = url_to_filename(url)
            assert "http" not in filename
            with pytest.raises(FileNotFoundError):
                filename_to_url(filename, cache_dir=self.TEST_DIR)
            pathlib.Path(os.path.join(self.TEST_DIR, filename)).touch()
            with pytest.raises(FileNotFoundError):
                filename_to_url(filename, cache_dir=self.TEST_DIR)
            json.dump({'url': url, 'etag': None},
                      open(os.path.join(self.TEST_DIR, filename + '.json'), 'w'))
            back_to_url, etag = filename_to_url(filename, cache_dir=self.TEST_DIR)
            assert back_to_url == url
            assert etag is None

    def test_url_to_filename_with_etags(self):
        for url in ['http://allenai.org', 'http://cool.org',
                    'https://www.google.com', 'http://pytorch.org']:
            filename = url_to_filename(url, etag="mytag")
            assert "http" not in filename
            pathlib.Path(os.path.join(self.TEST_DIR, filename)).touch()
            json.dump({'url': url, 'etag': 'mytag'},
                      open(os.path.join(self.TEST_DIR, filename + '.json'), 'w'))
            back_to_url, etag = filename_to_url(filename, cache_dir=self.TEST_DIR)
            assert back_to_url == url
            assert etag == "mytag"
        baseurl = 'http://allenai.org/'
        assert url_to_filename(baseurl + '1') != url_to_filename(baseurl, etag='1')

    def test_url_to_filename_with_etags_eliminates_quotes(self):
        for url in ['http://allenai.org', 'http://cool.org',
                    'https://www.google.com', 'http://pytorch.org']:
            filename = url_to_filename(url, etag='"mytag"')
            assert "http" not in filename
            pathlib.Path(os.path.join(self.TEST_DIR, filename)).touch()
            json.dump({'url': url, 'etag': 'mytag'},
                      open(os.path.join(self.TEST_DIR, filename + '.json'), 'w'))
            back_to_url, etag = filename_to_url(filename, cache_dir=self.TEST_DIR)
            assert back_to_url == url
            assert etag == "mytag"

    def test_url_to_filename_stays_within_name_max(self):
        # eCryptfs limits filenames to 143 bytes; make sure we stay under that
        # even with a long URL and etag.
        long_url = "https://s3-us-west-2.amazonaws.com/bucket/" + "a" * 300 + "/file.npz"
        long_etag = "x" * 300
        filename = url_to_filename(long_url, etag=long_etag)
        assert len(filename) <= 143
        assert filename.endswith(".npz")
        # also without etag
        filename_no_etag = url_to_filename(long_url)
        assert len(filename_no_etag) <= 143

    def test_url_to_filename_no_extension(self):
        # URLs without a file extension should still produce a valid filename
        filename = url_to_filename("https://example.com/data/somefile")
        assert len(filename) == 64  # just the sha256 hex digest
        assert "." not in filename

    def test_legacy_cache_files_still_found(self):
        from scispacy.file_cache import _find_legacy_cache_path

        url = "https://example.com/data/model.bin"
        etag = "some-etag"
        # Create a file with the old naming scheme
        old_filename = (
            "b6794c9b5101703824700fe53156f28b7c5c2ef432467c1399f30142e7db9977"
            ".700ccb3dacaae313fbd70ea50e5646377634d6f144ea63acaf30d8e7ecf1cc4e"
            ".model.bin"
        )
        old_path = os.path.join(self.TEST_DIR, old_filename)
        pathlib.Path(old_path).touch()

        found = _find_legacy_cache_path(url, etag, self.TEST_DIR)
        assert found == old_path
