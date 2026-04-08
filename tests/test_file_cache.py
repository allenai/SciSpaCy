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

    def test_url_to_filename_length_under_ecryptfs_limit(self):
        """Filenames (including .json sidecar) must stay under 143 bytes for eCryptfs.
        See: https://github.com/allenai/scispacy/issues/539
        """
        # These are the actual URLs used by scispacy linkers
        urls = [
            'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/nmslib_index.bin',
            'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/tfidf_vectorizer.joblib',
            'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/tfidf_vectors_sparse.npz',
            'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/concept_aliases.json',
        ]
        # Simulate a realistic 64-char hex ETag
        long_etag = '"d41d8cd98f00b204e9800998ecf8427ed41d8cd98f00b204e9800998ecf8427e"'
        for url in urls:
            filename = url_to_filename(url, etag=long_etag)
            meta_filename = filename + ".json"
            assert len(meta_filename) < 143, (
                f"Metadata filename too long for eCryptfs ({len(meta_filename)} >= 143): {meta_filename}"
            )
