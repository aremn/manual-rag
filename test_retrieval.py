import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from avia_agent.tools import ListFiles, ReadTextFile, Search, dispatch_tool


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.docs = self.root / "manuals"
        self.docs.mkdir()
        self.manual = self.docs / "demo.txt"
        self.manual.write_text("Fuel must be checked before departure.\n", encoding="utf-8")
        self.state = {"docs_root": str(self.docs)}

    def test_search_returns_document_evidence(self):
        with patch.dict(os.environ, {"AVIA_SEARCH_BACKEND": "python"}):
            result = dispatch_tool(Search(root=str(self.docs), query="fuel"), self.state)
        self.assertTrue(result.ok)
        self.assertTrue(result.payload["results"])
        self.assertIn("demo.txt", str(result.payload["results"]))

    def test_read_outside_document_root_is_rejected(self):
        private = self.root / "outside.txt"
        private.write_text("outside document root", encoding="utf-8")
        result = dispatch_tool(ReadTextFile(path=str(private)), self.state)
        self.assertFalse(result.ok)
        self.assertNotIn("text", result.payload)

    def test_file_listing_stays_inside_document_root(self):
        result = dispatch_tool(ListFiles(root=str(self.docs)), self.state)
        self.assertTrue(result.ok)
        self.assertEqual(result.payload["count"], 1)
        self.assertFalse(dispatch_tool(ListFiles(root=str(self.root)), self.state).ok)


if __name__ == "__main__":
    unittest.main()
