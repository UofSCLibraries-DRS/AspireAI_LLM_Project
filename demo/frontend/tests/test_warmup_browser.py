"""Offline browser regressions: python -m unittest discover -s demo/frontend/tests.

Requires Chrome or Chromium. No Node, browser-driver package, server, or AWS
credentials are required. Every fetch from the demo iframe is intercepted.
"""

import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class WarmupBrowserTests(unittest.TestCase):
    def test_demo_in_headless_chrome(self):
        browser = os.environ.get("CHROME_BIN") or next(
            (
                path
                for name in ("google-chrome", "chromium", "chromium-browser")
                if (path := shutil.which(name))
            ),
            None,
        )
        if browser is None:
            self.skipTest("Chrome/Chromium is required (or set CHROME_BIN)")
        directory = Path(__file__).parent
        source = json.dumps((directory.parent / "index.html").read_text()).replace(
            "</", "<\\/"
        )
        harness = (directory / "warmup.browser.js").read_text()
        with tempfile.TemporaryDirectory(prefix="aspireai-browser-test-") as temporary:
            page = Path(temporary) / "tests.html"
            page.write_text(
                '<!doctype html><html><body><pre id="results" data-status="running"></pre>'
                f"<script>const DEMO_HTML = {source};\n{harness}</script></body></html>"
            )
            process = subprocess.run(
                [
                    browser,
                    "--headless",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-background-networking",
                    "--no-first-run",
                    "--no-default-browser-check",
                    f"--user-data-dir={temporary}/profile",
                    "--allow-file-access-from-files",
                    "--virtual-time-budget=15000",
                    "--dump-dom",
                    page.as_uri(),
                ],
                capture_output=True,
                text=True,
                timeout=40,
                check=False,
            )
        result = re.search(
            r'<pre id="results" data-status="([^"]+)">(.*?)</pre>',
            process.stdout,
            re.DOTALL,
        )
        self.assertIsNotNone(result, process.stderr[-2000:])
        status, details = result.groups()
        self.assertEqual(status, "passed", html.unescape(details))
        self.assertEqual(len(json.loads(html.unescape(details))), 8)


if __name__ == "__main__":
    unittest.main()
