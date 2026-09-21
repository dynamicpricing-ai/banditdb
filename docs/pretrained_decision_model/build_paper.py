#!/usr/bin/env python3
"""Render paper.html to PDF.

The figures are inlined as data URIs so the PDF is self-contained, then Chrome
is used headlessly to print it. MathJax is fetched from a CDN at render time, so
this step needs network access; without it the equations come out blank.

    python3 build_paper.py [output.pdf]
"""
import base64
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = {
    "{{FIG_PRIOR}}":    "fig_prior.png",
    "{{FIG_OPE}}":      "fig_ope.png",
    "{{FIG_CAPACITY}}": "fig_capacity.png",
    "{{FIG_CATALOG}}":  "fig_catalog.png",
    "{{FIG_ENTROPY}}":  "fig_entropy.png",
}
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "chromium", "chromium-browser",
]


def find_chrome():
    for c in CHROME_CANDIDATES:
        if os.path.exists(c):
            return c
        found = shutil.which(c)
        if found:
            return found
    sys.exit("no Chrome/Chromium found; install one or set CHROME=/path/to/chrome")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "BanditDB_Pretrained_Decision_Model.pdf")

    html = open(os.path.join(HERE, "paper.html")).read()
    for token, name in FIGURES.items():
        path = os.path.join(HERE, "figures", name)
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        html = html.replace(token, f"data:image/png;base64,{b64}")
    if "{{" in html:
        sys.exit("a figure placeholder was left unsubstituted")

    built = os.path.join(HERE, "paper_built.html")
    open(built, "w").write(html)

    chrome = os.environ.get("CHROME") or find_chrome()
    subprocess.run([
        chrome, "--headless", "--disable-gpu",
        # MathJax typesets asynchronously; the budget gives it time to finish.
        "--virtual-time-budget=30000", "--no-pdf-header-footer",
        f"--print-to-pdf={out}", f"file://{built}",
    ], check=True, stderr=subprocess.DEVNULL)

    size = os.path.getsize(out)
    with open(out, "rb") as fh:
        data = fh.read()
    pages = data.count(b"/Type /Page") - data.count(b"/Type /Pages")
    images = data.count(b"/Subtype /Image")
    print(f"wrote {out}  ({size // 1024} KB, {pages} pages, {images} figures embedded)")
    if images != len(FIGURES):
        print("WARNING: expected", len(FIGURES), "figures", file=sys.stderr)


if __name__ == "__main__":
    main()
