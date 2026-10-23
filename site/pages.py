#!/usr/bin/env python3
import argparse
from html import escape
from pathlib import Path
import shutil

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
DEMO = "demo/private-info-detector"


def normalize_base(value):
    value = value.rstrip('/')
    if value and (not value.startswith('/') or value.startswith('//') or any(c in value for c in '\\?#') or any(part in ('.', '..') for part in value.split('/'))):
        raise ValueError("Pages base must be empty or an absolute URL path without traversal, query, or fragment.")
    return value


def redirect(target, status=""):
    target = escape(target, quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<meta http-equiv="refresh" content="0;url={target}">
<title>Decisive</title>
<style>body{{margin:0;min-height:100vh;display:grid;place-items:center;font:15px/1.5 system-ui,sans-serif;color:#344054;background:#f6f7f9}}a{{color:#4f46e5}}@media(prefers-color-scheme:dark){{body{{color:#d0d5dd;background:#0b0e14}}a{{color:#818cf8}}}}</style>
</head>
<body><p>{status}<a href="{target}">Open the Decisive privacy demo →</a></p></body>
</html>
"""

def build_pages(dist, out, base=""):
    dist, out = Path(dist).resolve(), Path(out).resolve()
    base = normalize_base(base)
    if not (dist / "index.html").is_file():
        raise ValueError("Run site/build.py first.")
    if out == dist or out.is_relative_to(dist) or dist.is_relative_to(out):
        raise ValueError("Pages output must not overlap its source or a source ancestor.")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(dist, out / DEMO)
    (out / ".nojekyll").touch()
    (out / "index.html").write_text(redirect(f"./{DEMO}/"))
    (out / "demo" / "index.html").write_text(redirect("./private-info-detector/"))
    (out / "404.html").write_text(redirect(f"{base}/{DEMO}/", "Page not found. "))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/pages")
    parser.add_argument("--base", default="")
    args = parser.parse_args()
    try:
        out = build_pages(SITE / "dist", args.out, args.base)
    except ValueError as error:
        parser.error(str(error))
    print(f"Pages site at {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}; demo under /{DEMO}/")


if __name__ == "__main__":
    main()
