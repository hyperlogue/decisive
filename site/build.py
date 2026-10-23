#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
parser = argparse.ArgumentParser()
parser.add_argument("--package", type=Path, default=ROOT / "artifacts/web/privacy-build/pkg")
args = parser.parse_args()
package = args.package.resolve()
metadata = json.loads((package / "build.json").read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if digest(package / "decisive_bg.wasm") != metadata["wasm_sha256"]:
    raise SystemExit("WASM hash does not match build.json.")
for name, expected in metadata["sources"].items():
    if digest(ROOT / name) != expected:
        raise SystemExit(f"Package does not match current source: {name}. Build an unchanged runtime first.")
if not {"decima", "pplx-flash", "wasm", "webgpu", "kernel-checks"} <= set(metadata["features"].split(",")):
    raise SystemExit("Package must include decima, pplx-flash, wasm, webgpu, and kernel-checks.")

dist = SITE / "dist"
if dist.exists():
    shutil.rmtree(dist)
dist.mkdir()
for name in ["index.html", "style.css", "app.js", "models.json"]:
    shutil.copyfile(SITE / name, dist / name)
for name in ["lib", "workers", "assets"]:
    shutil.copytree(SITE / name, dist / name)

fontsource = SITE / "node_modules/@fontsource/noto-sans-sc"
if not (fontsource / "400.css").is_file():
    raise SystemExit("Missing font package. Run: npm ci --prefix site --ignore-scripts")
fonts = dist / "assets/fonts"
fonts.mkdir()
faces = []
for face in re.findall(r"@font-face\s*\{[^}]*\}", (fontsource / "400.css").read_text()):
    name = re.search(r"url\(\./files/([\w-]+\.woff2)\)", face).group(1)
    ranges = re.search(r"unicode-range:\s*([^;]+);", face).group(1)
    shutil.copyfile(fontsource / "files" / name, fonts / name)
    faces.append('@font-face{font-family:"Noto Sans SC";font-style:normal;font-weight:400;font-display:swap;'
                 'src:local("Noto Sans CJK SC"),local("PingFang SC"),local("Microsoft YaHei"),'
                 f'url("./assets/fonts/{name}") format("woff2");unicode-range:{ranges}}}')
if not faces:
    raise SystemExit("No font faces found in @fontsource/noto-sans-sc/400.css.")
shutil.copyfile(fontsource / "LICENSE", fonts / "LICENSE.txt")
(dist / "style.css").write_text("\n".join(faces) + "\n" + (dist / "style.css").read_text())
(dist / "runtime").mkdir()
for name in ["runtime.js", "embedding-cache.js", "read-worker.js"]:
    shutil.copyfile(ROOT / "examples/pplx-flash" / name, dist / "runtime" / name)
(dist / "pkg").mkdir()
for name in ["decisive.js", "decisive_bg.wasm", "build.json"]:
    shutil.copyfile(package / name, dist / "pkg" / name)
for name in ["LICENSE-MIT", "LICENSE-APACHE", "NOTICE"]:
    shutil.copyfile(ROOT / name, dist / name)
(dist / ".nojekyll").touch()

files = sorted(p for p in dist.rglob("*") if p.is_file())
version = hashlib.sha256(("".join(digest(p) for p in files) + digest(SITE / "sw-template.js")).encode()).hexdigest()[:16]
release = dist / "releases" / version
for file in files:
    if file.name in ["index.html", ".nojekyll"]:
        continue
    destination = release / file.relative_to(dist)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(file, destination)
html = (dist / "index.html").read_text()
html = html.replace('href="./style.css"', f'href="./releases/{version}/style.css"')
html = html.replace('src="./app.js"', f'src="./releases/{version}/app.js"')
html = html.replace('href="./assets/favicon.svg"', f'href="./releases/{version}/assets/favicon.svg"')
(dist / "index.html").write_text(html)
font_dir = f"./releases/{version}/assets/fonts/"
assets = ["./index.html", *sorted(path for path in ("./" + p.relative_to(dist).as_posix() for p in release.rglob("*") if p.is_file()) if not path.startswith(font_dir))]
(dist / "release.json").write_text(json.dumps({"version":version}) + "\n")
sw = (SITE / "sw-template.js").read_text().replace("__ASSETS__", json.dumps(assets)).replace("__VERSION__", version).replace("__FONTS__", font_dir)
(dist / "sw.js").write_text(sw)
print(f"Built site/dist ({sum(p.stat().st_size for p in dist.rglob('*') if p.is_file()):,} bytes); model weights remain on Hugging Face.")
