"""Generate or edit images through the TAMUS AI chat gateway.

Single run:
  python gen.py "a watercolor lighthouse at sunset"
  python gen.py "make the sky purple" --image photo.png

Batch (every image in a folder x every prompt):
  python gen.py --folder base --prompts-file prompts.txt --outdir out

Prompts can be labeled as "label | prompt text" (in a prompts file or -p). The
label becomes the output suffix: photo1.jpg + "no_makeup | ..." -> photo1_no_makeup.png.
Unlabeled prompts get p01, p02, ... Labels must be unique.
A manifest.csv in the output folder records input, label, prompt, output and status.
Re-running skips outputs that already exist, so an interrupted batch can resume.
"""
import argparse
import base64
import csv
import mimetypes
import os
import re
import sys
import time

import requests

from dotenv import load_dotenv

# Option 1: paste your key between the quotes below (don't commit/share this file).
# Option 2: leave it empty and set the environment variable instead:
#   $env:TAMUS_AI_CHAT_API_KEY = "..."

load_dotenv() 

HARDCODED_API_KEY = ""

API_KEY = HARDCODED_API_KEY or os.environ.get("TAMUS_AI_CHAT_API_KEY", "")
if not API_KEY:
    sys.exit("No API key. Set HARDCODED_API_KEY in the script or the TAMUS_AI_CHAT_API_KEY env variable.")

ENDPOINT = os.environ.get("TAMUS_AI_CHAT_API_ENDPOINT", "https://chat-api.tamu.ai").rstrip("/")
DEFAULT_MODEL = "protected.gemini-3.1-flash-image"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


class GenerationError(Exception):
    pass


def image_to_data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def build_messages(prompt, image_path):
    if not image_path:
        return [{"role": "user", "content": prompt}]
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
        ],
    }]


def chat(model, messages):
    body = {"model": model, "stream": False, "messages": messages}
    r = requests.post(f"{ENDPOINT}/api/chat/completions", headers=HEADERS, json=body, timeout=300)
    if not r.ok:
        raise GenerationError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def fetch(url):
    if not url.startswith(("http://", "https://")):
        url = f"{ENDPOINT}{url}"
    r = requests.get(url, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=60)
    r.raise_for_status()
    return r.content


def extract_image_bytes(resp):
    """Return image bytes from a chat response, or raise GenerationError."""
    msg = resp["choices"][0]["message"]

    for img in msg.get("images") or []:
        url = (img.get("image_url") or {}).get("url") or img.get("url") or ""
        m = re.match(r"data:image/[a-zA-Z+]+;base64,(.+)", url, re.S)
        if m:
            return base64.b64decode(m.group(1))
        if url:
            return fetch(url)

    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))

    m = re.search(r"data:image/[a-zA-Z+]+;base64,([A-Za-z0-9+/=]+)", content)
    if m:
        return base64.b64decode(m.group(1))
    m = re.search(r"!\[[^\]]*\]\(([^)]+)\)", content) or re.search(r"(https?://\S+\.(?:png|jpg|jpeg|webp))", content)
    if m:
        return fetch(m.group(1))

    raise GenerationError(f"No image in reply: {content[:200]!r}")


def detect_ext(data):
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def generate(model, prompt, image_path=None, retries=2):
    """Call the model with retries. Returns image bytes."""
    last = None
    for attempt in range(retries + 1):
        try:
            return extract_image_bytes(chat(model, build_messages(prompt, image_path)))
        except (GenerationError, requests.RequestException) as e:
            last = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise GenerationError(str(last))


def next_numbered_path(outdir, ext=".png"):
    os.makedirs(outdir, exist_ok=True)
    nums = [int(m.group(1)) for f in os.listdir(outdir) if (m := re.fullmatch(r"image_(\d+)\.\w+", f))]
    return os.path.join(outdir, f"image_{max(nums, default=0) + 1:03d}{ext}")


def parse_prompt(text, idx):
    """'label | prompt text' -> (label, prompt). Without a label, uses p01, p02, ..."""
    if "|" in text:
        label, _, prompt = text.partition("|")
        label = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip()).strip("_")
        if label and prompt.strip():
            return label, prompt.strip()
    return f"p{idx:02d}", text.strip()


def read_prompts(args):
    raw = []
    if args.single_prompt:
        raw.append(args.single_prompt)
    raw += list(args.prompt or [])
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            raw += [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    prompts = [parse_prompt(t, i) for i, t in enumerate(raw, start=1)]
    labels = [l for l, _ in prompts]
    if len(labels) != len(set(labels)):
        sys.exit(f"Duplicate prompt labels would overwrite each other: {labels}")
    return prompts


def existing_output(outdir, stem, label):
    base = f"{stem}_{label}"
    for ext in (".png", ".jpg", ".webp"):
        if os.path.exists(os.path.join(outdir, base + ext)):
            return base + ext
    return None


def run_batch(args, prompts):
    if not os.path.isdir(args.folder):
        sys.exit(f"Folder not found: {args.folder}")
    images = sorted(
        os.path.join(args.folder, f) for f in os.listdir(args.folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not images:
        sys.exit(f"No images ({', '.join(sorted(IMAGE_EXTS))}) found in {args.folder}")

    os.makedirs(args.outdir, exist_ok=True)
    manifest_path = os.path.join(args.outdir, "manifest.csv")
    header = ["input", "label", "prompt", "output", "status"]
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="", encoding="utf-8") as f:
            first = next(csv.reader(f), [])
        if first != header:  # older manifest format: keep it, start a fresh one
            os.replace(manifest_path, os.path.join(args.outdir, "manifest_old.csv"))
    new_manifest = not os.path.exists(manifest_path)
    total = len(images) * len(prompts)
    done = failed = skipped = 0
    print(f"{len(images)} images x {len(prompts)} prompts = {total} jobs using {args.model}")

    with open(manifest_path, "a", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        if new_manifest:
            writer.writerow(header)

        n = 0
        for img in images:
            stem = os.path.splitext(os.path.basename(img))[0]
            for label, prompt in prompts:
                n += 1
                tag = f"[{n}/{total}] {os.path.basename(img)} {label}"
                if existing_output(args.outdir, stem, label):
                    print(f"{tag}: skipped (already exists)")
                    skipped += 1
                    continue
                try:
                    data = generate(args.model, prompt, img, retries=args.retries)
                    out_name = f"{stem}_{label}{detect_ext(data)}"
                    with open(os.path.join(args.outdir, out_name), "wb") as f:
                        f.write(data)
                    writer.writerow([os.path.basename(img), label, prompt, out_name, "ok"])
                    print(f"{tag}: saved {out_name}")
                    done += 1
                except Exception as e:
                    writer.writerow([os.path.basename(img), label, prompt, "", f"failed: {e}"])
                    print(f"{tag}: FAILED ({e})", file=sys.stderr)
                    failed += 1
                mf.flush()
                if args.delay:
                    time.sleep(args.delay)

    print(f"\nDone. {done} saved, {skipped} skipped, {failed} failed. Manifest: {manifest_path}")
    if failed:
        print("Re-run the same command to retry failures (finished outputs are skipped).")


def run_single(args, prompts):
    if len(prompts) != 1:
        sys.exit("Single mode needs exactly one prompt (or use --folder for batches).")
    if args.image and not os.path.isfile(args.image):
        sys.exit(f"Input image not found: {args.image}")
    try:
        data = generate(args.model, prompts[0][1], args.image, retries=args.retries)
    except GenerationError as e:
        sys.exit(f"Failed: {e}")
    path = next_numbered_path(args.outdir, detect_ext(data))
    with open(path, "wb") as f:
        f.write(data)
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser(description="Generate or edit images via the TAMUS AI gateway.")
    p.add_argument("single_prompt", nargs="?", help="prompt for a single run")
    p.add_argument("-p", "--prompt", action="append", help="a prompt (repeat the flag for several)")
    p.add_argument("--prompts-file", help="text file with one prompt per line (# comments allowed)")
    p.add_argument("--image", help="input image to edit (single mode)")
    p.add_argument("--folder", help="folder of input images (batch mode)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--outdir", default=".", help="output folder")
    p.add_argument("--retries", type=int, default=2, help="retries per job on failure")
    p.add_argument("--delay", type=float, default=1.0, help="seconds to pause between batch jobs")
    args = p.parse_args()

    prompts = read_prompts(args)
    if not prompts:
        p.error("Give a prompt, -p/--prompt, or --prompts-file.")

    if args.folder:
        run_batch(args, prompts)
    else:
        run_single(args, prompts)


if __name__ == "__main__":
    main()