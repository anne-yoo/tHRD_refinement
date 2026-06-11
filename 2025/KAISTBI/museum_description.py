#!/usr/bin/env python3
"""Scrape object descriptions from Met collection pages and append them to a CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Optional, Tuple

import requests


DEFAULT_INPUT = "/home/jiye/jiye/KAIST/BIteamproject/MET_datacleaning.csv"
DEFAULT_OUTPUT = "/home/jiye/jiye/KAIST/BIteamproject/MET_datacleaning_with_description.csv"
DEFAULT_LINK_COLUMN = "Link Resource"
DEFAULT_DESCRIPTION_COLUMN = "Description"

OBJECT_ID_RE = re.compile(r"/search/(\d+)")
CHECKPOINT_TEXT = "Vercel Security Checkpoint"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


class ReadMoreContentParser(HTMLParser):
    """Collect text inside Met's object overview read-more block."""

    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "ol",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.capture_depth = 0
        self.current: List[str] = []
        self.blocks: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attrs_dict = dict(attrs)
        if self.capture_depth:
            if tag in self.BLOCK_TAGS:
                self.current.append(" ")
            self.capture_depth += 1
        elif attrs_dict.get("data-testid") == "read-more-content":
            self.capture_depth = 1
            self.current = []

    def handle_endtag(self, tag: str) -> None:
        if not self.capture_depth:
            return

        if tag in self.BLOCK_TAGS:
            self.current.append(" ")

        self.capture_depth -= 1
        if self.capture_depth == 0:
            text = normalize_text("".join(self.current))
            if text:
                self.blocks.append(text)
            self.current = []

    def handle_data(self, data: str) -> None:
        if self.capture_depth:
            self.current.append(data)


@dataclass
class ScrapeResult:
    object_id: str
    url: str
    description: str
    status_code: Optional[int]
    error: str = ""

    @property
    def cacheable(self) -> bool:
        return self.status_code in {200, 404}

    def as_json(self) -> str:
        payload = {
            "object_id": self.object_id,
            "url": self.url,
            "description": self.description,
            "status_code": self.status_code,
            "error": self.error,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        return json.dumps(payload, ensure_ascii=False)


thread_local = threading.local()


def normalize_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split()).strip()


def object_id_from_link(link: str) -> Optional[str]:
    match = OBJECT_ID_RE.search(link or "")
    return match.group(1) if match else None


def canonical_url(object_id: str) -> str:
    return f"https://www.metmuseum.org/art/collection/search/{object_id}"


def get_session() -> requests.Session:
    session = getattr(thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(BROWSER_HEADERS)
        thread_local.session = session
    return session


def extract_description(html_text: str) -> str:
    parser = ReadMoreContentParser()
    parser.feed(html_text)

    for block in parser.blocks:
        if block and not block.lower().startswith("show more"):
            return block
    return ""


def fetch_description(object_id: str, timeout: int, retries: int, sleep: float) -> ScrapeResult:
    url = canonical_url(object_id)
    session = get_session()
    last_error = ""
    status_code: Optional[int] = None

    for attempt in range(retries + 1):
        if sleep:
            time.sleep(sleep)

        try:
            response = session.get(url, timeout=timeout)
            status_code = response.status_code
            text = response.text

            if status_code == 200 and CHECKPOINT_TEXT not in text:
                return ScrapeResult(
                    object_id=object_id,
                    url=response.url,
                    description=extract_description(text),
                    status_code=status_code,
                )

            if status_code == 404:
                return ScrapeResult(object_id=object_id, url=response.url, description="", status_code=status_code)

            if CHECKPOINT_TEXT in text:
                last_error = "met_security_checkpoint"
            else:
                last_error = f"http_{status_code}"
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < retries:
            time.sleep(min(30, 2**attempt))

    return ScrapeResult(object_id=object_id, url=url, description="", status_code=status_code, error=last_error)


def load_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        return list(reader.fieldnames), list(reader)


def load_cache(path: str) -> Dict[str, ScrapeResult]:
    cached: Dict[str, ScrapeResult] = {}
    if not os.path.exists(path):
        return cached

    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid cache line {line_number}: {path}", file=sys.stderr)
                continue

            object_id = str(record.get("object_id", "")).strip()
            if not object_id:
                continue

            cached[object_id] = ScrapeResult(
                object_id=object_id,
                url=str(record.get("url", "")),
                description=str(record.get("description", "")),
                status_code=record.get("status_code"),
                error=str(record.get("error", "")),
            )
    return cached


def append_cache(path: str, result: ScrapeResult) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(result.as_json())
        handle.write("\n")


def write_csv(
    path: str,
    fieldnames: List[str],
    rows: Iterable[Dict[str, str]],
    descriptions_by_id: Dict[str, ScrapeResult],
    link_column: str,
    description_column: str,
) -> None:
    output_fields = list(fieldnames)
    if description_column not in output_fields:
        output_fields.append(description_column)

    output_dir = os.path.dirname(path) or "."
    os.makedirs(output_dir, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(prefix=".museum_description_", suffix=".csv", dir=output_dir)
    os.close(fd)

    try:
        with open(temp_path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            for row in rows:
                row_out = dict(row)
                object_id = object_id_from_link(row.get(link_column, ""))
                if object_id and object_id in descriptions_by_id:
                    row_out[description_column] = descriptions_by_id[object_id].description
                else:
                    row_out[description_column] = row_out.get(description_column, "")
                writer.writerow(row_out)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input CSV path. Default: {DEFAULT_INPUT}")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output CSV path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--cache", default="", help="JSONL cache path. Default: <output>.cache.jsonl")
    parser.add_argument("--link-column", default=DEFAULT_LINK_COLUMN, help="Column containing Met object URLs.")
    parser.add_argument("--description-column", default=DEFAULT_DESCRIPTION_COLUMN, help="Column to add/update.")
    parser.add_argument("--workers", type=int, default=6, help="Number of concurrent requests.")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per object.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep before each request in each worker.")
    parser.add_argument("--write-every", type=int, default=500, help="Rewrite output every N newly fetched pages.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of uncached pages to fetch.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_path = args.cache or f"{os.path.splitext(args.output)[0]}.cache.jsonl"

    fieldnames, rows = load_rows(args.input)
    if args.link_column not in fieldnames:
        raise ValueError(f"Column '{args.link_column}' not found in {args.input}")

    cached = load_cache(cache_path)
    descriptions_by_id = dict(cached)

    object_ids: List[str] = []
    seen = set()
    for row in rows:
        object_id = object_id_from_link(row.get(args.link_column, ""))
        if object_id and object_id not in seen:
            seen.add(object_id)
            object_ids.append(object_id)

    pending = [object_id for object_id in object_ids if object_id not in descriptions_by_id]
    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"Input rows: {len(rows):,}")
    print(f"Unique Met object IDs: {len(object_ids):,}")
    print(f"Cached object IDs: {len(cached):,}")
    print(f"Pending fetches this run: {len(pending):,}")
    print(f"Output: {args.output}")
    print(f"Cache: {cache_path}")

    if pending:
        started = time.time()
        fetched = 0
        found = sum(1 for result in descriptions_by_id.values() if result.description)
        errors = 0

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(fetch_description, object_id, args.timeout, args.retries, args.sleep): object_id
                for object_id in pending
            }

            for future in as_completed(futures):
                result = future.result()
                fetched += 1

                if result.error:
                    errors += 1
                if result.description:
                    found += 1

                descriptions_by_id[result.object_id] = result
                if result.cacheable:
                    append_cache(cache_path, result)

                if fetched == 1 or fetched % 100 == 0 or fetched == len(pending):
                    elapsed = max(time.time() - started, 0.001)
                    rate = fetched / elapsed
                    print(
                        "Progress: "
                        f"{fetched:,}/{len(pending):,} fetched, "
                        f"{found:,} descriptions found total, "
                        f"{errors:,} errors this run, "
                        f"{rate:.2f} pages/sec"
                    )

                if args.write_every > 0 and fetched % args.write_every == 0:
                    write_csv(
                        args.output,
                        fieldnames,
                        rows,
                        descriptions_by_id,
                        args.link_column,
                        args.description_column,
                    )

    write_csv(args.output, fieldnames, rows, descriptions_by_id, args.link_column, args.description_column)

    found_final = sum(1 for result in descriptions_by_id.values() if result.description)
    print(f"Done. Descriptions found: {found_final:,}/{len(object_ids):,}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
