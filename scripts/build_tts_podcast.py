#!/usr/bin/env python3
import datetime as dt
import hashlib
import html
import json
import os
import pathlib
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# ==========
# Config
# ==========

READER_TOKEN = os.environ["READWISE_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

READER_LIST_URL = "https://readwise.io/api/v3/list/"

SITE_ROOT = pathlib.Path("docs")
AUDIO_DIR = SITE_ROOT / "audio"
STATE_PATH = pathlib.Path("state/processed.json")
PODCAST_XML_PATH = SITE_ROOT / "podcast.xml"

PODCAST_TITLE = os.environ.get("PODCAST_TITLE", "Personal TTS Podcast")
PODCAST_BASE_URL = os.environ.get("PODCAST_BASE_URL", "https://USERNAME.github.io/tts-podcast/").rstrip("/") + "/"
PODCAST_DESCRIPTION = os.environ.get(
    "PODCAST_DESCRIPTION", "Personal TTS feed generated from Readwise Reader tags."
)
PODCAST_LANGUAGE = os.environ.get("PODCAST_LANGUAGE", "en-gb")
PODCAST_AUTHOR = os.environ.get("PODCAST_AUTHOR", "Tom Selmes")
PODCAST_IMAGE_URL = os.environ.get("PODCAST_IMAGE_URL", "")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini-tts")
OPENAI_VOICE = os.environ.get("OPENAI_VOICE", "fable")
RESPONSE_FORMAT = "mp3"

MAX_CHARS_PER_CHUNK = int(os.environ.get("MAX_CHARS_PER_CHUNK", "5500"))
MIN_TEXT_CHARS = int(os.environ.get("MIN_TEXT_CHARS", "150"))
MAX_DOCS_PER_RUN = int(os.environ.get("MAX_DOCS_PER_RUN", "25"))
RETAIN_MAX_ITEMS = int(os.environ.get("RETAIN_MAX_ITEMS", "200"))

READWISE_RETRY_ATTEMPTS = int(os.environ.get("READWISE_RETRY_ATTEMPTS", "4"))
OPENAI_RETRY_ATTEMPTS = int(os.environ.get("OPENAI_RETRY_ATTEMPTS", "3"))

TAG_CONFIG = {
    "tts-en": {
        "lang": "en",
        "title_prefix": "[EN] ",
        "instructions": (
            "Read clearly and naturally in English. "
            "Use a calm, neutral speaking style. "
            "Pause slightly at headings. "
            "Read abbreviations naturally."
        ),
    },
    "tts-de": {
        "lang": "de",
        "title_prefix": "[DE] ",
        "instructions": (
            "Read clearly and naturally in German. "
            "Use standard High German pronunciation. "
            "Keep a calm, neutral speaking style. "
            "Pause slightly at headings."
        ),
    },
}


# ==========
# Helpers
# ==========


def ensure_dirs() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)



def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)



def iso_now() -> str:
    return now_utc().replace(microsecond=0).isoformat().replace("+00:00", "Z")



def iso_to_dt(value: str) -> dt.datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return dt.datetime.fromisoformat(value)



def dt_to_rfc2822(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")



def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()



def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "-", text)
    return text[:60].strip("-") or "untitled"



def build_audio_url(filename: str) -> str:
    return PODCAST_BASE_URL.rstrip("/") + "/audio/" + filename



def build_feed_url() -> str:
    return PODCAST_BASE_URL.rstrip("/") + "/podcast.xml"



def extract_doc_tags(doc: Dict[str, Any]) -> Set[str]:
    tags: Set[str] = set()
    for tag in doc.get("tags") or []:
        if isinstance(tag, dict):
            value = tag.get("name")
            if value:
                tags.add(str(value))
        else:
            tags.add(str(tag))
    return tags



def safe_unlink(path: pathlib.Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()



def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    params: Dict[str, Any],
    timeout: int,
    attempts: int,
) -> requests.Response:
    backoff = 2
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            response = session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                timeout=timeout,
            )
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(f"Request failed after {attempts} attempts: {last_error}")



def synthesize_chunk_with_retry(
    client: OpenAI,
    text: str,
    instructions: str,
    out_path: pathlib.Path,
    attempts: int,
) -> None:
    backoff = 2
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            with client.audio.speech.with_streaming_response.create(
                model=OPENAI_MODEL,
                voice=OPENAI_VOICE,
                input=text,
                instructions=instructions,
                response_format=RESPONSE_FORMAT,
            ) as response:
                response.stream_to_file(out_path)
            return
        except Exception as exc:
            last_error = exc
            safe_unlink(out_path)
            if attempt == attempts:
                break
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(f"OpenAI TTS failed after {attempts} attempts: {last_error}")



def clean_html_to_speech_text(raw_html: str, fallback_title: str, source_url: str = "") -> str:
    soup = BeautifulSoup(raw_html or "", "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = html.unescape(text)

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    junk_patterns = [
        r"^subscribe\b",
        r"^sign up\b",
        r"^log in\b",
        r"^share\b",
        r"^advertisement\b",
        r"^cookie\b",
        r"^all rights reserved\b",
    ]

    filtered: List[str] = []
    for line in lines:
        short = line.lower()
        if any(re.search(p, short) for p in junk_patterns):
            continue
        if re.fullmatch(r"https?://\S+", line):
            continue
        filtered.append(line)

    text = "\n\n".join(filtered)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    intro = [fallback_title]
    if source_url:
        intro.append(f"Source: {source_url}")
    intro_text = ". ".join(intro) + ".\n\n"

    return intro_text + text



def split_long_sentence(sentence: str, max_chars: int) -> List[str]:
    parts: List[str] = []
    cursor = 0
    while cursor < len(sentence):
        parts.append(sentence[cursor : cursor + max_chars].strip())
        cursor += max_chars
    return [part for part in parts if part]



def split_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                oversized_parts = [sentence]
                if len(sentence) > max_chars:
                    oversized_parts = split_long_sentence(sentence, max_chars)

                for part in oversized_parts:
                    if current_len + len(part) + 2 > max_chars and current:
                        chunks.append("\n\n".join(current).strip())
                        current = []
                        current_len = 0

                    current.append(part)
                    current_len += len(part) + 2
            continue

        if current_len + len(para) + 2 > max_chars and current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current).strip())

    return chunks



def file_size(path: pathlib.Path) -> int:
    return path.stat().st_size


# ==========
# State
# ==========


def empty_state() -> Dict[str, Any]:
    return {"version": 2, "docs": {}}



def normalize_state(raw: Dict[str, Any]) -> Dict[str, Any]:
    state = empty_state()

    if raw.get("version") == 2 and isinstance(raw.get("docs"), dict):
        state["docs"] = raw["docs"]
        return state

    # Migration path from v1 shape: {"processed": {doc_id: {...}}}
    processed = raw.get("processed")
    if isinstance(processed, dict):
        for doc_id, record in processed.items():
            if not isinstance(record, dict):
                continue
            state["docs"][str(doc_id)] = {
                "doc_id": str(doc_id),
                "title": record.get("title", "Untitled"),
                "queue_tag": record.get("queue_tag", ""),
                "source_url": record.get("source_url", ""),
                "fingerprint": record.get("fingerprint", ""),
                "processed_at": record.get("processed_at", iso_now()),
                "parts": record.get("parts", []),
            }

    return state



def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return empty_state()

    raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return normalize_state(raw)



def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")



def state_referenced_filenames(state: Dict[str, Any]) -> Set[str]:
    filenames: Set[str] = set()
    for doc in state["docs"].values():
        for part in doc.get("parts", []):
            filename = part.get("filename")
            if filename:
                filenames.add(filename)
    return filenames



def remove_doc_assets(doc_entry: Dict[str, Any]) -> None:
    for part in doc_entry.get("parts", []):
        filename = part.get("filename")
        if filename:
            safe_unlink(AUDIO_DIR / filename)


# ==========
# Readwise
# ==========


def reader_headers() -> Dict[str, str]:
    return {"Authorization": f"Token {READER_TOKEN}"}



def fetch_docs_by_tag(session: requests.Session, tag: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    next_page: Optional[str] = None

    while True:
        params: Dict[str, Any] = {"tag": tag, "withHtmlContent": "true"}
        if next_page:
            params["pageCursor"] = next_page

        response = request_with_retry(
            session=session,
            method="GET",
            url=READER_LIST_URL,
            headers=reader_headers(),
            params=params,
            timeout=60,
            attempts=READWISE_RETRY_ATTEMPTS,
        )
        response.raise_for_status()
        data = response.json()

        docs.extend(data.get("results", []))
        next_page = data.get("nextPageCursor")
        if not next_page:
            break

    return docs


# ==========
# Feed
# ==========


def collect_feed_items_from_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for doc in state["docs"].values():
        for part in doc.get("parts", []):
            items.append(part)

    items.sort(key=lambda x: x.get("processed_at", ""), reverse=True)
    return items



def build_feed(items: List[Dict[str, Any]]) -> None:
    ET.register_namespace("itunes", "http://www.itunes.com/dtds/podcast-1.0.dtd")
    ET.register_namespace("atom", "http://www.w3.org/2005/Atom")

    rss = ET.Element(
        "rss",
        {
            "version": "2.0",
            "xmlns:itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
            "xmlns:atom": "http://www.w3.org/2005/Atom",
        },
    )
    channel = ET.SubElement(rss, "channel")

    ET.SubElement(channel, "title").text = PODCAST_TITLE
    ET.SubElement(channel, "link").text = PODCAST_BASE_URL
    ET.SubElement(channel, "description").text = PODCAST_DESCRIPTION
    ET.SubElement(channel, "language").text = PODCAST_LANGUAGE
    ET.SubElement(channel, "lastBuildDate").text = dt_to_rfc2822(now_utc())

    ET.SubElement(
        channel,
        "{http://www.w3.org/2005/Atom}link",
        {
            "href": build_feed_url(),
            "rel": "self",
            "type": "application/rss+xml",
        },
    )

    ET.SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}author").text = PODCAST_AUTHOR
    ET.SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}summary").text = PODCAST_DESCRIPTION
    ET.SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}explicit").text = "false"

    if PODCAST_IMAGE_URL:
        ET.SubElement(
            channel,
            "{http://www.itunes.com/dtds/podcast-1.0.dtd}image",
            {"href": PODCAST_IMAGE_URL},
        )

    for item in items:
        entry = ET.SubElement(channel, "item")
        ET.SubElement(entry, "title").text = item["title"]
        ET.SubElement(entry, "description").text = item["description"]
        ET.SubElement(entry, "guid").text = item["guid"]

        processed_at = iso_to_dt(item["processed_at"])
        ET.SubElement(entry, "pubDate").text = dt_to_rfc2822(processed_at)

        enclosure = ET.SubElement(entry, "enclosure")
        enclosure.set("url", item["url"])
        enclosure.set("length", str(item["length"]))
        enclosure.set("type", "audio/mpeg")

        ET.SubElement(entry, "{http://www.itunes.com/dtds/podcast-1.0.dtd}author").text = PODCAST_AUTHOR
        ET.SubElement(entry, "{http://www.itunes.com/dtds/podcast-1.0.dtd}explicit").text = "false"

    tree = ET.ElementTree(rss)
    ET.indent(tree, space="  ", level=0)
    tree.write(PODCAST_XML_PATH, encoding="utf-8", xml_declaration=True)


# ==========
# Processing
# ==========


def choose_queue_tag(doc: Dict[str, Any], fetched_tags: Set[str]) -> Optional[str]:
    doc_tags = extract_doc_tags(doc) | fetched_tags
    has_en = "tts-en" in doc_tags
    has_de = "tts-de" in doc_tags

    if has_en and has_de:
        return None
    if has_en:
        return "tts-en"
    if has_de:
        return "tts-de"
    return None



def make_filename(doc_id: str, title: str, fingerprint: str, part_index: int, timestamp: dt.datetime) -> str:
    date_prefix = timestamp.strftime("%Y%m%d")
    fp8 = fingerprint[:8]
    slug = slugify(title)
    return f"{date_prefix}_doc{doc_id}_{fp8}_{slug}_p{part_index:02d}.mp3"



def process_doc(
    client: OpenAI,
    doc: Dict[str, Any],
    queue_tag: str,
    state: Dict[str, Any],
) -> Tuple[str, int]:
    doc_id = str(doc["id"])
    cfg = TAG_CONFIG[queue_tag]

    title = (doc.get("title") or "Untitled").strip()
    source_url = doc.get("url") or ""
    html_content = doc.get("html_content") or doc.get("htmlContent") or ""

    clean_text = clean_html_to_speech_text(html_content, fallback_title=title, source_url=source_url)
    if len(clean_text.strip()) < MIN_TEXT_CHARS:
        return "too_short", 0

    fingerprint = sha1_text(clean_text)
    existing = state["docs"].get(doc_id)
    if existing and existing.get("fingerprint") == fingerprint:
        return "unchanged", 0

    chunks = split_text(clean_text)
    if not chunks:
        return "too_short", 0

    processed_at = iso_now()
    processed_dt = iso_to_dt(processed_at)

    parts: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        part_suffix = f" (Part {idx})" if len(chunks) > 1 else ""
        episode_title = f'{cfg["title_prefix"]}{title}{part_suffix}'

        filename = make_filename(
            doc_id=doc_id,
            title=title,
            fingerprint=fingerprint,
            part_index=idx,
            timestamp=processed_dt,
        )
        out_path = AUDIO_DIR / filename

        synthesize_chunk_with_retry(
            client=client,
            text=chunk,
            instructions=cfg["instructions"],
            out_path=out_path,
            attempts=OPENAI_RETRY_ATTEMPTS,
        )

        guid = f"reader-{doc_id}-{fingerprint[:8]}-part-{idx}"
        description = source_url or f"Generated from Readwise Reader item {doc_id}"

        parts.append(
            {
                "doc_id": doc_id,
                "part": idx,
                "title": episode_title,
                "description": description,
                "guid": guid,
                "url": build_audio_url(filename),
                "length": file_size(out_path),
                "filename": filename,
                "processed_at": processed_at,
                "source_url": source_url,
                "queue_tag": queue_tag,
            }
        )

    state["docs"][doc_id] = {
        "doc_id": doc_id,
        "title": title,
        "queue_tag": queue_tag,
        "source_url": source_url,
        "fingerprint": fingerprint,
        "processed_at": processed_at,
        "parts": parts,
    }

    # Only remove prior files after successful synthesis and state update.
    if existing:
        remove_doc_assets(existing)

    return "processed", len(parts)



def enforce_retention(state: Dict[str, Any], max_items: int) -> int:
    if max_items <= 0:
        return 0

    all_parts: List[Dict[str, Any]] = collect_feed_items_from_state(state)
    if len(all_parts) <= max_items:
        return 0

    keep_guids = {item["guid"] for item in all_parts[:max_items]}
    removed_files = 0

    docs_to_delete: List[str] = []
    for doc_id, doc in state["docs"].items():
        kept_parts: List[Dict[str, Any]] = []
        for part in doc.get("parts", []):
            if part.get("guid") in keep_guids:
                kept_parts.append(part)
            else:
                filename = part.get("filename")
                if filename:
                    safe_unlink(AUDIO_DIR / filename)
                    removed_files += 1

        if kept_parts:
            doc["parts"] = kept_parts
        else:
            docs_to_delete.append(doc_id)

    for doc_id in docs_to_delete:
        del state["docs"][doc_id]

    return removed_files



def remove_orphan_audio_files(state: Dict[str, Any]) -> int:
    referenced = state_referenced_filenames(state)
    removed = 0

    for path in AUDIO_DIR.glob("*.mp3"):
        if path.name not in referenced:
            safe_unlink(path)
            removed += 1

    return removed



def main() -> None:
    ensure_dirs()
    state = load_state()
    client = OpenAI(api_key=OPENAI_API_KEY)
    session = requests.Session()

    fetched_docs: Dict[str, Dict[str, Any]] = {}
    fetched_tags_by_doc: Dict[str, Set[str]] = {}

    for queue_tag in TAG_CONFIG.keys():
        docs = fetch_docs_by_tag(session, queue_tag)
        for doc in docs:
            doc_id = str(doc.get("id"))
            if not doc_id:
                continue
            fetched_docs[doc_id] = doc
            fetched_tags_by_doc.setdefault(doc_id, set()).add(queue_tag)

    doc_ids = list(fetched_docs.keys())

    summary = {
        "processed_docs": 0,
        "processed_parts": 0,
        "unchanged": 0,
        "ambiguous": 0,
        "too_short": 0,
        "failed": 0,
        "failed_doc_ids": [],
    }

    processed_counter = 0
    for doc_id in doc_ids:
        if processed_counter >= MAX_DOCS_PER_RUN:
            break

        doc = fetched_docs[doc_id]
        queue_tag = choose_queue_tag(doc, fetched_tags_by_doc.get(doc_id, set()))
        if queue_tag is None:
            summary["ambiguous"] += 1
            continue

        try:
            status, part_count = process_doc(client, doc, queue_tag, state)
            if status == "processed":
                summary["processed_docs"] += 1
                summary["processed_parts"] += part_count
                processed_counter += 1
            elif status == "unchanged":
                summary["unchanged"] += 1
            elif status == "too_short":
                summary["too_short"] += 1
        except Exception as exc:
            summary["failed"] += 1
            summary["failed_doc_ids"].append(doc_id)
            print(f"Failed for doc {doc_id}: {exc}")

    pruned_by_retention = enforce_retention(state, RETAIN_MAX_ITEMS)
    removed_orphans = remove_orphan_audio_files(state)

    items = collect_feed_items_from_state(state)
    build_feed(items)
    save_state(state)

    print("Run summary")
    print(f"  processed docs: {summary['processed_docs']}")
    print(f"  processed parts: {summary['processed_parts']}")
    print(f"  unchanged docs: {summary['unchanged']}")
    print(f"  ambiguous docs skipped: {summary['ambiguous']}")
    print(f"  too-short docs skipped: {summary['too_short']}")
    print(f"  failed docs: {summary['failed']}")
    print(f"  retention-pruned files: {pruned_by_retention}")
    print(f"  orphan files removed: {removed_orphans}")
    if summary["failed_doc_ids"]:
        print("  failed doc ids: " + ", ".join(summary["failed_doc_ids"]))


if __name__ == "__main__":
    main()
