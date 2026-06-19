"""Source ingestion: parse the Razorpay ToS PDF and build a clause index."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

_NOISE_PATTERNS = [
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},?\s+\d.*Razorpay Terms & Conditions\s*$"),
    re.compile(r"^https://razorpay\.com/terms/?\s*\d*/?\d*\s*$"),
    re.compile(r"^-{2,}\s*\d+\s+of\s+\d+\s*-{2,}\s*$"),
    re.compile(r"^\s*\d+/\d+\s*$"),
]
_NOISE_SUBSTRINGS = (
    "Agentic Stack Payments Banking",
    "Login\tAgentic Stack",
)


@dataclass(frozen=True)
class SourceDocument:
    """Parsed source text plus provenance for reproducibility."""

    path: str
    filename: str
    sha256: str
    parsed_at: str
    text: str
    num_pages: int


def _clean(raw_text: str) -> str:
    out = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(p.match(stripped) for p in _NOISE_PATTERNS):
            continue
        if any(sub in stripped for sub in _NOISE_SUBSTRINGS):
            continue
        out.append(stripped)
    return "\n".join(re.sub(r"[ \t\xa0]+", " ", ln) for ln in out)


def load_source(pdf_path: str | Path) -> SourceDocument:
    """Read the PDF, returning cleaned text and provenance metadata."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Source PDF not found: {path}")

    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = _clean("\n".join(pages))

    return SourceDocument(
        path=str(path),
        filename=path.name,
        sha256=sha256,
        parsed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        text=text,
        num_pages=len(reader.pages),
    )


def save_parsed_text(doc: SourceDocument, out_dir: str | Path) -> Path:
    """Persist the cleaned text to the source cache dir for inspection/caching."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tos_parsed.txt"
    header = (
        f"# source: {doc.filename}\n"
        f"# sha256: {doc.sha256}\n"
        f"# parsed_at: {doc.parsed_at}\n"
        f"# pages: {doc.num_pages}\n\n"
    )
    out_path.write_text(header + doc.text, encoding="utf-8")
    return out_path


_SECTION_RE = re.compile(r"^(?P<title>[A-Z][A-Z &/,''()\-\.]*?)\s*(?P<num>\d{1,2})\.$")
_CLAUSE_RE = re.compile(r"^(?P<num>\d{1,2}\.\d{1,2}[A-Z]?)\s+(?P<rest>.+)$")
_PART_RE = re.compile(r"^PART\s+(?P<part>[AB])\b", re.IGNORECASE)
_DEF_RE = re.compile("^[\u201c\u201d\"](?P<term>[^\u201c\u201d\"]{2,80})[\u201c\u201d\"]\\s+(?P<rest>.+)$")


@dataclass
class Clause:
    clause_id: str
    part: str
    section: str
    section_title: str
    number: str
    title: str
    text: str


@dataclass
class ClauseIndex:
    clauses: dict[str, Clause] = field(default_factory=dict)
    definitions: dict[str, str] = field(default_factory=dict)
    source_filename: str = ""
    source_sha256: str = ""
    full_text: str = ""

    def get(self, clause_id: str) -> Clause | None:
        return self.clauses.get(clause_id)

    def exists(self, clause_id: str) -> bool:
        return clause_id in self.clauses

    def quote_contains(self, clause_id: str, quote: str) -> bool:
        """True if ``quote`` is a verbatim substring of the clause (whitespace-normalised)."""
        clause = self.clauses.get(clause_id)
        if clause is None:
            return False
        return _norm(quote) in _norm(clause.text)

    def __len__(self) -> int:
        return len(self.clauses)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _short_title(text: str) -> str:
    first = re.split(r"(?<=[.;:])\s", text.strip(), maxsplit=1)[0]
    words = first.split()
    return " ".join(words[:8]) + ("…" if len(words) > 8 else "")


def build_clause_index(doc: SourceDocument) -> ClauseIndex:
    index = ClauseIndex(
        source_filename=doc.filename,
        source_sha256=doc.sha256,
        full_text=doc.text,
    )

    part = "A"
    section_num = "0"
    section_title = "PREAMBLE"
    in_definitions = False
    seen_sections: set[str] = set()

    current: Clause | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current, buffer
        if current is not None:
            current.text = " ".join(buffer).strip()
            current.title = _short_title(current.text)
            if current.text:
                index.clauses[current.clause_id] = current
        current, buffer = None, []

    for line in doc.text.splitlines():
        line = line.strip()
        if not line:
            continue

        m_part = _PART_RE.match(line)
        if m_part:
            flush()
            part = m_part.group("part").upper()
            in_definitions = False
            continue

        if line.upper().startswith("DEFINITIONS"):
            flush()
            in_definitions = True
            continue

        section_candidate = line.lstrip(". ").strip()
        m_section = _SECTION_RE.match(section_candidate)
        if m_section and len(section_candidate) < 70:
            flush()
            in_definitions = False
            section_num = m_section.group("num")
            section_title = m_section.group("title").strip()
            seen_sections.add(section_num)
            continue

        if in_definitions:
            m_def = _DEF_RE.match(line)
            if m_def:
                index.definitions[m_def.group("term").strip()] = (
                    m_def.group("term").strip() + " " + m_def.group("rest").strip()
                )
            else:
                if index.definitions:
                    last = next(reversed(index.definitions))
                    index.definitions[last] += " " + line
            continue

        m_clause = _CLAUSE_RE.match(line)
        if m_clause:
            number = m_clause.group("num")
            sec_prefix = number.split(".")[0]
            if sec_prefix not in seen_sections and sec_prefix != section_num:
                if current is not None:
                    buffer.append(line)
                continue
            flush()
            current = Clause(
                clause_id=f"Part{part}/{number}",
                part=part,
                section=sec_prefix,
                section_title=section_title if sec_prefix == section_num else section_title,
                number=number,
                title="",
                text="",
            )
            buffer = [m_clause.group("rest").strip()]
            continue

        if current is not None:
            buffer.append(line)

    flush()
    return index


def select_quote(index: ClauseIndex, clause_id: str, hint: str | None = None) -> str:
    """Return a verbatim sentence from the clause, preferring one containing ``hint``."""
    clause = index.get(clause_id)
    if clause is None:
        return ""
    sentences = re.split(r"(?<=[.;])\s+", clause.text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 25]
    if not sentences:
        return clause.text[:200].strip()
    if hint:
        hint_l = hint.lower()
        for s in sentences:
            if hint_l in s.lower():
                return s
    return sentences[0]
