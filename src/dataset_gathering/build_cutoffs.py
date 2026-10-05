import json
import re
from pathlib import Path

PERSONAS_FILE = Path("data/personas_p.jsonl")
CUESTIONARIOS_DIR = Path("data/cuestionarios")
OUTPUT = Path("data/cutoffs.json")

DATE_RE = re.compile(r"<DATE>\s*([\d\-: ]+?)\s*</DATE>")

xml_by_name = {p.stem.lower(): p for p in CUESTIONARIOS_DIR.rglob("*.xml")}


def get_cutoff(username: str) -> str | None:
    xml_path = xml_by_name.get(username.lower())
    if not xml_path:
        return None
    dates = []
    content = xml_path.read_text(errors="ignore")
    for m in DATE_RE.finditer(content):
        dates.append(m.group(1).strip())
    return max(dates) if dates else None


usernames = [json.loads(l)["username"] for l in PERSONAS_FILE.read_text().splitlines() if l.strip()]

cutoffs = {}
missing = []
for username in usernames:
    cutoff = get_cutoff(username)
    if cutoff:
        cutoffs[username] = cutoff
    else:
        missing.append(username)

OUTPUT.write_text(json.dumps(cutoffs, indent=2))
print(f"Written {len(cutoffs)} cutoffs to {OUTPUT}")
if missing:
    print(f"No XML found for: {missing}")
