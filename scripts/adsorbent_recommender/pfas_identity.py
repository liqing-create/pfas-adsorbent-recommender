"""Bounded public identity lookup. Results are candidates, never confirmed identities."""
from __future__ import annotations
import json
import re
import threading
import time
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_request_lock = threading.Lock()
_last_request = 0.0

class IdentityLookupError(ValueError):
    pass

def _get_json(url: str) -> dict:
    global _last_request
    for attempt in range(2):
        try:
            # Application-wide serialization stays below PubChem's 5 requests/s limit.
            with _request_lock:
                time.sleep(max(0.0, 0.5 - (time.monotonic() - _last_request)))
                _last_request = time.monotonic()
                request = Request(url, headers={"Accept": "application/json", "User-Agent": "PFAS-Adsorbent-Recommender/1.0"})
                with urlopen(request, timeout=10) as response:
                    return json.loads(response.read(2_000_000))
        except HTTPError as exc:
            if exc.code == 404:
                return {}
            if exc.code in {429, 503} and attempt == 0:
                time.sleep(1)
                continue
            raise IdentityLookupError("PubChem is unavailable or rate-limited. Retry later, or paste the parent SMILES.") from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise IdentityLookupError("PubChem could not be reached or returned an unreadable response. Paste SMILES or retry later.") from exc
    return {}

@lru_cache(maxsize=256)
def _resolve(query: str, kind: str, hour: int) -> tuple[tuple, ...]:
    data = _get_json(f"{BASE_URL}/compound/name/{quote(query, safe='')}/cids/JSON?name_type=complete")
    ids = sorted(set(data.get("IdentifierList", {}).get("CID", [])))
    if len(ids) > 20:
        raise IdentityLookupError("This identifier matches more than 20 compounds. Enter a more specific name, CAS number, or SMILES.")
    if not ids:
        return ()
    identifiers = ",".join(str(int(cid)) for cid in ids)
    data = _get_json(f"{BASE_URL}/compound/cid/{identifiers}/property/Title,SMILES,InChIKey/JSON")
    rows = []
    for row in data.get("PropertyTable", {}).get("Properties", []):
        smiles = row.get("SMILES") or row.get("IsomericSMILES")
        if smiles:
            rows.append((int(row["CID"]), str(row.get("Title", row["CID"])), smiles, row.get("InChIKey", "")))
    if not rows:
        raise IdentityLookupError("PubChem returned no usable molecular structures. Paste the parent SMILES.")
    return tuple(rows)

def resolve_public_identity(query: str, kind: str) -> list[dict]:
    query = query.strip()
    if not query or len(query) > 512 or kind not in {"name", "cas"}:
        raise IdentityLookupError("Enter a name or CAS number (at most 512 characters).")
    if kind == "cas":
        if not re.fullmatch(r"\d{2,7}-\d{2}-\d", query):
            raise IdentityLookupError("Enter a CAS number in the form 335-67-1, or use Name/SMILES.")
        digits = query.replace("-", "")
        if sum((i + 1) * int(n) for i, n in enumerate(reversed(digits[:-1]))) % 10 != int(digits[-1]):
            raise IdentityLookupError("The CAS checksum is invalid. Check the number, or use Name/SMILES.")
    return [{"cid": cid, "title": title, "smiles": smiles, "inchikey": key,
             "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
             "query": query, "identity_type": kind, "source": "pubchem"}
            for cid, title, smiles, key in _resolve(query, kind, int(time.time() // 3600))]
