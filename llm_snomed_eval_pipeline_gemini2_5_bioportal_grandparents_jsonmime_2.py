#!/usr/bin/env python3
# python llm_snomed_eval_pipeline_gemini2_5_bioportal_grandparents_jsonmime_2.py --base-dir "C:/Users/esnam/OneDrive/Desktop/research_gemini" --keys-txt "C:/Users/esnam/OneDrive/Desktop/research_gemini/keys.txt" --step all

"""
LLM + SNOMED eval pipeline (4 steps), regenerated from the original notebook structure.

Edits implemented:
- LLM: Gemini 2.5 (via google-genai)
- SNOMED: BioPortal SNOMEDCT REST API (no RF2 files)
- "cousins" replaced with "grandparents"
- Keys supported via environment variables OR keys.txt

Outputs:
_outputs_llm_eval/
  run_<RUN_ID>/
    step1_llm_set1/set1_llm_output.csv
    step2_llm_set2/set2_llm_output.csv
    step3_snomed_ground_truth/snomed_ground_truth.csv
    step4_comparison/comparison_results.csv (+ .tsv, _long.csv, _wide.csv)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import quote

import requests
import pandas as pd

# ---------------------------
# Key loading (env or keys.txt)
# ---------------------------

def load_keys_from_txt(path: Path) -> Dict[str, str]:
    keys: Dict[str, str] = {}
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        keys[k.strip()] = v.strip()
    return keys


def get_key(name: str, keys_txt: Optional[Path]) -> Optional[str]:
    v = os.environ.get(name)
    if v:
        return v
    if keys_txt:
        d = load_keys_from_txt(keys_txt)
        return d.get(name)
    return None


# ---------------------------
# Helpers
# ---------------------------

def normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", (term or "").strip().lower())


def safe_json_extract(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object found in a model response.
    Accepts responses that may include extra text; grabs first {...} block.
    """
    m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model response.")
    return json.loads(m.group(0))


def list_to_pipe(items: Iterable[str]) -> str:
    cleaned = []
    for x in items or []:
        if x is None:
            continue
        s = str(x).replace("|", " ").strip()
        if s:
            cleaned.append(s)
    return "|".join(cleaned)


def split_pipe(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split("|") if x.strip()]


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


# ---------------------------
# BioPortal SNOMEDCT client
# ---------------------------

BIOPORTAL_BASE = "https://data.bioontology.org"


@dataclass
class BioPortalClient:
    apikey: str
    timeout: int = 30

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = dict(params or {})
        params["apikey"] = self.apikey
        url = f"{BIOPORTAL_BASE}{path}"
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def follow(self, full_url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = dict(params or {})
        params["apikey"] = self.apikey
        r = requests.get(full_url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def encode_snomed_iri(concept_id: str) -> str:
        iri = f"http://purl.bioontology.org/ontology/SNOMEDCT/{concept_id}"
        return quote(iri, safe="")

    def search_snomed(self, q: str, page: int = 1, pagesize: int = 10) -> Dict[str, Any]:
        return self.get("/search", {
            "q": q,
            "ontologies": "SNOMEDCT",
            "page": page,
            "pagesize": pagesize,
        })

    def get_class_by_concept_id(self, concept_id: str, display: str = "all") -> Dict[str, Any]:
        enc = self.encode_snomed_iri(concept_id)
        return self.get(f"/ontologies/SNOMEDCT/classes/{enc}", {"display": display})

    @staticmethod
    def _extract_concept_id_from_iri(iri: str) -> Optional[str]:
        """
        Extract the SNOMED concept ID from a purl IRI like:
        http://purl.bioontology.org/ontology/SNOMEDCT/50043002
        Returns '50043002' or None if not parseable.
        """
        if not iri:
            return None
        seg = iri.rsplit("/", 1)[-1]
        return seg if seg.isdigit() else None

    def _collect_items(self, response: Any) -> List[Dict[str, Any]]:
        """
        BioPortal sometimes returns {"collection": [...]} and sometimes
        returns a plain list [...]. Normalise both into a plain list.
        """
        if isinstance(response, list):
            return response
        if isinstance(response, dict):
            return response.get("collection", []) or []
        return []

    def get_class_by_concept_id_safe(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a SNOMED class using the proper BioPortal data.bioontology.org
        API (never the purl.bioontology.org URL, which returns 403).
        Returns None on any error.
        """
        try:
            return self.get_class_by_concept_id(concept_id, display="all")
        except Exception:
            return None

    def get_sons_and_grandparents(self, concept_id: str) -> Dict[str, Any]:
        """
        sons        = direct is-a children of the concept
        grandparents = parents of the concept's direct parents (two levels up)

        Always uses the BioPortal data.bioontology.org API.
        Never follows raw purl.bioontology.org URLs directly (those return 403).
        Instead, extracts the SNOMED concept ID from the IRI and re-fetches
        via the proper API endpoint.
        """
        cls = self.get_class_by_concept_id(concept_id, display="all")
        links = cls.get("links", {}) or {}

        # ── sons: follow the children link ────────────────────────────────
        sons: List[str] = []
        if links.get("children"):
            try:
                children_resp = self.follow(links["children"])
                for item in self._collect_items(children_resp):
                    lab = item.get("prefLabel") or item.get("label") or ""
                    if lab:
                        sons.append(lab)
            except Exception:
                pass  # log nothing; sons stays empty

        # ── grandparents: two hops up via concept IDs (not purl URLs) ────
        grandparents: List[str] = []
        gp_concept_ids: Set[str] = set()

        # Step 1: get the direct parents of this concept
        if links.get("parents"):
            try:
                parents_resp = self.follow(links["parents"])
                parent_items = self._collect_items(parents_resp)

                # Step 2: for each parent, fetch its parents via the proper API
                for p in parent_items:
                    # Extract SNOMED concept ID from the IRI in @id
                    parent_iri = p.get("@id") or ""
                    parent_cid = self._extract_concept_id_from_iri(parent_iri)
                    if not parent_cid:
                        continue

                    # Fetch the parent using the safe BioPortal API (no purl)
                    parent_obj = self.get_class_by_concept_id_safe(parent_cid)
                    if not parent_obj:
                        continue

                    parent_links = parent_obj.get("links", {}) or {}
                    if not parent_links.get("parents"):
                        continue

                    # Step 3: collect grandparent concept IDs
                    try:
                        gps_resp = self.follow(parent_links["parents"])
                        for g in self._collect_items(gps_resp):
                            gp_iri = g.get("@id") or ""
                            gp_cid = self._extract_concept_id_from_iri(gp_iri)
                            if gp_cid:
                                gp_concept_ids.add(gp_cid)
                    except Exception:
                        continue

            except Exception:
                pass

        # Step 4: fetch each grandparent's label via the proper API (no purl)
        gp_concept_ids.discard(concept_id)  # safety: remove self
        for gp_cid in gp_concept_ids:
            gp_obj = self.get_class_by_concept_id_safe(gp_cid)
            if not gp_obj:
                continue
            lab = gp_obj.get("prefLabel") or gp_obj.get("label") or ""
            if lab:
                grandparents.append(lab)
            time.sleep(0.1)  # be gentle with the API during grandparent fetching

        return {
            "concept_id": concept_id,
            "prefLabel": cls.get("prefLabel") or "",
            "sons": sons,
            "grandparents": grandparents,
        }


def extract_concept_id_from_hit(hit: Dict[str, Any]) -> Optional[str]:
    iri = hit.get("@id") or hit.get("id") or ""
    if not iri:
        return None
    # concept id is last segment
    return iri.rsplit("/", 1)[-1] if "/" in iri else None


def pick_best_hit(search_json: Dict[str, Any], preferred_label: Optional[str] = None) -> Optional[Dict[str, Any]]:
    hits = search_json.get("collection", []) or []
    if not hits:
        return None

    if preferred_label:
        pref_norm = normalize_term(preferred_label)
        # Try exact prefLabel match first
        for h in hits:
            if normalize_term(h.get("prefLabel") or "") == pref_norm:
                return h

    return hits[0]


# ---------------------------
# Gemini 2.5 client
# ---------------------------

def build_gemini_client(gemini_api_key: str):
    # Lazy import so the script can still run Step 3/4 without gemini installed.
    from google import genai
    return genai.Client(api_key=gemini_api_key)


def gemini_generate_json(client, model: str, prompt: str) -> Tuple[str, Dict[str, Any]]:
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    text = getattr(resp, "text", "") or ""
    data = safe_json_extract(text)
    return text, data


def make_prompt_set1(concept_term: str) -> str:
    return f"""
Return STRICT JSON only (no markdown, no prose) for the clinical concept:

CONCEPT: {concept_term}

Schema:
{{
  "sons": ["..."],
  "grandparents": ["..."]
}}

Definitions:
- "sons": direct child concepts (more specific is-a children)
- "grandparents": parents of the concept's parent(s) (two levels up in is-a)

Rules:
- Be concise and clinically reasonable.
- Provide 5–20 items if possible.
""".strip()


def make_prompt_set2(concept_term: str) -> str:
    return f"""
Return STRICT JSON only (no markdown, no prose) for the clinical concept:

CONCEPT: {concept_term}

Schema:
{{
  "sons": ["..."],
  "grandparents": ["..."]
}}

Definitions:
- "sons": direct child concepts (more specific is-a children)
- "grandparents": parents of the concept's parent(s) (two levels up in is-a)

Constraints:
- Up to 30 items per list.
- Prefer SNOMED-like clinically common phrasing.
- Avoid overly generic items.
""".strip()


# ---------------------------
# Step 0: Config / paths
# ---------------------------

@dataclass
class PipelinePaths:
    base_dir: Path
    run_id: str
    run_dir: Path
    step1_dir: Path
    step2_dir: Path
    step3_dir: Path
    step4_dir: Path
    step1_out: Path
    step2_out: Path
    step3_out: Path
    step4_out_main: Path
    step4_out_tsv: Path
    step4_out_long: Path
    step4_out_wide: Path


def init_paths(base_dir: Path, run_id: Optional[str]) -> PipelinePaths:
    base_dir = base_dir.expanduser().resolve()
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_root = base_dir / "_outputs_llm_eval"
    run_dir = pipeline_root / f"run_{run_id}"

    step1_dir = run_dir / "step1_llm_set1"
    step2_dir = run_dir / "step2_llm_set2"
    step3_dir = run_dir / "step3_snomed_ground_truth"
    step4_dir = run_dir / "step4_comparison"

    for d in [step1_dir, step2_dir, step3_dir, step4_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return PipelinePaths(
        base_dir=base_dir,
        run_id=run_id,
        run_dir=run_dir,
        step1_dir=step1_dir,
        step2_dir=step2_dir,
        step3_dir=step3_dir,
        step4_dir=step4_dir,
        step1_out=step1_dir / "set1_llm_output.csv",
        step2_out=step2_dir / "set2_llm_output.csv",
        step3_out=step3_dir / "snomed_ground_truth.csv",
        step4_out_main=step4_dir / "comparison_results.csv",
        step4_out_tsv=step4_dir / "comparison_results.tsv",
        step4_out_long=step4_dir / "comparison_results_long.csv",
        step4_out_wide=step4_dir / "comparison_results_wide.csv",
    )


# ---------------------------
# Step 1 & 2: LLM calls (Gemini)
# ---------------------------

def run_llm_step(
    *,
    paths: PipelinePaths,
    concept_terms: List[str],
    prompt_set: str,
    model_name: str,
    gemini_api_key: str,
    out_csv: Path,
    log_path: Path,
) -> None:
    client = build_gemini_client(gemini_api_key)

    # Resume logic
    done_terms: Set[str] = set()
    existing_df: Optional[pd.DataFrame] = None
    if out_csv.exists():
        existing_df = pd.read_csv(out_csv, dtype=str).fillna("")
        if "concept_term" in existing_df.columns:
            done_terms = set(existing_df["concept_term"].astype(str).tolist())

    rows: List[Dict[str, Any]] = []

    for concept_term in concept_terms:
        if concept_term in done_terms:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as logf:
                logf.write(f"{datetime.now().isoformat()}\t{concept_term}\tSKIP (already processed)\n")
            continue

        prompt = make_prompt_set1(concept_term) if prompt_set == "set1" else make_prompt_set2(concept_term)

        try:
            raw_text, parsed = gemini_generate_json(client, model=model_name, prompt=prompt)
            sons = parsed.get("sons", []) or []
            grandparents = parsed.get("grandparents", []) or []

            row = {
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "prompt_set": prompt_set,
                "concept_term": concept_term,
                "raw_output": (raw_text or "").replace("\n", " ").strip(),
                "extracted_sons": list_to_pipe(sons),
                "extracted_grandparents": list_to_pipe(grandparents),
            }
            rows.append(row)

            with log_path.open("a", encoding="utf-8") as logf:
                logf.write(f"{datetime.now().isoformat()}\t{concept_term}\tOK\n")

        except Exception as e:
            with log_path.open("a", encoding="utf-8") as logf:
                logf.write(f"{datetime.now().isoformat()}\t{concept_term}\tERROR\t{e}\n")

        # gentle pacing
        time.sleep(0.2)

    if rows:
        out_df = pd.DataFrame(rows)
        if existing_df is not None and not existing_df.empty:
            combined = pd.concat([existing_df, out_df], ignore_index=True)
        elif out_csv.exists():
            # output exists but couldn't read earlier for some reason
            prior = pd.read_csv(out_csv, dtype=str).fillna("")
            combined = pd.concat([prior, out_df], ignore_index=True)
        else:
            combined = out_df
        combined.to_csv(out_csv, index=False)


# ---------------------------
# Step 3: BioPortal ground truth
# ---------------------------

def run_ground_truth_step(
    *,
    paths: PipelinePaths,
    concept_terms: List[str],
    bioportal_api_key: str,
) -> None:
    bp = BioPortalClient(apikey=bioportal_api_key)

    rows: List[Dict[str, Any]] = []

    for concept_term in concept_terms:
        note = ""
        try:
            sr = bp.search_snomed(concept_term, page=1, pagesize=10)
            hit = pick_best_hit(sr, preferred_label=concept_term)
            if not hit:
                rows.append({
                    "concept_term": concept_term,
                    "concept_id": "",
                    "prefLabel": "",
                    "sons_terms": "",
                    "sons_count": 0,
                    "grandparents_terms": "",
                    "grandparents_count": 0,
                    "note": "No BioPortal SNOMEDCT search hit",
                })
                continue

            concept_id = extract_concept_id_from_hit(hit) or ""
            if not concept_id:
                rows.append({
                    "concept_term": concept_term,
                    "concept_id": "",
                    "prefLabel": "",
                    "sons_terms": "",
                    "sons_count": 0,
                    "grandparents_terms": "",
                    "grandparents_count": 0,
                    "note": "Search hit missing concept id (@id)",
                })
                continue

            truth = bp.get_sons_and_grandparents(concept_id)

            sons_terms = truth.get("sons", []) or []
            gp_terms = truth.get("grandparents", []) or []

            rows.append({
                "concept_term": concept_term,
                "concept_id": concept_id,
                "prefLabel": truth.get("prefLabel") or "",
                "sons_terms": list_to_pipe(sons_terms),
                "sons_count": len(sons_terms),
                "grandparents_terms": list_to_pipe(gp_terms),
                "grandparents_count": len(gp_terms),
                "note": note,
            })

        except Exception as e:
            rows.append({
                "concept_term": concept_term,
                "concept_id": "",
                "prefLabel": "",
                "sons_terms": "",
                "sons_count": 0,
                "grandparents_terms": "",
                "grandparents_count": 0,
                "note": f"ERROR: {e}",
            })

        time.sleep(0.1)

    pd.DataFrame(rows).to_csv(paths.step3_out, index=False)


# ---------------------------
# Step 4: Comparison
# ---------------------------

def compare_one(llm_row: pd.Series, truth_row: pd.Series) -> Dict[str, Any]:
    concept_term = str(llm_row.get("concept_term", ""))

    llm_sons = split_pipe(str(llm_row.get("extracted_sons", "")))
    llm_gp = split_pipe(str(llm_row.get("extracted_grandparents", "")))

    truth_sons = split_pipe(str(truth_row.get("sons_terms", "")))
    truth_gp = split_pipe(str(truth_row.get("grandparents_terms", "")))

    llm_sons_norm = {normalize_term(x) for x in llm_sons}
    llm_gp_norm = {normalize_term(x) for x in llm_gp}

    truth_sons_norm = {normalize_term(x) for x in truth_sons}
    truth_gp_norm = {normalize_term(x) for x in truth_gp}

    sons_inter = sorted(truth_sons_norm & llm_sons_norm)
    sons_missed = sorted(truth_sons_norm - llm_sons_norm)
    sons_extra = sorted(llm_sons_norm - truth_sons_norm)

    gp_inter = sorted(truth_gp_norm & llm_gp_norm)
    gp_missed = sorted(truth_gp_norm - llm_gp_norm)
    gp_extra = sorted(llm_gp_norm - truth_gp_norm)

    sons_correct = len(sons_inter)
    sons_truth = len(truth_sons_norm)
    sons_llm = len(llm_sons_norm)

    gp_correct = len(gp_inter)
    gp_truth = len(truth_gp_norm)
    gp_llm = len(llm_gp_norm)

    return {
        "timestamp": llm_row.get("timestamp", ""),
        "model": llm_row.get("model", ""),
        "prompt_set": llm_row.get("prompt_set", ""),
        "concept_term": concept_term,

        "truth_concept_id": truth_row.get("concept_id", ""),
        "truth_prefLabel": truth_row.get("prefLabel", ""),
        "truth_note": truth_row.get("note", ""),

        "sons_truth_count": sons_truth,
        "sons_llm_count": sons_llm,
        "sons_correct_count": sons_correct,
        "sons_recall": safe_div(sons_correct, sons_truth),
        "sons_precision": safe_div(sons_correct, sons_llm),
        "sons_intersection": list_to_pipe(sons_inter),
        "sons_missed": list_to_pipe(sons_missed),
        "sons_extra": list_to_pipe(sons_extra),

        "grandparents_truth_count": gp_truth,
        "grandparents_llm_count": gp_llm,
        "grandparents_correct_count": gp_correct,
        "grandparents_recall": safe_div(gp_correct, gp_truth),
        "grandparents_precision": safe_div(gp_correct, gp_llm),
        "grandparents_intersection": list_to_pipe(gp_inter),
        "grandparents_missed": list_to_pipe(gp_missed),
        "grandparents_extra": list_to_pipe(gp_extra),
    }


def run_comparison_step(paths: PipelinePaths, concept_terms: List[str]) -> None:
    if not paths.step1_out.exists():
        raise FileNotFoundError(f"Missing {paths.step1_out} (run step1 first).")
    if not paths.step2_out.exists():
        raise FileNotFoundError(f"Missing {paths.step2_out} (run step2 first).")
    if not paths.step3_out.exists():
        raise FileNotFoundError(f"Missing {paths.step3_out} (run step3 first).")

    set1 = pd.read_csv(paths.step1_out, dtype=str).fillna("")
    set2 = pd.read_csv(paths.step2_out, dtype=str).fillna("")
    truth = pd.read_csv(paths.step3_out, dtype=str).fillna("")

    truth_map: Dict[str, pd.Series] = {}
    for _, r in truth.iterrows():
        truth_map[str(r.get("concept_term", ""))] = r

    results: List[Dict[str, Any]] = []

    for df in [set1, set2]:
        for _, row in df.iterrows():
            ct = str(row.get("concept_term", ""))
            trow = truth_map.get(ct)
            if trow is None:
                # still emit a row noting missing truth
                results.append({
                    "timestamp": row.get("timestamp", ""),
                    "model": row.get("model", ""),
                    "prompt_set": row.get("prompt_set", ""),
                    "concept_term": ct,
                    "truth_note": "No ground truth row found for concept_term",
                })
                continue
            results.append(compare_one(row, trow))

    comp_df = pd.DataFrame(results)
    comp_df.to_csv(paths.step4_out_main, index=False)
    comp_df.to_csv(paths.step4_out_tsv, index=False, sep="\t")

    # long format
    long_rows: List[Dict[str, Any]] = []
    for _, r in comp_df.iterrows():
        for rel in ["sons", "grandparents"]:
            for status in ["intersection", "missed", "extra"]:
                col = f"{rel}_{status}"
                items = split_pipe(str(r.get(col, "")))
                for item in items:
                    long_rows.append({
                        "concept_term": r.get("concept_term", ""),
                        "prompt_set": r.get("prompt_set", ""),
                        "relation": rel,
                        "status": status,
                        "item": item,
                    })
    pd.DataFrame(long_rows).to_csv(paths.step4_out_long, index=False)

    # wide format (excel-friendly)
    wide_rows: List[Dict[str, Any]] = []
    list_cols = [
        "sons_intersection", "sons_missed", "sons_extra",
        "grandparents_intersection", "grandparents_missed", "grandparents_extra",
    ]
    for _, r in comp_df.iterrows():
        base = {k: r.get(k, "") for k in comp_df.columns if k not in list_cols}
        # expand list cols
        for col in list_cols:
            items = split_pipe(str(r.get(col, "")))
            for i, item in enumerate(items, start=1):
                base[f"{col}_{i}"] = item
        wide_rows.append(base)
    pd.DataFrame(wide_rows).to_csv(paths.step4_out_wide, index=False)


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4-step LLM+SNOMED evaluation pipeline (Gemini 2.5 + BioPortal)")
    p.add_argument("--base-dir", required=True, help="Base directory for outputs (e.g., project folder).")
    p.add_argument("--run-id", default=None, help="Run ID to reuse (e.g., 20260219_162401). If omitted, a new one is created.")
    p.add_argument("--keys-txt", default=None, help="Optional path to keys.txt (fallback if env vars not set).")
    p.add_argument("--concepts", default=None, help="Optional path to a text file with one concept term per line. If omitted, uses a small demo list.")
    p.add_argument("--gemini-model", default="gemini-2.5-pro", help="Gemini model name (e.g., gemini-2.5-pro or gemini-2.5-flash).")
    p.add_argument("--step", choices=["0", "1", "2", "3", "4", "all"], default="all", help="Which step(s) to run.")
    return p.parse_args()


def load_concepts(path: Optional[str]) -> List[str]:
    if not path:
        return ["Asthma", "Type 2 diabetes mellitus", "Hypertensive disorder"]
    p = Path(path).expanduser().resolve()
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir)
    paths = init_paths(base_dir=base_dir, run_id=args.run_id)

    keys_txt = Path(args.keys_txt).expanduser().resolve() if args.keys_txt else None
    bioportal_key = get_key("BIOPORTAL_API_KEY", keys_txt)
    gemini_key = get_key("GEMINI_API_KEY", keys_txt)

    concept_terms = load_concepts(args.concepts)

    print(f"Run directory: {paths.run_dir}")
    print(f"Concept count: {len(concept_terms)}")

    if args.step in ("0", "all"):
        # Step 0 is path initialization; already done
        print("Step 0 complete (paths initialized).")

    if args.step in ("1", "all"):
        if not gemini_key:
            raise RuntimeError("Missing GEMINI_API_KEY (env var or keys.txt).")
        run_llm_step(
            paths=paths,
            concept_terms=concept_terms,
            prompt_set="set1",
            model_name=args.gemini_model,
            gemini_api_key=gemini_key,
            out_csv=paths.step1_out,
            log_path=paths.step1_dir / "logs.txt",
        )
        print(f"Step 1 complete. Output: {paths.step1_out}")

    if args.step in ("2", "all"):
        if not gemini_key:
            raise RuntimeError("Missing GEMINI_API_KEY (env var or keys.txt).")
        run_llm_step(
            paths=paths,
            concept_terms=concept_terms,
            prompt_set="set2",
            model_name=args.gemini_model,
            gemini_api_key=gemini_key,
            out_csv=paths.step2_out,
            log_path=paths.step2_dir / "logs.txt",
        )
        print(f"Step 2 complete. Output: {paths.step2_out}")

    if args.step in ("3", "all"):
        if not bioportal_key:
            raise RuntimeError("Missing BIOPORTAL_API_KEY (env var or keys.txt).")
        run_ground_truth_step(
            paths=paths,
            concept_terms=concept_terms,
            bioportal_api_key=bioportal_key,
        )
        print(f"Step 3 complete. Output: {paths.step3_out}")

    if args.step in ("4", "all"):
        run_comparison_step(paths=paths, concept_terms=concept_terms)
        print("Step 4 complete.")
        print("Comparison outputs:")
        print(f"  {paths.step4_out_main}")
        print(f"  {paths.step4_out_tsv}")
        print(f"  {paths.step4_out_long}")
        print(f"  {paths.step4_out_wide}")

    print("Done.")


if __name__ == "__main__":
    main()
