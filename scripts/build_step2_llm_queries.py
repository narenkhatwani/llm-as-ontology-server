#!/usr/bin/env python3
"""
Build step2_llm_queries.ipynb (combined Set 1 + Set 2) for each testing_* folder.
Uses the correct LLM client and API call pattern per folder.

Step 2 reads validated concepts from Step 1 output (validated_concepts.csv),
so it only queries the LLM for concepts that have ground truth.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Config: finds shared ground truth + creates per-LLM run directory
CONFIG_SOURCE = [
    "import os\n",
    "import re\n",
    "import time\n",
    "from datetime import datetime\n",
    "from pathlib import Path\n",
    "\n",
    "import pandas as pd\n",
    "\n",
    "# ============================================================\n",
    "# Configuration\n",
    "# ============================================================\n",
    "\n",
    "# --- Detect LLM folder and paths ---\n",
    "_cwd = Path(\".\").resolve()\n",
    "if _cwd.name.startswith(\"testing_\"):\n",
    "    REPO_ROOT = _cwd.parent\n",
    "    LLM_NAME = _cwd.name.replace(\"testing_\", \"\")\n",
    "else:\n",
    "    REPO_ROOT = _cwd\n",
    "    LLM_NAME = \"gpt\"\n",
    "\n",
    "OUTPUT_ROOT = (REPO_ROOT / \"output\").resolve()\n",
    "PIPELINE_ROOT = OUTPUT_ROOT / LLM_NAME\n",
    "OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\n",
    "PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "# Shared ground truth (from step1_ground_truth.ipynb in ground_truth/ folder)\n",
    "GT_ROOT = OUTPUT_ROOT / \"ground_truth\"\n",
    "\n",
    "# Create NEW run directory for this LLM\n",
    "existing_runs = [d for d in PIPELINE_ROOT.iterdir() if d.is_dir() and d.name.startswith(\"run_\")]\n",
    "RUN_ID = len(existing_runs) + 1\n",
    "RUN_DIR = PIPELINE_ROOT / f\"run_{RUN_ID:03d}\"\n",
    "\n",
    "# Step directories\n",
    "SET1_DIR = RUN_DIR / \"step2_llm_set1\"\n",
    "SET2_DIR = RUN_DIR / \"step2_llm_set2\"\n",
    "\n",
    "# Output files\n",
    "SET1_OUT = SET1_DIR / \"set1_llm_output.csv\"\n",
    "SET2_OUT = SET2_DIR / \"set2_llm_output.csv\"\n",
    "\n",
    "# Create directories\n",
    "for d in [SET1_DIR, SET2_DIR]:\n",
    "    d.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "print(\"RUN_DIR:\", RUN_DIR)\n",
    "print(\"GT_ROOT:\", GT_ROOT)\n",
]

# Concept loading: read validated concepts from shared ground truth
CONCEPT_LOAD_SOURCE = [
    "# ============================================================\n",
    "# Load validated concepts from shared Ground Truth (Step 1)\n",
    "# ============================================================\n",
    "\n",
    "VALIDATED_CONCEPTS_PATH = GT_ROOT / \"validated_concepts.csv\"\n",
    "if not VALIDATED_CONCEPTS_PATH.exists():\n",
    "    raise FileNotFoundError(\n",
    "        f\"Validated concepts not found at {VALIDATED_CONCEPTS_PATH}\\n\"\n",
    "        f\"Run step1_ground_truth.ipynb in the ground_truth/ folder first!\"\n",
    "    )\n",
    "\n",
    "val_df = pd.read_csv(VALIDATED_CONCEPTS_PATH, dtype=str).fillna(\"\")\n",
    "# Only concepts with ground truth (non-empty snomed_id); step1 may include not_found rows for resume\n",
    "val_df = val_df[val_df[\"snomed_id\"].str.strip() != \"\"]\n",
    "CONCEPT_TERMS = val_df[\"concept_term\"].tolist()\n",
    "\n",
    "# Show replacement summary\n",
    "replaced = val_df[val_df[\"status\"] == \"replaced\"]\n",
    "if not replaced.empty:\n",
    "    print(\"Replaced concepts (original -> replacement):\")\n",
    "    for _, row in replaced.iterrows():\n",
    "        print(f\"  {row['original_term']} -> {row['concept_term']}\")\n",
    "\n",
    "print(f\"\\nTotal concepts to query: {len(CONCEPT_TERMS)}\")\n",
]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.split("\n") if isinstance(s, str) else s}

def code(s):
    lines = s if isinstance(s, list) else [s]
    return {"cell_type": "code", "metadata": {}, "outputs": [], "source": lines}

# Client and API call snippets per LLM
CLIENTS = {
    "gpt": {
        "import": "from openai import OpenAI\n",
        "init": (
            "client = OpenAI(api_key=os.environ.get(\"OPENAI_API_KEY\"))\n"
            "if not client.api_key:\n"
            "    raise EnvironmentError(\"OPENAI_API_KEY is not set.\")\n"
            "\n"
            "print(\"OpenAI client initialized.\")\n"
        ),
        "api_call_set1": (
            "        response = client.chat.completions.create(\n"
            "            model=MODEL_NAME,\n"
            "            messages=[{\"role\": \"user\", \"content\": prompt}],\n"
            "            temperature=0.5,\n"
            "        )\n"
            "        raw_output = response.choices[0].message.content or \"\"\n"
        ),
        "api_call_set2": (
            "        response = client.chat.completions.create(\n"
            "            model=MODEL_NAME,\n"
            "            messages=[{\"role\": \"user\", \"content\": prompt}],\n"
            "            temperature=0.5,\n"
            "        )\n"
            "        raw_output = response.choices[0].message.content or \"\"\n"
        ),
    },
    "claude": {
        "import": "from anthropic import Anthropic\n",
        "init": (
            "client = Anthropic(api_key=os.environ.get(\"ANTHROPIC_API_KEY\"))\n"
            "if not os.environ.get(\"ANTHROPIC_API_KEY\"):\n"
            "    raise EnvironmentError(\"ANTHROPIC_API_KEY is not set.\")\n"
            "\n"
            "print(\"Anthropic (Claude) client initialized.\")\n"
        ),
        "api_call_set1": (
            "        response = client.messages.create(\n"
            "            model=MODEL_NAME,\n"
            "            max_tokens=4096,\n"
            "            messages=[{\"role\": \"user\", \"content\": prompt}],\n"
            "        )\n"
            "        raw_output = response.content[0].text if response.content else \"\"\n"
        ),
        "api_call_set2": (
            "        response = client.messages.create(\n"
            "            model=MODEL_NAME,\n"
            "            max_tokens=4096,\n"
            "            messages=[{\"role\": \"user\", \"content\": prompt}],\n"
            "        )\n"
            "        raw_output = response.content[0].text if response.content else \"\"\n"
        ),
    },
    "gemini": {
        "import": "import google.generativeai as genai\n",
        "init": (
            "genai.configure(api_key=os.environ.get(\"GOOGLE_API_KEY\"))\n"
            "if not os.environ.get(\"GOOGLE_API_KEY\"):\n"
            "    raise EnvironmentError(\"GOOGLE_API_KEY is not set.\")\n"
            "model = genai.GenerativeModel(MODEL_NAME)\n"
            "\n"
            "print(\"Google Gemini client initialized.\")\n"
        ),
        "api_call_set1": (
            "        response = model.generate_content(prompt)\n"
            "        raw_output = response.text if response.text else \"\"\n"
        ),
        "api_call_set2": (
            "        response = model.generate_content(prompt)\n"
            "        raw_output = response.text if response.text else \"\"\n"
        ),
    },
    "deepseek": {
        "import": "from openai import OpenAI\n",
        "init": (
            "client = OpenAI(\n"
            "    api_key=os.environ.get(\"DEEPSEEK_API_KEY\"),\n"
            "    base_url=\"https://api.deepseek.com\",\n"
            ")\n"
            "if not client.api_key:\n"
            "    raise EnvironmentError(\"DEEPSEEK_API_KEY is not set.\")\n"
            "\n"
            "print(\"DeepSeek client initialized (OpenAI-compatible API).\")\n"
        ),
        "api_call_set1": (
            "        response = client.chat.completions.create(\n"
            "            model=MODEL_NAME,\n"
            "            messages=[{\"role\": \"user\", \"content\": prompt}],\n"
            "            temperature=0.5,\n"
            "        )\n"
            "        raw_output = response.choices[0].message.content or \"\"\n"
        ),
        "api_call_set2": (
            "        response = client.chat.completions.create(\n"
            "            model=MODEL_NAME,\n"
            "            messages=[{\"role\": \"user\", \"content\": prompt}],\n"
            "            temperature=0.5,\n"
            "        )\n"
            "        raw_output = response.choices[0].message.content or \"\"\n"
        ),
    },
}


def build_cells(llm_key):
    api = CLIENTS[llm_key]
    cells = []

    # Title
    cells.append(md(
        "# Step 2: LLM Queries – Set 1 (A1–A7) and Set 2 (B1–B7)\n\n"
        "This notebook runs **both** prompt sets in one place:\n"
        "- **Set 1**: A1 (FSN), A2 (semantic tag), A3 (definition status), A4–A7 (parents, grandparents, children, siblings).\n"
        "- **Set 2**: B1 (official name), B2 (kind), B3 (category), B4–B7 (broader, grandparents, narrower, peers).\n\n"
        "**Prerequisites**: Run `step1_ground_truth.ipynb` in the `ground_truth/` folder first.\n\n"
        "Run **Step 3** (`step3_accuracy.ipynb`) after this notebook."
    ))
    cells.append(md("## Configuration"))
    cells.append(code(CONFIG_SOURCE))
    cells.append(md("## Load Validated Concepts from Shared Ground Truth"))
    cells.append(code(CONCEPT_LOAD_SOURCE))

    # Set 1 prompt
    cells.append(md("## Set 1 – Prompt and Helpers"))
    cells.append(code([
        "assert SET1_DIR.exists(), \"SET1_DIR missing.\"\n",
        "LOG_PATH_SET1 = SET1_DIR / \"logs.txt\"\n",
        "MODEL_NAME = \"gpt-4o\"  # or gpt-4o-mini, claude-3-5-sonnet, gemini-1.5-pro, deepseek-chat, etc.\n",
        "PROMPT_TEMPLATE_SET1 = \"\"\"\n",
        "You are acting as a SNOMED CT ontology browser.\n",
        "\n",
        "Given the concept: \"{CONCEPT_TERM}\"\n",
        "\n",
        "Return ONLY the following fields.\n",
        "Use ONLY \"is-a\" taxonomic relationships.\n",
        "Do NOT explain anything.\n",
        "\n",
        "A1) FSN-style name (include semantic tag)\n",
        "A2) Semantic tag\n",
        "A3) Definition status (Primitive / Fully defined)\n",
        "A4) Immediate parent concept(s) (depth -1)\n",
        "A5) Grandparent concept(s) (depth -2, parents of parents)\n",
        "A6) Immediate child concept(s) (depth +1)\n",
        "A7) Near siblings (same parent)\n",
        "\n",
        "Rules:\n",
        "- Bullet lists for A4–A7\n",
        "- Exact labels A1–A7\n",
        "- No extra text\n",
        "\"\"\".strip()\n",
    ]))

    # Helpers (A + B)
    # We write the correct helper code directly
    cells.append(code([
        "def _csv_safe(x):\n",
        "    if x is None:\n",
        "        return \"\"\n",
        "    return str(x).replace(\"\\r\", \" \").replace(\"\\n\", \" \").strip()\n",
        "\n",
        "A_LABELS = [\"A1\", \"A2\", \"A3\", \"A4\", \"A5\", \"A6\", \"A7\"]\n",
        "\n",
        "def parse_A1_A7(raw: str) -> dict:\n",
        "    text = (raw or \"\").replace(\"\\r\\n\", \"\\n\").replace(\"\\r\", \"\\n\").strip()\n",
        "    out = {k: \"\" for k in A_LABELS}\n",
        "    label_re = re.compile(r\"(?m)^\\s*(A[1-7])\\)\\s*(.*)$\")\n",
        "    matches = list(label_re.finditer(text))\n",
        "    if not matches:\n",
        "        return out\n",
        "    idx = {m.group(1): {\"start\": m.start(), \"after\": m.group(2).strip()} for m in matches}\n",
        "    def section(label):\n",
        "        if label not in idx:\n",
        "            return \"\"\n",
        "        start = idx[label][\"start\"]\n",
        "        ends = [idx[k][\"start\"] for k in idx if idx[k][\"start\"] > start]\n",
        "        end = min(ends) if ends else len(text)\n",
        "        return text[start:end]\n",
        "    for k in [\"A1\", \"A2\", \"A3\"]:\n",
        "        val = idx[k][\"after\"] if k in idx else \"\"\n",
        "        out[k] = val.strip() if val else \"\"\n",
        "    bullet_re = re.compile(r\"(?m)^\\s*[-*\u2022]\\s+(.*)$\")\n",
        "    for k in [\"A4\", \"A5\", \"A6\", \"A7\"]:\n",
        "        block = section(k)\n",
        "        items = [m.group(1).strip() for m in bullet_re.finditer(block)]\n",
        "        items = [i.replace(\"|\", \" \") for i in items]\n",
        "        out[k] = \"|\".join(items) if items else \"\"\n",
        "    return out\n",
        "\n",
        "B_LABELS = [\"B1\", \"B2\", \"B3\", \"B4\", \"B5\", \"B6\", \"B7\"]\n",
        "\n",
        "def parse_B1_B7(raw: str) -> dict:\n",
        "    text = (raw or \"\").replace(\"\\r\\n\", \"\\n\").replace(\"\\r\", \"\\n\").strip()\n",
        "    out = {k: \"\" for k in B_LABELS}\n",
        "    label_re = re.compile(r\"(?m)^\\s*(B[1-7])\\)\\s*(.*)$\")\n",
        "    matches = list(label_re.finditer(text))\n",
        "    if not matches:\n",
        "        return out\n",
        "    idx = {m.group(1): {\"start\": m.start(), \"after\": m.group(2).strip()} for m in matches}\n",
        "    def section(label):\n",
        "        if label not in idx:\n",
        "            return \"\"\n",
        "        start = idx[label][\"start\"]\n",
        "        ends = [idx[k][\"start\"] for k in idx if idx[k][\"start\"] > start]\n",
        "        end = min(ends) if ends else len(text)\n",
        "        return text[start:end]\n",
        "    for k in [\"B1\", \"B2\", \"B3\"]:\n",
        "        val = idx[k][\"after\"] if k in idx else \"\"\n",
        "        out[k] = val.strip() if val else \"\"\n",
        "    bullet_re = re.compile(r\"(?m)^\\s*[-*\u2022]\\s+(.*)$\")\n",
        "    for k in [\"B4\", \"B5\", \"B6\", \"B7\"]:\n",
        "        block = section(k)\n",
        "        items = [m.group(1).strip() for m in bullet_re.finditer(block)]\n",
        "        items = [i.replace(\"|\", \" \") for i in items]\n",
        "        out[k] = \"|\".join(items) if items else \"\"\n",
        "    return out\n",
    ]))

    # Client init
    cells.append(md("## Initialize LLM Client"))
    cells.append(code(api["import"] + api["init"]))

    # --- Set 1 ---
    cells.append(md("## Set 1 – Resume and Process"))
    cells.append(code([
        "if SET1_OUT.exists():\n",
        "    existing_df_set1 = pd.read_csv(SET1_OUT, dtype=str).fillna(\"\")\n",
        "    done_terms_set1 = set(existing_df_set1[\"concept_term\"].tolist())\n",
        "else:\n",
        "    existing_df_set1 = pd.DataFrame()\n",
        "    done_terms_set1 = set()\n",
        "print(f\"Set 1 – concepts remaining: {len(CONCEPT_TERMS) - len(done_terms_set1)}\")\n",
    ]))
    cells.append(code(
        "rows_set1 = []\n"
        "\n"
        "for concept_term in CONCEPT_TERMS:\n"
        "    if concept_term in done_terms_set1:\n"
        "        with LOG_PATH_SET1.open(\"a\") as f:\n"
        "            f.write(f\"{datetime.now().isoformat()}\\t{concept_term}\\tSKIP\\n\")\n"
        "        continue\n"
        "\n"
        "    prompt = PROMPT_TEMPLATE_SET1.format(CONCEPT_TERM=concept_term)\n"
        "\n"
        "    try:\n"
        + api["api_call_set1"] +
        "        parsed = parse_A1_A7(raw_output)\n"
        "\n"
        "        rows_set1.append({\n"
        "            \"timestamp\": datetime.now().isoformat(),\n"
        "            \"model\": MODEL_NAME,\n"
        "            \"prompt_set\": \"set1\",\n"
        "            \"concept_term\": concept_term,\n"
        "            \"A1_fsn\": _csv_safe(parsed[\"A1\"]),\n"
        "            \"A2_semantic_tag\": _csv_safe(parsed[\"A2\"]),\n"
        "            \"A3_definition_status\": _csv_safe(parsed[\"A3\"]),\n"
        "            \"A4_parents\": _csv_safe(parsed[\"A4\"]),\n"
        "            \"A5_grandparents\": _csv_safe(parsed[\"A5\"]),\n"
        "            \"A6_children\": _csv_safe(parsed[\"A6\"]),\n"
        "            \"A7_siblings\": _csv_safe(parsed[\"A7\"]),\n"
        "        })\n"
        "\n"
        "        with LOG_PATH_SET1.open(\"a\") as f:\n"
        "            f.write(\n"
        "                \"\\n\" + \"=\"*80 + \"\\n\" + datetime.now().isoformat() + \"\\n\"\n"
        "                + \"CONCEPT: \" + concept_term + \"\\n\\n\" + raw_output.strip() + \"\\n\"\n"
        "            )\n"
        "\n"
        "    except Exception as e:\n"
        "        with LOG_PATH_SET1.open(\"a\") as f:\n"
        "            f.write(f\"{datetime.now().isoformat()}\\t{concept_term}\\tERROR\\t{e}\\n\")\n"
        "\n"
        "    time.sleep(0.2)\n"
        "\n"
        "print(f\"Set 1 processed {len(rows_set1)} new concepts.\")\n"
    ))
    cells.append(code([
        "if rows_set1:\n",
        "    out_df = pd.DataFrame(rows_set1)\n",
        "    combined = pd.concat([existing_df_set1, out_df], ignore_index=True) if not existing_df_set1.empty else out_df\n",
        "    combined = combined[[\"timestamp\", \"model\", \"prompt_set\", \"concept_term\", \"A1_fsn\", \"A2_semantic_tag\", \"A3_definition_status\", \"A4_parents\", \"A5_grandparents\", \"A6_children\", \"A7_siblings\"]]\n",
        "    combined.to_csv(SET1_OUT, index=False)\n",
        "print(\"Set 1 complete.\", SET1_OUT)\n",
    ]))

    # --- Set 2 ---
    cells.append(md("## Set 2 – Prompt, Resume and Process"))
    cells.append(code([
        "assert SET2_DIR.exists(), \"SET2_DIR missing.\"\n",
        "LOG_PATH_SET2 = SET2_DIR / \"logs.txt\"\n",
        "PROMPT_TEMPLATE_SET2 = \"\"\"\n",
        "For the term: \"{CONCEPT_TERM}\"\n",
        "\n",
        "Answer ONLY with the items below.\n",
        "Do NOT explain.\n",
        "\n",
        "B1) Most precise official-style name\n",
        "B2) What kind of thing it is (semantic type)\n",
        "B3) Category type (Primitive / Fully defined)\n",
        "B4) More general terms (immediate broader concepts)\n",
        "B5) Grandparent terms (broader concepts two levels up)\n",
        "B6) More specific terms (immediate narrower concepts)\n",
        "B7) Terms at the same generality level (peers / siblings)\n",
        "\n",
        "Rules:\n",
        "- Bullet lists where applicable\n",
        "- Exact labels B1\u2013B7\n",
        "- No extra text\n",
        "\"\"\".strip()\n",
    ]))
    cells.append(code([
        "if SET2_OUT.exists():\n",
        "    existing_df_set2 = pd.read_csv(SET2_OUT, dtype=str).fillna(\"\")\n",
        "    done_terms_set2 = set(existing_df_set2[\"concept_term\"].tolist())\n",
        "else:\n",
        "    existing_df_set2 = pd.DataFrame()\n",
        "    done_terms_set2 = set()\n",
        "print(f\"Set 2 – concepts remaining: {len(CONCEPT_TERMS) - len(done_terms_set2)}\")\n",
    ]))
    cells.append(code(
        "rows_set2 = []\n"
        "\n"
        "for concept_term in CONCEPT_TERMS:\n"
        "    if concept_term in done_terms_set2:\n"
        "        with LOG_PATH_SET2.open(\"a\") as f:\n"
        "            f.write(f\"{datetime.now().isoformat()}\\t{concept_term}\\tSKIP\\n\")\n"
        "        continue\n"
        "\n"
        "    prompt = PROMPT_TEMPLATE_SET2.format(CONCEPT_TERM=concept_term)\n"
        "\n"
        "    try:\n"
        + api["api_call_set2"] +
        "        parsed = parse_B1_B7(raw_output)\n"
        "\n"
        "        rows_set2.append({\n"
        "            \"timestamp\": datetime.now().isoformat(),\n"
        "            \"model\": MODEL_NAME,\n"
        "            \"prompt_set\": \"set2\",\n"
        "            \"concept_term\": concept_term,\n"
        "            \"B1_official_name\": _csv_safe(parsed[\"B1\"]),\n"
        "            \"B2_kind\": _csv_safe(parsed[\"B2\"]),\n"
        "            \"B3_category_type\": _csv_safe(parsed[\"B3\"]),\n"
        "            \"B4_immediate_broader\": _csv_safe(parsed[\"B4\"]),\n"
        "            \"B5_grandparents\": _csv_safe(parsed[\"B5\"]),\n"
        "            \"B6_immediate_narrower\": _csv_safe(parsed[\"B6\"]),\n"
        "            \"B7_peer_terms\": _csv_safe(parsed[\"B7\"]),\n"
        "        })\n"
        "\n"
        "        with LOG_PATH_SET2.open(\"a\") as f:\n"
        "            f.write(\n"
        "                \"\\n\" + \"=\"*80 + \"\\n\" + datetime.now().isoformat() + \"\\n\"\n"
        "                + \"CONCEPT: \" + concept_term + \"\\n\\n\" + raw_output.strip() + \"\\n\"\n"
        "            )\n"
        "\n"
        "    except Exception as e:\n"
        "        with LOG_PATH_SET2.open(\"a\") as f:\n"
        "            f.write(f\"{datetime.now().isoformat()}\\t{concept_term}\\tERROR\\t{e}\\n\")\n"
        "\n"
        "    time.sleep(0.2)\n"
        "\n"
        "print(f\"Set 2 processed {len(rows_set2)} new concepts.\")\n"
    ))
    cells.append(code([
        "if rows_set2:\n",
        "    out_df = pd.DataFrame(rows_set2)\n",
        "    combined = pd.concat([existing_df_set2, out_df], ignore_index=True) if not existing_df_set2.empty else out_df\n",
        "    combined = combined[[\"timestamp\", \"model\", \"prompt_set\", \"concept_term\", \"B1_official_name\", \"B2_kind\", \"B3_category_type\", \"B4_immediate_broader\", \"B5_grandparents\", \"B6_immediate_narrower\", \"B7_peer_terms\"]]\n",
        "    combined.to_csv(SET2_OUT, index=False)\n",
        "print(\"Set 2 complete.\", SET2_OUT)\n",
        "print(\"Run step3_accuracy.ipynb next.\")\n",
    ]))

    return cells


def main():
    for folder in ["testing_gpt", "testing_claude", "testing_gemini", "testing_deepseek"]:
        llm = folder.replace("testing_", "")
        path = REPO / folder / "step2_llm_queries.ipynb"
        nb = {
            "cells": build_cells(llm),
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10.0"},
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }
        with open(path, "w") as f:
            json.dump(nb, f, indent=2)
        print("Wrote", path)


if __name__ == "__main__":
    main()
