from __future__ import annotations

import json
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import typer
from huggingface_hub import hf_hub_download
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import math
import fasttext

app = typer.Typer(add_completion=False, no_args_is_help=True)


# -------------------------
# MT metrics (SacreBLEU)
# -------------------------

def _sacre_score(refs: List[str], hyps: List[str], scorer) -> Optional[float]:
    hyps_ = [(h or "").strip() for h in hyps]
    refs_ = [[(r or "").strip() for r in refs]]
    if not hyps_:
        return None
    return scorer.corpus_score(hyps_, refs_).score

def spbleu_corpus_score(refs: List[str], hyps: List[str]) -> Dict[str, Optional[float]]:
    spbleu = BLEU(tokenize="flores200")
    return {"SpBLEU_corpus_score": _sacre_score(refs, hyps, spbleu)}

def chrf_corpus_score(refs: List[str], hyps: List[str]) -> Dict[str, Optional[float]]:
    chrf =CHRF(char_order=6,
               word_order=2,
               beta=2)
    return {"ChrF_corpus_score": _sacre_score(refs, hyps, chrf)}

# -------------------------
# I/O
# -------------------------
def load_files(path: Path, csv_field: str) -> List[str]:
    if path.suffix.lower() == ".txt" or path.suffix.lower() == ".out":
        return path.read_text(encoding="utf-8").splitlines()
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path, sep=',',quotechar='"', engine='python')[csv_field].tolist()
    raise typer.BadParameter("Unsupported file type. Use .txt or .csv.")


def normalize_text(x):
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)

# -------------------------
# ADI scorer
# -------------------------

@dataclass
class ADIMaps:
    DIALECTS: List[str]
    COUNTRY2DIALECT: Dict[str, str]
    DIALECT2COUNTRY: Dict[str, str]
    COUNTRY2MACRO_DIALECT: Dict[str, str]
    MICROLANGUAGE_MAP: Dict[str, List[str]]


class ADIScorer:
    def __init__(
        self,
        *,
        aldi_model_id: str,
        nadi_model_id: str,
        target_lang: str,
        maps: ADIMaps,
        device: Optional[str] = None,
        fasttext_repo_id: str = "facebook/fasttext-language-identification",
        fasttext_filename: str = "model.bin",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_lang = target_lang
        self.maps = maps

        # LID model (fastText)
        ft_path = hf_hub_download(repo_id=fasttext_repo_id,
                                  filename=fasttext_filename)
        self.lid_model = fasttext.load_model(ft_path)

        # ALDi / NADI
        self.aldi_tokenizer = AutoTokenizer.from_pretrained(aldi_model_id)
        self.aldi_model = AutoModelForSequenceClassification.from_pretrained(aldi_model_id).to(self.device).eval()

        self.nadi_tokenizer = AutoTokenizer.from_pretrained(nadi_model_id)
        self.nadi_model = AutoModelForSequenceClassification.from_pretrained(nadi_model_id).to(self.device).eval()

    @staticmethod
    def clean_text(text: str) -> str:
        return (text or "").replace("\n", " ").strip()

    @staticmethod
    def _fasttext_code(pred_out) -> str:
        # pred_out like: (('__label__ara',), array([0.99]))
        label = pred_out[0][0]
        ### output __label__arb_Arab
        ### we extract arb
        return label[len("__label__"):len("__label__")+3] if label.startswith("__label__") else label

    def run_lid_gate(self, text: str) -> int:
        code = self._fasttext_code(self.lid_model.predict(text))
        allowed = set(self.maps.MICROLANGUAGE_MAP.get(self.target_lang, []))
        return 1 if code in allowed else 0

    def _dialect2index(self, dialect_country_code: str) -> int:
        dia = self.maps.COUNTRY2DIALECT[dialect_country_code]
        return self.maps.DIALECTS.index(dia)

    def _macro_prob(self, probs: List[float], dialect_country_code: str) -> float:
        target_macro = self.maps.COUNTRY2MACRO_DIALECT[dialect_country_code]
        total = 0.0
        for i, dia in enumerate(self.maps.DIALECTS):
            country = self.maps.DIALECT2COUNTRY[dia]
            if self.maps.COUNTRY2MACRO_DIALECT[country] == target_macro:
                total += probs[i]
        return total

    @torch.no_grad()
    def run_aldi(self, text: str) -> float:
        inputs = self.aldi_tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        logits = self.aldi_model(**inputs).logits
        return float(min(max(0.0, logits[0][0].item()), 1.0))

    @torch.no_grad()
    def run_nadi(self, text: str, dialect_country_code: str) -> Tuple[float, float]:
        inputs = self.nadi_tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        logits = self.nadi_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).flatten().tolist()
        idx = self._dialect2index(dialect_country_code)
        prob = float(probs[idx])
        macro_prob = float(self._macro_prob(probs, dialect_country_code))
        return prob, macro_prob

    def score_outputs(
        self,
        outputs: List[str],
        *,
        dialect: str,
        require_target_lang: bool = True,
        allow_msa: bool = False,
    ) -> Dict[str, float]:
        if (not allow_msa) and dialect == "msa":
            return {"prob": 0.0, "dialectness": 0.0, "score": 0.0, "macro_score": 0.0}

        prob_list: List[float] = []
        dness_list: List[float] = []
        score_list: List[float] = []
        mscore_list: List[float] = []

        for out in outputs:
            text = self.clean_text(normalize_text(out))
            if not text:
                prob_list.append(0.0); dness_list.append(0.0); score_list.append(0.0); mscore_list.append(0.0)
                continue
            # Check if output is in right language
            if self.run_lid_gate(text) == 0:
                prob_list.append(0.0); dness_list.append(0.0); score_list.append(0.0); mscore_list.append(0.0)
                continue

            prob, macro_prob = self.run_nadi(text, dialect)
            dness = self.run_aldi(text)

            prob_list.append(prob)
            dness_list.append(dness)
            score_list.append(prob * dness)
            mscore_list.append(macro_prob * dness)

        return {
            "prob": float(np.mean(prob_list)) if prob_list else 0.0,
            "dialectness": float(np.mean(dness_list)) if dness_list else 0.0,
            "score": float(np.mean(score_list)) if score_list else 0.0,
            "macro_score": float(np.mean(mscore_list)) if mscore_list else 0.0,
        }


def load_project_maps() -> ADIMaps:
    """
    Import variables from maps.py (from the original al-qasida github)
    """
    from maps import DIALECTS, COUNTRY2DIALECT, DIALECT2COUNTRY, COUNTRY2MACRO_DIALECT, MICROLANGUAGE_MAP
    return ADIMaps(
        DIALECTS=DIALECTS,
        COUNTRY2DIALECT=COUNTRY2DIALECT,
        DIALECT2COUNTRY=DIALECT2COUNTRY,
        COUNTRY2MACRO_DIALECT=COUNTRY2MACRO_DIALECT,
        MICROLANGUAGE_MAP=MICROLANGUAGE_MAP,
    )


# -------------------------
# Commands
# -------------------------

@app.command("score-adi")
def score_adi(
    outputs_path: Path = typer.Argument(..., exists=True, readable=True),
    dialect: str = typer.Option(..., help="Dialect country code (e.g., egy, mar, dza, ...)."),
    target_lang: str = typer.Option("ara", help="Key used for MICROLANGUAGE_MAP (e.g., ara)."),
    csv_field_hypothesis: str = typer.Option("generations", help="Field to read from JSONL."),
    require_target_lang: bool = typer.Option(True, help="Gate scoring with fastText LID."),
    allow_msa: bool = typer.Option(False, help="If False, returns zeros when dialect == msa."),
    aldi_model_id: str = typer.Option("AMR-KELEG/Sentence-ALDi"),
    nadi_model_id: str = typer.Option("AMR-KELEG/NADI2024-baseline"),
    device: Optional[str] = typer.Option(None, help="cpu/cuda (default: auto)."),
):
    hyps = load_files(outputs_path, csv_field=csv_field_hypothesis)

    maps = load_project_maps()
    scorer = ADIScorer(
        aldi_model_id=aldi_model_id,
        nadi_model_id=nadi_model_id,
        target_lang=target_lang,
        maps=maps,
        device=device,
    )
    scores = scorer.score_outputs(
        hyps, dialect=dialect, require_target_lang=require_target_lang, allow_msa=allow_msa
    )
    payload = {"n": len(hyps), "dialect": dialect, "target_lang": target_lang, **scores}
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("score-mt")
def score_mt(
    outputs_path: Path = typer.Argument(..., exists=True, readable=True),
    refs_path: Path = typer.Argument(..., exists=True, readable=True),
    csv_field_hypothesis: str = typer.Option("generations", help="Field to read from csv."),
    csv_field_reference: str = typer.Option("completion", help="Field to read from csv."),
):
    hyps = load_files(outputs_path, csv_field=csv_field_hypothesis)
    refs = load_files(refs_path, csv_field=csv_field_reference)

    if len(hyps) != len(refs):
        raise typer.BadParameter(f"outputs ({len(hyps)}) and refs ({len(refs)}) must have the same length.")

    scores: Dict[str, Any] = {"n": len(hyps)}
    scores.update(spbleu_corpus_score(refs, hyps))
    scores.update(chrf_corpus_score(refs, hyps))
    typer.echo(json.dumps(scores, ensure_ascii=False, indent=2))


@app.command("score-all")
def score_all(
    outputs_path: Path = typer.Argument(..., exists=True, readable=True),
    dialect: str = typer.Option(...),
    refs_path: Optional[Path] = typer.Option(None, help="If provided, also computes SpBLEU + chrF."),
    target_lang: str = typer.Option("ara"),
    csv_field_hypothesis: str = typer.Option("completion"),
    require_target_lang: bool = typer.Option(True),
    allow_msa: bool = typer.Option(False),
    aldi_model_id: str = typer.Option("AMR-KELEG/Sentence-ALDi"),
    nadi_model_id: str = typer.Option("AMR-KELEG/NADI2024-baseline"),
    device: Optional[str] = typer.Option(None),
):
    hyps = load_files(outputs_path, csv_filed=csv_field_hypothesis)

    out: Dict[str, Any] = {"n": len(hyps), "dialect": dialect, "target_lang": target_lang}

    # ADI
    maps = load_project_maps()
    scorer = ADIScorer(
        aldi_model_id=aldi_model_id,
        nadi_model_id=nadi_model_id,
        target_lang=target_lang,
        maps=maps,
        device=device,
    )
    out.update(scorer.score_outputs(hyps, dialect=dialect, require_target_lang=require_target_lang, allow_msa=allow_msa))

    # MT (optional)
    if refs_path is not None:
        refs = load_files(refs_path)
        if len(hyps) != len(refs):
            raise typer.BadParameter(f"outputs ({len(hyps)}) and refs ({len(refs)}) must have the same length.")
        out.update(spbleu_corpus_score(refs, hyps))
        out.update(chrf_corpus_score(refs, hyps))

    typer.echo(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    app()

