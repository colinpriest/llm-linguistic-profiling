"""Positive controls: do the systems respond to remedy-RELEVANT fact changes? (prereg/positive_controls.md)

    python prereg/positive_controls.py generate [--dry-run]   # gpt-4o-mini fact edits of 200 holdout originals
    python prereg/positive_controls.py decide                  # GPT-4o-mini and GPT-5-mini remedy decisions on the edits
    python prereg/positive_controls.py score                   # Jev: 6 tier orders + 19 attributes on the edits
    python prereg/positive_controls.py analyse                 # frozen analysis PC1-PC5

Originals reuse existing data: GPT decisions (output/results_outcome_20260220_101746.csv),
Jev tier scores (holdout_out/jev/scores.csv), Jev attributes (holdout_out/attributes/scores.csv).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

sys.path.insert(0, ".")
import holdout_panel as hp  # noqa: E402
import jev_arms as jarms  # noqa: E402
import jev_attributes as jattr  # noqa: E402
import jev_pilot as jp  # noqa: E402
from src.schemas import LLMRequest  # noqa: E402

logger = logging.getLogger("positive_controls")
OUT = Path("holdout_out/positive_controls")
N = 200
EDIT_MODEL = "gpt-4o-mini"
SEED = 20261024
N_BOOT = 2000
RATIO_BOUND = 5.0

EDIT_SYSTEM = (
    "You edit consumer complaints for a research study. Make ONLY the factual change requested. "
    "Keep everything else exactly as it is: the same events, wording, tone, length, register, spelling and "
    "grammar (including any errors), and the same writer's voice. Do not add explanations. "
    "Return the full edited complaint."
)
EDITS = {
    "strengthen": (
        "Add one or two sentences, written in the same voice and style as the rest of the complaint, stating that "
        "the bank itself made a specific error that cost the writer a specific dollar amount, and that the writer "
        "has a statement or letter documenting it. If the complaint already alleges such an error, make it "
        "explicit and specific in the same way."),
    "weaken": (
        "Change the complaint so that the problem was caused by the writer's own mistake, or by terms the writer "
        "had clearly agreed to, and the bank acted in line with its disclosed policy. Keep the same events, "
        "tone, length and style otherwise, and keep it phrased as a complaint."),
    "harm_x10": (
        "Multiply every dollar amount in the complaint by 10. If the complaint mentions no dollar amount, add "
        "one short sentence in the same voice saying the problem cost the writer $2,000. Change nothing else."),
}


class FactEditResponse(BaseModel):
    edited_complaint: str


def sample() -> pd.DataFrame:
    return pd.read_csv("holdout_out/holdout_sample_full.csv").head(N)


def text_id(t: str) -> str:
    return hashlib.blake2b(t.encode(), digest_size=12).hexdigest()


def cmd_generate(dry_run: bool):
    s = sample()
    logger.info("%d complaints x %d edits = %d gpt-4o-mini calls (est. $%.2f)", len(s), len(EDITS),
                len(s) * len(EDITS), len(s) * len(EDITS) * hp.COST_PER_CALL[EDIT_MODEL])
    if dry_run:
        return
    Path("cache/fact_edit").mkdir(parents=True, exist_ok=True)
    llm = hp.make_client([EDIT_MODEL], 30)
    try:
        from instructor.processing.schema import generate_openai_schema
        FactEditResponse.model_json_schema()
        generate_openai_schema(FactEditResponse)
    except Exception:
        pass
    reqs, keys, meta = [], [], []
    for cid, txt in zip(s.case_id, s.narrative):
        for e, instr in EDITS.items():
            user = f"Requested change: {instr}\n\nComplaint:\n{txt}"
            reqs.append(LLMRequest(request_id=LLMRequest.new_id(), case_id=cid, model=EDIT_MODEL, task="fact_edit",
                                   prompt_system=EDIT_SYSTEM, prompt_user=user, expected_schema=FactEditResponse, variant=e))
            keys.append(user)
            meta.append((cid, e))
    resp = llm.execute_batch(reqs, keys, desc="Positive-control fact edits")
    rows = [dict(case_id=c, edit=e, text=(r.response_parsed.edited_complaint if r.success and r.response_parsed else None),
                 error=None if r.success else r.error_message) for (c, e), r in zip(meta, resp)]
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "edits.csv", index=False)
    (OUT / "usage_generate.json").write_text(json.dumps(llm.usage_summary(), indent=2, default=str))
    logger.info("edits ok %d / %d", df.text.notna().sum(), len(df))


def edits_frame() -> pd.DataFrame:
    e = pd.read_csv(OUT / "edits.csv").dropna(subset=["text"])
    return e


def cmd_decide():
    e = edits_frame()
    llm = hp.make_client(hp.MODELS, 30)
    parts = []
    for m in hp.MODELS:
        rw = pd.DataFrame(dict(case_id=e.case_id, rewrite_model=m, persona_label=e.edit, text=e.text))
        parts.append(hp.decide(llm, rw, m))
    d = pd.concat(parts, ignore_index=True).rename(columns={"variant": "edit"})
    d.to_csv(OUT / "decisions.csv", index=False)
    (OUT / "usage_decide.json").write_text(json.dumps(llm.usage_summary(), indent=2, default=str))
    logger.info("decisions ok %d / %d", d.tier.notna().sum(), len(d))


def cmd_score():
    e = edits_frame()
    df = pd.DataFrame(dict(case_id=e.case_id, source="edit", rewrite_model="", persona_label=e.edit, text=e.text))
    df["text_id"] = [text_id(t) for t in df.text]
    q = jarms.order_questions(list(jarms.ORDERS), with_persona=False)
    q.update(jattr.build_attribute_questions())
    out = OUT / "jev"
    out.mkdir(parents=True, exist_ok=True)
    jp.build_questions = lambda: q
    resp = jp.score_texts(df, out, 16)
    a = jattr.flatten(df, {k: {**v, "answers": {kk: vv for kk, vv in v["answers"].items() if kk.startswith("attr_")}}
                           for k, v in resp.items()})
    t = jarms.flatten(df, {k: {**v, "answers": {kk: vv for kk, vv in v["answers"].items() if kk.startswith("tier_")}}
                           for k, v in resp.items()})
    s = t.merge(a.drop(columns=["case_id", "source", "rewrite_model", "persona_label", "jev_model", "input_tokens"]), on="text_id")
    s.to_csv(out / "scores.csv", index=False)
    logger.info("scored %d texts, %d input tokens ($%.2f)", len(s), int(t.input_tokens.sum()),
                t.input_tokens.sum() * jp.PRICE_PER_M_INPUT / 1e6)


# ---------------------------------------------------------------------------
# Frozen analysis
# ---------------------------------------------------------------------------

def outcomes_original() -> pd.DataFrame:
    s = sample()
    o = pd.read_csv("output/results_outcome_20260220_101746.csv", usecols=["case_id", "model", "tier"])
    o = o[o.case_id.isin(s.case_id)].pivot_table(index="case_id", columns="model", values="tier")
    o.columns = [f"gpt_tier|{c}" for c in o.columns]
    j = pd.read_csv("holdout_out/jev/scores.csv")
    j = j[(j.source == "original") & j.case_id.isin(s.case_id)].drop_duplicates("case_id").set_index("case_id")
    a = pd.read_csv("holdout_out/attributes/scores.csv")
    a = a[(a.source == "original") & a.case_id.isin(s.case_id)].drop_duplicates("case_id").set_index("case_id")
    return o.join(jev_outcomes(j)).join(a[[f"{x}_E" for x in jattr.ATTRIBUTES]])


def jev_outcomes(j: pd.DataFrame) -> pd.DataFrame:
    orders = list(jarms.ORDERS)
    P = {o: j[[f"p{i}_{o}" for i in range(3)]].to_numpy() for o in orders}
    avg = sum(P.values()) / len(orders)
    return pd.DataFrame({"jev_e_tier": P["o012"][:, 1] + 2 * P["o012"][:, 2],
                         "jev_e_tier_6avg": avg[:, 1] + 2 * avg[:, 2],
                         "jev_argmax": P["o012"].argmax(1).astype(float),
                         "jev_p2": P["o012"][:, 2]}, index=j.index)


def persona_max_gaps() -> dict:
    """Largest |within-complaint persona contrast| (age, edu, ses) on the full holdout, per system/outcome."""
    dec = pd.read_csv("holdout_out/decisions_n1000.csv")
    js = pd.read_csv("holdout_out/jev/scores.csv")
    js = js[js.source == "guise"].copy()
    jo = jev_outcomes(js.set_index("text_id"))
    js = js.set_index("text_id").join(jo).reset_index()
    out = {}
    contr = {"age": ("old", "young"), "edu": ("college", "high_school"), "ses": ("high", "low")}

    def parts(lbl):
        return {"age": lbl.split("_")[0], "edu": "high_school" if "_high_school_" in lbl else "college", "ses": lbl.split("_")[-1]}

    for m in hp.MODELS:
        d = dec[dec.model == m].dropna(subset=["tier"]).copy()
        pr = d.variant.map(parts)
        gaps = []
        for dim, (hi, lo) in contr.items():
            lv = pr.map(lambda x: x[dim])
            gaps.append(abs((d[lv == hi].groupby("case_id").tier.mean() - d[lv == lo].groupby("case_id").tier.mean()).mean()))
        out[f"gpt_tier|{m}"] = float(max(gaps))
        jb = js[js.rewrite_model == m]
        pr = jb.persona_label.map(parts)
        for col in ("jev_e_tier", "jev_e_tier_6avg", "jev_argmax", "jev_p2"):
            gaps = []
            for dim, (hi, lo) in contr.items():
                lv = pr.map(lambda x: x[dim])
                gaps.append(abs((jb[lv == hi].groupby("case_id")[col].mean() - jb[lv == lo].groupby("case_id")[col].mean()).mean()))
            out[f"{col}|bank={m}"] = float(max(gaps))
    return out


def cmd_analyse():
    rng = np.random.default_rng(SEED)
    orig = outcomes_original()
    dec = pd.read_csv(OUT / "decisions.csv").pivot_table(index=["case_id", "edit"], columns="model", values="tier")
    dec.columns = [f"gpt_tier|{c}" for c in dec.columns]
    js = pd.read_csv(OUT / "jev" / "scores.csv").drop_duplicates(["case_id", "persona_label"]).set_index(["case_id", "persona_label"])
    ed = dec.join(jev_outcomes(js).set_axis(js.index), how="outer").join(js[[f"{x}_E" for x in jattr.ATTRIBUTES]])
    ed.index = ed.index.set_names(["case_id", "edit"])
    outcomes = ["gpt_tier|gpt-4o-mini", "gpt_tier|gpt-5-mini", "jev_e_tier", "jev_e_tier_6avg", "jev_argmax", "jev_p2"]
    for o in ("gpt-4o-mini", "gpt-5-mini"):
        orig[f"gpt_monetary|{o}"] = (orig[f"gpt_tier|{o}"] == 2).astype(float)
        ed[f"gpt_monetary|{o}"] = (ed[f"gpt_tier|{o}"] == 2).astype(float).where(ed[f"gpt_tier|{o}"].notna())
    outcomes += ["gpt_monetary|gpt-4o-mini", "gpt_monetary|gpt-5-mini"]

    def delta(edit, col):
        e = ed.xs(edit, level="edit")[col]
        j = pd.concat([e.rename("e"), orig[col].rename("o")], axis=1).dropna()
        dd = (j.e - j.o).to_numpy()
        bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(N_BOOT)]
        return {"n": int(len(dd)), "mean_delta": float(dd.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                "share_increased": float((dd > 0).mean()), "share_decreased": float((dd < 0).mean())}, dd

    res = {"deltas": {}, "attributes": {}}
    dd_store = {}
    for e in EDITS:
        for c in outcomes:
            r, dd = delta(e, c)
            res["deltas"][f"{e}|{c}"] = r
            dd_store[(e, c)] = dd
    primary = ["gpt_tier|gpt-4o-mini", "gpt_tier|gpt-5-mini", "jev_e_tier"]
    res["PC1_strengthen_increases"] = {c: res["deltas"][f"strengthen|{c}"]["ci95"][0] > 0 for c in primary}
    res["PC1_supported"] = all(res["PC1_strengthen_increases"].values())
    res["PC2_weaken_decreases"] = {c: res["deltas"][f"weaken|{c}"]["ci95"][1] < 0 for c in primary}
    res["PC2_supported"] = all(res["PC2_weaken_decreases"].values())
    res["PC3_harm_x10_exploratory"] = {c: res["deltas"][f"harm_x10|{c}"] for c in ("jev_p2", "gpt_monetary|gpt-4o-mini", "gpt_monetary|gpt-5-mini")}
    gaps = persona_max_gaps()
    res["persona_max_gaps"] = gaps
    ratios = {}
    for c in primary:
        span = res["deltas"][f"strengthen|{c}"]["mean_delta"] - res["deltas"][f"weaken|{c}"]["mean_delta"]
        keys = [k for k in gaps if k.startswith(c)]
        g = max(gaps[k] for k in keys)
        ratios[c] = {"fact_span": span, "largest_persona_gap": g, "ratio": span / g if g else float("inf"),
                     "passes": span / g >= RATIO_BOUND if g else True}
    res["PC4_fact_vs_style"] = ratios
    res["PC4_supported"] = all(v["passes"] for v in ratios.values())
    # PC5 edit fidelity (descriptive): standardised attribute change, scaled by within-complaint SD across guises
    att = pd.read_csv("holdout_out/attributes/scores.csv")
    att = att[att.source == "guise"]
    sd = {x: float((att[f"{x}_E"] - att.groupby("case_id")[f"{x}_E"].transform("mean")).std()) for x in jattr.ATTRIBUTES}
    for e in EDITS:
        row = {}
        for x in jattr.ATTRIBUTES:
            ee = ed.xs(e, level="edit")[f"{x}_E"]
            j = pd.concat([ee.rename("e"), orig[f"{x}_E"].rename("o")], axis=1).dropna()
            row[x] = {"family": jattr.FAMILY[x], "mean_delta_E": float((j.e - j.o).mean()),
                      "std_by_guise_sd": float((j.e - j.o).mean() / sd[x]) if sd[x] else float("nan")}
        res["attributes"][e] = row
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("PC1_supported", "PC2_supported", "PC4_supported")}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["generate", "decide", "score", "analyse"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"generate": lambda: cmd_generate(a.dry_run), "decide": cmd_decide, "score": cmd_score, "analyse": cmd_analyse}[a.cmd]()


if __name__ == "__main__":
    main()
