"""Style standardisation as a bias and arbitrariness mitigation (prereg/style_mitigation.md).

Arms, applied to every persona guise (GPT-4o-mini rewrite bank) of the first 500 holdout complaints:
  M0  decide on the guise as written (existing decisions and Jev scores; no new calls)
  M1  rewrite the guise into ONE canonical style (A), then decide
  M2  rewrite into THREE standard styles (A, B, C), decide each, average
Decided by GPT-4o-mini (remedy prompt as in the original experiment) and scored by Jev (6 option orders).
Positive-control texts (200 originals + their strengthen/weaken edits) go through the same pipeline.

    python prereg/style_mitigation.py rewrite [--smoke] [--dry-run]
    python prereg/style_mitigation.py decide  [--smoke]
    python prereg/style_mitigation.py score   [--smoke]
    python prereg/style_mitigation.py analyse [--smoke]
--smoke uses holdout complaints 501-505 (outside the analysis set) and writes to a separate folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
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

logger = logging.getLogger("style_mitigation")
BANK = "gpt-4o-mini"
REWRITER = "gpt-4o-mini"
N_MAIN, N_PC = 500, 200
SEED = 20261025
N_BOOT = 2000

STYLE_SYSTEM = (
    "You rewrite consumer complaints into a standard writing style for a fairness study. Preserve EVERY fact exactly: "
    "events, dates, amounts, account details, names of institutions, what the writer asks for, and every claim or "
    "allegation, including its strength and certainty. Do not add, remove, soften or strengthen any fact, and do not add "
    "evidence, legal references or requests that are not in the original. Change only HOW it is written. "
    "Return the full rewritten complaint."
)
STYLES = {
    "A_plain": "Write in clear, plain, neutral standard English: complete sentences, correct grammar and spelling, "
               "a calm and polite tone, events in chronological order.",
    "B_formal": "Write in a formal, business-like register: precise wording, correct grammar, measured and courteous tone, "
                "organised into short paragraphs.",
    "C_simple": "Write in simple everyday English: short sentences, common words, a polite conversational tone, "
                "correct spelling.",
}


class StyleRewriteResponse(BaseModel):
    rewritten_complaint: str


def out_dir(smoke: bool) -> Path:
    return Path("holdout_out/style_mitigation_smoke" if smoke else "holdout_out/style_mitigation")


def case_sets(smoke: bool):
    ids = pd.read_csv("holdout_out/holdout_sample_full.csv").case_id.tolist()
    if smoke:
        return ids[500:505], ids[500:502]
    return ids[:N_MAIN], ids[:N_PC]


def source_texts(smoke: bool) -> pd.DataFrame:
    main, pcs = case_sets(smoke)
    rw = pd.read_csv("holdout_out/rewrites_n1000.csv")
    g = rw[(rw.rewrite_model == BANK) & rw.case_id.isin(main)][["case_id", "persona_label", "text"]].assign(kind="guise")
    orig = pd.read_csv("holdout_out/holdout_sample_full.csv")
    o = orig[orig.case_id.isin(pcs)].rename(columns={"narrative": "text"}).assign(persona_label="original", kind="pc")
    e = pd.read_csv("holdout_out/positive_controls/edits.csv").dropna(subset=["text"])
    e = e[e.case_id.isin(pcs) & e.edit.isin(["strengthen", "weaken"])]
    e = e.rename(columns={"edit": "persona_label"}).assign(kind="pc")
    return pd.concat([g, o[["case_id", "persona_label", "text", "kind"]],
                      e[["case_id", "persona_label", "text", "kind"]]], ignore_index=True)


def tid(t: str) -> str:
    return hashlib.blake2b(t.encode(), digest_size=12).hexdigest()


def cmd_rewrite(smoke: bool, dry_run: bool):
    src = source_texts(smoke)
    n = len(src) * len(STYLES)
    logger.info("%d source texts x %d styles = %d rewrites (est. $%.2f)", len(src), len(STYLES), n, n * 0.00036)
    if dry_run:
        return
    Path("cache/style_rewrite").mkdir(parents=True, exist_ok=True)
    llm = hp.make_client([REWRITER], 30)
    try:
        from instructor.processing.schema import generate_openai_schema
        StyleRewriteResponse.model_json_schema()
        generate_openai_schema(StyleRewriteResponse)
    except Exception:
        pass
    reqs, keys, meta = [], [], []
    for r in src.itertuples(index=False):
        for st, instr in STYLES.items():
            user = f"Target style: {instr}\n\nComplaint:\n{r.text}"
            reqs.append(LLMRequest(request_id=LLMRequest.new_id(), case_id=r.case_id, model=REWRITER, task="style_rewrite",
                                   prompt_system=STYLE_SYSTEM, prompt_user=user, expected_schema=StyleRewriteResponse,
                                   variant=st))
            keys.append(user)
            meta.append((r.case_id, r.persona_label, r.kind, st))
    resp = llm.execute_batch(reqs, keys, desc="Style standardisation rewrites")
    rows = [dict(case_id=c, persona_label=p, kind=k, style=st,
                 text=(x.response_parsed.rewritten_complaint if x.success and x.response_parsed else None))
            for (c, p, k, st), x in zip(meta, resp)]
    o = out_dir(smoke)
    o.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(o / "rewrites.csv", index=False)
    (o / "usage_rewrite.json").write_text(json.dumps(llm.usage_summary(), indent=2, default=str))
    failed = int(df.text.isna().sum())
    logger.info("rewrites ok %d / %d", len(df) - failed, len(df))
    if failed:
        raise SystemExit(3)


def cmd_decide(smoke: bool):
    o = out_dir(smoke)
    rw = pd.read_csv(o / "rewrites.csv").dropna(subset=["text"])
    llm = hp.make_client([BANK], 30)
    frame = pd.DataFrame(dict(case_id=rw.case_id, rewrite_model=BANK,
                              persona_label=rw["persona_label"] + "||" + rw["style"], text=rw["text"]))
    d = hp.decide(llm, frame, BANK)
    d[["persona_label", "style"]] = d.variant.str.split(r"\|\|", expand=True)
    d.to_csv(o / "decisions.csv", index=False)
    (o / "usage_decide.json").write_text(json.dumps(llm.usage_summary(), indent=2, default=str))
    failed = int(d.tier.isna().sum())
    logger.info("decisions ok %d / %d", len(d) - failed, len(d))
    if failed:
        raise SystemExit(3)


def cmd_score(smoke: bool):
    o = out_dir(smoke)
    rw = pd.read_csv(o / "rewrites.csv").dropna(subset=["text"])
    df = pd.DataFrame(dict(case_id=rw.case_id, source=rw.kind, rewrite_model=rw["style"],
                           persona_label=rw.persona_label, text=rw.text))
    df["text_id"] = [tid(t) for t in df.text]
    q = jarms.order_questions(list(jarms.ORDERS), with_persona=False)
    attr_ids = set(df.loc[(df.rewrite_model == "A_plain") | (df.source == "pc"), "text_id"])
    qa = dict(q, **jattr.build_attribute_questions())
    out = o / "jev"
    for sub in ("tier_only", "tier_attr"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    jp.build_questions = lambda: q
    r1 = jp.score_texts(df[~df.text_id.isin(attr_ids)], out / "tier_only", 16)
    jp.build_questions = lambda: qa
    r2 = jp.score_texts(df[df.text_id.isin(attr_ids)], out / "tier_attr", 16)
    resp = {**r1, **r2}
    tiers = jarms.flatten(df, {k: {**v, "answers": {a: b for a, b in v["answers"].items() if a.startswith("tier_")}} for k, v in resp.items()})
    attrs = jattr.flatten(df[df.text_id.isin(attr_ids)], {k: {**v, "answers": {a: b for a, b in v["answers"].items() if a.startswith("attr_")}} for k, v in r2.items()})
    s = tiers.merge(attrs[["text_id"] + [c for c in attrs.columns if c.endswith("_E")]], on="text_id", how="left")
    s.to_csv(out / "scores.csv", index=False)
    logger.info("scored %d texts; tokens %d ($%.2f)", len(s), int(tiers.input_tokens.sum()), tiers.input_tokens.sum() * jp.PRICE_PER_M_INPUT / 1e6)


# ---------------------------------------------------------------------------
# Frozen analysis
# ---------------------------------------------------------------------------

CONTR = {"age": ("old", "young"), "edu": ("college", "high_school"), "ses": ("high", "low")}


def parts(lbl):
    return {"age": lbl.split("_")[0], "edu": "high_school" if "_high_school_" in lbl else "college", "ses": lbl.split("_")[-1]}


def e_tier(df, order="o012"):
    return df[f"p1_{order}"] + 2 * df[f"p2_{order}"]


def arm_tables(smoke: bool):
    """Return complaint x persona matrices of outcomes for M0, M1, M2 (GPT tier, Jev E[tier])."""
    o = out_dir(smoke)
    main, _ = case_sets(smoke)
    personas = sorted(pd.read_csv("holdout_out/rewrites_n1000.csv").persona_label.unique())
    # M0
    d0 = pd.read_csv("holdout_out/decisions_n1000.csv")
    d0 = d0[(d0.model == BANK) & d0.case_id.isin(main)].rename(columns={"variant": "persona_label"})
    j0 = pd.read_csv("holdout_out/jev/scores.csv")
    j0 = j0[(j0.source == "guise") & (j0.rewrite_model == BANK) & j0.case_id.isin(main)].assign(e=lambda x: e_tier(x))
    # M1 / M2
    d = pd.read_csv(o / "decisions.csv")
    d = d[d.case_id.isin(main) & ~d.persona_label.isin(["original", "strengthen", "weaken"])]
    j = pd.read_csv(o / "jev" / "scores.csv")
    j = j[(j.source == "guise") & j.case_id.isin(main)].assign(e=lambda x: e_tier(x))
    j = j.rename(columns={"rewrite_model": "style"})

    def mat(frame, val, styles=None):
        f = frame if styles is None else frame[frame["style"].isin(styles)]
        w = f.groupby(["case_id", "persona_label"])[val].mean().unstack()
        return w.reindex(index=main, columns=personas)

    return {
        "gpt": {"M0": mat(d0, "tier"), "M1": mat(d, "tier", ["A_plain"]), "M2": mat(d, "tier", list(STYLES))},
        "jev": {"M0": mat(j0, "e"), "M1": mat(j, "e", ["A_plain"]), "M2": mat(j, "e", list(STYLES))},
    }, personas


def contrast_vec(W: pd.DataFrame, dim):
    hi, lo = CONTR[dim]
    lv = np.array([parts(p)[dim] for p in W.columns])
    X = W.to_numpy()
    return np.nanmean(X[:, lv == hi], 1) - np.nanmean(X[:, lv == lo], 1)


def mpad(W: pd.DataFrame):
    """Per-complaint mean absolute pairwise difference across guises (arbitrariness, any outcome scale)."""
    X = W.to_numpy()
    iu = np.triu_indices(X.shape[1], 1)
    return np.nanmean(np.abs(X[:, iu[0]] - X[:, iu[1]]), 1)


def boot(vals_fn, C, rng):
    return np.array([vals_fn(rng.integers(0, C, C)) for _ in range(N_BOOT)])


def ci(b):
    return [float(np.nanquantile(b, .025)), float(np.nanquantile(b, .975))]


def cmd_analyse(smoke: bool):
    rng = np.random.default_rng(SEED)
    T, personas = arm_tables(smoke)
    res = {"n_complaints": len(T["gpt"]["M0"]), "profiling": {}, "arbitrariness": {}}
    for sysn, arms in T.items():
        keep = np.ones(len(arms["M0"]), bool)
        for W in arms.values():
            keep &= W.notna().all(1).to_numpy()
        arms = {k: v[keep] for k, v in arms.items()}
        C = int(keep.sum())
        for dim in CONTR:
            cv = {k: contrast_vec(W, dim) for k, W in arms.items()}
            r = {k: {"gap": float(v.mean()), "ci95": ci(boot(lambda ix, v=v: v[ix].mean(), C, rng))} for k, v in cv.items()}
            for k in ("M1", "M2"):
                dv = cv["M0"] - cv[k]
                r[f"reduction_M0_minus_{k}"] = {"estimate": float(dv.mean()), "ci95": ci(boot(lambda ix, dv=dv: dv[ix].mean(), C, rng))}
            res["profiling"][f"{sysn}:{dim}"] = r
        mp = {k: mpad(W) for k, W in arms.items()}
        r = {k: {"mpad": float(v.mean()), "ci95": ci(boot(lambda ix, v=v: v[ix].mean(), C, rng))} for k, v in mp.items()}
        for a, b_ in (("M0", "M1"), ("M1", "M2"), ("M0", "M2")):
            dv = mp[a] - mp[b_]
            r[f"{a}_minus_{b_}"] = {"estimate": float(dv.mean()), "ci95": ci(boot(lambda ix, dv=dv: dv[ix].mean(), C, rng))}
        res["arbitrariness"][sysn] = r
    res["fact_sensitivity"] = fact_sensitivity(smoke, rng)
    res["fidelity"] = fidelity(smoke)
    # Pre-registered verdicts (see prereg/style_mitigation.md)
    p = res["profiling"]
    res["MH1_profiling_removed_M1"] = {d: bool(p[f"gpt:{d}"]["reduction_M0_minus_M1"]["ci95"][0] > 0 and
                                              abs(p[f"gpt:{d}"]["M1"]["gap"]) <= 0.5 * abs(p[f"gpt:{d}"]["M0"]["gap"]))
                                       for d in ("edu", "ses")}
    res["MH1_supported"] = all(res["MH1_profiling_removed_M1"].values())
    a = res["arbitrariness"]
    res["MH2_arbitrariness_M1_lt_M0"] = {s: bool(a[s]["M0_minus_M1"]["ci95"][0] > 0) for s in ("gpt", "jev")}
    res["MH2_supported"] = all(res["MH2_arbitrariness_M1_lt_M0"].values())
    fs = res["fact_sensitivity"]
    res["MH3_fact_retention"] = {k: bool(v["retention_ci95"][0] >= 0.5) for k, v in fs.items() if "retention_ci95" in v}
    res["MH3_supported"] = all(res["MH3_fact_retention"].values()) if res["MH3_fact_retention"] else None
    res["MH5_M2_lt_M1"] = {s: bool(a[s]["M1_minus_M2"]["ci95"][0] > 0) for s in ("gpt", "jev")}
    o = out_dir(smoke)
    (o / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k.startswith("MH")}, indent=2))


def fact_sensitivity(smoke: bool, rng):
    """Fact span = mean(outcome | strengthen) - mean(outcome | weaken), paired by complaint, for M0 (texts as edited)
    and for M1/M2 (edited texts standardised); retention = span_Mk / span_M0."""
    o = out_dir(smoke)
    _, pcs = case_sets(smoke)
    d = pd.read_csv(o / "decisions.csv")
    d = d[d.case_id.isin(pcs) & d.persona_label.isin(["strengthen", "weaken"])]
    pc_dec = pd.read_csv("holdout_out/positive_controls/decisions.csv")
    pc_dec = pc_dec[(pc_dec.model == BANK) & pc_dec.case_id.isin(pcs)].rename(columns={"edit": "persona_label"})
    j = pd.read_csv(o / "jev" / "scores.csv")
    j = j[(j.source == "pc") & j.case_id.isin(pcs) & j.persona_label.isin(["strengthen", "weaken"])].assign(e=lambda x: e_tier(x))
    pj = pd.read_csv("holdout_out/positive_controls/jev/scores.csv")
    pj = pj[pj.case_id.isin(pcs) & pj.persona_label.isin(["strengthen", "weaken"])].assign(e=lambda x: e_tier(x))
    out = {}
    for sysn, base, arm, val, stylecol in (("gpt", pc_dec, d, "tier", "style"), ("jev", pj, j, "e", "rewrite_model")):
        def span(frame, styles=None):
            f = frame if styles is None else frame[frame[stylecol].isin(styles)]
            w = f.groupby(["case_id", "persona_label"])[val].mean().unstack().reindex(columns=["strengthen", "weaken"])
            return (w["strengthen"] - w["weaken"]).reindex(pcs)
        s0 = span(base)
        for k, styles in (("M1", ["A_plain"]), ("M2", list(STYLES))):
            sk = span(arm, styles)
            both = pd.concat([s0.rename("a"), sk.rename("b")], axis=1).dropna()
            A, B = both.a.to_numpy(), both.b.to_numpy()
            C = len(A)
            b = boot(lambda ix: B[ix].mean() / A[ix].mean() if A[ix].mean() else np.nan, C, rng) if C else np.array([np.nan])
            out[f"{sysn}:{k}"] = {"span_M0": float(A.mean()) if C else None, f"span_{k}": float(B.mean()) if C else None,
                                  "retention": float(B.mean() / A.mean()) if C and A.mean() else None,
                                  "retention_ci95": ci(b), "n": C}
    return out


def fidelity(smoke: bool):
    """Descriptive: content attributes should be preserved by standardisation; delivery attributes should collapse
    across personas. Within-complaint persona eta^2 for M0 guises vs M1 (A_plain) texts."""
    o = out_dir(smoke)
    main, _ = case_sets(smoke)
    a0 = pd.read_csv("holdout_out/attributes/scores.csv")
    a0 = a0[(a0.source == "guise") & (a0.rewrite_model == BANK) & a0.case_id.isin(main)]
    j = pd.read_csv(o / "jev" / "scores.csv")
    a1 = j[(j.source == "guise") & (j.rewrite_model == "A_plain") & j.case_id.isin(main)]

    def eta(frame, col):
        W = frame.pivot_table(index="case_id", columns="persona_label", values=col).dropna().to_numpy()
        if W.size == 0:
            return float("nan")
        rm, cm, gm = W.mean(1, keepdims=True), W.mean(0, keepdims=True), W.mean()
        ss_p = W.shape[0] * ((cm - gm) ** 2).sum()
        ss_r = ((W - rm - cm + gm) ** 2).sum()
        return float(ss_p / (ss_p + ss_r)) if ss_p + ss_r > 0 else float("nan")
    out = {}
    for x in jattr.ATTRIBUTES:
        col = f"{x}_E"
        m0 = a0.groupby("case_id")[col].mean()
        m1 = a1.groupby("case_id")[col].mean() if col in a1 else pd.Series(dtype=float)
        jj = pd.concat([m0.rename("m0"), m1.rename("m1")], axis=1).dropna()
        out[x] = {"family": jattr.FAMILY[x], "eta2_persona_M0": eta(a0, col), "eta2_persona_M1": eta(a1, col) if col in a1 else None,
                  "complaint_level_corr_M0_M1": float(jj.m0.corr(jj.m1)) if len(jj) > 2 else None,
                  "mean_shift_M1_minus_M0": float((jj.m1 - jj.m0).mean()) if len(jj) else None}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["rewrite", "decide", "score", "analyse"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"rewrite": lambda: cmd_rewrite(a.smoke, a.dry_run), "decide": lambda: cmd_decide(a.smoke),
     "score": lambda: cmd_score(a.smoke), "analyse": lambda: cmd_analyse(a.smoke)}[a.cmd]()


if __name__ == "__main__":
    main()
