# OpenReview submission form: FAccT 2027 (field-by-field guidance)

The form registers the abstract and takes the PDF in one place. **The PDF field is optional now.** Submit the form with
title, abstract and the other fields by the abstract deadline, then edit the submission to upload the PDF before the
full-paper deadline. Title, abstract and focus areas can be edited until then; the author list cannot.

| Deadline | AoE (official) | Brisbane (AEST) |
|---|---|---|
| Abstract registration (this form, without PDF) | 27 Oct 2026, 23:59 | **Wed 28 Oct, 21:59** |
| Full paper (upload the PDF by editing the submission) | 3 Nov 2026, 23:59 | **Wed 4 Nov, 21:59** |

The form header shows "Deadline: Oct 29, 2026, 12:00:00 AM", which is later than the official abstract deadline, so treat
the official CFP times above as binding.

---

**Title** (required)
`Same Facts, Different Remedy: Linguistic Profiling and Arbitrariness in LLM and System One Decisions`

**Authors** (required)
Only you, ~Colin_Priest1, with your institution added. Anyone you add must already have an OpenReview profile, and the
list cannot be changed after the abstract deadline.

**Keywords** (required; the field asks for a *comma*-separated list, so do not use semicolons)
`linguistic profiling, arbitrariness, matched guise, large language models, fairness audit, consumer complaints, pre-registration`

**TL;DR** (optional; one sentence)
`Matched-guise audits of three LLM remedy-decision systems show that small average gaps by writing style coexist with substantial individual instability, that smoother probabilistic scores barely stabilise actual decisions, and that style standardisation reduces some gaps while reversing others.`

**Abstract** (required; plain text; TeX is only rendered inside `$...$`, so `%` and other symbols are fine)
Paste the text below. It matches the current paper's abstract and is about 2,000 characters, within the 5,000 limit.

> Large language models are increasingly used to triage and resolve consumer complaints, and the way complainants write varies systematically with age, education and socioeconomic status (SES). We audit three remedy-decision systems on sociolinguistic rewrites of US consumer financial complaints, each instructed to preserve the facts: two generative LLMs that return a single label (GPT-4o-mini, GPT-5-mini) and a probabilistic "System One" model that returns a distribution over remedies (Jev). Using hypotheses pre-registered and tested once on 1,000 held-out complaints (72,000 rewrites), we separate two outcomes that single-label audits conflate. At the group scale, older, college-educated and higher-SES writing styles receive slightly more generous remedies, but every held-out gap is below 0.5 percentage points of the monetary-remedy rate. At the individual scale, two style versions of the same complaint receive different monetary decisions 2.6–6.3% of the time; in a variance-component model, persona components shared across the two rewrite models account for an estimated 1.6–1.7% of within-complaint variation. In an independent audit by two LLMs from outside the audited systems, individual auditors flagged a material fact change in 3.3% and 11.7% of rewrites, and instability was similar among rewrites that passed this screen. The group gaps are associated mainly with delivery (vocabulary, grammar, politeness) rather than content, and positive controls show that all three systems respond 49–387× more strongly to remedy-relevant facts than to style. Graded probabilistic outputs make scores smoother but, at matched monetary volume, barely change how often actual remedy decisions disagree. In a pre-registered mitigation, rewriting each complaint into one standard style reduced GPT-4o-mini's education and SES gaps by 94% and 87% (point estimates) while retaining fact sensitivity, but it did not detectably stabilise individual decisions and it reversed two of the probabilistic model's gaps. We argue that audits should report average disparities and individual instability as separate outcomes, evaluate both on the decisions people actually receive, and audit any mitigation with the same design.

If you keep your own edited version instead, check it against the paper: 1.6–1.7% (not just 1.6%), and "passed this screen" rather than "neither flagged".

**PDF** (optional now)
Leave it empty for the abstract registration. Before the full-paper deadline, edit the submission and upload the anonymous
PDF. That PDF must use the ACM template `\documentclass[manuscript,screen,review,anonymous]{acmart}`, single column, with at
most 14 pages plus one page of endmatter, followed by references and appendices in the same file.

**Profile Confirmed** (required)
Tick **Yes** only once your OpenReview profile has current emails, work history (employer and dates) and relationships
(collaborators and colleagues from the last two years). These details are used to detect conflicts of interest.

**Reviewer Sign Up** (required)
Tick only after you have **submitted** the separate reviewer sign-up form. You are the only author, so the confirmation
covers you alone.

**Primary Focus Area** (required)
**Evaluations and evaluation practices.** The paper is an audit and evaluation of LLM decision systems.

**Secondary Focus Area** (optional)
**System development and deployment** is reasonable, because the mitigation and decision-rule results concern deployed
systems. You can also leave it blank.

**Generative AI Disclosure Statement** (required)
Tick. The statement is in the paper's endmatter (§7) and names Claude Opus 5.5 and ChatGPT GPT-6 Sol.

**Non Archival**
**Leave unticked**, so the paper is archival and appears in the ACM proceedings. This cannot be changed after registration.
Tick it only if you plan to publish a version in a journal.

**Email Sharing** (required) and **Data Release** (required)
Select the authorise options. Both are conditions of submission.

**License** (required)
Choose **CC BY 4.0** if it is offered. ACM publishes FAccT papers open access, and CC BY is the standard choice. This
licence applies to the OpenReview record; the camera-ready licence is handled separately with ACM.

**Readers / Signatures**
Leave the defaults: readers are "ACM FAccT 2027 Conference" and the authors, and the signature is your profile.

**Submit**
Then check that the submission appears under your OpenReview tasks/console. You should also receive a confirmation email.
