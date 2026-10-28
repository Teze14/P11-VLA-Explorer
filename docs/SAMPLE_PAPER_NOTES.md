# Lessons from the professor's six sample papers

Sources are the actual PDFs and links downloaded from the course's read-only Overleaf project. These are design takeaways, not claims that P11 replicated the papers' evaluations.

| Course sample | Relevant design idea | Application in P11 |
|---|---|---|
| Transformer Explainer (Cho et al., CHI 2026) | Connect a model overview to lower-level mathematical operations and direct hyperparameter experiments | Five-stage trace, tensor dimensions, cosine table, temperature slider |
| Diffusion Explainer (Lee et al., arXiv:2305.03509) | Let users compare prompt variants and move between levels of abstraction | Save two instruction-conditioned outputs on the same image; guided explanation |
| ExplainML folder: WebSHAP (Wang and Chau, WWW Companion 2023) | Client-side model execution and interactive explanations reduce server dependence | Browser-local model execution and explicit data-handling explanation; P11 does not implement SHAP |
| UNIPO (arXiv:2605.11549) | Coordinate overview, detailed inspection, and comparison using consistent notation | Shared region IDs and weights across heatmap, score table, target computation, and saved comparisons |
| VISGRADER (Hull et al., arXiv:2310.12347) | Verify interactive behavior and the underlying values, not only the screenshot | Automated invariance checks for opacity, gain, normalization, caching, and export |
| Orca (Jiang and Xia, CHI 2026) | Preserve user agency and source context while supporting comparisons | Learner predicts and chooses interventions; inspectable provenance and retained comparison snapshots |

Links: https://poloclub.github.io/transformer-explainer/ ; https://poloclub.github.io/diffusion-explainer/ ; https://poloclub.github.io/webshap/ ; https://poloclub.github.io/unipo/ ; https://github.com/poloclub/visgrader ; https://orca.jiang.pl .

The main course report uses the actual supplied `vgtc` class and its bibliography style. No findings from these papers are presented as P11 results. Their learner studies cannot substitute for evaluating this toolkit.
