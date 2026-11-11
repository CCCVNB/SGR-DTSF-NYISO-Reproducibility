# Draft response — Reviewer #1, Comment 2

**Do not submit this draft until the repository has been pushed to a public host, the URL has been tested without authentication, and the final URL has replaced the placeholder below.**

Thank you for raising this important concern regarding implementation transparency and reproducibility.

In response, we have prepared a selective public source-code repository for the NYISO H288→P288 experiments. It contains the SGR-DTSF implementation, the proposed-model ablation code, and the author-implemented baseline models used in the NYISO comparisons, including persistence, linear, fusion, MLP, Transformer, temporal, and graph-based baselines.

We clarify that these baseline models are task-adapted author implementations rather than verbatim reproductions of the original authors' official repositories. The repository provides a method-to-source mapping and implementation-fidelity notes to make the NYISO comparison implementations inspectable.

The public release is intentionally limited to NYISO code. It does not include GEFCom2012 source code, raw datasets, processed arrays, fitted scalers, trained checkpoints, or raw-data preprocessing and dataset-generation programs. Instead, it documents the preprocessed NYISO input interface required by the model and baseline runners. Accordingly, the repository is a selective algorithm-code release rather than a complete archive of all datasets and historical experimental artifacts.

The public repository is available at:

**[INSERT VERIFIED PUBLIC REPOSITORY URL]**

We sincerely appreciate the reviewer's suggestion, which has helped us improve the transparency and accessibility of the NYISO algorithm and comparative implementations.
