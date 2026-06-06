# Overleaf Instructions

Upload the files in this directory to Overleaf.

- Main diagram file: `methodology_diagrams.tex`
- Compiler: **pdfLaTeX**
- TeX Live: use Overleaf's latest available version

`methodology_diagrams.tex` produces a four-page PDF:

1. Real-dataset curation and deterministic 10% selection.
2. Dual-domain temporal-frequency model.
3. Training objectives and NVIDIA L4 optimization.
4. Leakage-aware evaluation and exported artifacts.

The diagrams are vector graphics and can be inserted into an ACM paper with:

```latex
\includegraphics[width=\linewidth,page=1]{methodology_diagrams.pdf}
```

Use `icca_paper_skeleton.tex` as the double-blind ACM paper starting point.
Before submission, verify the exact ACM rights and CCS blocks required by the
conference template.

`acmart.cls` and `ACM-Reference-Format.bst` were extracted from ICCA's official
2026 LaTeX archive. The downloaded archive had SHA-256:
`2078369af92a0c626fc98d009a11ffbec5edeb7ac454a4cc460ad190019373b0`.
