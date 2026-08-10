# Manuscript (Classical and Quantum Gravity)

Physics manuscript targeting **Classical and Quantum Gravity** (IOP Publishing), using the official `iopjournal` submission class.

| File | Role |
|------|------|
| [`paper-cqg.tex`](paper-cqg.tex) / [`paper-cqg.pdf`](paper-cqg.pdf) | **Primary CQG manuscript** (`iopjournal` + `iopart-num`) |
| [`iopjournal.cls`](iopjournal.cls) | IOP submission class |
| [`iopart-num.bst`](iopart-num.bst) | Numeric IOP bibliography style |
| [`references.bib`](references.bib) | Bibliography |
| [`derived_equations_summary.tex`](derived_equations_summary.tex) | Compact Einstein-tensor summary (optional `\input`) |
| [`energy_conditions_summary.tex`](energy_conditions_summary.tex) | Compact NEC/WEC summary (optional `\input`) |
| [`appendix_tensor_components.tex`](appendix_tensor_components.tex) | Appendix: equatorial Einstein tensor |
| [`appendix_energy_conditions.tex`](appendix_energy_conditions.tex) | Appendix: NEC/WEC contractions |
| [`figures/`](figures/) | Illustration PDFs |
| [`path3_continuum_report.md`](path3_continuum_report.md) | Path-3 continuum spectral campaign |

Compile from this directory:

```bash
pdflatex paper-cqg
bibtex paper-cqg
pdflatex paper-cqg
pdflatex paper-cqg
```

Requires a TeX distribution with `amsmath` / `amsthm` (MiKTeX / TeX Live). The class file `iopjournal.cls`, style `iopart-num.bst`, and `orcid.pdf` must sit beside `paper-cqg.tex`.

## Path-3 continuum stress (unnormalized \(\widetilde{K}\))

Numerical campaign (`RenormPolicy::None`, \(\nu=0\), \(N\le 1024\), \(m^2\to 0\)):

- Report: [`path3_continuum_report.md`](path3_continuum_report.md)
- Regenerate: `cargo test --release path3_extended -- --nocapture`
