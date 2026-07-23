# Manuscript (SoftwareX OSP)

| File | Role |
|------|------|
| [`paper.tex`](paper.tex) | Original Software Publication manuscript |
| [`references.bib`](references.bib) | Bibliography |
| [`elsarticle.cls`](elsarticle.cls) | Elsevier document class |
| [`elsarticle-num.bst`](elsarticle-num.bst) | Numbered BibTeX style |
| [`figures/`](figures/) | Optional illustration PDFs |

Compile:

```bash
pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper
```

Describes `ctc-discrete-monodromy` **v0.1.0** — https://github.com/mauramyrez/ctc-discrete-monodromy
