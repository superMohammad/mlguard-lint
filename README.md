# MLGuard

A lightweight static analyzer for **silent methodological errors** in scikit-learn ML workflows —
data leakage, invalid evaluation, and unsound train/test splits that run without raising an error
but quietly inflate your results.

Zero runtime dependencies (Python standard library only). MLGuard is a heuristic surfacing tool,
not a prover: every diagnostic carries a `confidence` (high / medium / low) and is meant as a
code-review prompt, not a proof of a bug.

## Install

```bash
python -m pip install -e .          # from the repo root
```

## Usage

```bash
mlguard scan_me.ipynb               # a single notebook or .py file
mlguard src/                        # a directory (scanned recursively)
mlguard notebooks/ --summary        # compact, one line per file (large scans)
mlguard . --fail-on critical        # exit code 2 on any critical finding (CI gate)
mlguard scan_me.ipynb --json out.json   # machine-readable output
mlguard scan_me.ipynb --no-color    # plain text (colors auto-off when piped)
```

`PATH` may be a `.ipynb`, a `.py`, or a directory. With `--fail-on`, the process exits `2` once a
finding at or above the given severity is present — useful as a CI gate.

## Rules

- **MLG001** — Transformer fitted before train/test split
- **MLG002** — Preprocessing outside cross-validation
- **MLG003** — Model evaluated on training data
- **MLG004** — Resampling before split/CV or outside an imblearn Pipeline
- **MLG005** — Classification split without `stratify`
- **MLG006** — Missing `random_state`
- **MLG007** — Possible group/entity leakage
- **MLG008** — `GridSearchCV` `best_score_` reported as final performance
- **MLG009** — Ordinal/label encoding of a nominal feature
- **MLG010** — Test set reused multiple times
- **MLG011** — Possible target/mean encoding without cross-fitting
- **MLG012** — No independent test set
- **MLG013** — Resampling applied to the test/validation set
- **MLG014** — Random split/CV on time-ordered data
- **MLG015** — Target column included in features
- **MLG016** — Transformer re-fit on the test set
- **MLG017** — Probability metric given hard predictions
- **MLG018** — Feature built from dataset-wide statistics before split
- **MLG019** — Misleading micro-average on imbalanced multiclass
- **MLG020** — ID/source column used as a feature
- **MLG021** — Rows duplicated/upsampled before split

## How it works

The scanner concatenates a notebook's code cells into a single module, parses it with the standard
`ast` module, performs one walk to collect calls, assignments, model `fit` arguments, and id-like
string constants, then runs ordered rule blocks that emit diagnostics. Because the whole notebook is
analyzed as one program (no per-cell execution semantics), cross-cell dataflow is approximate and
line numbers are notebook-global.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_rules.py -q
```

Each rule has a synthetic fixture under `tests/notebooks/` plus clean controls that must stay silent.

## Limitation

This is a heuristic static analyzer. It is useful for surfacing risks, not for proving that every
warning is a real bug. Treat diagnostics as prompts for a closer look during code review.

## License

MIT — see [LICENSE](LICENSE).

The methodology behind the rules is documented in
[docs/Silent-Methodological-Errors-in-scikit-learn-Workflows.pdf](docs/Silent-Methodological-Errors-in-scikit-learn-Workflows.pdf).
