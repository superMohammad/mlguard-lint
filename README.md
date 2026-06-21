# MLGuard

A lightweight static analyzer for **silent methodological errors** in scikit-learn ML workflows —
data leakage, invalid evaluation, and unsound train/test splits that run without raising an error
but quietly inflate your results.

Zero runtime dependencies (Python standard library only). MLGuard is a heuristic surfacing tool,
not a prover: every diagnostic carries a `confidence` (high / medium / low) and is meant as a
code-review prompt, not a proof of a bug.

## Install

```bash
pip install mlguard-lint
```

That's it — same command on Linux, macOS, and Windows. Requires Python 3.9+.

## Use it

Point it at a single file or a whole folder:

```bash
mlguard-lint notebook.ipynb        # scan one notebook or .py file
mlguard-lint src/                  # scan a directory (recursive)
```

By default you get one clean line per issue:

```
mlguard-lint — notebook.ipynb

notebook.ipynb
  ✗ line 5    MLG001  Transformer fitted before split
  ⚠ line 6    MLG006  Missing random_state
  ⚠ line 6    MLG005  Classification split without stratify

3 issue(s) in 1 of 1 file(s) · 1 critical, 2 warnings
Tip: add --explain for why each matters and how to fix.
```

### Options

```bash
mlguard-lint notebook.ipynb --explain    # add the code, why it matters, and how to fix it
mlguard-lint src/ --summary              # one line per file (handy for large folders)
mlguard-lint . --fail-on critical        # exit code 2 on any critical finding (CI gate)
mlguard-lint . --json out.json           # machine-readable output
mlguard-lint notebook.ipynb --no-color   # plain text (colors auto-off when piped)
```

If the `mlguard-lint` command isn't on your PATH, the module form always works:

```bash
python -m mlguard_lint notebook.ipynb
```

> **Windows:** use **Windows Terminal** or **PowerShell** so the `✗ ⚠` symbols and colors render
> correctly. On the legacy `cmd.exe` console, pass `--no-color` (or run `chcp 65001` once for UTF-8).

## Rules

- **MLG001** — Transformer fitted before split
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

## Limitation

This is a heuristic static analyzer. It is useful for surfacing risks, not for proving that every
warning is a real bug. Treat diagnostics as prompts for a closer look during code review.

## Development

```bash
git clone <repo> && cd mlguard
pip install -e ".[dev]"          # editable install + pytest/build/twine
python -m pytest tests/test_rules.py -q
```

Each rule has a synthetic fixture under `tests/notebooks/` plus clean controls that must stay silent.

## License

MIT — see [LICENSE](LICENSE).

The methodology behind the rules is documented in
[docs/Silent-Methodological-Errors-in-scikit-learn-Workflows.pdf](docs/Silent-Methodological-Errors-in-scikit-learn-Workflows.pdf).
