import argparse, json, sys, pathlib
from collections import Counter, defaultdict
from .scanner import scan_path

# severity -> (symbol, ansi color, sort rank)
_SEV = {
    "critical": ("✗", "\033[31m", 0),  # ✗ red
    "warning": ("⚠", "\033[33m", 1),   # ⚠ yellow
    "info": ("ℹ", "\033[34m", 2),      # ℹ blue
}
_RESET, _BOLD, _DIM, _GREEN = "\033[0m", "\033[1m", "\033[2m", "\033[32m"

_HEURISTIC_NOTE = "MLGuard is a heuristic surfacing tool — treat findings as review prompts, weighted by confidence."


def _sev_meta(severity):
    return _SEV.get(severity, ("•", "", 3))


def _count_targets(path):
    p = pathlib.Path(path)
    if p.is_dir():
        return len(list(p.rglob("*.ipynb")) + list(p.rglob("*.py")))
    return 1 if p.exists() else 0


def _tally(diags):
    """'X critical, Y warnings, Z info' for the severities actually present."""
    sev = Counter(d.severity for d in diags)
    labels = {"critical": "critical", "warning": "warnings", "info": "info"}
    parts = [f"{sev[s]} {labels[s]}" for s in ("critical", "warning", "info") if sev.get(s)]
    return ", ".join(parts)


def _render_text(diags, path, use_color, mode):
    """mode: 'default' (one line per issue), 'explain' (full detail), 'summary' (one line per file)."""
    def paint(text, color):
        return f"{color}{text}{_RESET}" if (use_color and color) else text

    total_files = _count_targets(path)
    by_file = defaultdict(list)
    for d in diags:
        by_file[d.path].append(d)
    lines = [paint(f"mlguard-lint — {path}", _BOLD)]

    if not diags:
        lines.append(paint(f"✓ No issues found in {total_files} file(s).", _GREEN))
        return "\n".join(lines)

    footer = f"{len(diags)} issue(s) in {len(by_file)} of {total_files} file(s) · {_tally(diags)}"

    # --- summary: one line per file ---
    if mode == "summary":
        lines.append("")
        for fpath in sorted(by_file):
            fsev = Counter(d.severity for d in by_file[fpath])
            badge = " ".join(
                paint(f"{_sev_meta(s)[0]}{fsev[s]}", _sev_meta(s)[1])
                for s in ("critical", "warning", "info") if fsev.get(s)
            )
            lines.append(f"  {badge:<24} {fpath}")
        lines.append("")
        lines.append(footer)
        return "\n".join(lines)

    # --- default & explain: grouped by file, severity then line order ---
    for fpath in sorted(by_file):
        lines.append("")
        lines.append(paint(fpath, _BOLD))
        for d in sorted(by_file[fpath], key=lambda x: (_sev_meta(x.severity)[2], x.line)):
            sym, col, _ = _sev_meta(d.severity)
            lines.append(f"  {paint(sym, col)} line {d.line:<4} {d.rule_id}  {d.title}")
            if mode == "explain":
                if d.code:
                    lines.append(f"        code: {d.code.strip()[:120]}")
                lines.append(f"        why:  {d.why_it_matters}")
                lines.append(f"        fix:  {d.suggested_fix}")
                lines.append(paint(f"        [{d.severity} · {d.confidence}]", _DIM))

    lines.append("")
    lines.append(footer)
    if mode == "default":
        lines.append(paint("Tip: add --explain for why each matters and how to fix.", _DIM))
    else:
        lines.append(paint(_HEURISTIC_NOTE, _DIM))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="mlguard-lint",
        description="Static linter for silent methodological errors in scikit-learn workflows.",
    )
    p.add_argument("path", help="Notebook, Python file, or directory to scan")
    p.add_argument("--explain", action="store_true", help="Show why each finding matters, the code, and how to fix it")
    p.add_argument("--summary", action="store_true", help="Compact one-line-per-file output (for large/directory scans)")
    p.add_argument("--json", dest="json_out", help="Write diagnostics JSON to this path")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--fail-on", choices=["none", "warning", "critical"], default="none")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors in text output")
    args = p.parse_args(argv)

    diags = scan_path(args.path)
    data = [d.to_dict() for d in diags]
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(data, indent=2), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(data, indent=2))
    else:
        mode = "summary" if args.summary else "explain" if args.explain else "default"
        use_color = sys.stdout.isatty() and not args.no_color
        print(_render_text(diags, args.path, use_color, mode))

    if args.fail_on == "critical" and any(d.severity == "critical" for d in diags):
        return 2
    if args.fail_on == "warning" and any(d.severity in {"critical", "warning"} for d in diags):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
