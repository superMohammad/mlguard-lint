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


def _sev_meta(severity):
    return _SEV.get(severity, ("•", "", 3))


def _count_targets(path):
    p = pathlib.Path(path)
    if p.is_dir():
        return len(list(p.rglob("*.ipynb")) + list(p.rglob("*.py")))
    return 1 if p.exists() else 0


def _render_text(diags, path, use_color, summary_only):
    def paint(text, color):
        return f"{color}{text}{_RESET}" if (use_color and color) else text

    total_files = _count_targets(path)
    by_file = defaultdict(list)
    for d in diags:
        by_file[d.path].append(d)
    sev = Counter(d.severity for d in diags)
    lines = []

    # --- header ---
    lines.append(paint(f"MLGuard — {path}", _BOLD))
    if not diags:
        lines.append(paint(f"✓ No issues found in {total_files} file(s).", _GREEN))
        return "\n".join(lines)
    tally = "  ".join(
        paint(f"{_sev_meta(s)[0]} {s}: {sev.get(s, 0)}", _sev_meta(s)[1])
        for s in ("critical", "warning", "info") if sev.get(s)
    )
    lines.append(f"{total_files} file(s) scanned · {len(by_file)} with findings · {len(diags)} diagnostics")
    lines.append("  " + tally)

    # --- per-rule summary table (titles, not bare ids), sorted by severity then count ---
    rule_ct = Counter(d.rule_id for d in diags)
    rule_meta = {d.rule_id: (d.severity, d.title) for d in diags}
    lines.append("")
    lines.append(paint("Findings by rule:", _BOLD))
    for rid in sorted(rule_ct, key=lambda r: (_sev_meta(rule_meta[r][0])[2], -rule_ct[r], r)):
        s, title = rule_meta[rid]
        sym, col, _ = _sev_meta(s)
        lines.append(f"  {paint(sym, col)} {rid}  {title[:48]:<48} {rule_ct[rid]:>4}")

    # --- compact mode: one line per file ---
    if summary_only:
        lines.append("")
        for fpath in sorted(by_file):
            fsev = Counter(d.severity for d in by_file[fpath])
            badge = " ".join(
                paint(f"{_sev_meta(s)[0]}{fsev[s]}", _sev_meta(s)[1])
                for s in ("critical", "warning", "info") if fsev.get(s)
            )
            lines.append(f"  {badge:<24} {fpath}")
        lines.append("")
        lines.append(paint("MLGuard is a heuristic surfacing tool — treat findings as review prompts, weighted by confidence.", _DIM))
        return "\n".join(lines)

    # --- detailed findings, grouped by file, severity-then-line within file ---
    for fpath in sorted(by_file):
        lines.append("")
        lines.append(paint(fpath, _BOLD))
        for d in sorted(by_file[fpath], key=lambda x: (_sev_meta(x.severity)[2], x.line)):
            sym, col, _ = _sev_meta(d.severity)
            lines.append(f"  {paint(sym, col)} {d.rule_id}  {d.title}   {paint(f'[{d.severity} · {d.confidence}]', _DIM)}")
            lines.append(f"      line {d.line}")
            if d.code:
                lines.append(f"      code:  {d.code.strip()[:120]}")
            lines.append(f"      why:   {d.why_it_matters}")
            lines.append(f"      fix:   {d.suggested_fix}")

    lines.append("")
    lines.append(paint("MLGuard is a heuristic surfacing tool — treat findings as review prompts, weighted by confidence.", _DIM))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="mlguard", description="Static linter for silent methodological errors in scikit-learn workflows.")
    p.add_argument("path", help="Notebook, Python file, or directory to scan")
    p.add_argument("--json", dest="json_out", help="Write diagnostics JSON to this path")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--fail-on", choices=["none", "warning", "critical"], default="none")
    p.add_argument("--summary", action="store_true", help="Compact one-line-per-file output (for large/directory scans)")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors in text output")
    args = p.parse_args(argv)

    diags = scan_path(args.path)
    data = [d.to_dict() for d in diags]
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(data, indent=2), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(data, indent=2))
    else:
        use_color = sys.stdout.isatty() and not args.no_color
        print(_render_text(diags, args.path, use_color, args.summary))

    if args.fail_on == "critical" and any(d.severity == "critical" for d in diags):
        return 2
    if args.fail_on == "warning" and any(d.severity in {"critical", "warning"} for d in diags):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
