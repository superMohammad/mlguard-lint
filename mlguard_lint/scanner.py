import ast, json, re, pathlib
from typing import List, Tuple, Dict, Any, Optional
from .rules import Diagnostic, RULES

TRANSFORMERS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer",
    "SimpleImputer", "KNNImputer", "IterativeImputer",
    "PCA", "TruncatedSVD", "NMF", "SelectKBest", "SelectPercentile", "RFE", "RFECV",
    "CountVectorizer", "TfidfVectorizer", "PolynomialFeatures", "OneHotEncoder", "OrdinalEncoder",
}
RESAMPLERS = {"SMOTE", "RandomOverSampler", "RandomUnderSampler", "ADASYN", "BorderlineSMOTE", "SMOTETomek", "SMOTEENN"}
STOCHASTIC = {"RandomForestClassifier", "RandomForestRegressor", "GradientBoostingClassifier", "GradientBoostingRegressor", "SGDClassifier", "SGDRegressor", "LogisticRegression", "SVC", "KMeans", "MLPClassifier", "MLPRegressor", "train_test_split"}
ID_PAT = re.compile(r"(patient|subject|user|customer|student|account|device|hospital|scanner|session|visit|image|file|filename).*_?id|id$", re.I)
PROB_METRICS = {"roc_auc_score", "log_loss", "average_precision_score"}
AVG_METRICS = {"f1_score", "precision_score", "recall_score", "fbeta_score", "jaccard_score"}
AGG_METHODS = {"mean", "std", "max", "min", "sum", "median", "var", "quantile"}
HOLDOUT_PAT = re.compile(r"test|valid|holdout|\beval\b", re.I)


def _name(n):
    if isinstance(n, ast.Name): return n.id
    if isinstance(n, ast.Attribute): return n.attr
    if isinstance(n, ast.Call): return _name(n.func)
    if isinstance(n, ast.Subscript): return _name(n.value)
    if isinstance(n, ast.Tuple): return tuple(_name(e) for e in n.elts)
    return None

def _code_segment(src, node):
    try:
        return ast.get_source_segment(src, node)
    except Exception:
        return None

def _call_name(call):
    if isinstance(call, ast.Call):
        return _name(call.func)
    return None

def _has_kw(call, kw):
    return any(k.arg == kw for k in getattr(call, 'keywords', []))

def _diag(rule_id, path, line, src, node=None, message=None, conf="high", meta=None):
    r = RULES[rule_id]
    return Diagnostic(rule_id, r["title"], r["severity"], r["category"], str(path), int(line), conf,
                      message or r["title"], r["why"], r["fix"], _code_segment(src, node) if node else None, meta or {})


def _string_const(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None

def _obj_of_call(call):
    # Name of the object a method is called on: `obj.method(...)` -> "obj".
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
        return _name(call.func.value)
    return None

def _target_column_source(val):
    # df["col"] -> ("df", "col"); only the high-precision subscript-with-string form.
    if isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name):
        key = val.slice
        if isinstance(key, ast.Index):  # py<3.9 compatibility
            key = key.value
        col = _string_const(key)
        if col is not None:
            return (val.value.id, col)
    return None

def _x_from_dataframe(val):
    # X = df            -> ("df", "WHOLE")          whole frame, nothing removed
    # X = df.drop(cols) -> ("df", frozenset(cols))  only when columns are explicit (columns= or axis=1)
    # anything else     -> None                     (not tracked; avoids false positives)
    if isinstance(val, ast.Name):
        return (val.id, "WHOLE")
    if isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute) and val.func.attr == "drop" and isinstance(val.func.value, ast.Name):
        dropped = set()
        explicit_columns = False
        col_nodes = []
        for kw in val.keywords:
            if kw.arg == "columns":
                explicit_columns = True
                col_nodes.append(kw.value)
            elif kw.arg == "labels":
                col_nodes.append(kw.value)
            elif kw.arg == "axis" and isinstance(kw.value, ast.Constant) and kw.value.value == 1:
                explicit_columns = True
        if val.args:
            col_nodes.append(val.args[0])
        for cn in col_nodes:
            s = _string_const(cn)
            if s is not None:
                dropped.add(s)
            elif isinstance(cn, (ast.List, ast.Tuple)):
                for e in cn.elts:
                    s2 = _string_const(e)
                    if s2 is not None:
                        dropped.add(s2)
        if not explicit_columns:
            return None
        return (val.func.value.id, frozenset(dropped))
    return None


def _agg_is_global(call):
    # A reduction with axis=1 (or "columns") collapses across columns per row, which is
    # row-local and not a leak. Anything else reduces across rows -> a dataset-wide statistic.
    for kw in getattr(call, "keywords", []):
        if kw.arg == "axis" and isinstance(kw.value, ast.Constant) and kw.value.value in (1, "columns"):
            return False
    return True


def _global_agg_in(node):
    # Return the name of a dataset-wide aggregate used anywhere inside `node`, else None.
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            attr = n.func.attr
            if attr in AGG_METHODS and _agg_is_global(n):
                return attr
            if attr == "transform" and n.args:
                s = _string_const(n.args[0])
                if s in AGG_METHODS:
                    return f"transform('{s}')"
    return None


def load_code(path: str) -> str:
    p = pathlib.Path(path)
    if p.suffix == ".ipynb":
        data = json.loads(p.read_text(encoding="utf-8"))
        chunks = []
        for i, cell in enumerate(data.get("cells", []), 1):
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                if isinstance(source, str): source = source.splitlines(True)
                chunks.append(f"\n# <cell {i}>\n" + "".join(source))
        return "\n".join(chunks)
    return p.read_text(encoding="utf-8", errors="ignore")


def scan_code(src: str, path: str = "<memory>") -> List[Diagnostic]:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [Diagnostic("PARSE", "Could not parse Python code", "info", "parse", path, e.lineno or 1, "high", str(e), "Some notebooks contain shell/magic syntax that must be stripped or parsed cell-by-cell.", "Remove or ignore non-Python cells/magics.")]

    diags: List[Diagnostic] = []
    calls: List[Tuple[int, ast.Call, str]] = []
    assignments: Dict[str, Tuple[int, ast.AST, Optional[str]]] = {}
    fitted_vars: Dict[str, Dict[str, Any]] = {}
    model_fit_args: Dict[str, List[Tuple[str, str, int, ast.Call]]] = {}
    test_evals: List[Tuple[int, ast.Call, str]] = []
    id_like_seen = False
    sklearn_pipeline_with_resampler = False
    df_of_target: Dict[str, Tuple[str, str]] = {}            # yvar -> (dataframe, target column)
    x_from_df: Dict[str, Tuple[str, Any]] = {}               # xvar -> (dataframe, "WHOLE" | frozenset(dropped cols))

    for node in ast.walk(tree):
        # string id-like columns
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and ID_PAT.search(node.value):
            id_like_seen = True

        if isinstance(node, ast.Call):
            cname = _call_name(node)
            if cname:
                calls.append((getattr(node, 'lineno', 1), node, cname))
                if cname == "Pipeline":
                    seg = _code_segment(src, node) or ""
                    if any(r in seg for r in RESAMPLERS) and "imblearn.pipeline" not in src:
                        sklearn_pipeline_with_resampler = True
                if cname in STOCHASTIC and not _has_kw(node, "random_state"):
                    diags.append(_diag("MLG006", path, node.lineno, src, node, f"{cname} is called without random_state.", conf="medium"))

        if isinstance(node, ast.Assign):
            val = node.value
            vname = None
            if isinstance(val, ast.Call):
                vname = _call_name(val)
            tgt_col = _target_column_source(val)
            x_src = _x_from_dataframe(val)
            for t in node.targets:
                target_name = _name(t)
                if isinstance(target_name, str):
                    assignments[target_name] = (node.lineno, val, vname)
                    if tgt_col is not None:
                        df_of_target[target_name] = tgt_col
                    if x_src is not None:
                        x_from_df[target_name] = x_src

    calls.sort(key=lambda x: x[0])
    split_lines = [line for line, call, name in calls if name == "train_test_split"]
    cv_lines = [line for line, call, name in calls if name in {"cross_val_score", "cross_validate", "GridSearchCV", "RandomizedSearchCV"}]

    # Transformer/encoder/target-encoder instances bound to variables, so we can detect
    # `imp = SimpleImputer(); imp.fit(X)` and not only the inline `SimpleImputer().fit(X)` form.
    transformer_vars = {name: vn for name, (ln, val, vn) in assignments.items() if vn in TRANSFORMERS}
    encoder_vars = {name: vn for name, (ln, val, vn) in assignments.items() if vn in {"LabelEncoder", "OrdinalEncoder"}}
    te_vars = {name for name, (ln, val, vn) in assignments.items() if vn == "TargetEncoder"}

    # Rule MLG001/2/4 fit_transform and fit_resample before split/CV
    for line, call, cname in calls:
        seg = _code_segment(src, call) or ""
        if cname in {"fit_transform", "fit"}:
            obj = _obj_of_call(call)
            used = [t for t in TRANSFORMERS if t in seg]
            via_var = transformer_vars.get(obj) if obj else None
            if via_var and via_var not in used:
                used.append(via_var)
            if used:
                if any(sl > line for sl in split_lines):
                    diags.append(_diag("MLG001", path, line, src, call, f"A transformer appears fitted before train_test_split at line {min([s for s in split_lines if s > line])}.", meta={"transformers": used}))
                if any(cl > line for cl in cv_lines):
                    diags.append(_diag("MLG002", path, line, src, call, f"A transformer appears fitted before cross-validation/model selection at line {min([c for c in cv_lines if c > line])}.", meta={"transformers": used}))
                if call.args:
                    arg_seg = _code_segment(src, call.args[0]) or ""
                    if HOLDOUT_PAT.search(arg_seg):
                        diags.append(_diag("MLG016", path, line, src, call, f"A transformer is {cname}'d on '{arg_seg[:40]}', which looks like evaluation data; only transform() should be applied to the test set.", conf="medium", meta={"variable": arg_seg[:80], "transformers": used}))
        if cname == "fit_resample" or any(r + "().fit_resample" in seg for r in RESAMPLERS):
            if any(sl > line for sl in split_lines) or any(cl > line for cl in cv_lines):
                diags.append(_diag("MLG004", path, line, src, call, "A resampler is applied before split/CV.", meta={"resamplers": [r for r in RESAMPLERS if r in seg]}))

    if sklearn_pipeline_with_resampler:
        line = src.find("Pipeline")
        diags.append(_diag("MLG004", path, src[:line].count('\n')+1 if line >= 0 else 1, src, None, "A resampler appears inside sklearn Pipeline rather than imblearn.pipeline.Pipeline.", conf="medium"))

    # train_test_split missing stratify and group leakage
    for line, call, cname in calls:
        if cname == "train_test_split":
            if len(call.args) >= 2 and not _has_kw(call, "stratify"):
                diags.append(_diag("MLG005", path, line, src, call, "train_test_split is used without stratify= for a likely supervised split.", conf="medium"))
            if id_like_seen:
                diags.append(_diag("MLG007", path, line, src, call, "ID/group-like column names exist, but a random train_test_split is used.", conf="low"))
        if cname in {"KFold", "StratifiedKFold"} and id_like_seen:
            diags.append(_diag("MLG007", path, line, src, call, f"{cname} used while ID/group-like columns are present.", conf="low"))

    # model fit/score dataflow
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            meth = node.func.attr
            obj = _name(node.func.value)
            if meth == "fit" and obj and len(node.args) >= 2:
                model_fit_args.setdefault(obj, []).append((_name(node.args[0]) or "?", _name(node.args[1]) or "?", node.lineno, node))
            if meth in {"score", "predict", "predict_proba"} and obj and len(node.args) >= 1:
                test_evals.append((node.lineno, node, _name(node.args[0]) or "?"))
                if obj in model_fit_args:
                    xeval = _name(node.args[0]) or "?"
                    yeval = _name(node.args[1]) if len(node.args) > 1 else None
                    for xfit, yfit, fitline, fitnode in model_fit_args[obj]:
                        # Flag only when the SAME X (and same y, if y is provided) is reused,
                        # checked against every recorded fit of this model object.
                        if xfit == xeval and (yeval is None or yeval == yfit):
                            conf = "high" if (yeval is not None and yeval == yfit) else "medium"
                            diags.append(_diag("MLG003", path, node.lineno, src, node, f"Model '{obj}' appears evaluated on the same data used in fit at line {fitline}.", conf=conf))
                            break

    # MLG012: a model is fit and scored on non-holdout data, with no train_test_split or CV anywhere.
    # If the score argument looks like a held-out set (X_test/val/...), a test set exists even when
    # the split call is out of scope, so we do not flag it.
    fit_lines_any = [line for line, call, name in calls if name == "fit"]
    if fit_lines_any and not split_lines and not cv_lines:
        for line, call, name in calls:
            if name != "score" or not call.args:
                continue
            argname = _name(call.args[0]) or ""
            if not re.search(r"test|valid|holdout|\beval\b", argname, re.I):
                diags.append(_diag("MLG012", path, line, src, call, "A model is fit and scored on non-holdout data, with no train_test_split or cross-validation anywhere; there is no independent test set."))
                break

    # MLG015: the feature matrix X still contains the target column.
    for obj_name, fits in model_fit_args.items():
        for xfit, yfit, fitline, fitnode in fits:
            if xfit in x_from_df and yfit in df_of_target:
                xdf, dropped = x_from_df[xfit]
                ydf, ycol = df_of_target[yfit]
                if xdf == ydf:
                    if dropped == "WHOLE":
                        diags.append(_diag("MLG015", path, fitline, src, fitnode, f"X is assigned the full DataFrame '{xdf}', which still contains the target column '{ycol}'.", conf="medium", meta={"dataframe": xdf, "target": ycol}))
                    elif isinstance(dropped, frozenset) and ycol not in dropped:
                        diags.append(_diag("MLG015", path, fitline, src, fitnode, f"X is built from '{xdf}.drop(...)' but the target column '{ycol}' is not among the dropped columns.", conf="low", meta={"dataframe": xdf, "target": ycol, "dropped": sorted(dropped)}))

    # MLG013: resampling applied to a test/validation set.
    for line, call, cname in calls:
        if cname == "fit_resample" and call.args:
            argname = _name(call.args[0]) or ""
            if re.search(r"test|valid|\beval\b|holdout", argname, re.I):
                diags.append(_diag("MLG013", path, line, src, call, f"A resampler is applied to '{argname}', which looks like evaluation data.", conf="medium", meta={"variable": argname}))

    # test set reuse
    x_test_counts: Dict[str, int] = {}
    for line, call, arg in test_evals:
        if re.search(r"test", arg or "", re.I):
            x_test_counts[arg] = x_test_counts.get(arg, 0) + 1
    for arg, count in x_test_counts.items():
        if count >= 2:
            first_line = next(line for line, call, a in test_evals if a == arg)
            diags.append(_diag("MLG010", path, first_line, src, None, f"{arg} is used in {count} evaluation/prediction calls.", conf="medium", meta={"count": count, "variable": arg}))

    # best_score_
    for m in re.finditer(r"\.best_score_", src):
        line = src[:m.start()].count("\n") + 1
        diags.append(_diag("MLG008", path, line, src, None, "GridSearchCV/RandomizedSearchCV best_score_ appears referenced; do not report it as final test performance.", conf="medium"))

    # MLG009: LabelEncoder/OrdinalEncoder applied to a feature rather than the target.
    for line, call, cname in calls:
        if cname not in {"fit_transform", "fit"}:
            continue
        obj = _obj_of_call(call)
        cls = encoder_vars.get(obj) if obj else None
        if cls is None and obj in {"LabelEncoder", "OrdinalEncoder"}:
            cls = obj
        if cls and call.args:
            arg_seg = _code_segment(src, call.args[0]) or ""
            if not re.search(r"\by\b|target|label|class", arg_seg, re.I):
                diags.append(_diag("MLG009", path, line, src, call, f"{cls} appears applied to an input feature rather than the target.", conf="medium", meta={"encoder": cls, "argument": arg_seg[:80]}))

    # MLG011: groupby target-mean encoding heuristic.
    for m in re.finditer(r"groupby\s*\([^\)]*\)\s*\[[^\]]*(target|label|y|class)[^\]]*\]\s*\.mean\s*\(", src, re.I):
        line = src[:m.start()].count("\n") + 1
        diags.append(_diag("MLG011", path, line, src, None, "A groupby target mean pattern may be target encoding without cross-fitting.", conf="medium"))

    # MLG011: TargetEncoder fit then transform on the same data disables internal cross-fitting.
    te_fit_x: Dict[str, set] = {}
    for line, call, cname in calls:
        obj = _obj_of_call(call)
        if obj in te_vars and cname == "fit" and call.args:
            te_fit_x.setdefault(obj, set()).add(_name(call.args[0]))
        elif obj in te_vars and cname == "transform" and call.args:
            xt = _name(call.args[0])
            if xt in te_fit_x.get(obj, set()):
                diags.append(_diag("MLG011", path, line, src, call, "TargetEncoder is fit and then transformed on the same data, disabling its internal cross-fitting.", conf="medium", meta={"variable": obj}))

    # MLG014: random split/CV on apparently time-ordered data.
    time_signal = re.search(r"to_datetime|parse_dates\s*=|\.dt\.|DatetimeIndex|\.resample\s*\(|sort_values\s*\(\s*[^)]*['\"](?:date|time|timestamp|datetime)", src, re.I)
    if time_signal and "TimeSeriesSplit" not in src:
        for line, call, cname in calls:
            if cname in {"train_test_split", "KFold", "StratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit", "cross_val_score", "cross_validate"}:
                diags.append(_diag("MLG014", path, line, src, call, f"{cname} is used on apparently time-ordered data; random splitting/CV leaks future information.", conf="low"))
                break

    # MLG017: probability/ranking metric handed hard predict() output instead of scores.
    for line, call, cname in calls:
        if cname in PROB_METRICS and len(call.args) >= 2:
            a2 = call.args[1]
            if isinstance(a2, ast.Call) and isinstance(a2.func, ast.Attribute) and a2.func.attr == "predict":
                diags.append(_diag("MLG017", path, line, src, call, f"{cname} is given predict() hard labels; it needs probabilities/scores from predict_proba or decision_function.", conf="medium", meta={"metric": cname}))

    # MLG018: a feature is built from a dataset-wide statistic before the train/test split or CV.
    split_cv_lines = split_lines + cv_lines
    boundary = min(split_cv_lines) if split_cv_lines else None
    if boundary is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and getattr(node, "lineno", 0) < boundary:
                col_targets = [t for t in node.targets if isinstance(t, ast.Subscript)]
                if not col_targets:
                    continue
                agg = _global_agg_in(node.value)
                if agg:
                    col = _string_const(col_targets[0].slice.value if isinstance(col_targets[0].slice, ast.Index) else col_targets[0].slice) or "a feature"
                    diags.append(_diag("MLG018", path, node.lineno, src, node, f"Feature '{col}' is built from a dataset-wide statistic ({agg}) computed before the split at line {boundary}.", conf="low", meta={"aggregate": agg, "column": col}))

    # MLG019: misleading micro-average on (likely imbalanced) multiclass metrics.
    for line, call, cname in calls:
        if cname in AVG_METRICS:
            for kw in call.keywords:
                if kw.arg == "average" and _string_const(kw.value) == "micro":
                    diags.append(_diag("MLG019", path, line, src, call, f"{cname}(average='micro') re-reports overall accuracy and hides minority-class performance.", conf="medium", meta={"metric": cname}))

    # MLG020: identifier/source-like columns left in the feature matrix (Clever Hans shortcut).
    if id_like_seen:
        for obj_name, fits in model_fit_args.items():
            for xfit, yfit, fitline, fitnode in fits:
                if xfit in x_from_df:
                    xdf, dropped = x_from_df[xfit]
                    if dropped == "WHOLE":
                        diags.append(_diag("MLG020", path, fitline, src, fitnode, f"Model is fit on the full DataFrame '{xdf}' while ID/source-like columns are present; identifier columns used as features cause shortcut learning.", conf="low", meta={"dataframe": xdf}))
                    elif isinstance(dropped, frozenset) and not any(ID_PAT.search(c) for c in dropped):
                        diags.append(_diag("MLG020", path, fitline, src, fitnode, f"X is built from '{xdf}.drop(...)' but no ID/source-like column is dropped, while identifier-like columns are present.", conf="low", meta={"dataframe": xdf, "dropped": sorted(dropped)}))

    # MLG021: rows duplicated/upsampled (with replacement) before the split.
    for line, call, cname in calls:
        later_split = [sl for sl in split_lines if sl > line]
        if not later_split:
            continue
        if (cname == "resample" and (_has_kw(call, "replace") or _has_kw(call, "n_samples"))) or (cname == "sample" and _has_kw(call, "replace")):
            diags.append(_diag("MLG021", path, line, src, call, f"Rows appear duplicated/upsampled (via {cname}) before train_test_split at line {min(later_split)}; near-identical rows can land on both sides.", conf="low", meta={"call": cname}))

    # Deduplicate line/rule/code
    seen = set(); out=[]
    for d in diags:
        key=(d.rule_id,d.line,d.message)
        if key not in seen:
            out.append(d); seen.add(key)
    return out


def scan_path(path: str) -> List[Diagnostic]:
    p = pathlib.Path(path)
    paths = []
    if p.is_dir():
        paths = list(p.rglob("*.ipynb")) + list(p.rglob("*.py"))
    else:
        paths = [p]
    all_diags=[]
    for fp in paths:
        try:
            src = load_code(str(fp))
            # strip simple magics/shell lines
            src = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("%", "!")))
            all_diags.extend(scan_code(src, str(fp)))
        except Exception as e:
            all_diags.append(Diagnostic("ERROR", "Scan error", "info", "scanner", str(fp), 1, "high", repr(e), "The scanner failed on this file.", "Open an issue with the traceback and a minimal notebook."))
    return all_diags
