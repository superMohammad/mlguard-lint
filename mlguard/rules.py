from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

@dataclass
class Diagnostic:
    rule_id: str
    title: str
    severity: str
    category: str
    path: str
    line: int
    confidence: str
    message: str
    why_it_matters: str
    suggested_fix: str
    code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return asdict(self)

RULES = {
    "MLG001": {
        "title": "Transformer fitted before split",
        "severity": "critical",
        "category": "data_leakage",
        "why": "A transformer learned statistics or structure from the full dataset before the test set was separated.",
        "fix": "Split first, then fit the transformer on X_train only, or place preprocessing inside a sklearn Pipeline/ColumnTransformer."
    },
    "MLG002": {
        "title": "Preprocessing outside cross-validation",
        "severity": "critical",
        "category": "data_leakage",
        "why": "Each validation fold can influence preprocessing if transformation happens before cross-validation.",
        "fix": "Pass a Pipeline containing preprocessing and model to cross_val_score/cross_validate/GridSearchCV."
    },
    "MLG003": {
        "title": "Model evaluated on training data",
        "severity": "critical",
        "category": "invalid_evaluation",
        "why": "Training data performance does not estimate generalization and can be highly optimistic.",
        "fix": "Use an independent test set, validation set, or cross-validation with no train/test overlap."
    },
    "MLG004": {
        "title": "Resampling before split/CV or outside imblearn Pipeline",
        "severity": "critical",
        "category": "data_leakage",
        "why": "Synthetic or selected samples can cross train/test boundaries and inflate metrics.",
        "fix": "Use imblearn.pipeline.Pipeline so resampling is applied only to training folds."
    },
    "MLG005": {
        "title": "Classification split without stratify",
        "severity": "warning",
        "category": "splitting",
        "why": "Class proportions may drift between train and test, especially with small or imbalanced data.",
        "fix": "Use train_test_split(..., stratify=y) for classification unless you have a reason not to."
    },
    "MLG006": {
        "title": "Missing random_state",
        "severity": "warning",
        "category": "reproducibility",
        "why": "Results can change across runs, making experiments hard to reproduce.",
        "fix": "Set random_state to an integer in splitters and stochastic estimators."
    },
    "MLG007": {
        "title": "Possible group/entity leakage",
        "severity": "warning",
        "category": "splitting",
        "why": "Rows from the same entity may appear in both train and test when random splitting is used.",
        "fix": "Use GroupShuffleSplit, GroupKFold, StratifiedGroupKFold, or an entity-level split."
    },
    "MLG008": {
        "title": "GridSearchCV best_score_ reported as final performance",
        "severity": "warning",
        "category": "model_selection_bias",
        "why": "The same CV used to tune hyperparameters is being treated as final evaluation, which is optimistically biased.",
        "fix": "Use nested CV or a truly untouched final test set for final performance."
    },
    "MLG009": {
        "title": "Ordinal/label encoding of a nominal feature",
        "severity": "warning",
        "category": "encoding",
        "why": "LabelEncoder/OrdinalEncoder impose an arbitrary numeric order on nominal categories, which linear, distance, and tree-threshold models can exploit as a false ranking.",
        "fix": "Use OneHotEncoder for nominal features; reserve LabelEncoder for the target, and use OrdinalEncoder(categories=[...]) only for truly ordinal features."
    },
    "MLG010": {
        "title": "Test set reused multiple times",
        "severity": "warning",
        "category": "holdout_reuse",
        "why": "Repeated adaptive use of test data turns it into validation data and biases final metrics.",
        "fix": "Use validation data for iteration and keep a final test set untouched until the end."
    },
    "MLG011": {
        "title": "Possible target/mean encoding without cross-fitting",
        "severity": "warning",
        "category": "target_leakage",
        "why": "Encoding categories using target means on the same rows can leak the target into features.",
        "fix": "Use cross-fitted target encoding, e.g. TargetEncoder.fit_transform inside a Pipeline."
    },
    "MLG012": {
        "title": "No independent test set",
        "severity": "critical",
        "category": "invalid_evaluation",
        "why": "A model is fit and then scored without any train_test_split or cross-validation, so the reported performance is measured on data the model has already seen.",
        "fix": "Hold out an independent test set with train_test_split, or use cross-validation, and report performance only on unseen data."
    },
    "MLG013": {
        "title": "Resampling applied to the test/validation set",
        "severity": "critical",
        "category": "invalid_evaluation",
        "why": "Resampling evaluation data (e.g. SMOTE on the test set) fabricates the metric so it no longer reflects real performance.",
        "fix": "Resample only the training data, inside an imblearn Pipeline so resampling is confined to training folds."
    },
    "MLG014": {
        "title": "Random split/CV on time-ordered data",
        "severity": "warning",
        "category": "data_leakage",
        "why": "Randomly splitting or shuffling time-ordered data trains on future observations to predict the past, leaking information.",
        "fix": "Use TimeSeriesSplit (and purged/embargoed CV for financial series) to respect temporal order."
    },
    "MLG015": {
        "title": "Target column included in features",
        "severity": "critical",
        "category": "target_leakage",
        "why": "Leaving the target column in the feature matrix lets the model read the answer directly, producing near-perfect but meaningless scores.",
        "fix": "Drop the target column from X (e.g. X = df.drop(columns=[target])) so features never contain the label."
    },
    "MLG016": {
        "title": "Transformer re-fit on the test set",
        "severity": "critical",
        "category": "data_leakage",
        "why": "Calling fit/fit_transform on evaluation data learns parameters from the test set and applies a different transformation than training saw, breaking the held-out guarantee.",
        "fix": "Fit the transformer on training data only, then call transform() (not fit/fit_transform) on the test/validation set."
    },
    "MLG017": {
        "title": "Probability metric given hard predictions",
        "severity": "warning",
        "category": "invalid_evaluation",
        "why": "roc_auc_score/log_loss/average_precision_score need probabilities or scores; feeding them predict() hard labels silently computes a wrong, usually worse and meaningless metric.",
        "fix": "Pass predict_proba(X)[:, 1] (or decision_function(X)) instead of predict(X) to ranking/probability metrics."
    },
    "MLG018": {
        "title": "Feature built from dataset-wide statistics before split",
        "severity": "critical",
        "category": "data_leakage",
        "why": "Computing a feature from a whole-column statistic (mean/std/max/median/groupby-transform) before the split mixes test-set information into training features, the most common silent pandas-level leak.",
        "fix": "Compute such statistics on X_train only (after the split), or inside a Pipeline/ColumnTransformer so each fold sees only its own training rows."
    },
    "MLG019": {
        "title": "Misleading micro-average on imbalanced multiclass",
        "severity": "warning",
        "category": "invalid_evaluation",
        "why": "average='micro' for single-label multiclass just re-reports overall accuracy and hides poor minority-class performance, the metric most affected by class imbalance.",
        "fix": "Use average='macro' (or 'weighted'), balanced_accuracy, or per-class scores to expose minority-class performance."
    },
    "MLG020": {
        "title": "ID/source column used as a feature",
        "severity": "warning",
        "category": "target_leakage",
        "why": "Leaving identifier or source columns (patient_id, hospital, scanner, filename) in X lets the model exploit spurious shortcuts (Clever Hans) that do not generalize.",
        "fix": "Drop identifier/source columns from X, and use them only as grouping keys for group-aware splitting."
    },
    "MLG021": {
        "title": "Rows duplicated/upsampled before split",
        "severity": "warning",
        "category": "data_leakage",
        "why": "Duplicating or upsampling rows before the split puts near-identical rows on both sides, so the test set is partly memorised training data.",
        "fix": "Split first, then resample/augment the training portion only (e.g. inside an imblearn Pipeline)."
    },
}
