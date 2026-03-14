"""
Course Scoring Module
=====================
Implements the decision model for ranking candidate courses during
semester planning. Three model structures are supported:
    A) Flat linear model
    B) Multi-dimensional model (necessity / interest / feasibility)
    C) Active-only model
    D) Top-K sub-factor model

Each course is scored across 9 decision factors grouped into 3 dimensions.
Python computes the factual factors (necessity, feasibility); the LLM
judges the subjective factors (interest).

Usage:
    from scoring import CourseScorer, FRAMEWORK_PROFILES

    # Default (linear — selected model):
    scorer = CourseScorer()

    # For experiments comparing model structures:
    scorer = CourseScorer(model="multidimensional")
    scorer = CourseScorer(model="active_only")
    scorer = CourseScorer(model="topk")

    scores = scorer.score_candidates(candidates)
"""

import numpy as np


# ─── Factor Definitions ─────────────────────────────────────

FACTORS = {
    # Necessity — weights reflect: critical path is most consequential,
    # scarcity has real cost, efficiency is optimization, CI-M is binary check
    "critical_path":     {"dimension": "necessity",    "computed_by": "python", "sub_weight": 3.0},
    "scarcity":          {"dimension": "necessity",    "computed_by": "python", "sub_weight": 2.5},
    "efficiency":        {"dimension": "necessity",    "computed_by": "python", "sub_weight": 2.0},
    "ci_m_need":         {"dimension": "necessity",    "computed_by": "python", "sub_weight": 1.0},
    # Interest — interest match is primary, career is speculative,
    # learning style is secondary
    "interest_match":    {"dimension": "interest",     "computed_by": "llm",    "sub_weight": 3.0},
    "career_relevance":  {"dimension": "interest",     "computed_by": "llm",    "sub_weight": 2.0},
    "learning_style_fit":{"dimension": "interest",     "computed_by": "llm",    "sub_weight": 1.0},
    # Feasibility — workload is primary, rating is supporting signal
    "workload_absolute": {"dimension": "feasibility",  "computed_by": "python", "sub_weight": 3.0},
    "rating":            {"dimension": "feasibility",  "computed_by": "python", "sub_weight": 2.0},
}

# Default sub-factor weights extracted from FACTORS for use in scoring
DEFAULT_SUB_WEIGHTS = {f: info["sub_weight"] for f, info in FACTORS.items()}

DIMENSIONS = ["necessity", "interest", "feasibility"]

FACTORS_BY_DIM = {
    dim: [f for f, info in FACTORS.items() if info["dimension"] == dim]
    for dim in DIMENSIONS
}


# ─── Framework Profiles ─────────────────────────────────────

FRAMEWORK_PROFILES = {
    "requirements_optimizer": {
        "dim_weights": {"necessity": 0.60, "interest": 0.10, "feasibility": 0.30},
        "description": "Prioritize graduating efficiently. Requirements first.",
    },
    "interest_explorer": {
        "dim_weights": {"necessity": 0.15, "interest": 0.55, "feasibility": 0.30},
        "description": "Explore what's interesting. Requirements are secondary.",
    },
    "workload_balancer": {
        "dim_weights": {"necessity": 0.20, "interest": 0.15, "feasibility": 0.65},
        "description": "Keep the semester manageable. Avoid overload.",
    },
    "career_strategist": {
        "dim_weights": {"necessity": 0.15, "interest": 0.55, "feasibility": 0.30},
        "description": "Build toward a career goal. Interest weighted toward career_relevance.",
        "sub_weight_overrides": {"career_relevance": 2.0, "interest_match": 1.0, "learning_style_fit": 0.5},
    },
    "balanced": {
        "dim_weights": {"necessity": 0.35, "interest": 0.35, "feasibility": 0.30},
        "description": "No single priority dominates.",
    },
}

# Default dimension weights when no profile is specified
DEFAULT_DIM_WEIGHTS = {"necessity": 0.35, "interest": 0.35, "feasibility": 0.30}


# ─── Scoring Functions ──────────────────────────────────────

def score_linear(factors, weights=None):
    """
    Approach A: Flat linear model.
    All factors weighted independently and summed.

    Args:
        factors: dict {factor_name: value (0-1)}
        weights: dict {factor_name: weight}. If None, equal weights.
    Returns:
        float score (0-1)
    """
    if weights is None:
        weights = {f: 1.0 for f in FACTORS}

    total = 0.0
    weight_sum = 0.0
    for f, v in factors.items():
        if f in weights and v is not None:
            total += weights[f] * v
            weight_sum += weights[f]

    return total / weight_sum if weight_sum > 0 else 0.0


def score_multidimensional(factors, dim_weights=None, sub_weight_overrides=None):
    """
    Approach B: Multi-dimensional model.
    Sub-factors are weighted within each dimension using DEFAULT_SUB_WEIGHTS
    (reflecting relative importance), then dimensions are weighted at top level.

    Args:
        factors: dict {factor_name: value (0-1)}
        dim_weights: dict {"necessity": W1, "interest": W2, "feasibility": W3}
        sub_weight_overrides: optional dict {factor_name: weight} to override
                              default sub-weights for specific factors
    Returns:
        float score (0-1), dict of dimension scores
    """
    if dim_weights is None:
        dim_weights = DEFAULT_DIM_WEIGHTS

    dim_scores = {}
    for dim in DIMENSIONS:
        dim_factors = FACTORS_BY_DIM[dim]
        values = []
        weights = []
        for f in dim_factors:
            if f in factors and factors[f] is not None:
                # Use sub_weight_overrides if provided, else default sub-weight
                if sub_weight_overrides and f in sub_weight_overrides:
                    w = sub_weight_overrides[f]
                else:
                    w = DEFAULT_SUB_WEIGHTS.get(f, 1.0)
                values.append(factors[f] * w)
                weights.append(w)

        if values:
            dim_scores[dim] = sum(values) / sum(weights)
        else:
            dim_scores[dim] = 0.0

    # Weighted combination of dimensions
    total = sum(dim_weights.get(d, 0) * dim_scores.get(d, 0) for d in DIMENSIONS)
    w_sum = sum(dim_weights.get(d, 0) for d in DIMENSIONS)
    score = total / w_sum if w_sum > 0 else 0.0

    return score, dim_scores


def score_active_only(factors, dim_weights=None, active_dims=None, active_factors=None):
    """
    Approach C: Partial information model.
    Only active dimensions/factors contribute to the score.
    Useful when information is incomplete.

    Args:
        factors: dict {factor_name: value (0-1)}
        dim_weights: dict of dimension weights (for active dims only)
        active_dims: list of active dimension names. If None, all active.
        active_factors: list of specific active factor names. Overrides active_dims.
    Returns:
        float score (0-1), dict of active dimension scores
    """
    if dim_weights is None:
        dim_weights = DEFAULT_DIM_WEIGHTS

    if active_factors is not None:
        # Score only on specified factors
        active_by_dim = {}
        for f in active_factors:
            dim = FACTORS.get(f, {}).get("dimension")
            if dim and f in factors and factors[f] is not None:
                if dim not in active_by_dim:
                    active_by_dim[dim] = []
                active_by_dim[dim].append(factors[f])

        dim_scores = {}
        for dim, vals in active_by_dim.items():
            dim_scores[dim] = np.mean(vals) if vals else 0.0

    elif active_dims is not None:
        # Score only on specified dimensions
        dim_scores = {}
        for dim in active_dims:
            dim_factors = FACTORS_BY_DIM.get(dim, [])
            vals = [factors[f] for f in dim_factors if f in factors and factors[f] is not None]
            dim_scores[dim] = np.mean(vals) if vals else 0.0
    else:
        # All dimensions active — same as multidimensional
        _, dim_scores = score_multidimensional(factors, dim_weights)

    # Weight only active dimensions
    active_w = {d: dim_weights.get(d, 0) for d in dim_scores}
    w_sum = sum(active_w.values())
    if w_sum > 0:
        score = sum(active_w[d] * dim_scores[d] for d in dim_scores) / w_sum
    else:
        score = 0.0

    return score, dim_scores


def score_topk(factors, dim_weights=None, sub_weight_overrides=None, k=2):
    """
    Approach D: Top-K sub-factor model.
    Within each dimension, only the k highest-valued sub-factors contribute
    to the dimension score. Lower-valued sub-factors are zeroed out.

    This prevents weak sub-factors from diluting a strong signal within a
    dimension. For example, if a course has critical_path=1.0 but
    scarcity=0.0, efficiency=0.0, ci_m_need=0.0, the standard
    multidimensional model averages these down. Top-K (k=1) would score
    necessity purely on critical_path.

    Args:
        factors: dict {factor_name: value (0-1)}
        dim_weights: dict {"necessity": W1, "interest": W2, "feasibility": W3}
        sub_weight_overrides: optional dict to override default sub-weights
        k: number of top sub-factors to keep per dimension
    Returns:
        float score (0-1), dict of dimension scores
    """
    if dim_weights is None:
        dim_weights = DEFAULT_DIM_WEIGHTS

    dim_scores = {}
    for dim in DIMENSIONS:
        dim_factors = FACTORS_BY_DIM[dim]

        # Collect (factor_name, value, sub_weight) for available factors
        available = []
        for f in dim_factors:
            if f in factors and factors[f] is not None:
                if sub_weight_overrides and f in sub_weight_overrides:
                    w = sub_weight_overrides[f]
                else:
                    w = DEFAULT_SUB_WEIGHTS.get(f, 1.0)
                available.append((f, factors[f], w))

        if not available:
            dim_scores[dim] = 0.0
            continue

        # Sort by value descending and keep top k
        available.sort(key=lambda x: x[1], reverse=True)
        top = available[:k]

        # Weighted average of top-k factors
        values = [v * w for _, v, w in top]
        weights = [w for _, _, w in top]
        dim_scores[dim] = sum(values) / sum(weights) if weights else 0.0

    # Weighted combination of dimensions
    total = sum(dim_weights.get(d, 0) * dim_scores.get(d, 0) for d in DIMENSIONS)
    w_sum = sum(dim_weights.get(d, 0) for d in DIMENSIONS)
    score = total / w_sum if w_sum > 0 else 0.0

    return score, dim_scores


# ─── Weight Adjustment ──────────────────────────────────────

BOOST_MULTIPLIER = 2.0
REDUCE_MULTIPLIER = 0.5

def apply_signals(current_weights, signals):
    """
    Apply BOOST/REDUCE/KEEP signals to dimension weights.
    Signals stack across conversation turns.

    Args:
        current_weights: dict {"necessity": w1, ...}
        signals: dict {"necessity": "BOOST", "interest": "REDUCE", ...}
    Returns:
        dict of adjusted, renormalized weights
    """
    multipliers = {"BOOST": BOOST_MULTIPLIER, "REDUCE": REDUCE_MULTIPLIER, "KEEP": 1.0}

    adjusted = {}
    for dim, w in current_weights.items():
        signal = signals.get(dim, "KEEP")
        adjusted[dim] = w * multipliers.get(signal, 1.0)

    # Renormalize to sum to 1
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {d: w / total for d, w in adjusted.items()}

    return adjusted


def detect_signals(message):
    """
    Detect weight adjustment signals from a student's message.
    Returns dict of {dimension: "BOOST" | "REDUCE"} for detected signals.
    Only includes dimensions where a signal was detected.
    """
    import re
    msg = message.lower()
    signals = {}

    boost_patterns = {
        "necessity": [
            r"need to graduate", r"running out of time", r"have to finish",
            r"requirement", r"what do i still need", r"efficient",
            r"on track", r"behind", r"can i graduate",
        ],
        "interest": [
            r"interested in", r"passionate about", r"curious about",
            r"want to explore", r"sounds cool", r"love",
            r"career", r"grad school", r"job",
        ],
        "feasibility": [
            r"workload", r"don't want to overload", r"light semester",
            r"busy with", r"athletics", r"club", r"job", r"research",
            r"manageable", r"balance", r"how hard", r"how heavy",
        ],
    }

    reduce_patterns = {
        "necessity": [
            r"no rush", r"plenty of time", r"not worried about graduating",
            r"don't care about requirements",
        ],
        "interest": [
            r"don't care what", r"just need to finish", r"doesn't matter what",
        ],
        "feasibility": [
            r"can handle", r"don't mind heavy", r"challenge me", r"bring it on",
            r"i can take a lot",
        ],
    }

    for dim, patterns in boost_patterns.items():
        for pat in patterns:
            if re.search(pat, msg):
                signals[dim] = "BOOST"
                break

    for dim, patterns in reduce_patterns.items():
        for pat in patterns:
            if re.search(pat, msg):
                signals[dim] = "REDUCE"
                break

    return signals


# ─── Course Scorer (Main Interface) ─────────────────────────

class CourseScorer:
    """
    Scores candidate courses using one of three model structures.

    The default model is "linear" (selected in Section 3.8 of the development
    notebook). The model parameter allows experiments to compare all three.

    Model types:
        "linear" — flat weighted average of all factors (default, used in production)
        "multidimensional" — factors grouped into dimensions, then dimensions weighted
        "active_only" — only active dimensions/factors contribute
        "topk" — within each dimension, only the k strongest sub-factors contribute

    The LLM can adjust weights at two granularities:
      - Dimension level: BOOST/REDUCE necessity/interest/feasibility
      - Factor level: override a specific factor's weight
    """

    def __init__(self, model="linear", k=2):
        """
        Args:
            model: scoring model structure — "linear", "multidimensional", "active_only", or "topk"
            k: for topk model, number of top sub-factors to keep per dimension
        """
        if model not in ("linear", "multidimensional", "active_only", "topk"):
            raise ValueError(f"Unknown model type: {model!r}. "
                             f"Must be 'linear', 'multidimensional', 'active_only', or 'topk'.")
        self.model = model
        self.dim_weights = dict(DEFAULT_DIM_WEIGHTS)
        self.sub_weight_overrides = None
        self.signal_history = []
        # For active_only model: which dimensions are currently active
        self._active_dims = None      # None = all active
        self._active_factors = None   # None = use _active_dims
        # For topk model: how many sub-factors to keep per dimension
        self._topk_k = k

    def set_framework(self, framework_name):
        """Set weights from a predefined framework profile."""
        profile = FRAMEWORK_PROFILES.get(framework_name)
        if profile:
            self.dim_weights = dict(profile["dim_weights"])
            self.sub_weight_overrides = profile.get("sub_weight_overrides")

    def set_active_dims(self, dims):
        """
        Set which dimensions are active (for active_only model).
        Pass None to activate all dimensions.

        Args:
            dims: list of dimension names, e.g. ["necessity", "feasibility"]
        """
        self._active_dims = list(dims) if dims is not None else None
        self._active_factors = None  # dims take precedence

    def set_active_factors(self, factors):
        """
        Set which specific factors are active (for active_only model).
        Overrides set_active_dims.

        Args:
            factors: list of factor names
        """
        self._active_factors = list(factors) if factors is not None else None

    def apply_signal(self, signals):
        """Apply BOOST/REDUCE signals and record in history."""
        self.signal_history.append(signals)
        self.dim_weights = apply_signals(self.dim_weights, signals)

    def apply_message(self, message):
        """Detect and apply signals from a student message."""
        signals = detect_signals(message)
        if signals:
            self.apply_signal(signals)
        return signals

    def _get_factor_weights(self):
        """
        Convert dimension weights + sub-weights into a flat factor weight vector.
        Each factor's effective weight = dim_weight * sub_weight (within its dimension).
        """
        factor_weights = {}
        for f, info in FACTORS.items():
            dim = info["dimension"]
            dim_w = self.dim_weights.get(dim, 1.0)
            if self.sub_weight_overrides and f in self.sub_weight_overrides:
                sub_w = self.sub_weight_overrides[f]
            else:
                sub_w = DEFAULT_SUB_WEIGHTS.get(f, 1.0)
            factor_weights[f] = dim_w * sub_w
        return factor_weights

    def score_course(self, factors):
        """
        Score a single course using the configured model.

        Returns:
            (score, details) where details includes dim_scores for explanation
        """
        if self.model == "linear":
            factor_weights = self._get_factor_weights()
            score = score_linear(factors, factor_weights)
            # Compute dimensional subtotals post-hoc for explainability
            dim_scores = self._compute_dim_scores(factors)
            return score, {"dim_scores": dim_scores}

        elif self.model == "multidimensional":
            score, dim_scores = score_multidimensional(
                factors, self.dim_weights, self.sub_weight_overrides
            )
            return score, {"dim_scores": dim_scores}

        elif self.model == "active_only":
            score, dim_scores = score_active_only(
                factors, self.dim_weights,
                active_dims=self._active_dims,
                active_factors=self._active_factors,
            )
            return score, {"dim_scores": dim_scores}

        elif self.model == "topk":
            score, dim_scores = score_topk(
                factors, self.dim_weights, self.sub_weight_overrides,
                k=self._topk_k,
            )
            return score, {"dim_scores": dim_scores}

        else:
            raise ValueError(f"Unknown model: {self.model}")

    def _compute_dim_scores(self, factors):
        """Compute per-dimension subtotals for explainability (used by linear model)."""
        dim_scores = {}
        for dim in DIMENSIONS:
            dim_factors = FACTORS_BY_DIM[dim]
            vals, weights = [], []
            for f in dim_factors:
                if f in factors and factors[f] is not None:
                    sub_w = DEFAULT_SUB_WEIGHTS.get(f, 1.0)
                    if self.sub_weight_overrides and f in self.sub_weight_overrides:
                        sub_w = self.sub_weight_overrides[f]
                    vals.append(factors[f] * sub_w)
                    weights.append(sub_w)
            dim_scores[dim] = sum(vals) / sum(weights) if weights else 0.0
        return dim_scores

    def score_candidates(self, candidates):
        """
        Score and rank a list of candidates.
        Args:
            candidates: dict {course_id: {factor_name: value}}
        Returns:
            list of (course_id, score, details) sorted by score descending
        """
        results = []
        for name, factors in candidates.items():
            score, details = self.score_course(factors)
            results.append((name, score, details))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_state(self):
        """Return current scorer state for debugging/display."""
        return {
            "model": self.model,
            "dim_weights": dict(self.dim_weights),
            "sub_weight_overrides": self.sub_weight_overrides,
            "factor_weights": self._get_factor_weights(),
            "signal_history": list(self.signal_history),
            "active_dims": self._active_dims,
            "active_factors": self._active_factors,
        }


# ─── Factor Computation from Raw Candidate Data ─────────────

def compute_candidate_factors(raw_candidates, max_units=48, planned_units=0):
    """
    Convert raw candidates from get_feasible_candidates() into normalized
    0-1 factor values for the scoring model.

    Args:
        raw_candidates: list of dicts from get_feasible_candidates(), each with:
            course_id, title, units, description, requirement_filled,
            critical_path ("HIGH"/"MEDIUM"/"NONE"), critical_detail,
            scarcity ("both"/"fall_only"/"spring_only"/"unknown"),
            efficiency ("DOUBLE_COUNTS"/"SINGLE"), efficiency_detail,
            ci_m_value ("NEEDED"/"AVAILABLE_BUT_MET"/"N/A"),
            rating (float or None), in_class_hours, out_of_class_hours
        max_units: student's max units per semester
        planned_units: units already planned this semester

    Returns:
        dict {course_id: {factor_name: normalized_value (0-1), ...}}
        Interest factors are set to None (must be scored by LLM).
    """
    results = {}

    for c in raw_candidates:
        cid = c["course_id"]
        factors = {}

        # ── Necessity ──

        # Critical path: HIGH=1.0, MEDIUM=0.5, NONE=0.0
        cp_map = {"HIGH": 1.0, "MEDIUM": 0.5, "NONE": 0.0}
        factors["critical_path"] = cp_map.get(c.get("critical_path", "NONE"), 0.0)

        # Scarcity: fall_only or spring_only = 1.0, both = 0.0
        scarcity = c.get("scarcity", "unknown")
        factors["scarcity"] = 1.0 if scarcity in ("fall_only", "spring_only") else 0.0

        # Efficiency: DOUBLE_COUNTS = 1.0, SINGLE = 0.0
        factors["efficiency"] = 1.0 if c.get("efficiency") == "DOUBLE_COUNTS" else 0.0

        # CI-M need: NEEDED = 1.0, else 0.0
        factors["ci_m_need"] = 1.0 if c.get("ci_m_value") == "NEEDED" else 0.0

        # ── Interest (set to None — must be scored by LLM) ──
        factors["interest_match"] = None
        factors["career_relevance"] = None
        factors["learning_style_fit"] = None

        # ── Feasibility ──

        # Workload absolute: invert so lighter = higher score
        in_hrs = c.get("in_class_hours") or 0
        out_hrs = c.get("out_of_class_hours") or 0
        total_hrs = in_hrs + out_hrs
        if total_hrs > 0:
            # Normalize: 20+ hrs/week = 0.0, 4 hrs/week = 1.0
            factors["workload_absolute"] = max(0.0, min(1.0, 1.0 - (total_hrs - 4) / 16))
        else:
            # No hour data — use units as proxy (12u = moderate, 15u = heavy)
            units = c.get("units", 12)
            factors["workload_absolute"] = max(0.0, min(1.0, 1.0 - (units - 6) / 12))

        # Workload relative: how much capacity remains after adding this course
        units = c.get("units", 12)
        remaining_after = max_units - planned_units - units
        factors["workload_relative"] = max(0.0, min(1.0, remaining_after / max_units))

        # Rating: FireRoad uses a 7-point scale, most courses fall in 4-7 range.
        # Normalize within that range: 4.0 → 0.0, 7.0 → 1.0
        rating = c.get("rating")
        if rating is not None:
            factors["rating"] = max(0.0, min(1.0, (rating - 4.0) / 3.0))
        else:
            factors["rating"] = 0.5  # default when no rating available

        results[cid] = {
            "factors": factors,
            "title": c.get("title", "Unknown"),
            "units": c.get("units", 12),
            "description": c.get("description", ""),
            "requirement_filled": c.get("requirement_filled", ""),
        }

    return results


def fill_interest_defaults(candidate_factors, default=0.5):
    """
    Fill None interest factors with a default value.
    Use when LLM scoring is not available.
    Returns a new dict with all factors as floats.
    """
    filled = {}
    for cid, data in candidate_factors.items():
        if isinstance(data, dict) and "factors" in data:
            factors = dict(data["factors"])
        else:
            factors = dict(data)
        for f in ["interest_match", "career_relevance", "learning_style_fit"]:
            if factors.get(f) is None:
                factors[f] = default
        filled[cid] = factors
    return filled


def compute_course_factors(course_ids, profile, courses_db, tracker=None, planner_obj=None,
                           max_units=48, planned_units=0):
    """
    Compute factor values for ANY set of courses given a student profile.
    Unlike compute_candidate_factors, this doesn't require pre-filtered feasible candidates.
    It computes what it can and flags what it can't (e.g., prereqs not met).

    Args:
        course_ids: list of course IDs to score
        profile: StudentProfile object
        courses_db: dict {course_id: course_info} from courses.json
        tracker: RequirementsTracker instance (optional)
        planner_obj: SemesterPlanner instance (optional)
        max_units: unit cap
        planned_units: units already planned

    Returns:
        dict {course_id: {"factors": {...}, "title": ..., "units": ..., "flags": [...]}}
    """
    results = {}

    # Get critical path info if planner available
    critical_set = set()
    critical_chain_length = {}
    if planner_obj and profile.major_id:
        try:
            paths = planner_obj.find_critical_paths(profile.major_id, profile.courses_taken)
            for path in paths:
                for i, cid in enumerate(path):
                    critical_set.add(cid)
                    courses_after = len(path) - i - 1
                    critical_chain_length[cid] = max(critical_chain_length.get(cid, 0), courses_after)
        except Exception:
            pass

    # Get requirement group counts for efficiency
    majors_data = {}
    if tracker and profile.major_id:
        try:
            import json, os
            majors_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "majors.json")
            if os.path.exists(majors_path):
                with open(majors_path) as f:
                    all_majors = json.load(f)
                majors_data = all_majors.get(profile.major_id, {})
        except Exception:
            pass

    course_group_count = {}
    for group in majors_data.get("select_groups", []):
        for cid in group.get("courses", []):
            course_group_count[cid] = course_group_count.get(cid, 0) + 1

    # CI-M status
    ci_m_done = False
    if tracker and profile.major_id:
        try:
            status = tracker.check_major(profile.major_id, profile.courses_taken)
            ci_m_done = status.get("ci_m", {}).get("done", False)
        except Exception:
            pass

    next_fall = profile.next_is_fall if profile.next_is_fall is not None else True

    for cid in course_ids:
        course = courses_db.get(cid, {})
        flags = []
        factors = {}

        if not course:
            flags.append("NOT_IN_CATALOG")
            factors = {f: 0.0 for f in FACTORS}
            factors["interest_match"] = None
            factors["career_relevance"] = None
            factors["learning_style_fit"] = None
            results[cid] = {"factors": factors, "title": cid, "units": 0,
                           "description": "", "requirement_filled": "None", "flags": flags}
            continue

        # Check prereqs
        if tracker:
            try:
                sat, missing = tracker.check_prereqs_satisfied(cid, profile.courses_taken)
                if not sat:
                    flags.append(f"PREREQS_NOT_MET: {', '.join(missing)}")
            except Exception:
                pass

        # Check if offered this semester
        sem_key = "offered_fall" if next_fall else "offered_spring"
        if not course.get(sem_key, False):
            flags.append("NOT_OFFERED_THIS_SEMESTER")

        # Check if already taken
        if cid in profile.courses_taken:
            flags.append("ALREADY_TAKEN")

        # ── Necessity ──
        if cid in critical_set:
            unlocks = critical_chain_length.get(cid, 0)
            factors["critical_path"] = 1.0 if unlocks >= 2 else 0.5
        else:
            factors["critical_path"] = 0.0

        scarcity = "both"
        if planner_obj:
            try:
                scarcity = planner_obj.get_offering_frequency(cid)
            except Exception:
                pass
        factors["scarcity"] = 1.0 if scarcity in ("fall_only", "spring_only") else 0.0

        factors["efficiency"] = 1.0 if course_group_count.get(cid, 0) > 1 else 0.0

        is_ci_m = bool(course.get("communication_requirement", "") and
                      "CI-M" in course.get("communication_requirement", ""))
        factors["ci_m_need"] = 1.0 if (is_ci_m and not ci_m_done) else 0.0

        # Check if course fills any requirement
        req_filled = "None"
        req_courses = set(majors_data.get("required_courses", []))
        if cid in req_courses:
            req_filled = "Required"
        else:
            for group in majors_data.get("select_groups", []):
                if cid in group.get("courses", []):
                    req_filled = group.get("name", "Elective")
                    break

        # Check HASS
        hass = course.get("hass_attribute", "")
        if hass and hass != "None" and req_filled == "None":
            req_filled = f"HASS ({hass})"

        # ── Interest (LLM scores) ──
        factors["interest_match"] = None
        factors["career_relevance"] = None
        factors["learning_style_fit"] = None

        # ── Feasibility ──
        in_hrs = course.get("in_class_hours") or 0
        out_hrs = course.get("out_of_class_hours") or 0
        total_hrs = in_hrs + out_hrs
        units = course.get("total_units", 12)

        if total_hrs > 0:
            factors["workload_absolute"] = max(0.0, min(1.0, 1.0 - (total_hrs - 4) / 16))
        else:
            factors["workload_absolute"] = max(0.0, min(1.0, 1.0 - (units - 6) / 12))

        # Rating
        rating = course.get("rating")
        if rating is not None:
            factors["rating"] = max(0.0, min(1.0, (rating - 4.0) / 3.0))
        else:
            factors["rating"] = 0.5

        results[cid] = {
            "factors": factors,
            "title": course.get("title", "Unknown"),
            "units": units,
            "description": course.get("description", "")[:300],
            "requirement_filled": req_filled,
            "flags": flags,
        }

    return results