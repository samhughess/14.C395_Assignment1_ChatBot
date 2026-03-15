"""
MIT Course Advisor Chatbot
==========================
Hybrid architecture:
    - Python router handles factual questions (course lookup, requirements, scheduling, planning)
    - LLM handles subjective/open-ended questions (recommendations, general advising)
    - LLM always presents the final response in natural language
"""

import json
import os
import re
import sys

# Add project root to path so we can import requirements.py, planner.py, etc.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from huggingface_hub import InferenceClient
from src.config import BASE_MODEL, MY_MODEL, HF_TOKEN, DATA_DIR


# ─── Load data ──────────────────────────────────────────────

def _load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


COURSES_LIST = _load_json("courses.json") or []
COURSES = {c["subject_id"]: c for c in COURSES_LIST}
MAJORS = _load_json("majors.json") or {}
GIRS = _load_json("girs.json") or {}

# ─── Load backend tools (graceful if not available) ─────────

try:
    from requirements import RequirementsTracker

    _tracker = RequirementsTracker(data_dir=DATA_DIR)
except Exception as e:
    print(f"[chat.py] Warning: RequirementsTracker not loaded: {e}")
    _tracker = None

try:
    from planner import SemesterPlanner

    _planner = SemesterPlanner(data_dir=DATA_DIR)
except Exception as e:
    print(f"[chat.py] Warning: SemesterPlanner not loaded: {e}")
    _planner = None

try:
    from scheduler import Scheduler

    _scheduler = Scheduler(os.path.join(DATA_DIR, "courses.json"))
except Exception as e:
    print(f"[chat.py] Warning: Scheduler not loaded: {e}")
    _scheduler = None

try:
    from scoring import (CourseScorer, compute_candidate_factors, fill_interest_defaults,
                         fill_interest_from_embeddings,
                         detect_signals, apply_signals, DEFAULT_DIM_WEIGHTS,
                         FRAMEWORK_PROFILES)

    _scoring_available = True
except Exception as e:
    print(f"[chat.py] Warning: scoring module not loaded: {e}")
    _scoring_available = False


# ─── Student Profile ────────────────────────────────────────

class SemesterPlan:
    """Tracks the state of an in-progress semester being built."""

    PROFILE_NAMES = ["requirements_optimizer", "interest_explorer",
                     "workload_balancer", "career_strategist", "balanced"]

    def __init__(self):
        self.active = False
        self.semester_type = None  # "fall" or "spring"
        self.planned_courses = []  # courses picked so far
        self.planned_units = 0
        self.priority = None  # student's stated priority/interests
        self.stage = None  # "initiated", "gathering_prefs", "suggesting", "finalizing"
        self.feasible_candidates = []  # pre-vetted by Python
        self.scorer = CourseScorer(model="multidimensional") if _scoring_available else None
        # Profile scores — running weights for each framework profile (sum to 1)
        # Start equal; BOOST/REDUCE signals accumulate across turns
        self.profile_scores = {p: 1.0 / len(self.PROFILE_NAMES)
                               for p in self.PROFILE_NAMES}
        self._active_profile = "balanced"
        # Stored suggested plans from gathering_prefs stage (for plan selection by name)
        self.suggested_plans = []  # list of plan dicts from _build_semester_plans

    def reset(self):
        self.__init__()

    def add_course(self, course_id, units):
        if course_id not in self.planned_courses:
            self.planned_courses.append(course_id)
            self.planned_units += units

    def summary(self):
        if not self.active:
            return "No semester plan in progress."
        parts = [f"Semester: {self.semester_type or 'TBD'}"]
        parts.append(f"Planned: {', '.join(self.planned_courses) if self.planned_courses else 'none yet'}")
        parts.append(f"Units: {self.planned_units}")
        if self.priority:
            parts.append(f"Priority: {self.priority}")
        parts.append(f"Stage: {self.stage}")
        parts.append(f"Active profile: {self._active_profile}")
        return "\n".join(parts)

    def apply_profile_signals(self, profile_signals):
        """
        Apply BOOST/REDUCE signals to profile scores and update active profile.
        Uses same mechanism as dimension weights: multiply, renormalize.

        Args:
            profile_signals: dict {profile_name: "BOOST" | "REDUCE"}
                             (profiles not mentioned are KEEP)
        """
        multipliers = {"BOOST": 2.0, "REDUCE": 0.5, "KEEP": 1.0}

        for p in self.profile_scores:
            signal = profile_signals.get(p, "KEEP")
            self.profile_scores[p] *= multipliers.get(signal, 1.0)

        # Renormalize to sum to 1
        total = sum(self.profile_scores.values())
        if total > 0:
            self.profile_scores = {p: s / total for p, s in self.profile_scores.items()}

        # Update active profile (argmax)
        new_active = max(self.profile_scores, key=self.profile_scores.get)
        if new_active != self._active_profile:
            self._active_profile = new_active
            # Apply the new profile's sub-weight overrides to the scorer
            if self.scorer:
                self.scorer.set_framework(new_active)


class StudentProfile:
    """Tracks what we know about the student across the conversation."""

    def __init__(self):
        self.major_id = None
        self.courses_taken = []
        self.year = None
        self.semesters_left = None
        self.next_is_fall = None
        self.max_units = 48
        self.interests = []
        self.semester_plan = SemesterPlan()
        # Cached ground truth — recomputed when profile changes
        self._cached_status = None
        self._status_dirty = True

    def is_complete(self):
        return self.major_id is not None and len(self.courses_taken) > 0

    def mark_dirty(self):
        """Call whenever major or courses_taken changes to invalidate cached status."""
        self._status_dirty = True

    def get_ground_truth(self):
        """
        Get the current ground truth requirement status.
        Recomputes from requirements tracker if profile has changed.
        Returns a formatted string suitable for injection into every LLM prompt.
        """
        if not self.is_complete() or not _tracker:
            return None

        if self._status_dirty or self._cached_status is None:
            status = _tracker.get_status(self.major_id, self.courses_taken)
            self._cached_status = self._format_ground_truth(status)
            self._status_dirty = False

        return self._cached_status

    def _format_ground_truth(self, status):
        """Format requirement status into a concise ground truth string."""
        parts = []
        major_name = MAJORS.get(self.major_id, {}).get("name", self.major_id)
        parts.append(f"GROUND TRUTH for {self.major_id} ({major_name}):")
        parts.append(f"Courses taken: {', '.join(self.courses_taken)}")

        # Major requirements
        req = status["major_status"]["required_courses"]
        if req["fulfilled"]:
            parts.append(f"Required courses DONE: {', '.join(req['fulfilled'])}")
        if req["remaining"]:
            parts.append(f"Required courses REMAINING: {', '.join(req['remaining'])}")
        else:
            parts.append("All required courses complete.")

        # Select groups
        for group in status["major_status"]["select_groups"]:
            if group["done"]:
                parts.append(f"  ✓ {group['name']}: complete ({', '.join(group['fulfilled'])})")
            else:
                opts = ', '.join(group.get('options', [])[:6])
                if group.get('flexible'):
                    parts.append(f"  ✗ {group['name']}: need {group['remaining_needed']} more (advisor choice)")
                else:
                    parts.append(f"  ✗ {group['name']}: need {group['remaining_needed']} more from: {opts}")

        # CI-M
        ci_m = status["major_status"]["ci_m"]
        if ci_m["done"]:
            parts.append(f"CI-M: complete ({', '.join(ci_m['fulfilled'])})")
        else:
            parts.append(f"CI-M: need {ci_m['remaining_needed']} more")

        # GIRs summary
        gir = status["gir_status"]
        gir_done = []
        gir_remaining = []
        for cat, info in gir.items():
            if cat == "HASS":
                if info["done"]:
                    gir_done.append("HASS")
                else:
                    gir_remaining.append(f"HASS ({info['count']}/{info['required']})")
            else:
                if info["done"]:
                    gir_done.append(cat)
                else:
                    gir_remaining.append(cat)
        if gir_done:
            parts.append(f"GIRs complete: {', '.join(gir_done)}")
        if gir_remaining:
            parts.append(f"GIRs remaining: {', '.join(gir_remaining)}")

        return "\n".join(parts)

    def summary(self):
        parts = []
        if self.major_id:
            name = MAJORS.get(self.major_id, {}).get("name", self.major_id)
            parts.append(f"Major: {self.major_id} ({name})")
        if self.year:
            parts.append(f"Year: {self.year}")
        if self.courses_taken:
            parts.append(f"Courses taken: {', '.join(self.courses_taken)}")
        if self.semesters_left:
            parts.append(f"Semesters remaining: {self.semesters_left}")
        if self.interests:
            parts.append(f"Interests: {', '.join(self.interests)}")
        return "\n".join(parts) if parts else "No student info provided yet."

    def update_from_form(self, major_id=None, courses_str=None, year=None,
                         semesters_left=None, next_is_fall=None):
        changed = False
        if major_id and major_id.strip():
            self.major_id = major_id.strip()
            changed = True
        if courses_str and courses_str.strip():
            self.courses_taken = _parse_course_list(courses_str)
            changed = True
        if year and year.strip():
            self.year = year.strip().lower()
        if semesters_left is not None:
            try:
                self.semesters_left = int(semesters_left)
            except (ValueError, TypeError):
                pass
        if next_is_fall is not None:
            self.next_is_fall = next_is_fall
        if changed:
            self.mark_dirty()

    def update_from_message(self, message):
        msg_lower = message.lower()

        # Detect major — try exact patterns first, then keyword matching
        major_patterns = {
            # EECS
            "6-3": [r"\b6-3\b", r"course 6-3", r"cs\s*and\s*eng"],
            "6-4": [r"\b6-4\b", r"course 6-4", r"ai\s*and\s*decision", r"ai\+d"],
            "6-5": [r"\b6-5\b", r"course 6-5", r"ee\s*with\s*comp"],
            "6-7": [r"\b6-7\b", r"course 6-7", r"cs\s*and\s*mol"],
            "6-9": [r"\b6-9\b", r"course 6-9", r"comp.*and\s*cog"],
            "6-14": [r"\b6-14\b", r"course 6-14", r"cs.*econ.*data"],
            # Engineering
            "1-ENG": [r"\b1-eng\b", r"course 1-eng", r"civil.*env.*eng"],
            "1-12": [r"\b1-12\b", r"climate\s*system"],
            "2": [r"\bcourse\s*2\b", r"mech.*eng"],
            "2-A": [r"\b2-a\b", r"course 2-a"],
            "2-OE": [r"\b2-oe\b", r"ocean\s*eng"],
            "3": [r"\bcourse\s*3\b", r"materials?\s*sci"],
            "3-A": [r"\b3-a\b", r"course 3-a"],
            "3-C": [r"\b3-c\b", r"archaeology.*material"],
            "10": [r"\bcourse\s*10\b", r"chem.*eng(?!tic)"],
            "10-B": [r"\b10-b\b", r"chem.*bio.*eng"],
            "10-C": [r"\b10-c\b"],
            "10-ENG": [r"\b10-eng\b"],
            "16": [r"\bcourse\s*16\b", r"aero.*eng", r"aerospace"],
            "16-ENG": [r"\b16-eng\b"],
            "20": [r"\bcourse\s*20\b", r"bio.*eng(?!tic)"],
            "22": [r"\bcourse\s*22\b", r"nuclear.*eng"],
            "22-ENG": [r"\b22-eng\b"],
            # Architecture
            "4": [r"\bcourse\s*4\b", r"architecture\b"],
            "4-B": [r"\b4-b\b", r"art\s*and\s*design"],
            "11": [r"\bcourse\s*11\b", r"\bplanning\b"],
            "11-6": [r"\b11-6\b", r"urban.*sci.*cs"],
            # Science
            "5": [r"\bcourse\s*5\b(?!-)", r"\bchemistry\b(?!.*bio)"],
            "5-7": [r"\b5-7\b", r"chem.*and\s*bio"],
            "7": [r"\bcourse\s*7\b", r"\bbiology\b"],
            "8": [r"\bcourse\s*8\b", r"\bphysics\b"],
            "9": [r"\bcourse\s*9\b", r"brain.*cog"],
            "12": [r"\bcourse\s*12\b", r"earth.*atmo", r"\beaps\b"],
            "18": [r"\bcourse\s*18\b(?!-)", r"\bmathematics\b"],
            "18-C": [r"\b18-c\b", r"math.*with.*cs", r"math.*comp.*sci"],
            # Economics / Sloan
            "14-1": [r"\b14-1\b", r"\beconomics\b(?!.*math)"],
            "14-2": [r"\b14-2\b", r"math.*econ"],
            "15-1": [r"\b15-1\b", r"\bmanagement\b"],
            "15-2": [r"\b15-2\b", r"business\s*analytics"],
            "15-3": [r"\b15-3\b", r"\bfinance\b"],
            # HASS
            "17": [r"\bcourse\s*17\b", r"poli.*sci"],
            "21": [r"\bcourse\s*21\b(?![\w-])", r"\bhumanities\b(?!.*eng|.*sci)"],
            "21A": [r"\b21a\b", r"anthropology"],
            "21E": [r"\b21e\b", r"humanities.*eng"],
            "21G": [r"\b21g\b", r"global.*lang"],
            "21H": [r"\b21h\b", r"\bhistory\b"],
            "21L": [r"\b21l\b", r"\bliterature\b"],
            "21M": [r"\b21m\b", r"\bmusic\b"],
            "21S": [r"\b21s\b", r"humanities.*sci"],
            "21T": [r"\b21t\b", r"theater"],
            "21W": [r"\b21w\b", r"\bwriting\b"],
            "24-1": [r"\b24-1\b", r"\bphilosophy\b"],
            "24-2": [r"\b24-2\b", r"linguistics"],
            "CMS": [r"\bcms\b", r"comparative\s*media"],
            "STS": [r"\bsts\b", r"science.*tech.*society"],
        }
        for mid, patterns in major_patterns.items():
            for pat in patterns:
                if re.search(pat, msg_lower):
                    if self.major_id != mid:
                        self.major_id = mid
                        self.mark_dirty()
                    break

        # Detect year
        year_map = {"freshman": 7, "sophomore": 5, "junior": 3, "senior": 1}
        for year_name, sems in year_map.items():
            if year_name in msg_lower:
                self.year = year_name
                if self.semesters_left is None:
                    self.semesters_left = sems
                break

        # Detect course numbers
        found_courses = re.findall(r'\b(\d{1,2}\.\w{2,5})\b', message)
        added_any = False
        for c in found_courses:
            if c in COURSES and c not in self.courses_taken:
                self.courses_taken.append(c)
                added_any = True
        if added_any:
            self.mark_dirty()


def _parse_course_list(text):
    return re.findall(r'(\d{1,2}\.\w{2,5})', text)


# ─── Intent Detection ───────────────────────────────────────

class Intent:
    COURSE_LOOKUP = "course_lookup"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    REQUIREMENTS = "requirements"
    SCHEDULING = "scheduling"
    PLANNING = "planning"
    SEMESTER_BUILD = "semester_build"
    PROFILE_UPDATE = "profile_update"
    GENERAL = "general"


def detect_intent(message):
    msg_lower = message.lower()

    course_ids = re.findall(r'\b(\d{1,2}\.\w{2,5})\b', message)
    valid_ids = [c for c in course_ids if c in COURSES]

    req_keywords = ["requirement", "still need", "remaining", "what do i need",
                    "how close", "progress", "left to take", "gir", "hass",
                    "ci-h", "ci-m", "rest requirement"]
    schedule_keywords = ["schedule", "conflict", "time slot", "overlap",
                         "fit together", "weekly"]
    plan_keywords = ["plan", "next semester", "graduate", "feasib",
                     "critical path", "how many semesters", "sequence",
                     "what order", "roadmap"]
    semester_build_keywords = ["what else", "fill in", "round out", "complete my semester",
                               "what should i add", "what should i take", "what course",
                               "other courses", "go with", "fourth class", "fourth course",
                               "pair with", "alongside", "take with", "what to take",
                               "suggest", "recommend"]
    profile_keywords = ["i'm a", "i am a", "my major", "i've taken",
                        "i have taken", "my courses", "i'm in course"]

    compare_keywords = ["compare", "versus", " vs ", " or ", "between",
                        "difference between", "which is better", "which one",
                        "choose between", "deciding between", "pick between",
                        "instead of", "rather than"]

    recommend_keywords = ["similar to", "like ", "courses like", "something like",
                          "alternative to", "alternatives to", "instead of",
                          "go deeper", "deeper into", "follow up", "follow-up",
                          "follow on", "follow-on", "build on", "builds on",
                          "after taking", "after finishing", "finished",
                          "enjoyed", "loved", "more like", "what next after",
                          "similar courses", "related to", "related courses"]

    if any(kw in msg_lower for kw in profile_keywords):
        return Intent.PROFILE_UPDATE, {"course_ids": valid_ids}
    if any(kw in msg_lower for kw in req_keywords):
        return Intent.REQUIREMENTS, {"course_ids": valid_ids}

    # Comparison: exactly 2 valid courses + comparison language (or just 2 courses with "or")
    if len(valid_ids) == 2 and any(kw in msg_lower for kw in compare_keywords):
        return Intent.COMPARISON, {"course_ids": valid_ids}

    # Recommendation: 1 anchor course + recommendation language
    if len(valid_ids) == 1 and any(kw in msg_lower for kw in recommend_keywords):
        return Intent.RECOMMENDATION, {"course_ids": valid_ids}

    # Semester build: student mentions specific courses + asking for suggestions
    # Also triggers if student mentions courses + planning keywords (they have courses in mind)
    if valid_ids and any(kw in msg_lower for kw in semester_build_keywords):
        return Intent.SEMESTER_BUILD, {"course_ids": valid_ids}
    if valid_ids and any(kw in msg_lower for kw in plan_keywords):
        return Intent.SEMESTER_BUILD, {"course_ids": valid_ids}

    if any(kw in msg_lower for kw in plan_keywords):
        return Intent.PLANNING, {"course_ids": valid_ids}
    if any(kw in msg_lower for kw in schedule_keywords):
        return Intent.SCHEDULING, {"course_ids": valid_ids}
    if valid_ids:
        return Intent.COURSE_LOOKUP, {"course_ids": valid_ids}

    return Intent.GENERAL, {"course_ids": []}


# ─── Tool Execution ─────────────────────────────────────────

def execute_course_lookup(course_ids):
    results = []
    for cid in course_ids[:5]:
        course = COURSES.get(cid)
        if course:
            results.append({
                "subject_id": course.get("subject_id", cid),
                "title": course.get("title", "Unknown"),
                "description": course.get("description", "No description."),
                "prerequisites": course.get("prerequisites", "None"),
                "total_units": course.get("total_units", "?"),
                "offered_fall": course.get("offered_fall", False),
                "offered_spring": course.get("offered_spring", False),
                "instructors": course.get("instructors", []),
                "rating": course.get("rating", None),
                "in_class_hours": course.get("in_class_hours", None),
                "out_of_class_hours": course.get("out_of_class_hours", None),
                "gir_attribute": course.get("gir_attribute", None),
                "hass_attribute": course.get("hass_attribute", None),
                "communication_requirement": course.get("communication_requirement", None),
            })
        else:
            results.append({"subject_id": cid, "error": "Course not found in catalog."})
    return results


def execute_requirements_check(profile):
    if not _tracker:
        return {"error": "Requirements tracker not available."}
    if not profile.major_id:
        return {"error": "I need to know your major to check requirements. What major are you in?"}
    if not profile.courses_taken:
        return {"error": "I need to know what courses you've taken. Could you list them?"}

    status = _tracker.get_status(profile.major_id, profile.courses_taken)
    gir = status["gir_status"]
    major = status["major_status"]
    parts = []

    # GIR summary
    gir_done, gir_remaining = [], []
    for cat, info in gir.items():
        if cat == "HASS":
            if info["done"]:
                gir_done.append(f"HASS ({info['count']}/{info['required']})")
            else:
                gir_remaining.append(f"HASS: need {info['required'] - info['count']} more")
        else:
            if info["done"]:
                gir_done.append(cat)
            else:
                gir_remaining.append(f"{cat}: need {info['required'] - len(info['fulfilled'])} more")

    parts.append(f"GIRs completed: {', '.join(gir_done) if gir_done else 'none'}")
    if gir_remaining:
        parts.append(f"GIRs still needed: {'; '.join(gir_remaining)}")

    # Major summary
    req = major["required_courses"]
    parts.append(f"Required courses: {len(req['fulfilled'])}/{req['total']} done")
    if req["remaining"]:
        parts.append(f"Still need: {', '.join(req['remaining'])}")

    for group in major["select_groups"]:
        if not group["done"]:
            opts = ", ".join(group["options"][:5])
            if len(group["options"]) > 5:
                opts += f" (+{len(group['options']) - 5} more)"
            parts.append(f"{group['name']}: need {group['remaining_needed']} more — options: {opts}")

    if not major["ci_m"]["done"]:
        parts.append(f"CI-M: need {major['ci_m']['remaining_needed']} more")

    takeable = [t for t in status["takeable_next"] if t["prereqs_met"]]
    if takeable:
        t_str = ", ".join(f"{t['course']} ({t['title'][:30]})" for t in takeable[:8])
        parts.append(f"Ready to take next (prereqs met): {t_str}")

    return {"summary": "\n".join(parts)}


def execute_planning(profile):
    if not _planner:
        return {"error": "Planner not available."}
    if not profile.major_id or not profile.courses_taken:
        return {"error": "I need your major and courses taken to build a plan."}

    semesters = profile.semesters_left or 4
    next_fall = profile.next_is_fall if profile.next_is_fall is not None else True

    feasibility = _planner.check_feasibility(
        profile.major_id, profile.courses_taken, semesters, profile.max_units
    )
    suggestion = _planner.suggest_next_semester(
        profile.major_id, profile.courses_taken, next_fall, profile.max_units
    )

    parts = []
    if feasibility["feasible"]:
        parts.append("Graduation feasibility: ON TRACK")
    else:
        parts.append("Graduation feasibility: AT RISK")
    parts.append(f"Minimum semesters needed: {feasibility['min_semesters_needed']}")
    parts.append(f"Remaining major courses: {feasibility['remaining_major_courses']}")
    parts.append(f"Remaining major units: ~{feasibility['remaining_major_units']}")

    if feasibility["critical_paths"]:
        for path in feasibility["critical_paths"][:2]:
            parts.append(f"Critical prereq chain: {' → '.join(path)} ({len(path)} semesters)")

    if feasibility["warnings"]:
        parts.append("Warnings:")
        for w in feasibility["warnings"]:
            parts.append(f"  - {w}")

    sem_type = "Fall" if next_fall else "Spring"
    parts.append(f"\nSuggested for next {sem_type} ({suggestion['total_units']} units):")
    for s in suggestion["suggestions"]:
        reasons = f" [{', '.join(s['reasons'])}]" if s.get("reasons") else ""
        parts.append(f"  {s['course']} - {s['title']} ({s['units']}u){reasons}")

    return {"summary": "\n".join(parts)}


def execute_scheduling(course_ids):
    """Check schedule conflicts between specific courses."""
    if not _scheduler:
        return {"summary": "Scheduler not available."}
    if len(course_ids) < 2:
        return {"summary": "Need at least 2 courses to check for conflicts."}

    # Check if courses have schedule data
    parts = []
    courses_with_schedules = []
    for cid in course_ids:
        course = COURSES.get(cid, {})
        title = course.get("title", cid)
        if course.get("schedule"):
            courses_with_schedules.append(cid)
        else:
            parts.append(f"{cid} ({title}): no schedule data available")

    if len(courses_with_schedules) < 2:
        parts.append("Not enough courses with schedule data to check conflicts.")
        return {"summary": "\n".join(parts)}

    # Try to find a conflict-free schedule
    results = _scheduler.find_schedules(courses_with_schedules)
    if results:
        total_units = results[0]["total_units"]
        parts.append(
            f"✅ No conflicts found! {len(courses_with_schedules)} courses can be taken together ({total_units} units).")
        for cid in courses_with_schedules:
            c = COURSES.get(cid, {})
            parts.append(f"  {cid} - {c.get('title', '?')} ({c.get('total_units', '?')}u)")
    else:
        parts.append(f"❌ Schedule conflicts detected among: {', '.join(courses_with_schedules)}")
        parts.append("Try dropping one course or look for alternative sections.")

    return {"summary": "\n".join(parts)}


def execute_comparison(course_ids, profile):
    """
    Compare two courses by computing key differentiators.

    Following the design from the planning notebook (Section 4):
    - Compute standard metrics for each course
    - Identify dimensions where they meaningfully differ (differentiators)
    - Rank differentiators by impact (prereq blockage > term exclusivity > workload > rating > critical path)
    - Build context that highlights the top differentiator for the LLM to center the conversation on

    Returns dict with comparison data and LLM instructions.
    """
    if len(course_ids) < 2:
        return {"error": "Need exactly 2 courses to compare."}

    cid_a, cid_b = course_ids[0], course_ids[1]
    course_a = COURSES.get(cid_a, {})
    course_b = COURSES.get(cid_b, {})

    if not course_a:
        return {"error": f"Course {cid_a} not found in catalog."}
    if not course_b:
        return {"error": f"Course {cid_b} not found in catalog."}

    # ── Compute metrics for each course ──
    def _course_metrics(cid, course):
        metrics = {
            "subject_id": cid,
            "title": course.get("title", "Unknown"),
            "units": course.get("total_units", 12),
            "in_class_hours": course.get("in_class_hours") or 0,
            "out_of_class_hours": course.get("out_of_class_hours") or 0,
            "rating": course.get("rating"),
            "offered_fall": course.get("offered_fall", False),
            "offered_spring": course.get("offered_spring", False),
            "description": course.get("description", "No description.")[:400],
            "instructors": course.get("instructors", []),
            "gir_attribute": course.get("gir_attribute"),
            "hass_attribute": course.get("hass_attribute"),
            "communication_requirement": course.get("communication_requirement"),
        }
        metrics["total_hours"] = metrics["in_class_hours"] + metrics["out_of_class_hours"]

        # Offering frequency
        if metrics["offered_fall"] and metrics["offered_spring"]:
            metrics["frequency"] = "both semesters"
        elif metrics["offered_fall"]:
            metrics["frequency"] = "fall only"
        elif metrics["offered_spring"]:
            metrics["frequency"] = "spring only"
        else:
            metrics["frequency"] = "unknown"

        # Prereq status (if student profile available)
        metrics["prereqs_met"] = True
        metrics["missing_prereqs"] = []
        if _tracker and profile and profile.courses_taken:
            sat, missing = _tracker.check_prereqs_satisfied(cid, profile.courses_taken)
            metrics["prereqs_met"] = sat
            metrics["missing_prereqs"] = missing

        # Downstream courses unlocked (critical path impact)
        metrics["downstream_count"] = 0
        if _planner and profile and profile.major_id:
            try:
                paths = _planner.find_critical_paths(profile.major_id, profile.courses_taken)
                for path in paths:
                    if cid in path:
                        idx = path.index(cid)
                        metrics["downstream_count"] = max(
                            metrics["downstream_count"], len(path) - idx - 1
                        )
            except Exception:
                pass

        return metrics

    ma = _course_metrics(cid_a, course_a)
    mb = _course_metrics(cid_b, course_b)

    # ── Identify differentiators ──
    # Each differentiator: (impact_rank, name, description)
    # Lower rank = higher impact. Only include dimensions where courses meaningfully differ.
    differentiators = []

    # 1. Prerequisite blockage (highest impact — constrains what student can actually do)
    if ma["prereqs_met"] != mb["prereqs_met"]:
        blocked = cid_a if not ma["prereqs_met"] else cid_b
        available = cid_b if blocked == cid_a else cid_a
        blocked_missing = ma["missing_prereqs"] if blocked == cid_a else mb["missing_prereqs"]
        differentiators.append((1, "prerequisite_status",
            f"{blocked} has unmet prerequisites ({', '.join(blocked_missing)}), "
            f"while {available} is available now. This is a hard constraint."))

    # 2. Term exclusivity (constrains timing)
    if ma["frequency"] != mb["frequency"]:
        if "only" in ma["frequency"] or "only" in mb["frequency"]:
            differentiators.append((2, "term_availability",
                f"{cid_a} is offered {ma['frequency']}; "
                f"{cid_b} is offered {mb['frequency']}. "
                f"A course offered only once per year should be prioritized when available."))

    # 3. Workload gap (>3 hrs/wk difference is meaningful)
    hrs_diff = abs(ma["total_hours"] - mb["total_hours"])
    if hrs_diff >= 3:
        lighter = cid_a if ma["total_hours"] < mb["total_hours"] else cid_b
        heavier = cid_b if lighter == cid_a else cid_a
        lighter_hrs = min(ma["total_hours"], mb["total_hours"])
        heavier_hrs = max(ma["total_hours"], mb["total_hours"])
        differentiators.append((3, "workload",
            f"{heavier} is significantly heavier (~{heavier_hrs:.0f} hrs/wk) "
            f"compared to {lighter} (~{lighter_hrs:.0f} hrs/wk). "
            f"A {hrs_diff:.0f} hr/wk gap matters for semester balance."))

    # 4. Rating difference (>0.5 points on 7-point scale)
    if ma["rating"] is not None and mb["rating"] is not None:
        rating_diff = abs(ma["rating"] - mb["rating"])
        if rating_diff >= 0.5:
            higher = cid_a if ma["rating"] > mb["rating"] else cid_b
            lower = cid_b if higher == cid_a else cid_a
            differentiators.append((4, "student_rating",
                f"{higher} is rated higher ({max(ma['rating'], mb['rating']):.1f}/7) "
                f"vs {lower} ({min(ma['rating'], mb['rating']):.1f}/7)."))

    # 5. Critical path impact (downstream courses unlocked)
    if ma["downstream_count"] != mb["downstream_count"]:
        more_unlock = cid_a if ma["downstream_count"] > mb["downstream_count"] else cid_b
        fewer_unlock = cid_b if more_unlock == cid_a else cid_a
        more_count = max(ma["downstream_count"], mb["downstream_count"])
        fewer_count = min(ma["downstream_count"], mb["downstream_count"])
        if more_count > 0:
            differentiators.append((5, "critical_path",
                f"{more_unlock} unlocks {more_count} downstream course(s), "
                f"while {fewer_unlock} unlocks {fewer_count}. "
                f"Taking {more_unlock} first keeps more options open."))

    # 6. Units difference
    if ma["units"] != mb["units"]:
        differentiators.append((6, "units",
            f"{cid_a} is {ma['units']} units; {cid_b} is {mb['units']} units."))

    # Sort by impact rank
    differentiators.sort(key=lambda x: x[0])

    # ── Build context string ──
    parts = []
    parts.append(f"COURSE COMPARISON: {cid_a} vs {cid_b}\n")

    # Side-by-side metrics
    parts.append(f"  {cid_a} — {ma['title']}")
    parts.append(f"    Units: {ma['units']}  |  Hours/wk: ~{ma['total_hours']:.0f}  |  "
                 f"Rating: {ma['rating']:.1f}/7" if ma['rating'] else
                 f"    Units: {ma['units']}  |  Hours/wk: ~{ma['total_hours']:.0f}  |  Rating: N/A")
    parts.append(f"    Offered: {ma['frequency']}  |  "
                 f"Prereqs met: {'Yes' if ma['prereqs_met'] else 'No (' + ', '.join(ma['missing_prereqs']) + ')'}")
    if ma["downstream_count"] > 0:
        parts.append(f"    Unlocks {ma['downstream_count']} downstream course(s)")
    tags_a = []
    if ma["gir_attribute"]:
        tags_a.append(ma["gir_attribute"])
    if ma["hass_attribute"]:
        tags_a.append(ma["hass_attribute"])
    if ma["communication_requirement"]:
        tags_a.append(ma["communication_requirement"])
    if tags_a:
        parts.append(f"    Attributes: {', '.join(tags_a)}")
    parts.append(f"    Description: {ma['description']}")
    parts.append("")

    parts.append(f"  {cid_b} — {mb['title']}")
    parts.append(f"    Units: {mb['units']}  |  Hours/wk: ~{mb['total_hours']:.0f}  |  "
                 f"Rating: {mb['rating']:.1f}/7" if mb['rating'] else
                 f"    Units: {mb['units']}  |  Hours/wk: ~{mb['total_hours']:.0f}  |  Rating: N/A")
    parts.append(f"    Offered: {mb['frequency']}  |  "
                 f"Prereqs met: {'Yes' if mb['prereqs_met'] else 'No (' + ', '.join(mb['missing_prereqs']) + ')'}")
    if mb["downstream_count"] > 0:
        parts.append(f"    Unlocks {mb['downstream_count']} downstream course(s)")
    tags_b = []
    if mb["gir_attribute"]:
        tags_b.append(mb["gir_attribute"])
    if mb["hass_attribute"]:
        tags_b.append(mb["hass_attribute"])
    if mb["communication_requirement"]:
        tags_b.append(mb["communication_requirement"])
    if tags_b:
        parts.append(f"    Attributes: {', '.join(tags_b)}")
    parts.append(f"    Description: {mb['description']}")
    parts.append("")

    # Key differentiators
    if differentiators:
        parts.append(f"KEY DIFFERENTIATORS (ranked by impact):")
        for rank, name, desc in differentiators:
            parts.append(f"  {rank}. [{name}] {desc}")
        parts.append("")
        top_diff = differentiators[0]
        parts.append(f"TOP DIFFERENTIATOR: {top_diff[2]}")
    else:
        parts.append("These courses are similar across all measured dimensions (units, workload, rating, availability, prerequisites).")
    parts.append("")

    # Shared properties (not differentiators — don't focus on these)
    shared = []
    if ma["frequency"] == mb["frequency"]:
        shared.append(f"Both offered {ma['frequency']}")
    if ma["prereqs_met"] == mb["prereqs_met"]:
        shared.append(f"Both {'available' if ma['prereqs_met'] else 'blocked by prereqs'}")
    if abs(ma["total_hours"] - mb["total_hours"]) < 3:
        shared.append(f"Similar workload (~{ma['total_hours']:.0f} vs ~{mb['total_hours']:.0f} hrs/wk)")
    if shared:
        parts.append(f"SHARED (not differentiators): {'; '.join(shared)}")
        parts.append("")

    # LLM instructions
    parts.append("INSTRUCTIONS:")
    parts.append("1. Present the side-by-side metrics clearly.")
    parts.append("2. Highlight the TOP DIFFERENTIATOR in a single sentence — make it the focal point.")
    parts.append("3. Do NOT recommend which course is better yet. Instead, end with TWO questions:")
    parts.append("   a) A targeted question about the top differentiator (e.g., 'Is workload a concern this semester?')")
    parts.append("   b) A framing question: 'Are you choosing one of these, or planning to take both at some point?'")
    parts.append("      This matters because choosing one = selection problem, taking both = ordering problem.")
    parts.append("4. Do NOT add generic advice. Do NOT suggest other courses. Focus only on these two.")

    return {"summary": "\n".join(parts), "metrics_a": ma, "metrics_b": mb,
            "differentiators": differentiators}


# ─── Course Recommendation ──────────────────────────────────

def _get_recommendation_embeddings():
    """
    Get embeddings for recommendation. Reuses the scoring module's
    cached embeddings if available, otherwise computes them.
    Returns (cid_list, embedding_matrix, cid_to_idx) or (None, None, None).
    """
    try:
        from scoring import _get_embed_model, _get_course_embeddings
        model = _get_embed_model()
        if model is None:
            return None, None, None
        cid_list, embed_matrix = _get_course_embeddings(COURSES)
        if cid_list is None:
            return None, None, None
        cid_to_idx = {cid: i for i, cid in enumerate(cid_list)}
        return cid_list, embed_matrix, cid_to_idx
    except Exception as e:
        print(f"[chat.py] Warning: recommendation embeddings not available: {e}")
        return None, None, None


def _get_downstream_courses(anchor_cid):
    """Find all courses that list anchor_cid in their prerequisites."""
    downstream = []
    for cid, course in COURSES.items():
        prereq_str = course.get("prerequisites", "")
        if prereq_str and anchor_cid in prereq_str:
            downstream.append(cid)
    return downstream


def _check_prereqs_met(cid, courses_taken):
    """Check if a course's prereqs are satisfied. Returns (met: bool, missing: list)."""
    if _tracker:
        return _tracker.check_prereqs_satisfied(cid, courses_taken)
    # Fallback: simple prereq extraction
    course = COURSES.get(cid, {})
    prereq_str = course.get("prerequisites", "")
    if not prereq_str or prereq_str.lower() == "none":
        return True, []
    mentioned = set(re.findall(r'([\d]+\.[\w.]+)', prereq_str))
    taken_set = set(courses_taken)
    missing = mentioned - taken_set
    return len(missing) == 0, list(missing)


def execute_recommendation(anchor_cid, profile):
    """
    Recommend courses similar to an anchor course.

    Implements the two-scenario model from the course recommendation notebook:
    - Scenario 1 (Alternatives): anchor NOT in student's transcript → find similar courses
    - Scenario 2 (Follow-ons): anchor IN student's transcript → find courses that build on it

    Each scenario uses sentence embeddings (all-MiniLM-L6-v2) for semantic similarity,
    plus prereq graph analysis for follow-ons.

    Returns dict with summary string for LLM context injection.
    """
    course = COURSES.get(anchor_cid)
    if not course:
        return {"error": f"Course {anchor_cid} not found in catalog."}

    anchor_title = course.get("title", "Unknown")
    courses_taken = profile.courses_taken if profile else []
    taken_set = set(courses_taken)

    # Scenario detection: has the student taken this course?
    is_followon = anchor_cid in taken_set

    # Get embeddings
    cid_list, embed_matrix, cid_to_idx = _get_recommendation_embeddings()
    if cid_list is None:
        return {"error": "Course similarity model not available (sentence-transformers not installed)."}

    if anchor_cid not in cid_to_idx:
        return {"error": f"No description embedding available for {anchor_cid}."}

    # Compute similarities
    from sklearn.metrics.pairwise import cosine_similarity
    anchor_idx = cid_to_idx[anchor_cid]
    sims = cosine_similarity(embed_matrix[anchor_idx:anchor_idx+1], embed_matrix)[0]

    # For follow-ons: find downstream courses (list anchor as prereq)
    downstream_set = set()
    if is_followon:
        downstream_set = set(_get_downstream_courses(anchor_cid))

    # Score and partition all courses
    DOWNSTREAM_BONUS = 0.2
    takeable = []
    blocked = []

    for i, sim_score in enumerate(sims):
        cid = cid_list[i]
        if cid == anchor_cid or cid in taken_set:
            continue

        is_ds = cid in downstream_set
        combined = float(sim_score) + (DOWNSTREAM_BONUS if is_ds else 0.0)

        met, missing = _check_prereqs_met(cid, courses_taken)
        hours = _get_course_hours(cid)
        c = COURSES.get(cid, {})

        entry = {
            "cid": cid,
            "title": c.get("title", "Unknown"),
            "units": c.get("total_units", 12),
            "hours": hours,
            "similarity": float(sim_score),
            "combined": combined,
            "is_downstream": is_ds,
            "prereqs_met": met,
            "missing_prereqs": missing,
            "description": c.get("description", "")[:250],
        }

        if met:
            takeable.append(entry)
        else:
            blocked.append(entry)

    # Sort by combined score
    takeable.sort(key=lambda x: x["combined"], reverse=True)
    blocked.sort(key=lambda x: x["combined"], reverse=True)

    # Deduplicate by title (e.g. 6.4210 and 6.4212 both "Robotic Manipulation")
    seen_titles = set()
    deduped_takeable = []
    for entry in takeable:
        if entry["title"] not in seen_titles:
            deduped_takeable.append(entry)
            seen_titles.add(entry["title"])
    takeable = deduped_takeable[:7]
    blocked = blocked[:3]

    # ── Build context string ──
    parts = []

    if is_followon:
        # Scenario 2: Follow-ons
        parts.append(f"FOLLOW-ON COURSES FROM {anchor_cid} — {anchor_title}")
        parts.append(f"The student has already taken {anchor_cid} and wants to go deeper.")
        parts.append("")
        parts.append("INSTRUCTIONS:")
        parts.append(f"- Split results into 'Courses that build directly on {anchor_cid}' (marked below)")
        parts.append("  and 'Related courses in the same area'.")
        parts.append(f"- For each course, write ONE sentence explaining what it adds beyond {anchor_cid}.")
        parts.append("  Use the description — don't just say 'related to the topic'.")
        parts.append("- Show hours/week where available. If hours are 0 or missing, omit the field entirely.")
        parts.append("- Mention blocked courses briefly as future goals with their missing prereqs.")
        parts.append("- End with: 'Are you more interested in the theoretical side, applied/project-based")
        parts.append("  work, or a specific application area (robotics, NLP, etc.)?'")
        parts.append("- Do NOT suggest courses not in the lists below.")
        parts.append("")
        parts.append("RECOMMENDED COURSES (prereqs met):\n")

        for e in takeable:
            hrs_str = f", ~{e['hours']:.0f} hrs/wk" if e["hours"] > 0 else ""
            tag = f"  ** Builds directly on {anchor_cid} **" if e["is_downstream"] else ""
            parts.append(f"  {e['cid']} — {e['title']} ({e['units']}u{hrs_str}){tag}")
            parts.append(f"    Description: {e['description']}")
            parts.append("")

    else:
        # Scenario 1: Alternatives
        parts.append(f"ALTERNATIVE COURSES TO {anchor_cid} — {anchor_title}")
        parts.append(f"The student has NOT taken {anchor_cid} and wants courses covering similar topics.")
        parts.append("")
        parts.append("INSTRUCTIONS:")
        parts.append("- Present all courses in a single ranked list (these are alternatives, not follow-ons).")
        parts.append(f"- For each course, write ONE sentence explaining what it covers and how it")
        parts.append(f"  relates to {anchor_cid}. Use the description — don't just say 'similar'.")
        parts.append("- Show hours/week where available. If hours are 0 or missing, omit the field entirely.")
        parts.append("- Mention blocked courses briefly as future goals with their missing prereqs.")
        parts.append("- End with: 'Are you looking for something with the same theoretical depth,")
        parts.append("  or more of a hands-on / project-based approach?'")
        parts.append("- Do NOT suggest courses not in the lists below.")
        parts.append("")
        parts.append("RECOMMENDED ALTERNATIVES (prereqs met):\n")

        for e in takeable:
            hrs_str = f", ~{e['hours']:.0f} hrs/wk" if e["hours"] > 0 else ""
            parts.append(f"  {e['cid']} — {e['title']} ({e['units']}u{hrs_str})  Similarity: {e['similarity']:.3f}")
            parts.append(f"    Description: {e['description']}")
            parts.append("")

    if blocked:
        parts.append("BLOCKED (prereqs not met — future goals):\n")
        for e in blocked:
            missing_str = ", ".join(e["missing_prereqs"][:3])
            parts.append(f"  {e['cid']} — {e['title']} (needs: {missing_str})")
        parts.append("")

    return {"summary": "\n".join(parts), "scenario": "followon" if is_followon else "alternatives",
            "anchor": anchor_cid, "takeable": takeable, "blocked": blocked}


def get_feasible_candidates(profile):
    """
    Get all feasible candidate courses scored across strategic dimensions.
    Python computes the factual scores; the scoring module ranks them;
    the LLM uses the ranked list + student preferences to make final recommendations.

    Dimensions scored:
      - critical_path: does this unlock future courses? (HIGH/MEDIUM/NONE)
      - scarcity: offering frequency (FALL_ONLY/SPRING_ONLY/BOTH)
      - efficiency: does it double-count across requirements? (DOUBLE_COUNTS/SINGLE)
      - ci_m_value: does student still need CI-M? (NEEDED/NOT_NEEDED)
      - rating: student rating from FireRoad (numeric or None)
      - workload: in-class + out-of-class hours
    """
    import io, contextlib

    if not _tracker:
        return []

    next_fall = profile.next_is_fall if profile.next_is_fall is not None else True
    plan = profile.semester_plan

    # Get remaining requirements with prereqs met
    status = _tracker.get_status(profile.major_id, profile.courses_taken)
    takeable = [t for t in status["takeable_next"] if t["prereqs_met"]]

    # Filter out already taken and already planned
    exclude = set(profile.courses_taken) | set(plan.planned_courses)
    candidates = [t for t in takeable if t["course"] not in exclude]

    # Filter to courses offered this semester
    sem_key = "offered_fall" if next_fall else "offered_spring"
    candidates = [t for t in candidates
                  if COURSES.get(t["course"], {}).get(sem_key, False)]

    # Check unit fit
    remaining_units = profile.max_units - plan.planned_units
    candidates = [t for t in candidates
                  if COURSES.get(t["course"], {}).get("total_units", 12) <= remaining_units]

    # ── Compute strategic dimensions ──

    # Critical path: which courses unlock future courses?
    critical_set = set()
    critical_chain_length = {}
    if _planner:
        paths = _planner.find_critical_paths(profile.major_id, profile.courses_taken)
        for path in paths:
            for i, cid in enumerate(path):
                critical_set.add(cid)
                # How many courses does this unlock? (position in chain)
                courses_after = len(path) - i - 1
                critical_chain_length[cid] = max(
                    critical_chain_length.get(cid, 0), courses_after
                )

    # Efficiency: check if any course appears in multiple select groups
    major = MAJORS.get(profile.major_id, {})
    course_group_count = {}
    for group in major.get("select_groups", []):
        for cid in group.get("courses", []):
            course_group_count[cid] = course_group_count.get(cid, 0) + 1

    # CI-M need
    ci_m_status = status["major_status"]["ci_m"]
    student_needs_ci_m = not ci_m_status["done"]

    # Check conflicts with planned courses
    can_check_conflicts = (_scheduler and plan.planned_courses and
                           any(COURSES.get(c, {}).get("schedule") for c in plan.planned_courses))

    # ── Build enriched candidate list ──
    enriched = []
    for t in candidates:
        cid = t["course"]
        course = COURSES.get(cid, {})
        units = course.get("total_units", 12)

        # Conflict check
        has_conflict = False
        if can_check_conflicts and course.get("schedule"):
            test_set = plan.planned_courses + [cid]
            with contextlib.redirect_stdout(io.StringIO()):
                test_results = _scheduler.find_schedules(test_set)
            if not test_results:
                has_conflict = True

        if has_conflict:
            continue

        freq = _planner.get_offering_frequency(cid) if _planner else "unknown"
        is_ci_m = bool(course.get("communication_requirement", "")
                       and "CI-M" in course.get("communication_requirement", ""))

        # Critical path scoring
        if cid in critical_set:
            unlocks = critical_chain_length.get(cid, 0)
            if unlocks >= 2:
                critical_path = "HIGH"
                critical_detail = f"unlocks {unlocks} downstream courses"
            else:
                critical_path = "MEDIUM"
                critical_detail = f"unlocks {unlocks} downstream course(s)"
        else:
            critical_path = "NONE"
            critical_detail = None

        # Efficiency scoring
        if course_group_count.get(cid, 0) > 1:
            efficiency = "DOUBLE_COUNTS"
            efficiency_detail = f"satisfies {course_group_count[cid]} requirement groups"
        else:
            efficiency = "SINGLE"
            efficiency_detail = None

        # CI-M value
        if is_ci_m and student_needs_ci_m:
            ci_m_value = "NEEDED"
        elif is_ci_m:
            ci_m_value = "AVAILABLE_BUT_MET"
        else:
            ci_m_value = "N/A"

        enriched.append({
            "course_id": cid,
            "title": course.get("title", "Unknown"),
            "units": units,
            "description": course.get("description", "No description available.")[:300],
            "requirement_filled": t["category"],
            # Strategic dimensions
            "critical_path": critical_path,
            "critical_detail": critical_detail,
            "scarcity": freq,  # "both", "fall_only", "spring_only", "unknown"
            "efficiency": efficiency,
            "efficiency_detail": efficiency_detail,
            "ci_m_value": ci_m_value,
            "rating": course.get("rating"),
            "in_class_hours": course.get("in_class_hours"),
            "out_of_class_hours": course.get("out_of_class_hours"),
            "instructors": course.get("instructors", []),
        })

    return enriched


# ─── LLM Signal Detection ───────────────────────────────────

_SIGNAL_SYSTEM_PROMPT = """You are an MIT course advisor analyzing a student's message to understand their priorities.

DIMENSION SIGNALS — The recommendation system has three dimensions. For each, output BOOST (student cares more), REDUCE (student cares less), or KEEP (no signal):
- NECESSITY: graduation requirements, prerequisite chains, timeline pressure, CI-M needs
- INTEREST: personal curiosity, career goals, topic preferences, learning style
- FEASIBILITY: workload concerns, time constraints, course difficulty, ratings

PROFILE SIGNALS — The system has five student profiles. For each, output BOOST (message matches this profile), REDUCE (message contradicts this profile), or KEEP (no signal):
- requirements_optimizer: focused on graduating efficiently, knocking out requirements
- interest_explorer: wants to take interesting courses, explore topics for curiosity
- workload_balancer: prioritizing a manageable semester, avoiding overload
- career_strategist: thinking about career or grad school goals, building toward a specific job
- balanced: no single priority dominates, wants a reasonable mix

Respond with EXACTLY this format, nothing else:
necessity: [BOOST/REDUCE/KEEP]
interest: [BOOST/REDUCE/KEEP]
feasibility: [BOOST/REDUCE/KEEP]
requirements_optimizer: [BOOST/REDUCE/KEEP]
interest_explorer: [BOOST/REDUCE/KEEP]
workload_balancer: [BOOST/REDUCE/KEEP]
career_strategist: [BOOST/REDUCE/KEEP]
balanced: [BOOST/REDUCE/KEEP]"""

_PROFILE_NAMES = ["requirements_optimizer", "interest_explorer",
                  "workload_balancer", "career_strategist", "balanced"]


def _parse_llm_signals(response):
    """Parse LLM signal detection response into dimension signals and profile signals."""
    dim_signals = {}
    profile_signals = {}
    for line in response.strip().split("\n"):
        line = line.strip().lower()

        # Check dimensions
        for dim in ["necessity", "interest", "feasibility"]:
            if dim in line and dim not in ["interest_explorer"]:
                # Avoid matching "interest" inside "interest_explorer"
                # Only match if it's the dimension keyword at the start
                if line.startswith(dim) or (dim + ":" in line and not any(p in line for p in _PROFILE_NAMES)):
                    if "boost" in line:
                        dim_signals[dim] = "BOOST"
                    elif "reduce" in line:
                        dim_signals[dim] = "REDUCE"
                    break

        # Check profiles
        for profile in _PROFILE_NAMES:
            if profile in line:
                if "boost" in line:
                    profile_signals[profile] = "BOOST"
                elif "reduce" in line:
                    profile_signals[profile] = "REDUCE"
                break

    return dim_signals, profile_signals


def _llm_detect_signals(message, client):
    """
    Use the LLM to detect dimension and profile signals from a student message.
    Falls back to regex detect_signals (dimension only) if the LLM call fails.

    Args:
        message: student's message text
        client: HuggingFace InferenceClient instance

    Returns:
        (dim_signals, profile_signals) tuple of dicts
        dim_signals: {dimension: "BOOST" | "REDUCE"}
        profile_signals: {profile_name: "BOOST" | "REDUCE"}
    """
    if client is None:
        return detect_signals(message), {}

    try:
        messages = [
            {"role": "system", "content": _SIGNAL_SYSTEM_PROMPT},
            {"role": "user", "content": f'Student message: "{message}"'},
        ]
        response = client.chat_completion(
            messages=messages,
            max_tokens=120,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        dim_signals, profile_signals = _parse_llm_signals(raw)
        return dim_signals, profile_signals
    except Exception:
        # Fall back to regex on any LLM failure (dimension only, no profile)
        return detect_signals(message), {}


def _build_requirements_summary(profile):
    """Build a concise requirements summary for the initiated stage."""
    if not _tracker or not profile.major_id:
        return None

    try:
        status = _tracker.get_status(profile.major_id, profile.courses_taken)
    except Exception:
        return None

    parts = []
    major_name = MAJORS.get(profile.major_id, {}).get("name", profile.major_id)
    parts.append(f"REQUIREMENTS SUMMARY for {profile.major_id} ({major_name}):")

    # Major progress
    req = status["major_status"]["required_courses"]
    remaining_count = len(req.get("remaining", []))
    total = req.get("total", 0)
    parts.append(f"  Major: {total - remaining_count}/{total} required courses done, {remaining_count} remaining")

    # Unfulfilled select groups
    unfulfilled_groups = [g for g in status["major_status"]["select_groups"] if not g["done"]]
    if unfulfilled_groups:
        group_names = [f"{g['name']} (need {g['remaining_needed']})" for g in unfulfilled_groups[:4]]
        parts.append(f"  Open requirement groups: {', '.join(group_names)}")

    # CI-M
    ci_m = status["major_status"]["ci_m"]
    if not ci_m["done"]:
        parts.append(f"  CI-M: need {ci_m['remaining_needed']} more")

    # GIRs
    gir = status["gir_status"]
    gir_remaining = []
    for cat, info in gir.items():
        if cat == "HASS":
            if not info["done"]:
                gir_remaining.append(f"HASS ({info['count']}/{info['required']})")
        else:
            if not info["done"]:
                gir_remaining.append(cat)
    if gir_remaining:
        parts.append(f"  GIRs still needed: {', '.join(gir_remaining)}")

    # Critical paths
    if _planner:
        try:
            paths = _planner.find_critical_paths(profile.major_id, profile.courses_taken)
            if paths:
                longest = paths[0]
                parts.append(f"  Critical prereq chain: {' → '.join(longest)} ({len(longest)} semesters)")
        except Exception:
            pass

    return "\n".join(parts)


def _get_missing_info(profile, plan):
    """
    Determine what information is missing for good recommendations.
    Returns a list of strings describing what to ask about, ordered by importance.
    """
    missing = []

    # Check if interests have been stated
    has_interest_signal = False
    if plan.scorer:
        # Check if any interest-related signals have been applied
        for signal_set in plan.scorer.signal_history:
            if "interest" in signal_set:
                has_interest_signal = True
                break
    if plan.priority and any(kw in plan.priority.lower() for kw in
                             ["interest", "like", "enjoy", "curious", "career", "goal"]):
        has_interest_signal = True

    if not has_interest_signal:
        missing.append(
            "Interests/goals: What topics or career directions interest the student? (e.g., ML, systems, theory, quant finance)")

    # Check if workload constraints are known
    has_feasibility_signal = False
    if plan.scorer:
        for signal_set in plan.scorer.signal_history:
            if "feasibility" in signal_set:
                has_feasibility_signal = True
                break

    if not has_feasibility_signal:
        missing.append("Workload constraints: Does the student have a UROP, job, athletics, or other time commitments?")

    # Check if they have specific courses in mind
    if not plan.planned_courses:
        missing.append("Specific courses: Does the student already have any courses in mind for this semester?")

    # If we somehow have everything, ask about preferences
    if not missing:
        missing.append(
            "Course format preference: Does the student prefer project-based, lecture-based, or lab-based courses?")

    return missing


def _get_course_hours(cid):
    """Get total estimated hours/week for a course."""
    c = COURSES.get(cid, {})
    ih = c.get("in_class_hours") or 0
    oh = c.get("out_of_class_hours") or 0
    return ih + oh


def _score_to_5(score, ranked_scores):
    """Convert a raw model score to a 1-5 scale based on the ranked score distribution."""
    if not ranked_scores:
        return 3
    min_s = min(ranked_scores)
    max_s = max(ranked_scores)
    if max_s == min_s:
        return 3
    normalized = (score - min_s) / (max_s - min_s)  # 0-1
    return max(1, min(5, round(1 + normalized * 4)))


def _build_semester_plans(ranked, candidates, profile, plan):
    """
    Construct 3 feasible semester plans from ranked candidates.

    Each plan has 4 courses: 3 major + 1 GIR/HASS (or 4 major if no HASS needed).
    No two courses in a plan fill the same requirement group.

    Plan A (Requirements-focused): top courses by overall score
    Plan B (Interest-oriented): top courses by interest dimension score
    Plan C (Lighter load): 4 courses with lowest total hours/week

    Returns list of plan dicts.
    """
    cid_to_info = {c["course_id"]: c for c in candidates}
    max_units = profile.max_units - plan.planned_units

    # Categorize courses
    major_pool = []
    hass_pool = []
    for cid, score, details in ranked:
        info = cid_to_info.get(cid, {})
        req = info.get("requirement_filled", "")
        hours = _get_course_hours(cid)
        entry = {"cid": cid, "score": score, "details": details, "req": req,
                 "units": COURSES.get(cid, {}).get("total_units", 12), "hours": hours}
        if "HASS" in req or req in ("REST", "CI-H"):
            hass_pool.append(entry)
        else:
            major_pool.append(entry)

    def _pick_plan(major_sorted, hass_sorted, target_count=4):
        """Pick courses avoiding duplicate requirement groups. Always tries to fill target_count courses.
        Prefers 3 major + 1 HASS, but fills with extra major courses if HASS pool is empty.
        If a greedy pick blocks the last slot due to unit cap, retries skipping the blocker."""

        def _try_pick(major_list, hass_list, skip_cids=None):
            skip_cids = skip_cids or set()
            reqs_used = set()
            picked = []
            total_units = 0

            hass_available = len([e for e in hass_list if e["cid"] not in skip_cids])
            n_major_target = target_count - (1 if hass_available > 0 else 0)

            # First pass: major courses
            for e in major_list:
                if len(picked) >= n_major_target:
                    break
                if e["cid"] in skip_cids:
                    continue
                if e["req"] and e["req"] in reqs_used:
                    continue
                if total_units + e["units"] > max_units:
                    continue
                picked.append(e)
                total_units += e["units"]
                if e["req"]:
                    reqs_used.add(e["req"])

            # Second pass: HASS/GIR courses
            for e in hass_list:
                if len(picked) >= target_count:
                    break
                if e["cid"] in skip_cids:
                    continue
                if e["req"] and e["req"] in reqs_used:
                    continue
                if total_units + e["units"] > max_units:
                    continue
                picked.append(e)
                total_units += e["units"]
                if e["req"]:
                    reqs_used.add(e["req"])

            # Third pass: backfill with more major courses if under target
            if len(picked) < target_count:
                for e in major_list:
                    if len(picked) >= target_count:
                        break
                    if e["cid"] in skip_cids:
                        continue
                    if any(e["cid"] == p["cid"] for p in picked):
                        continue
                    if e["req"] and e["req"] in reqs_used:
                        continue
                    if total_units + e["units"] > max_units:
                        continue
                    picked.append(e)
                    total_units += e["units"]
                    if e["req"]:
                        reqs_used.add(e["req"])

            total_hours = sum(e["hours"] for e in picked)
            return picked, total_units, total_hours

        # First attempt: greedy
        picked, total_units, total_hours = _try_pick(major_sorted, hass_sorted)

        # If we didn't fill target_count, retry by skipping the largest-unit course
        if len(picked) < target_count:
            # Find courses in the pick that are above 12 units (standard) and try skipping them
            for big_course in sorted(picked, key=lambda e: e["units"], reverse=True):
                if big_course["units"] > 12:
                    retry, ru, rh = _try_pick(major_sorted, hass_sorted, skip_cids={big_course["cid"]})
                    if len(retry) >= target_count:
                        return retry, ru, rh
                    # Also try skipping the big course entirely from pool
            # If still can't fill, return what we have

        return picked, total_units, total_hours

    plans = []

    # Debug: show pool sizes
    print(
        f"[DEBUG _build_semester_plans] major_pool={len(major_pool)}, hass_pool={len(hass_pool)}, max_units={max_units}")
    for e in major_pool[:5]:
        print(f"  major: {e['cid']} req={e['req']} hrs={e['hours']:.0f}")
    for e in hass_pool[:5]:
        print(f"  hass: {e['cid']} req={e['req']} hrs={e['hours']:.0f}")

    # Plan A: Requirements-focused — by overall score (already sorted)
    plan_a, units_a, hours_a = _pick_plan(major_pool, hass_pool)
    print(f"[DEBUG] Plan A: {len(plan_a)} courses, {units_a}u, {hours_a:.0f}hrs — {[e['cid'] for e in plan_a]}")
    if plan_a:
        plans.append({
            "label": "Requirements-focused",
            "description": "Prioritizes courses that make the most progress toward graduation",
            "courses": [(e["cid"], e["units"], e["hours"], e["req"]) for e in plan_a],
            "total_units": units_a,
            "total_hours": hours_a,
        })

    # Plan B: Interest-oriented — sort by interest score, break ties with rating + lower hours
    # This ensures Plan B differs from Plan A even when interest defaults to 0.5
    major_by_interest = sorted(major_pool,
                               key=lambda e: (
                                   e["details"].get("dim_scores", {}).get("interest", 0),
                                   e["details"].get("dim_scores", {}).get("feasibility", 0),  # prefer lighter courses
                               ), reverse=True)
    hass_by_interest = sorted(hass_pool,
                              key=lambda e: (
                                  e["details"].get("dim_scores", {}).get("interest", 0),
                                  e["details"].get("dim_scores", {}).get("feasibility", 0),
                              ), reverse=True)
    plan_b, units_b, hours_b = _pick_plan(major_by_interest, hass_by_interest)
    print(f"[DEBUG] Plan B: {len(plan_b)} courses, {units_b}u, {hours_b:.0f}hrs — {[e['cid'] for e in plan_b]}")
    plan_a_set = set(e["cid"] for e in plan_a) if plan_a else set()
    plan_b_set = set(e["cid"] for e in plan_b) if plan_b else set()
    # If Plan B is identical to Plan A, try a more aggressive alternative:
    # pick courses with highest feasibility (rating + low workload) that aren't in Plan A
    if plan_b_set == plan_a_set:
        major_alt = sorted(
            [e for e in major_pool if e["cid"] not in plan_a_set],
            key=lambda e: e["details"].get("dim_scores", {}).get("interest", 0), reverse=True
        ) + [e for e in major_pool if e["cid"] in plan_a_set]
        plan_b, units_b, hours_b = _pick_plan(major_alt, hass_by_interest)
        plan_b_set = set(e["cid"] for e in plan_b) if plan_b else set()
        print(
            f"[DEBUG] Plan B (alt): {len(plan_b)} courses, {units_b}u, {hours_b:.0f}hrs — {[e['cid'] for e in plan_b]}")
    if plan_b and plan_b_set != plan_a_set:
        plans.append({
            "label": "Interest-oriented",
            "description": "Prioritizes courses matching the student's stated interests",
            "courses": [(e["cid"], e["units"], e["hours"], e["req"]) for e in plan_b],
            "total_units": units_b,
            "total_hours": hours_b,
        })

    # Plan C: Lighter load — same number of courses, sorted by lowest hours
    major_by_hours = sorted(major_pool, key=lambda e: e["hours"])
    hass_by_hours = sorted(hass_pool, key=lambda e: e["hours"])
    plan_c, units_c, hours_c = _pick_plan(major_by_hours, hass_by_hours)
    print(f"[DEBUG] Plan C: {len(plan_c)} courses, {units_c}u, {hours_c:.0f}hrs — {[e['cid'] for e in plan_c]}")
    plan_c_set = set(e["cid"] for e in plan_c) if plan_c else set()
    if plan_c and plan_c_set != plan_a_set and plan_c_set != plan_b_set:
        plans.append({
            "label": "Lighter load",
            "description": "Same number of courses, but picks the ones with the fewest hours/week",
            "courses": [(e["cid"], e["units"], e["hours"], e["req"]) for e in plan_c],
            "total_units": units_c,
            "total_hours": hours_c,
        })

    # Plan D: Technical heavy — 4 major courses, prioritize necessity (no HASS slot)
    major_by_necessity = sorted(major_pool,
                                key=lambda e: e["details"].get("dim_scores", {}).get("necessity", 0), reverse=True)
    plan_d, units_d, hours_d = _pick_plan(major_by_necessity, [], target_count=4)  # empty hass pool forces all major
    print(f"[DEBUG] Plan D: {len(plan_d)} courses, {units_d}u, {hours_d:.0f}hrs — {[e['cid'] for e in plan_d]}")
    plan_d_set = set(e["cid"] for e in plan_d) if plan_d else set()
    existing_sets = [s for s in [plan_a_set, plan_b_set, plan_c_set] if s]
    if plan_d and plan_d_set not in existing_sets:
        plans.append({
            "label": "Technical heavy",
            "description": "All 4 courses are major requirements — maximum progress toward degree",
            "courses": [(e["cid"], e["units"], e["hours"], e["req"]) for e in plan_d],
            "total_units": units_d,
            "total_hours": hours_d,
        })

    return plans


def _detect_plan_selection(msg_lower, suggested_plans):
    """
    Detect if the student is selecting one of the suggested plans by name.
    Returns the matching plan dict, or None if no match.

    Matches on plan labels (e.g., "requirements-focused", "lighter load"),
    plan numbers ("plan 1", "plan A", "the first plan", "option 2"),
    and descriptive references ("the light one", "the ML plan", "interest").
    """
    if not suggested_plans:
        return None

    # Check for plan number references
    number_patterns = [
        (r'\bplan\s*(?:a|1|one)\b|\bfirst\s+(?:plan|option)\b|\boption\s*(?:a|1|one)\b', 0),
        (r'\bplan\s*(?:b|2|two)\b|\bsecond\s+(?:plan|option)\b|\boption\s*(?:b|2|two)\b', 1),
        (r'\bplan\s*(?:c|3|three)\b|\bthird\s+(?:plan|option)\b|\boption\s*(?:c|3|three)\b', 2),
        (r'\bplan\s*(?:d|4|four)\b|\bfourth\s+(?:plan|option)\b|\boption\s*(?:d|4|four)\b', 3),
    ]
    for pattern, idx in number_patterns:
        if re.search(pattern, msg_lower) and idx < len(suggested_plans):
            return suggested_plans[idx]

    # Check for label keyword matches
    label_keywords = {
        "requirements": ["requirement", "progress", "graduate", "efficient"],
        "interest": ["interest", "ml", "ai", "machine learning", "explore", "fun"],
        "lighter": ["light", "easy", "manageable", "less work", "fewer hours"],
        "technical": ["technical", "heavy", "all major", "maximum progress", "hardcore"],
    }

    for plan in suggested_plans:
        plan_label = plan["label"].lower()
        for label_key, keywords in label_keywords.items():
            if label_key in plan_label:
                for kw in keywords:
                    if kw in msg_lower and any(
                            trigger in msg_lower for trigger in
                            ["go with", "like", "pick", "choose", "let's do", "sounds good",
                             "that one", "prefer", "want the", "take the"]
                    ):
                        return plan

    return None


def build_semester_context(profile, stage_override=None, llm_client=None):
    """
    Build the full context string for the semester planning conversation.
    This gets injected into the LLM prompt. Python provides the facts,
    LLM provides the judgment about interests and recommendations.

    If llm_client is provided, uses LLM-based signal detection (preferred).
    Otherwise falls back to regex-based detect_signals.
    """
    plan = profile.semester_plan
    stage = stage_override or plan.stage
    next_fall = profile.next_is_fall if profile.next_is_fall is not None else True
    semester_type = "Fall" if next_fall else "Spring"
    parts = []

    # Current plan state
    parts.append(f"SEMESTER PLANNING — {semester_type}")
    parts.append(f"Unit cap: {profile.max_units}")

    if plan.planned_courses:
        parts.append(f"\nCurrently planned courses:")
        for cid in plan.planned_courses:
            course = COURSES.get(cid, {})
            title = course.get("title", "Unknown")
            units = course.get("total_units", "?")
            parts.append(f"  {cid} - {title} ({units}u)")
        parts.append(f"Total planned: {plan.planned_units} units")
        parts.append(f"Remaining capacity: {profile.max_units - plan.planned_units} units")
    else:
        parts.append("No courses planned yet.")

    if plan.priority:
        parts.append(f"\nStudent's stated priority: {plan.priority}")

    # Stage-specific instructions for the LLM
    if stage == "initiated":
        # Build requirements summary for context
        req_summary = _build_requirements_summary(profile)
        if req_summary:
            parts.append(f"\n{req_summary}")

        # Determine what info is missing for targeted follow-up
        missing_info = _get_missing_info(profile, plan)

        parts.append(f"\nINSTRUCTIONS: The student wants to plan their {semester_type} semester.")
        parts.append("1. Briefly confirm their profile (major, year, courses remaining).")
        parts.append(
            "2. Highlight the most urgent items: critical prerequisite chains, fall/spring-only courses, CI-M if needed.")
        parts.append("3. Ask a SPECIFIC follow-up question to gather the missing information listed below.")
        parts.append("   Do NOT suggest specific courses yet — you need their preferences first.")
        parts.append("   Do NOT give generic advice like 'check prerequisites' or 'talk to your advisor'.")
        parts.append(f"\nMISSING INFORMATION — ask about ONE of these:")
        for item in missing_info:
            parts.append(f"  - {item}")
        parts.append("\nPick the most important missing item and ask about it naturally in conversation.")

    elif stage == "gathering_prefs":
        parts.append(f"\nINSTRUCTIONS: The student has shared their preferences. Present course recommendations.")
        parts.append("")
        parts.append("RESPONSE FORMAT:")
        parts.append("1. Present the SCORED CANDIDATE LIST below. For each course, use the score (1-5),")
        parts.append("   requirement, and description to write a one-sentence reason explaining why this")
        parts.append("   specific course fits THIS student. Do NOT repeat the same reason for every course.")
        parts.append("   Differentiate: what makes each course a distinct option?")
        parts.append("")
        parts.append("2. Present the SUGGESTED PLANS below exactly as provided (they are pre-built by our")
        parts.append("   system to avoid schedule conflicts and duplicate requirements). Show the total hours.")
        parts.append("")
        parts.append("3. End with ONE specific follow-up question that references a concrete tradeoff between")
        parts.append("   the plans. Example: 'Plan A gets you through the prereq chain fastest, but Plan B")
        parts.append("   includes the ML courses you mentioned — which matters more to you right now?'")
        parts.append("")
        parts.append("Do NOT:")
        parts.append("- Suggest courses not in the candidate list")
        parts.append("- Modify the suggested plans (add/remove courses)")
        parts.append("- Add generic advice ('check prerequisites', 'talk to your advisor')")
        parts.append("- Use the same reason for multiple courses\n")

        candidates = get_feasible_candidates(profile)
        plan.feasible_candidates = candidates

        if candidates and _scoring_available and plan.scorer:
            # Apply signals from the student's priority statement
            if plan.priority:
                if llm_client:
                    dim_signals, profile_signals = _llm_detect_signals(plan.priority, llm_client)
                else:
                    dim_signals = detect_signals(plan.priority)
                    profile_signals = {}
                if dim_signals:
                    plan.scorer.apply_signal(dim_signals)
                if profile_signals:
                    plan.apply_profile_signals(profile_signals)

            # Score candidates
            raw_factors = compute_candidate_factors(candidates, profile.max_units, plan.planned_units)

            # Use embedding-based interest scoring if student has stated interests
            if plan.priority:
                filled = fill_interest_from_embeddings(raw_factors, plan.priority, COURSES)
            else:
                filled = fill_interest_defaults(raw_factors)

            ranked = plan.scorer.score_candidates(filled)

            # State info
            state = plan.scorer.get_state()
            w = state["dim_weights"]
            parts.append(
                f"CURRENT WEIGHTS: necessity={w['necessity']:.2f}, interest={w['interest']:.2f}, feasibility={w['feasibility']:.2f}")
            parts.append(f"ACTIVE PROFILE: {plan._active_profile}")
            ps = plan.profile_scores
            parts.append(
                f"PROFILE SCORES: {', '.join(f'{p}={s:.2f}' for p, s in sorted(ps.items(), key=lambda x: -x[1]))}\n")

            # Convert scores to 1-5 scale
            all_scores = [score for _, score, _ in ranked]

            # Build scorecard with 1-5 scores, hours, and description
            top_n = ranked[:7]  # show top 6-7 candidates
            parts.append(f"SCORED CANDIDATE LIST ({len(top_n)} of {len(ranked)} feasible courses):\n")
            for rank, (cid, score, details) in enumerate(top_n, 1):
                c = next((x for x in candidates if x["course_id"] == cid), {})
                ds = details.get("dim_scores", {})
                score_5 = _score_to_5(score, all_scores)
                hours = _get_course_hours(cid)

                parts.append(
                    f"  #{rank}  {cid} — {c.get('title', '?')} ({c.get('units', '?')}u, ~{hours:.0f} hrs/wk)  Score: {score_5}/5")
                parts.append(f"    Requirement: {c.get('requirement_filled', '?')}")

                # Key context for differentiated reasons
                tags = []
                if c.get("critical_path", "NONE") != "NONE":
                    tags.append(f"Critical path: {c.get('critical_detail', 'unlocks future courses')}")
                if c.get("scarcity") not in ("both", "unknown", None):
                    tags.append(f"Only offered in {c['scarcity'].replace('_', ' ')} — take now or wait a year")
                if c.get("efficiency") == "DOUBLE_COUNTS":
                    tags.append(f"Double-counts: {c.get('efficiency_detail', 'satisfies multiple groups')}")
                if c.get("ci_m_value") == "NEEDED":
                    tags.append("Fulfills CI-M requirement")
                if c.get("rating"):
                    tags.append(f"Student rating: {c['rating']:.1f}/7")
                if tags:
                    parts.append(f"    Key facts: {'; '.join(tags)}")
                parts.append(f"    Description: {c.get('description', '')[:250]}")
                parts.append("")

            # Build plans in Python
            semester_plans = _build_semester_plans(ranked, candidates, profile, plan)
            plan.suggested_plans = semester_plans  # store for plan selection by name
            if semester_plans:
                parts.append("SUGGESTED PLANS (pre-built, no requirement conflicts):\n")
                for sp in semester_plans:
                    parts.append(f"  {sp['label']} ({sp['total_units']}u, ~{sp['total_hours']:.0f} hrs/wk total):")
                    parts.append(f"    {sp['description']}")
                    for cid, units, hours, req in sp["courses"]:
                        title = COURSES.get(cid, {}).get("title", "?")
                        parts.append(f"    - {cid} — {title} ({units}u, ~{hours:.0f} hrs/wk) [{req}]")
                    parts.append("")

        elif candidates:
            parts.append(f"CANDIDATE LIST ({len(candidates)} courses):\n")
            for c in candidates:
                parts.append(f"  {c['course_id']} — {c['title']} ({c['units']}u) — {c['requirement_filled']}")
                parts.append(f"    Description: {c['description'][:200]}")
                parts.append("")
        else:
            parts.append("No additional feasible candidates found.")

    elif stage == "suggesting":
        remaining_units = profile.max_units - plan.planned_units
        remaining_slots = max(0, 4 - len(plan.planned_courses))
        total_planned_hours = sum(_get_course_hours(cid) for cid in plan.planned_courses)

        # Show current plan
        parts.append(
            f"\nCURRENT PLAN ({len(plan.planned_courses)}/4 courses, {plan.planned_units}u, ~{total_planned_hours:.0f} hrs/wk):")
        for cid in plan.planned_courses:
            course = COURSES.get(cid, {})
            title = course.get("title", "?")
            units = course.get("total_units", "?")
            hours = _get_course_hours(cid)
            parts.append(f"  ✓ {cid} — {title} ({units}u, ~{hours:.0f} hrs/wk)")

        if remaining_slots == 0 or remaining_units < 12:
            # Plan is full
            parts.append(
                f"\nThe semester plan is complete ({plan.planned_units}u, ~{total_planned_hours:.0f} hrs/wk total).")
            parts.append("\nINSTRUCTIONS: The plan is full. Summarize the final plan.")
            parts.append("Show all 4 courses with units, hours, and requirements.")
            parts.append("Ask if the student is happy with this plan or wants to swap anything.")

        else:
            # Still have slots to fill — show scored candidates for remaining slots
            parts.append(f"\n{remaining_slots} slot(s) remaining ({remaining_units}u available)\n")

            candidates = get_feasible_candidates(profile)
            plan.feasible_candidates = candidates

            if candidates and _scoring_available and plan.scorer:
                # Score remaining candidates
                raw_factors = compute_candidate_factors(candidates, profile.max_units, plan.planned_units)
                if plan.priority:
                    filled = fill_interest_from_embeddings(raw_factors, plan.priority, COURSES)
                else:
                    filled = fill_interest_defaults(raw_factors)
                ranked = plan.scorer.score_candidates(filled)
                all_scores = [score for _, score, _ in ranked]

                # Show top candidates for remaining slots
                n_show = min(remaining_slots + 3, len(ranked), 6)  # show a few more than slots remaining
                parts.append(f"INSTRUCTIONS: The student has {remaining_slots} slot(s) left.")
                if remaining_slots == 1:
                    parts.append("Recommend the single best course from the list below with a clear reason.")
                    parts.append("Also mention the runner-up as an alternative.")
                else:
                    parts.append(f"Recommend the top {remaining_slots} courses, with a brief reason for each.")
                    parts.append("Also mention 1-2 alternatives if the student doesn't like the top picks.")
                parts.append("For each course, use the score, hours, and key facts to explain the recommendation.")
                parts.append("End with a specific question: ask if they want these courses or prefer an alternative.")
                parts.append("Do NOT suggest courses not in this list. Do NOT add boilerplate advice.\n")

                parts.append(f"TOP CANDIDATES FOR REMAINING SLOT(S):\n")
                for rank, (cid, score, details) in enumerate(ranked[:n_show], 1):
                    c = next((x for x in candidates if x["course_id"] == cid), {})
                    score_5 = _score_to_5(score, all_scores)
                    hours = _get_course_hours(cid)

                    parts.append(
                        f"  #{rank}  {cid} — {c.get('title', '?')} ({c.get('units', '?')}u, ~{hours:.0f} hrs/wk)  Score: {score_5}/5")
                    parts.append(f"    Requirement: {c.get('requirement_filled', '?')}")

                    tags = []
                    if c.get("critical_path", "NONE") != "NONE":
                        tags.append(f"Critical path: {c.get('critical_detail', 'unlocks future courses')}")
                    if c.get("scarcity") not in ("both", "unknown", None):
                        tags.append(f"Only offered in {c['scarcity'].replace('_', ' ')}")
                    if c.get("ci_m_value") == "NEEDED":
                        tags.append("Fulfills CI-M requirement")
                    if c.get("rating"):
                        tags.append(f"Rating: {c['rating']:.1f}/7")
                    if tags:
                        parts.append(f"    Key facts: {'; '.join(tags)}")
                    parts.append(f"    Description: {c.get('description', '')[:200]}")

                    # Show what total plan would look like with this course
                    new_total_hrs = total_planned_hours + hours
                    new_total_units = plan.planned_units + (c.get("units") or 12)
                    parts.append(f"    → Plan with this course: {new_total_units}u, ~{new_total_hrs:.0f} hrs/wk total")
                    parts.append("")

            elif candidates:
                parts.append(f"REMAINING OPTIONS ({len(candidates)}):")
                for c in candidates[:6]:
                    hours = _get_course_hours(c["course_id"])
                    parts.append(
                        f"  {c['course_id']} — {c['title']} ({c['units']}u, ~{hours:.0f} hrs/wk) — {c['requirement_filled']}")
            else:
                parts.append("No additional feasible candidates found.")

    elif stage == "finalizing":
        total_hours = sum(_get_course_hours(cid) for cid in plan.planned_courses)
        remaining_units = profile.max_units - plan.planned_units
        parts.append(f"\nFINAL SEMESTER PLAN ({plan.planned_units}u, ~{total_hours:.0f} hrs/wk total):")
        for cid in plan.planned_courses:
            course = COURSES.get(cid, {})
            title = course.get("title", "?")
            units = course.get("total_units", "?")
            hours = _get_course_hours(cid)
            parts.append(f"  {cid} — {title} ({units}u, ~{hours:.0f} hrs/wk)")
        parts.append(f"\nINSTRUCTIONS: Present the final semester plan clearly.")
        parts.append("For each course, mention what requirement it fulfills.")
        parts.append("Give the total units and estimated hours per week.")
        if len(plan.planned_courses) < 4 and remaining_units >= 12:
            parts.append(
                f"Note: the student has {remaining_units} units of remaining capacity — they could add another course (HASS, elective, etc.) if they want.")
        else:
            parts.append(
                f"This plan is COMPLETE at {len(plan.planned_courses)} courses and {plan.planned_units} units. The semester is full. Do not suggest adding more courses.")
        parts.append("Ask if they're happy with this plan or want to swap any courses.")

    return "\n".join(parts)


# ─── System Prompt ──────────────────────────────────────────

SYSTEM_PROMPT = """You are an MIT course advisor chatbot helping students navigate degree planning across all MIT majors. You can help with:
- Understanding course content, prerequisites, and workload
- Tracking progress toward degree requirements
- Planning which courses to take each semester
- Navigating GIR, HASS, CI-H, CI-M, REST, and Lab requirements
- Recommending courses based on interests

You know about all MIT undergraduate majors including:
- Engineering: 1-ENG, 2, 2-A, 2-OE, 3, 3-A, 3-C, 10, 10-B, 10-C, 10-ENG, 16, 16-ENG, 20, 22, 22-ENG
- EECS/Computing: 6-3, 6-4, 6-5, 6-7, 6-9, 6-14, 11-6, 18-C
- Science: 5, 5-7, 7, 8, 9, 12, 18
- HASS: 14-1, 14-2, 17, 21, 21A, 21E, 21G, 21H, 21L, 21M, 21S, 21T, 21W, 24-1, 24-2, CMS, STS
- Sloan: 15-1, 15-2, 15-3
- Architecture: 4, 4-B, 11
- Interdisciplinary: 1-12

IMPORTANT RULES:
1. When you receive FACTUAL DATA from our tools, present ONLY that data. Do NOT add courses, requirements, or suggestions that are not in the tool output.
2. NEVER invent or hallucinate course numbers, course names, or degree requirements. If a course is not in the tool data, do not mention it.
3. If you don't have enough info, ask the student rather than guessing.
4. If the student hasn't shared their major or courses taken, ask them.
5. Be concise but helpful.
6. Use official course numbers (e.g., 6.3900).
7. Clearly distinguish facts from opinions.
8. For majors where we have partial data, let the student know they should verify with their department advisor.
"""


# ─── Chatbot Class ──────────────────────────────────────────

class Chatbot:

    def __init__(self):
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.profile = StudentProfile()

    def format_prompt(self, user_input, tool_context=None, history=None):
        messages = []

        # System prompt + profile + ground truth
        system = SYSTEM_PROMPT
        if self.profile.is_complete():
            system += f"\n\nCURRENT STUDENT PROFILE:\n{self.profile.summary()}"

            # Inject ground truth requirements — this is the authoritative source
            ground_truth = self.profile.get_ground_truth()
            if ground_truth:
                system += f"\n\n{ground_truth}"
                system += ("\n\nThe GROUND TRUTH above is the authoritative source for this student's "
                           "requirements. ONLY reference requirements listed above. Do NOT mention "
                           "requirements from other majors.")

        elif self.profile.major_id or self.profile.courses_taken:
            system += f"\n\nPARTIAL STUDENT PROFILE:\n{self.profile.summary()}"
            system += "\n(Profile incomplete — ask for missing info if needed.)"

        messages.append({"role": "system", "content": system})

        # Conversation history (last 12 messages ≈ 6 turns)
        if history:
            # Gradio 6 format: list of {"role": "user"/"assistant", "content": "..."}
            # Gradio 4/5 format: list of [user_msg, bot_msg] pairs
            recent = history[-12:] if len(history) > 12 else history
            for item in recent:
                if isinstance(item, dict):
                    # Gradio 6 format
                    role = item.get("role", "")
                    content = item.get("content", "")
                    if role in ("user", "assistant") and content:
                        messages.append({"role": role, "content": content})
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    # Gradio 4/5 format
                    user_msg, bot_msg = item
                    if user_msg:
                        messages.append({"role": "user", "content": user_msg})
                    if bot_msg:
                        messages.append({"role": "assistant", "content": bot_msg})

        # Current message with tool context
        if tool_context:
            content = (
                f"The student asked: {user_input}\n\n"
                f"Factual data from our system:\n---\n{tool_context}\n---\n\n"
                f"Present ONLY the information above clearly and helpfully. "
                f"Do NOT suggest any courses, requirements, or facts that are not in the data above. "
                f"If the student needs more information beyond what the tool returned, "
                f"say so and ask a follow-up question rather than guessing."
            )
        else:
            content = user_input

        messages.append({"role": "user", "content": content})
        return messages

    def get_response(self, user_input, history=None):
        # Step 1: Update profile
        self.profile.update_from_message(user_input)

        # Step 2: Detect intent
        intent, data = detect_intent(user_input)
        tool_context = None

        # Debug: trace routing
        plan = self.profile.semester_plan
        print(f"[DEBUG ROUTING] intent={intent}, plan.active={plan.active}, plan.stage={plan.stage}")
        print(f"[DEBUG ROUTING] planned_courses={plan.planned_courses}, planned_units={plan.planned_units}")
        print(f"[DEBUG ROUTING] suggested_plans={len(plan.suggested_plans)}, priority={plan.priority is not None}")

        # Handle ongoing semester planning FIRST — this takes priority over intent detection
        # because messages during planning (plan selection, course picks, preferences) often
        # contain keywords that match other intents ("plan", course IDs, etc.)
        if self.profile.semester_plan.active and tool_context is None:
            plan = self.profile.semester_plan
            msg_lower = user_input.lower()

            # Apply LLM signal detection on every message during planning
            if plan.scorer and _scoring_available:
                dim_signals, profile_signals = _llm_detect_signals(user_input, self.client)
                if dim_signals:
                    plan.scorer.apply_signal(dim_signals)
                if profile_signals:
                    plan.apply_profile_signals(profile_signals)

            # Check if student is adding a course by mentioning course IDs
            course_ids = re.findall(r'\b(\d{1,2}\.\w{2,5})\b', user_input)
            valid_ids = [c for c in course_ids if c in COURSES]

            # Check if student is selecting a suggested plan by name
            selected_plan = _detect_plan_selection(msg_lower, plan.suggested_plans)

            print(f"[DEBUG get_response] stage={plan.stage}, valid_ids={valid_ids}")
            print(f"[DEBUG get_response] suggested_plans={len(plan.suggested_plans)} plans")
            print(f"[DEBUG get_response] plan labels: {[p['label'] for p in plan.suggested_plans]}")
            print(f"[DEBUG get_response] selected_plan={'YES: ' + selected_plan['label'] if selected_plan else 'None'}")
            print(f"[DEBUG get_response] msg_lower: {msg_lower[:80]}")

            if selected_plan:
                # Student picked a plan — add all its courses
                for cid, units, hours, req in selected_plan["courses"]:
                    plan.add_course(cid, COURSES.get(cid, {}).get("total_units", units))
                plan.stage = "finalizing"
                tool_context = build_semester_context(self.profile, llm_client=self.client)

            elif valid_ids and plan.stage in ("gathering_prefs", "suggesting"):
                # Student picked course(s)
                for cid in valid_ids:
                    course = COURSES.get(cid, {})
                    if course:
                        plan.add_course(cid, course.get("total_units", 12))
                if len(plan.planned_courses) >= 4 or plan.planned_units >= self.profile.max_units - 6:
                    plan.stage = "finalizing"
                else:
                    plan.stage = "suggesting"
                tool_context = build_semester_context(self.profile, llm_client=self.client)

            elif plan.stage == "initiated":
                plan.priority = user_input
                plan.stage = "gathering_prefs"
                tool_context = build_semester_context(self.profile, llm_client=self.client)

            elif plan.stage == "gathering_prefs":
                plan.priority = user_input
                plan.stage = "gathering_prefs"
                tool_context = build_semester_context(self.profile, llm_client=self.client)

            elif any(kw in msg_lower for kw in ["done", "looks good", "that's it", "happy with",
                                                "finalize", "good semester", "all set"]):
                plan.stage = "finalizing"
                tool_context = build_semester_context(self.profile, llm_client=self.client)

            else:
                tool_context = build_semester_context(self.profile, llm_client=self.client)

        # Step 3: Execute tools (only if not already handled by active planning above)
        if intent == Intent.COURSE_LOOKUP and data["course_ids"] and tool_context is None:
            results = execute_course_lookup(data["course_ids"])
            tool_context = json.dumps(results, indent=2)

        elif intent == Intent.REQUIREMENTS and tool_context is None:
            if self.profile.is_complete():
                result = execute_requirements_check(self.profile)
                tool_context = result.get("summary", result.get("error", ""))
            else:
                missing = []
                if not self.profile.major_id:
                    missing.append("your major (e.g., 6-3, 6-4)")
                if not self.profile.courses_taken:
                    missing.append("what courses you've taken")
                tool_context = f"MISSING INFO: To check requirements, I need: {', '.join(missing)}."

        elif intent == Intent.PLANNING and tool_context is None:
            if self.profile.is_complete():
                # Route to the scored semester planning flow
                plan = self.profile.semester_plan
                plan.active = True
                plan.semester_type = "fall" if self.profile.next_is_fall else "spring"
                plan.stage = "initiated"

                # Apply signals from this message
                if plan.scorer and _scoring_available:
                    dim_signals, profile_signals = _llm_detect_signals(user_input, self.client)
                    if dim_signals:
                        plan.scorer.apply_signal(dim_signals)
                    if profile_signals:
                        plan.apply_profile_signals(profile_signals)

                tool_context = build_semester_context(self.profile, llm_client=self.client)
            else:
                missing = []
                if not self.profile.major_id:
                    missing.append("your major")
                if not self.profile.courses_taken:
                    missing.append("courses you've taken")
                tool_context = f"MISSING INFO: To build a plan, I need: {', '.join(missing)}."

        elif intent == Intent.SCHEDULING and data["course_ids"] and tool_context is None:
            result = execute_scheduling(data["course_ids"])
            tool_context = result.get("summary", "")

        elif intent == Intent.COMPARISON and data["course_ids"] and tool_context is None:
            result = execute_comparison(data["course_ids"], self.profile)
            tool_context = result.get("summary", result.get("error", ""))

        elif intent == Intent.RECOMMENDATION and data["course_ids"] and tool_context is None:
            anchor = data["course_ids"][0]
            result = execute_recommendation(anchor, self.profile)
            tool_context = result.get("summary", result.get("error", ""))

        elif intent == Intent.SEMESTER_BUILD and data["course_ids"] and tool_context is None:
            if not self.profile.is_complete():
                missing = []
                if not self.profile.major_id:
                    missing.append("your major")
                if not self.profile.courses_taken:
                    missing.append("courses you've taken")
                tool_context = f"MISSING INFO: To suggest courses for your semester, I need: {', '.join(missing)}."
            else:
                plan = self.profile.semester_plan
                # Initialize or update the semester plan
                if not plan.active:
                    plan.active = True
                    plan.semester_type = "fall" if self.profile.next_is_fall else "spring"

                # Add any mentioned courses to the plan
                for cid in data["course_ids"]:
                    course = COURSES.get(cid, {})
                    if course:
                        plan.add_course(cid, course.get("total_units", 12))

                # Apply signals from this message
                if plan.scorer and _scoring_available:
                    dim_signals, profile_signals = _llm_detect_signals(user_input, self.client)
                    if dim_signals:
                        plan.scorer.apply_signal(dim_signals)
                    if profile_signals:
                        plan.apply_profile_signals(profile_signals)

                # Student already has courses — go straight to suggesting remaining
                plan.stage = "suggesting"
                tool_context = build_semester_context(self.profile, llm_client=self.client)

        elif intent == Intent.PROFILE_UPDATE and tool_context is None:
            if self.profile.is_complete():
                tool_context = (
                    f"Student profile updated:\n{self.profile.summary()}\n\n"
                    f"Acknowledge the update and ask what they'd like help with."
                )
            else:
                tool_context = (
                    f"Partial profile:\n{self.profile.summary()}\n\n"
                    f"Thank them and ask for any missing details (major, courses taken)."
                )

        # Step 4: Assemble prompt and call LLM
        messages = self.format_prompt(user_input, tool_context, history)

        # Use more tokens for semester planning responses (scored lists + plans are long)
        planning_active = self.profile.semester_plan.active
        token_limit = 1500 if planning_active else 800

        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=token_limit,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "overloaded" in error_msg.lower():
                return ("I'm currently busy — please try again in a few seconds. "
                        "(Free-tier model gets overloaded sometimes.)")
            elif "401" in error_msg or "token" in error_msg.lower():
                return "API auth error — check your HF_TOKEN in the .env file."
            else:
                return f"Sorry, I hit an error: {error_msg}"

    def update_profile_from_form(self, major_id=None, courses_str=None,
                                 year=None, semesters_left=None, next_is_fall=None):
        self.profile.update_from_form(major_id, courses_str, year,
                                      semesters_left, next_is_fall)