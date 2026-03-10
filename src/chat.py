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


# ─── Student Profile ────────────────────────────────────────

class SemesterPlan:
    """Tracks the state of an in-progress semester being built."""

    def __init__(self):
        self.active = False
        self.semester_type = None       # "fall" or "spring"
        self.planned_courses = []       # courses picked so far
        self.planned_units = 0
        self.priority = None            # student's stated priority/interests
        self.stage = None               # "initiated", "gathering_prefs", "suggesting", "finalizing"
        self.feasible_candidates = []   # pre-vetted by Python

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
        return "\n".join(parts)


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
                              "what should i add", "other courses", "go with",
                              "pair with", "alongside", "take with"]
    profile_keywords = ["i'm a", "i am a", "my major", "i've taken",
                       "i have taken", "my courses", "i'm in course"]

    if any(kw in msg_lower for kw in profile_keywords):
        return Intent.PROFILE_UPDATE, {"course_ids": valid_ids}
    if any(kw in msg_lower for kw in req_keywords):
        return Intent.REQUIREMENTS, {"course_ids": valid_ids}

    # Semester build: student mentions specific courses + asking for suggestions
    if valid_ids and any(kw in msg_lower for kw in semester_build_keywords):
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
                opts += f" (+{len(group['options'])-5} more)"
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
        parts.append(f"✅ No conflicts found! {len(courses_with_schedules)} courses can be taken together ({total_units} units).")
        for cid in courses_with_schedules:
            c = COURSES.get(cid, {})
            parts.append(f"  {cid} - {c.get('title', '?')} ({c.get('total_units', '?')}u)")
    else:
        parts.append(f"❌ Schedule conflicts detected among: {', '.join(courses_with_schedules)}")
        parts.append("Try dropping one course or look for alternative sections.")

    return {"summary": "\n".join(parts)}


def get_feasible_candidates(profile):
    """
    Get all feasible candidate courses scored across strategic dimensions.
    Python computes the factual scores; the LLM uses them + student preferences
    to make final recommendations.

    Dimensions scored:
      - critical_path: does this unlock future courses? (HIGH/MEDIUM/NONE)
      - scarcity: offering frequency (FALL_ONLY/SPRING_ONLY/BOTH)
      - efficiency: does it double-count across requirements? (DOUBLE_COUNTS/SINGLE)
      - ci_m_value: does student still need CI-M? (NEEDED/NOT_NEEDED)
      - timeline_pressure: how tight is the student's remaining schedule? (HIGH/MEDIUM/LOW)
      - rating: student rating from FireRoad (numeric or None)
      - units: course weight (numeric)
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

    # Timeline pressure
    semesters_left = profile.semesters_left or 4
    remaining_groups = sum(1 for g in status["major_status"]["select_groups"] if not g["done"])
    remaining_required = len(status["major_status"]["required_courses"]["remaining"])
    total_remaining = remaining_groups + remaining_required
    if semesters_left <= 2 and total_remaining > 4:
        timeline_pressure = "HIGH"
    elif semesters_left <= 3 and total_remaining > 6:
        timeline_pressure = "MEDIUM"
    else:
        timeline_pressure = "LOW"

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
            "timeline_pressure": timeline_pressure,
            "rating": course.get("rating"),
            "in_class_hours": course.get("in_class_hours"),
            "out_of_class_hours": course.get("out_of_class_hours"),
            "instructors": course.get("instructors", []),
        })

    return enriched


def build_semester_context(profile, stage_override=None):
    """
    Build the full context string for the semester planning conversation.
    This gets injected into the LLM prompt. Python provides the facts,
    LLM provides the judgment about interests and recommendations.
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
        parts.append(f"\nINSTRUCTIONS: The student just started planning their {semester_type} semester.")
        parts.append("Confirm the planned course(s), show units and what requirement they fill.")
        parts.append("Then ask the student what their priority is for the rest of the semester:")
        parts.append("- Knock out core/GIR requirements")
        parts.append("- Explore an area of interest (ask what interests them)")
        parts.append("- Keep the workload balanced")
        parts.append("- Something specific they have in mind")
        parts.append("Do NOT suggest any courses yet. Just ask about priorities.")

    elif stage == "gathering_prefs":
        parts.append(f"\nINSTRUCTIONS: The student stated their priorities. Now recommend courses.")
        parts.append("Below is a SCORECARD of feasible candidates with strategic scores across multiple dimensions.")
        parts.append("Use ONLY these candidates. Do NOT suggest any other courses.")
        parts.append("Consider the student's stated priorities AND the strategic scores to pick 3-5 best options.")
        parts.append("For each recommendation, explain WHY it's a good fit — reference both the student's interests")
        parts.append("and the strategic factors (e.g., 'this is fall-only so take it now' or 'this unlocks 2 future courses').")
        parts.append("If a student's interest conflicts with strategic urgency, note the tradeoff and let them decide.")
        parts.append("Ask which one(s) they'd like to add.\n")

        candidates = get_feasible_candidates(profile)
        plan.feasible_candidates = candidates

        if candidates:
            parts.append(f"CANDIDATE SCORECARD ({len(candidates)} courses, all prereqs met, no schedule conflicts):")
            parts.append(f"Timeline pressure for this student: {candidates[0]['timeline_pressure'] if candidates else 'LOW'}\n")
            for c in candidates:
                parts.append(f"  {c['course_id']} — {c['title']} ({c['units']}u)")
                parts.append(f"    Requirement: {c['requirement_filled']}")

                # Strategic tags
                tags = []
                if c["critical_path"] != "NONE":
                    tags.append(f"Critical path: {c['critical_path']} ({c['critical_detail']})")
                if c["scarcity"] not in ("both", "unknown"):
                    tags.append(f"Scarcity: {c['scarcity'].upper().replace('_', ' ')} — take now or wait a year")
                if c["efficiency"] == "DOUBLE_COUNTS":
                    tags.append(f"Efficiency: {c['efficiency_detail']}")
                if c["ci_m_value"] == "NEEDED":
                    tags.append("CI-M: student still needs this — counts toward CI-M requirement")
                if c.get("rating"):
                    tags.append(f"Rating: {c['rating']}/5")
                if c.get("in_class_hours") and c.get("out_of_class_hours"):
                    total_hrs = (c["in_class_hours"] or 0) + (c["out_of_class_hours"] or 0)
                    tags.append(f"Workload: ~{total_hrs:.0f} hrs/week")

                if tags:
                    for tag in tags:
                        parts.append(f"    • {tag}")
                else:
                    parts.append(f"    • No special strategic flags")

                parts.append(f"    Description: {c['description']}")
                parts.append("")
        else:
            parts.append("No additional feasible candidates found.")

    elif stage == "suggesting":
        parts.append(f"\nINSTRUCTIONS: The student chose a course to add.")
        parts.append("Confirm the addition, show updated plan with total units.")

        remaining = profile.max_units - plan.planned_units
        if remaining >= 12:
            parts.append("Then show remaining options with their strategic scores and ask if they want more.\n")
            candidates = get_feasible_candidates(profile)
            plan.feasible_candidates = candidates
            if candidates:
                parts.append(f"REMAINING CANDIDATES ({len(candidates)}):")
                for c in candidates[:10]:
                    tags = []
                    if c["critical_path"] != "NONE":
                        tags.append(f"critical:{c['critical_path']}")
                    if c["scarcity"] not in ("both", "unknown"):
                        tags.append(f"{c['scarcity']}")
                    if c["efficiency"] == "DOUBLE_COUNTS":
                        tags.append("double-counts")
                    if c["ci_m_value"] == "NEEDED":
                        tags.append("CI-M needed")
                    tag_str = f" [{', '.join(tags)}]" if tags else ""
                    parts.append(f"  {c['course_id']} — {c['title']} ({c['units']}u) — {c['requirement_filled']}{tag_str}")
                    parts.append(f"    Description: {c['description'][:150]}")
        else:
            parts.append("The semester is nearly full. Ask if they're happy with the plan or want to swap anything.")

    elif stage == "finalizing":
        parts.append(f"\nINSTRUCTIONS: Summarize the final semester plan.")
        parts.append("List all planned courses with units and requirements filled.")
        parts.append("Mention total units and any notes (e.g., consider adding a HASS/CI-H if needed).")

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

        # Step 3: Execute tools
        if intent == Intent.COURSE_LOOKUP and data["course_ids"]:
            results = execute_course_lookup(data["course_ids"])
            tool_context = json.dumps(results, indent=2)

        elif intent == Intent.REQUIREMENTS:
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

        elif intent == Intent.PLANNING:
            if self.profile.is_complete():
                result = execute_planning(self.profile)
                tool_context = result.get("summary", result.get("error", ""))
            else:
                missing = []
                if not self.profile.major_id:
                    missing.append("your major")
                if not self.profile.courses_taken:
                    missing.append("courses you've taken")
                tool_context = f"MISSING INFO: To build a plan, I need: {', '.join(missing)}."

        elif intent == Intent.SCHEDULING and data["course_ids"]:
            result = execute_scheduling(data["course_ids"])
            tool_context = result.get("summary", "")

        elif intent == Intent.SEMESTER_BUILD and data["course_ids"]:
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
                    plan.stage = "initiated"

                # Add any mentioned courses to the plan
                for cid in data["course_ids"]:
                    course = COURSES.get(cid, {})
                    if course:
                        plan.add_course(cid, course.get("total_units", 12))

                # Build context for LLM
                tool_context = build_semester_context(self.profile)

                # Advance stage for next turn
                if plan.stage == "initiated":
                    plan.stage = "gathering_prefs"

        # Handle ongoing semester planning (student responding to preference question or picking courses)
        elif self.profile.semester_plan.active:
            plan = self.profile.semester_plan
            msg_lower = user_input.lower()

            # Check if student is adding a course by mentioning course IDs
            course_ids = re.findall(r'\b(\d{1,2}\.\w{2,5})\b', user_input)
            valid_ids = [c for c in course_ids if c in COURSES]

            if valid_ids and plan.stage in ("gathering_prefs", "suggesting"):
                # Student picked course(s)
                for cid in valid_ids:
                    course = COURSES.get(cid, {})
                    if course:
                        plan.add_course(cid, course.get("total_units", 12))
                plan.stage = "suggesting"
                tool_context = build_semester_context(self.profile)

            elif plan.stage == "gathering_prefs":
                # Student stated their preferences — save and move to suggesting
                plan.priority = user_input
                plan.stage = "gathering_prefs"  # stays here — build_semester_context uses this to show candidates
                tool_context = build_semester_context(self.profile)

            elif any(kw in msg_lower for kw in ["done", "looks good", "that's it", "happy with",
                                                  "finalize", "good semester", "all set"]):
                plan.stage = "finalizing"
                tool_context = build_semester_context(self.profile)

            else:
                # General follow-up during planning — show current state + candidates
                tool_context = build_semester_context(self.profile)

        elif intent == Intent.PROFILE_UPDATE:
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

        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=800,
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