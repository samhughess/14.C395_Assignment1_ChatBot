"""
MIT Requirements Tracker
========================
Takes a student's major and courses taken, determines:
1. Which requirements are fulfilled
2. What's remaining (GIRs + major requirements)
3. Which remaining courses the student can take next (prereqs satisfied)

Uses:
    data/courses.json   — course catalog from FireRoad
    data/majors.json    — major requirements
    data/girs.json      — GIR structure

Usage:
    python requirements.py

    from requirements import RequirementsTracker
    tracker = RequirementsTracker()
    status = tracker.get_status("6-3", ["6.100A", "6.100B", "6.1010", "18.01", "18.02", "8.01", "8.02"])
"""

import json
import os
import re


class RequirementsTracker:
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")

        with open(os.path.join(data_dir, "courses.json"), "r") as f:
            courses_list = json.load(f)
        self.courses = {c["subject_id"]: c for c in courses_list}

        with open(os.path.join(data_dir, "majors.json"), "r") as f:
            all_majors = json.load(f)
        # Filter out _meta key — it's not a major
        self.majors = {k: v for k, v in all_majors.items() if k != "_meta"}

        with open(os.path.join(data_dir, "girs.json"), "r") as f:
            self.girs = json.load(f)

    # ─── GIR Checking ──────────────────────────────────────

    def check_girs(self, courses_taken):
        """
        Check which GIRs are fulfilled by courses taken.
        Returns dict with each GIR category and its status.
        """
        taken = set(courses_taken)
        results = {}

        # Science core (6 subjects)
        for key, req in self.girs["science_core"].items():
            if key == "_comment":
                continue
            satisfied_by = set(req["satisfied_by"])
            fulfilled = taken & satisfied_by
            results[req["category"]] = {
                "required": req["courses_required"],
                "fulfilled": list(fulfilled),
                "done": len(fulfilled) >= req["courses_required"],
            }

        # REST (2 subjects)
        rest_courses = [sid for sid in taken
                        if self.courses.get(sid, {}).get("gir_attribute") == "REST"]
        results["REST"] = {
            "required": 2,
            "fulfilled": rest_courses,
            "done": len(rest_courses) >= 2,
        }

        # Lab (1 subject)
        lab_courses = [sid for sid in taken
                       if self.courses.get(sid, {}).get("gir_attribute") == "LAB"]
        results["Institute Lab"] = {
            "required": 1,
            "fulfilled": lab_courses,
            "done": len(lab_courses) >= 1,
        }

        # HASS (8 subjects total, at least 1 each of A, H, S)
        hass_a = [sid for sid in taken
                  if self.courses.get(sid, {}).get("hass_attribute") == "HASS-A"]
        hass_h = [sid for sid in taken
                  if self.courses.get(sid, {}).get("hass_attribute") == "HASS-H"]
        hass_s = [sid for sid in taken
                  if self.courses.get(sid, {}).get("hass_attribute") == "HASS-S"]
        hass_e = [sid for sid in taken
                  if self.courses.get(sid, {}).get("hass_attribute") == "HASS-E"]
        all_hass = hass_a + hass_h + hass_s + hass_e

        results["HASS"] = {
            "required": 8,
            "fulfilled": all_hass,
            "count": len(all_hass),
            "done": len(all_hass) >= 8 and len(hass_a) >= 1 and len(hass_h) >= 1 and len(hass_s) >= 1,
            "distribution": {
                "HASS-A": {"required": 1, "count": len(hass_a), "done": len(hass_a) >= 1},
                "HASS-H": {"required": 1, "count": len(hass_h), "done": len(hass_h) >= 1},
                "HASS-S": {"required": 1, "count": len(hass_s), "done": len(hass_s) >= 1},
            }
        }

        # CI-H (2 subjects)
        ci_h_courses = [sid for sid in taken
                        if self.courses.get(sid, {}).get("communication_requirement") in ("CI-H", "CI-HW")]
        results["CI-H"] = {
            "required": 2,
            "fulfilled": ci_h_courses,
            "done": len(ci_h_courses) >= 2,
        }

        return results

    # ─── Major Checking ─────────────────────────────────────

    def check_major(self, major_id, courses_taken):
        """
        Check which major requirements are fulfilled.
        Returns dict with each requirement group and its status.
        """
        if major_id not in self.majors:
            return {"error": f"Major {major_id} not found"}

        major = self.majors[major_id]
        taken = set(courses_taken)
        results = {}

        # Required courses
        required = major.get("required_courses", [])
        req_fulfilled = [c for c in required if c in taken]
        req_remaining = [c for c in required if c not in taken]
        results["required_courses"] = {
            "total": len(required),
            "fulfilled": req_fulfilled,
            "remaining": req_remaining,
            "done": len(req_remaining) == 0,
        }

        # Select groups
        results["select_groups"] = []
        for group in major.get("select_groups", []):
            # Handle track-based groups (pick_track: True)
            if group.get("pick_track"):
                tracks = group.get("tracks", {})
                # Check if student has completed any track
                best_track = None
                best_count = 0
                for track_name, track_courses in tracks.items():
                    track_set = set(track_courses)
                    fulfilled_in_track = list(taken & track_set)
                    if len(fulfilled_in_track) > best_count:
                        best_count = len(fulfilled_in_track)
                        best_track = track_name

                all_track_courses = []
                for tc in tracks.values():
                    all_track_courses.extend(tc)

                results["select_groups"].append({
                    "name": group["name"],
                    "pick": 1,
                    "pick_track": True,
                    "best_track": best_track,
                    "best_track_progress": best_count,
                    "fulfilled": list(taken & set(all_track_courses)),
                    "remaining_needed": 1 if best_count == 0 else 0,
                    "options": [c for c in all_track_courses if c not in taken],
                    "done": best_track is not None and best_count >= len(tracks.get(best_track, [])),
                    "notes": group.get("notes", ""),
                    "tracks": {tn: {"courses": tc, "fulfilled": list(taken & set(tc)),
                                    "remaining": [c for c in tc if c not in taken]}
                               for tn, tc in tracks.items()},
                })
                continue

            # Standard select group
            group_courses = group.get("courses", [])
            pick = group.get("pick", 1)

            # Handle empty course lists (flexible programs)
            if not group_courses:
                results["select_groups"].append({
                    "name": group["name"],
                    "pick": pick,
                    "fulfilled": [],
                    "remaining_needed": pick,
                    "options": [],
                    "done": False,
                    "notes": group.get("notes", "Choose with advisor"),
                    "flexible": True,
                })
                continue

            group_set = set(group_courses)
            fulfilled = list(taken & group_set)
            done = len(fulfilled) >= pick

            results["select_groups"].append({
                "name": group["name"],
                "pick": pick,
                "fulfilled": fulfilled,
                "remaining_needed": max(0, pick - len(fulfilled)),
                "options": [c for c in group_courses if c not in taken],
                "done": done,
                "notes": group.get("notes", ""),
            })

        # Restricted electives (separate field, some majors have this)
        if "restricted_electives" in major:
            re_info = major["restricted_electives"]
            results["restricted_electives"] = {
                "pick": re_info.get("pick", 0),
                "notes": re_info.get("notes", ""),
                "done": False,  # Can't easily verify without knowing the full elective list
            }

        # CI-M
        ci_m_list = set(major.get("ci_m_courses", []))
        ci_m_taken = list(taken & ci_m_list)
        ci_m_needed = major.get("ci_m_required", 2)
        results["ci_m"] = {
            "required": ci_m_needed,
            "fulfilled": ci_m_taken,
            "remaining_needed": max(0, ci_m_needed - len(ci_m_taken)),
            "done": len(ci_m_taken) >= ci_m_needed,
            "options": [c for c in major.get("ci_m_courses", []) if c not in taken],
        }

        # SERC (6-4 specific)
        if "serc_courses" in major:
            serc_list = set(major["serc_courses"])
            serc_taken = list(taken & serc_list)
            serc_needed = major.get("serc_required", 1)
            results["serc"] = {
                "required": serc_needed,
                "fulfilled": serc_taken,
                "done": len(serc_taken) >= serc_needed,
                "options": [c for c in major["serc_courses"] if c not in taken],
            }

        return results

    # ─── Prerequisite Checking ──────────────────────────────

    def parse_prerequisites(self, prereq_str):
        """
        Parse a FireRoad prerequisite string into a structure we can evaluate.

        Formats:
            "6.1010" → simple single course
            "6.1010/6.1210" → OR
            "(6.1010/6.1210), (18.03/18.06)" → (A OR B) AND (C OR D)
            "GIR:CAL1" → GIR requirement
            "permission of instructor" → always satisfiable (return True)
            "None" → no prereqs

        Returns a nested structure of AND/OR lists, or None if no prereqs.
        """
        if not prereq_str or prereq_str.lower() in ("none", ""):
            return None

        # Permission of instructor = no hard prereq
        if "permission of instructor" in prereq_str.lower():
            return None

        return prereq_str

    def check_prereqs_satisfied(self, subject_id, courses_taken):
        """
        Check if a student has satisfied the prerequisites for a course.
        Returns (satisfied: bool, missing: list of descriptions).
        """
        course = self.courses.get(subject_id)
        if not course:
            return False, [f"Course {subject_id} not found"]

        prereq_str = course.get("prerequisites", "")
        if not prereq_str or prereq_str.lower() == "none":
            return True, []

        taken = set(courses_taken)
        missing = []

        # Handle GIR prerequisites
        gir_map = {
            "GIR:CAL1": set(self.girs["science_core"]["calculus_i"]["satisfied_by"]),
            "GIR:CAL2": set(self.girs["science_core"]["calculus_ii"]["satisfied_by"]),
            "GIR:PHY1": set(self.girs["science_core"]["physics_i"]["satisfied_by"]),
            "GIR:PHY2": set(self.girs["science_core"]["physics_ii"]["satisfied_by"]),
            "GIR:CHEM": set(self.girs["science_core"]["chemistry"]["satisfied_by"]),
            "GIR:BIOL": set(self.girs["science_core"]["biology"]["satisfied_by"]),
        }

        # Simple parsing: split by comma (AND groups), then slash (OR within group)
        # Handle parentheses by stripping them for now
        cleaned = prereq_str.replace("(", "").replace(")", "")

        # Split on ", " for AND groups
        and_groups = [g.strip() for g in cleaned.split(",")]

        for group in and_groups:
            group = group.strip()
            if not group:
                continue

            # Check for GIR references
            if group.startswith("GIR:"):
                gir_courses = gir_map.get(group, set())
                if not (taken & gir_courses):
                    missing.append(group)
                continue

            # Skip "permission of instructor" type strings
            if "permission" in group.lower() or "coreq" in group.lower():
                continue

            # Split on "/" for OR options
            options = [o.strip() for o in group.split("/")]

            # Check if any option is satisfied
            option_satisfied = False
            for opt in options:
                # Extract course number (might have extra text)
                course_match = re.match(r"([\d]+\.[\w.]+|[A-Z]+\.[\w.]+)", opt)
                if course_match:
                    if course_match.group(1) in taken:
                        option_satisfied = True
                        break

            if not option_satisfied:
                # Filter to just valid course numbers for the missing report
                valid_options = []
                for opt in options:
                    course_match = re.match(r"([\d]+\.[\w.]+|[A-Z]+\.[\w.]+)", opt)
                    if course_match:
                        valid_options.append(course_match.group(1))
                if valid_options:
                    missing.append(" or ".join(valid_options))

        return len(missing) == 0, missing

    def get_takeable_courses(self, major_id, courses_taken):
        """
        From remaining major requirements, find which courses the student
        can take next (prerequisites satisfied).
        """
        major_status = self.check_major(major_id, courses_taken)
        if "error" in major_status:
            return major_status

        takeable = []

        # Check remaining required courses
        for course_id in major_status["required_courses"]["remaining"]:
            satisfied, missing = self.check_prereqs_satisfied(course_id, courses_taken)
            takeable.append({
                "course": course_id,
                "title": self.courses.get(course_id, {}).get("title", "Unknown"),
                "category": "Required",
                "prereqs_met": satisfied,
                "missing_prereqs": missing,
            })

        # Check remaining select group options
        for group in major_status["select_groups"]:
            if group["done"]:
                continue
            for course_id in group["options"]:
                # Skip if already in takeable list
                if any(t["course"] == course_id for t in takeable):
                    continue
                satisfied, missing = self.check_prereqs_satisfied(course_id, courses_taken)
                takeable.append({
                    "course": course_id,
                    "title": self.courses.get(course_id, {}).get("title", "Unknown"),
                    "category": group["name"],
                    "prereqs_met": satisfied,
                    "missing_prereqs": missing,
                })

        return takeable

    # ─── Full Status ────────────────────────────────────────

    def get_status(self, major_id, courses_taken):
        """
        Get complete graduation status for a student.
        Returns GIR status, major status, and what they can take next.
        """
        return {
            "major": major_id,
            "courses_taken": courses_taken,
            "gir_status": self.check_girs(courses_taken),
            "major_status": self.check_major(major_id, courses_taken),
            "takeable_next": self.get_takeable_courses(major_id, courses_taken),
        }

    # ─── Pretty Printing ───────────────────────────────────

    def print_status(self, status):
        """Pretty-print a student's graduation status."""
        print(f"\n{'='*60}")
        print(f"Major: {status['major']} — {self.majors.get(status['major'], {}).get('name', 'Unknown')}")
        print(f"Courses taken: {len(status['courses_taken'])}")
        print(f"{'='*60}")

        # GIR Status
        print(f"\n── GIR Status ──")
        gir = status["gir_status"]
        for category, info in gir.items():
            if category == "HASS":
                checkmark = "✅" if info["done"] else "❌"
                print(f"  {checkmark} {category}: {info['count']}/{info['required']}")
                for dist, dinfo in info["distribution"].items():
                    d_check = "✅" if dinfo["done"] else "❌"
                    print(f"      {d_check} {dist}: {dinfo['count']}/{dinfo['required']}")
            else:
                checkmark = "✅" if info["done"] else "❌"
                count = len(info["fulfilled"])
                required = info["required"]
                print(f"  {checkmark} {category}: {count}/{required}  {info['fulfilled'] if info['fulfilled'] else ''}")

        # Major Status
        print(f"\n── Major Requirements ──")
        major = status["major_status"]

        req = major["required_courses"]
        print(f"  Required courses: {len(req['fulfilled'])}/{req['total']}")
        if req["remaining"]:
            print(f"    Still need: {', '.join(req['remaining'])}")

        for group in major["select_groups"]:
            checkmark = "✅" if group["done"] else "❌"
            filled = len(group["fulfilled"])
            print(f"  {checkmark} {group['name']}: {filled}/{group['pick']}")
            if group["fulfilled"]:
                print(f"    Taken: {', '.join(group['fulfilled'])}")
            if not group["done"] and group["options"]:
                options_preview = group["options"][:5]
                more = f" (+{len(group['options'])-5} more)" if len(group["options"]) > 5 else ""
                print(f"    Options: {', '.join(options_preview)}{more}")

        ci_m = major["ci_m"]
        checkmark = "✅" if ci_m["done"] else "❌"
        print(f"  {checkmark} CI-M: {len(ci_m['fulfilled'])}/{ci_m['required']}")
        if ci_m["fulfilled"]:
            print(f"    Taken: {', '.join(ci_m['fulfilled'])}")

        if "serc" in major:
            serc = major["serc"]
            checkmark = "✅" if serc["done"] else "❌"
            print(f"  {checkmark} SERC: {len(serc['fulfilled'])}/{serc['required']}")

        # Takeable Next
        print(f"\n── Can Take Next (prereqs satisfied) ──")
        takeable = status["takeable_next"]
        ready = [t for t in takeable if t["prereqs_met"]]
        blocked = [t for t in takeable if not t["prereqs_met"]]

        if ready:
            print(f"  Ready to take ({len(ready)}):")
            for t in ready[:15]:
                print(f"    {t['course']:12s} {t['title'][:40]:40s} [{t['category']}]")
            if len(ready) > 15:
                print(f"    ... and {len(ready)-15} more")

        if blocked:
            print(f"\n  Blocked by prereqs ({len(blocked)}):")
            for t in blocked[:10]:
                missing_str = "; ".join(t["missing_prereqs"])
                print(f"    {t['course']:12s} {t['title'][:35]:35s} needs: {missing_str}")
            if len(blocked) > 10:
                print(f"    ... and {len(blocked)-10} more")


# ─── Test Examples ──────────────────────────────────────────

def test_freshman_6_3():
    """Simulate a 6-3 freshman who has completed first-year courses."""
    print("\n" + "="*60)
    print("TEST: 6-3 Freshman (completed first year)")
    print("="*60)

    tracker = RequirementsTracker()
    courses_taken = [
        # First year science core
        "18.01",    # Calculus I
        "18.02",    # Calculus II
        "8.01",     # Physics I
        "8.02",     # Physics II
        "5.111",    # Chemistry
        "7.012",    # Biology
        # First year EECS
        "6.100A",   # Intro to CS
        "6.100B",   # Computational Thinking
    ]

    status = tracker.get_status("6-3", courses_taken)
    tracker.print_status(status)


def test_sophomore_6_4():
    """Simulate a 6-4 sophomore partway through the major."""
    print("\n" + "="*60)
    print("TEST: 6-4 Sophomore (partway through major)")
    print("="*60)

    tracker = RequirementsTracker()
    courses_taken = [
        # GIRs
        "18.01", "18.02", "8.01", "8.02", "5.111", "7.012",
        # EECS core
        "6.100A", "6.100B", "6.1010", "6.1200", "6.1210",
        "18.C06",   # Linear Algebra
        "6.3700",   # Probability
        # Started on centers
        "6.3900",   # ML (data-centric)
    ]

    status = tracker.get_status("6-4", courses_taken)
    tracker.print_status(status)


def test_junior_6_9():
    """Simulate a 6-9 junior well into the major."""
    print("\n" + "="*60)
    print("TEST: 6-9 Junior (well into major)")
    print("="*60)

    tracker = RequirementsTracker()
    courses_taken = [
        # GIRs
        "18.01", "18.02", "8.01", "8.02", "5.111", "7.012",
        # Math
        "18.06",    # Linear Algebra
        "6.1200",   # Discrete Math
        "6.3700",   # Probability
        # EECS
        "6.100A", "6.1010", "6.1210",
        "6.3900",   # ML
        "6.3000",   # Signal Processing (EECS breadth)
        # BCS
        "9.01",     # Intro Neuroscience
        "9.13",     # Human Brain
        "9.66",     # Computational Cognitive Science
    ]

    status = tracker.get_status("6-9", courses_taken)
    tracker.print_status(status)


def test_prereq_check():
    """Test prerequisite checking for specific courses."""
    print("\n" + "="*60)
    print("TEST: Prerequisite Checking")
    print("="*60)

    tracker = RequirementsTracker()

    test_cases = [
        ("6.1010", ["6.100A", "6.100B"]),           # Should pass
        ("6.1010", ["6.100A"]),                       # Should fail (needs 6.100B or 6.1000)
        ("6.1210", ["6.100A", "6.1200"]),            # Should pass
        ("6.1210", ["6.100A"]),                       # Should fail (needs 6.1200)
        ("6.3900", ["6.1010", "18.06"]),             # Should pass
        ("6.3900", ["6.1010"]),                       # Should fail (needs linear algebra)
        ("6.1200", ["18.01"]),                        # Should pass (needs GIR:CAL1)
        ("6.1200", []),                               # Should fail
    ]

    for course_id, taken in test_cases:
        satisfied, missing = tracker.check_prereqs_satisfied(course_id, taken)
        status = "✅ PASS" if satisfied else "❌ FAIL"
        course_title = tracker.courses.get(course_id, {}).get("title", "Unknown")
        print(f"  {status}  {course_id} ({course_title[:30]})")
        print(f"         Taken: {taken}")
        if missing:
            print(f"         Missing: {missing}")
        print()


if __name__ == "__main__":
    test_freshman_6_3()
    test_sophomore_6_4()
    test_junior_6_9()
    test_prereq_check()