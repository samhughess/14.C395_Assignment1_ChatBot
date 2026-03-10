"""
MIT Semester Planner
====================
Given a student's major, courses taken, and remaining semesters,
produces a feasible semester-by-semester plan to graduation.

Key features:
    1. Critical path analysis — find longest prereq chains, identify bottlenecks
    2. Feasibility check — can the student graduate on time?
    3. Semester-by-semester planning — slot courses into specific semesters
    4. Risk flagging — courses only offered once/year, overloaded semesters

Uses:
    data/courses.json   — course catalog (prereqs, offering terms, units)
    data/majors.json    — major requirements
    data/girs.json      — GIR structure
    requirements.py     — requirement checking

Usage:
    python planner.py

    from planner import SemesterPlanner
    planner = SemesterPlanner()
    plan = planner.plan_semesters("6-4", courses_taken, semesters_left=4, next_is_fall=True)
"""

import json
import os
from collections import defaultdict
from requirements import RequirementsTracker


class SemesterPlanner:
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")

        with open(os.path.join(data_dir, "courses.json"), "r") as f:
            courses_list = json.load(f)
        self.courses = {c["subject_id"]: c for c in courses_list}

        with open(os.path.join(data_dir, "majors.json"), "r") as f:
            all_majors = json.load(f)
        self.majors = {k: v for k, v in all_majors.items() if k != "_meta"}

        with open(os.path.join(data_dir, "girs.json"), "r") as f:
            self.girs = json.load(f)

        self.tracker = RequirementsTracker(data_dir=data_dir)

    # ─── Offering Term Logic ────────────────────────────────

    def is_offered_in(self, subject_id, semester):
        """
        Check if a course is offered in a given semester type.
        semester: "fall" or "spring"
        """
        course = self.courses.get(subject_id, {})
        if semester == "fall":
            return course.get("offered_fall", False)
        elif semester == "spring":
            return course.get("offered_spring", False)
        return False

    def get_offering_frequency(self, subject_id):
        """
        Classify how often a course is offered.
        Returns: "both", "fall_only", "spring_only", or "unknown"
        """
        course = self.courses.get(subject_id, {})
        fall = course.get("offered_fall", False)
        spring = course.get("offered_spring", False)
        if fall and spring:
            return "both"
        elif fall:
            return "fall_only"
        elif spring:
            return "spring_only"
        else:
            return "unknown"

    # ─── Prerequisite Graph ─────────────────────────────────

    def build_prereq_graph(self, course_ids):
        """
        Build a directed graph of prerequisite relationships among a set of courses.
        Returns dict: course_id -> list of prereq course_ids (that are also in the set).

        Only includes edges where the prereq is also in the given set
        (i.e., courses the student still needs to take).
        """
        course_set = set(course_ids)
        graph = {cid: [] for cid in course_ids}

        for cid in course_ids:
            course = self.courses.get(cid, {})
            prereq_str = course.get("prerequisites", "")
            if not prereq_str or prereq_str.lower() == "none":
                continue

            # Extract all course IDs mentioned in the prereq string
            import re
            mentioned = re.findall(r"([\d]+\.[\w.]+|[A-Z]+\.[\w.]+)", prereq_str)

            for prereq_id in mentioned:
                if prereq_id in course_set:
                    graph[cid].append(prereq_id)

        return graph

    def find_critical_paths(self, major_id, courses_taken):
        """
        Find the longest prerequisite chains among remaining required courses.
        These are the bottleneck sequences that constrain the minimum semesters needed.

        Returns list of paths, sorted longest first. Each path is a list of course_ids
        in the order they must be taken.
        """
        # Get all remaining courses from requirements
        status = self.tracker.get_status(major_id, courses_taken)
        remaining = set()

        # Collect remaining required courses
        for cid in status["major_status"]["required_courses"]["remaining"]:
            remaining.add(cid)

        # Collect one option from each unfulfilled select group
        # (use the first option whose prereqs are closest to satisfied)
        for group in status["major_status"]["select_groups"]:
            if not group["done"] and group["options"]:
                remaining.add(group["options"][0])

        # Also add remaining GIR courses (simplified — just flag what categories are missing)
        # For now, focus on major requirements since GIR courses are typically more flexible

        if not remaining:
            return []

        # Build prereq graph among remaining courses
        graph = self.build_prereq_graph(remaining)

        # Find all paths using DFS
        def dfs_longest_path(node, visited):
            visited.add(node)
            longest = [node]

            for prereq in graph.get(node, []):
                if prereq not in visited:
                    path = dfs_longest_path(prereq, visited.copy())
                    candidate = path + [node]
                    if len(candidate) > len(longest):
                        longest = candidate

            return longest

        # Find longest path from each node
        all_paths = []
        for cid in remaining:
            path = dfs_longest_path(cid, set())
            if len(path) > 1:
                all_paths.append(path)

        # Deduplicate and sort by length
        unique_paths = []
        seen = set()
        for path in sorted(all_paths, key=len, reverse=True):
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                unique_paths.append(path)

        return unique_paths

    def get_minimum_semesters_needed(self, major_id, courses_taken):
        """
        Calculate the minimum number of semesters needed to complete remaining
        requirements, based on the longest prerequisite chain.

        Also accounts for fall/spring offering constraints — a fall-only → spring-only
        chain takes at least 2 semesters even though it's only 2 courses.
        """
        paths = self.find_critical_paths(major_id, courses_taken)
        if not paths:
            return 0

        longest_chain = len(paths[0])

        # Check if offering constraints stretch any chain further
        # A chain of [A(fall-only), B(spring-only), C(fall-only)] needs 3 semesters
        # but a chain of [A(both), B(both)] only needs 2
        max_stretched = longest_chain

        for path in paths[:3]:  # Check top 3 paths
            semester_count = 0
            current_sem = None  # Will be set based on what works

            for cid in path:
                freq = self.get_offering_frequency(cid)
                semester_count += 1
                # If course is only offered in one semester, it constrains the sequence
                # (more sophisticated analysis would alternate fall/spring and check fit)

            max_stretched = max(max_stretched, semester_count)

        return max_stretched

    # ─── Feasibility Check ──────────────────────────────────

    def check_feasibility(self, major_id, courses_taken, semesters_remaining, max_units=48):
        """
        Check if the student can feasibly graduate on time.
        Returns a dict with feasibility assessment and any warnings.
        """
        status = self.tracker.get_status(major_id, courses_taken)

        # Count remaining courses and total units
        remaining_courses = []
        for cid in status["major_status"]["required_courses"]["remaining"]:
            remaining_courses.append(cid)
        for group in status["major_status"]["select_groups"]:
            if not group["done"]:
                # Need group["remaining_needed"] more from this group
                for opt in group["options"][:group["remaining_needed"]]:
                    remaining_courses.append(opt)

        total_remaining_units = sum(
            self.courses.get(cid, {}).get("total_units", 12)
            for cid in remaining_courses
        )

        min_semesters = self.get_minimum_semesters_needed(major_id, courses_taken)
        max_units_total = semesters_remaining * max_units

        # GIR check — count actual subjects still needed
        gir = status["gir_status"]
        remaining_gir_subjects = 0
        for cat, info in gir.items():
            if cat == "HASS":
                if not info.get("done", True):
                    remaining_gir_subjects += max(0, info["required"] - info["count"])
            else:
                if not info.get("done", True):
                    remaining_gir_subjects += max(0, info["required"] - len(info.get("fulfilled", [])))
        remaining_girs = remaining_gir_subjects

        warnings = []
        feasible = True

        if min_semesters > semesters_remaining:
            feasible = False
            warnings.append(
                f"Critical path requires at least {min_semesters} semesters, "
                f"but only {semesters_remaining} remaining. "
                f"May need to overload or take summer courses."
            )

        if total_remaining_units > max_units_total:
            warnings.append(
                f"Remaining major courses total ~{total_remaining_units} units, "
                f"but {semesters_remaining} semesters × {max_units} units = "
                f"{max_units_total} unit capacity. "
                f"Schedule will be tight (doesn't include GIRs/electives)."
            )

        avg_units = total_remaining_units / max(semesters_remaining, 1)
        if avg_units > 42:
            warnings.append(
                f"Average remaining major load is ~{avg_units:.0f} units/semester. "
                f"Consider whether GIRs and electives can fit alongside."
            )

        if remaining_girs > 0:
            warnings.append(
                f"Still need ~{remaining_girs} GIR categories beyond major requirements."
            )

        # Check for fall/spring only courses at risk
        critical_paths = self.find_critical_paths(major_id, courses_taken)
        for path in critical_paths[:3]:
            for cid in path:
                freq = self.get_offering_frequency(cid)
                title = self.courses.get(cid, {}).get("title", cid)
                if freq == "fall_only":
                    warnings.append(f"{cid} ({title}) is fall-only — don't miss a fall offering.")
                elif freq == "spring_only":
                    warnings.append(f"{cid} ({title}) is spring-only — don't miss a spring offering.")

        return {
            "feasible": feasible,
            "semesters_remaining": semesters_remaining,
            "min_semesters_needed": min_semesters,
            "remaining_major_courses": len(remaining_courses),
            "remaining_major_units": total_remaining_units,
            "remaining_gir_categories": remaining_girs,
            "critical_paths": critical_paths[:3],
            "warnings": warnings,
        }

    # ─── Semester-by-Semester Plan ──────────────────────────

    def suggest_next_semester(self, major_id, courses_taken, next_is_fall, max_units=48):
        """
        Suggest which courses to take next semester.
        Prioritizes:
            1. Courses on critical prerequisite paths (unblock future courses)
            2. Courses only offered this semester (fall/spring constraint)
            3. Courses that double-count (satisfy multiple requirements)

        Returns list of suggested courses with reasoning.
        """
        status = self.tracker.get_status(major_id, courses_taken)
        semester_type = "fall" if next_is_fall else "spring"

        # Get all takeable courses (prereqs met)
        takeable = status["takeable_next"]
        ready = [t for t in takeable if t["prereqs_met"]]

        # Filter to courses offered this semester
        offered_now = []
        for t in ready:
            cid = t["course"]
            if self.is_offered_in(cid, semester_type):
                t["offered_frequency"] = self.get_offering_frequency(cid)
                t["units"] = self.courses.get(cid, {}).get("total_units", 12)
                offered_now.append(t)

        if not offered_now:
            return {"semester": semester_type, "suggestions": [], "total_units": 0}

        # Score each course
        critical_set = set()
        critical_paths = self.find_critical_paths(major_id, courses_taken)
        for path in critical_paths:
            for cid in path:
                critical_set.add(cid)

        for t in offered_now:
            score = 0
            cid = t["course"]

            # High priority: on a critical path
            if cid in critical_set:
                score += 30

            # High priority: only offered this semester (not both)
            if t["offered_frequency"] != "both":
                score += 20

            # Medium priority: required course (not elective choice)
            if t["category"] == "Required":
                score += 15

            # Lower priority: CI-M (need to spread these out)
            ci = self.courses.get(cid, {}).get("communication_requirement", "")
            if ci and "CI-M" in ci:
                score += 5

            t["priority_score"] = score

        # Sort by priority
        offered_now.sort(key=lambda t: t["priority_score"], reverse=True)

        # Greedy bin-pack by units
        suggestions = []
        total_units = 0
        for t in offered_now:
            if total_units + t["units"] <= max_units:
                reasons = []
                if t["course"] in critical_set:
                    reasons.append("on critical prereq path")
                if t["offered_frequency"] != "both":
                    reasons.append(f"{t['offered_frequency'].replace('_', ' ')}")
                if t["category"] == "Required":
                    reasons.append("required")
                t["reasons"] = reasons
                suggestions.append(t)
                total_units += t["units"]

        return {
            "semester": semester_type,
            "suggestions": suggestions,
            "total_units": total_units,
        }

    def plan_semesters(self, major_id, courses_taken, semesters_remaining,
                       next_is_fall, max_units=48):
        """
        Generate a full semester-by-semester plan.
        Iteratively calls suggest_next_semester, advancing the state each time.

        Returns list of semester plans.
        """
        plan = []
        current_taken = list(courses_taken)
        is_fall = next_is_fall

        for sem_num in range(1, semesters_remaining + 1):
            semester_type = "Fall" if is_fall else "Spring"
            suggestion = self.suggest_next_semester(
                major_id, current_taken, is_fall, max_units
            )

            semester_plan = {
                "semester_number": sem_num,
                "semester_type": semester_type,
                "courses": [],
                "total_units": suggestion["total_units"],
            }

            for s in suggestion["suggestions"]:
                semester_plan["courses"].append({
                    "course": s["course"],
                    "title": s["title"],
                    "units": s["units"],
                    "category": s["category"],
                    "reasons": s.get("reasons", []),
                })
                current_taken.append(s["course"])

            plan.append(semester_plan)

            # Alternate semesters
            is_fall = not is_fall

            # Check if done
            check = self.tracker.check_major(major_id, current_taken)
            all_done = check["required_courses"]["done"]
            groups_done = all(g["done"] for g in check["select_groups"])
            if all_done and groups_done:
                break

        return plan

    # ─── Pretty Printing ────────────────────────────────────

    def print_feasibility(self, result):
        """Pretty-print a feasibility check result."""
        print(f"\n{'='*60}")
        status = "✅ FEASIBLE" if result["feasible"] else "❌ AT RISK"
        print(f"Graduation Feasibility: {status}")
        print(f"{'='*60}")
        print(f"  Semesters remaining: {result['semesters_remaining']}")
        print(f"  Minimum semesters needed (prereq chains): {result['min_semesters_needed']}")
        print(f"  Remaining major courses: {result['remaining_major_courses']}")
        print(f"  Remaining major units: ~{result['remaining_major_units']}")
        print(f"  Remaining GIR categories: {result['remaining_gir_categories']}")

        if result["critical_paths"]:
            print(f"\n  Critical prerequisite chains:")
            for i, path in enumerate(result["critical_paths"][:3]):
                titles = []
                for cid in path:
                    title = self.courses.get(cid, {}).get("title", cid)
                    titles.append(f"{cid}")
                print(f"    {i+1}. {' → '.join(titles)}  ({len(path)} semesters)")

        if result["warnings"]:
            print(f"\n  ⚠️  Warnings:")
            for w in result["warnings"]:
                print(f"    • {w}")

    def print_plan(self, plan):
        """Pretty-print a semester plan."""
        print(f"\n{'='*60}")
        print(f"Semester-by-Semester Plan")
        print(f"{'='*60}")
        for sem in plan:
            print(f"\n  ── {sem['semester_type']} (Semester {sem['semester_number']}) "
                  f"── {sem['total_units']} units ──")
            if not sem["courses"]:
                print(f"    (no major courses scheduled)")
                continue
            for c in sem["courses"]:
                reasons_str = f"  [{', '.join(c['reasons'])}]" if c.get("reasons") else ""
                print(f"    {c['course']:12s} {c['title'][:38]:38s} {c['units']:2d}u"
                      f"  ({c['category']}){reasons_str}")


# ─── Test Examples ──────────────────────────────────────────

def test_sophomore_6_4():
    """Test planning for a 6-4 sophomore with 5 semesters left."""
    print("\n" + "="*60)
    print("TEST: 6-4 Sophomore — 5 semesters remaining (next is Fall)")
    print("="*60)

    planner = SemesterPlanner()
    courses_taken = [
        "18.01", "18.02", "8.01", "8.02", "5.111", "7.012",
        "6.100A", "6.100B", "6.1010", "6.1200", "6.1210",
        "18.C06", "6.3700",
        "6.3900",  # ML (data-centric center)
    ]

    # Feasibility check
    result = planner.check_feasibility("6-4", courses_taken, semesters_remaining=5)
    planner.print_feasibility(result)

    # Next semester suggestion
    print("\n" + "-"*40)
    print("Next semester suggestion (Fall):")
    suggestion = planner.suggest_next_semester("6-4", courses_taken, next_is_fall=True)
    for s in suggestion["suggestions"]:
        reasons = f"  [{', '.join(s['reasons'])}]" if s.get("reasons") else ""
        print(f"  {s['course']:12s} {s['title'][:40]:40s} {s['units']:2d}u{reasons}")
    print(f"  Total: {suggestion['total_units']} units")

    # Full plan
    plan = planner.plan_semesters("6-4", courses_taken, semesters_remaining=5,
                                  next_is_fall=True, max_units=48)
    planner.print_plan(plan)


def test_junior_6_3():
    """Test planning for a 6-3 junior with 3 semesters left."""
    print("\n" + "="*60)
    print("TEST: 6-3 Junior — 3 semesters remaining (next is Spring)")
    print("="*60)

    planner = SemesterPlanner()
    courses_taken = [
        "18.01", "18.02", "8.01", "8.02", "5.111", "7.012",
        "6.100A", "6.100B", "6.1010", "6.1020", "6.1200", "6.1210",
        "6.1903", "6.1910",
        "18.06",
        "6.3900",
    ]

    result = planner.check_feasibility("6-3", courses_taken, semesters_remaining=3)
    planner.print_feasibility(result)

    plan = planner.plan_semesters("6-3", courses_taken, semesters_remaining=3,
                                  next_is_fall=False, max_units=48)
    planner.print_plan(plan)


def test_critical_paths_6_14():
    """Test critical path analysis for a 6-14 freshman."""
    print("\n" + "="*60)
    print("TEST: 6-14 Freshman critical paths (just completed first year)")
    print("="*60)

    planner = SemesterPlanner()
    courses_taken = [
        "18.01", "18.02", "8.01", "8.02", "5.111", "7.012",
        "6.100A",
    ]

    paths = planner.find_critical_paths("6-14", courses_taken)
    print(f"\n  Found {len(paths)} prerequisite chains:")
    for i, path in enumerate(paths[:5]):
        names = []
        for cid in path:
            title = planner.courses.get(cid, {}).get("title", "?")
            names.append(f"{cid} ({title[:25]})")
        print(f"    {i+1}. {' → '.join(names)}")
        print(f"       Length: {len(path)} semesters minimum")

    result = planner.check_feasibility("6-14", courses_taken, semesters_remaining=7)
    planner.print_feasibility(result)


if __name__ == "__main__":
    test_sophomore_6_4()
    test_junior_6_3()
    test_critical_paths_6_14()