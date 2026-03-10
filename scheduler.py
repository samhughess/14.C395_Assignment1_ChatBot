"""
MIT Course Scheduler
====================
Given a list of candidate courses, finds valid (conflict-free) schedule
combinations for the current semester.

A course is considered available this semester if it has schedule data.
If no schedule data exists, we don't know its times and can't schedule it.

Uses course data from data/courses.json (fetched from FireRoad API).

Usage:
    python scheduler.py 6.3900 6.1200 6.1010 18.06
    python scheduler.py 6.3900 6.1200 6.1010 --max-units 48
    python scheduler.py 6.3900 6.1200 6.1010 --max-hours 40

    from scheduler import Scheduler
    s = Scheduler("data/courses.json")
    results = s.find_schedules(["6.3900", "6.1200", "6.1010"])

Schedule string format from FireRoad:
    "Lecture,26-100/TR/0/2.30-4;Recitation,26-168/WF/0/10,38-166/WF/0/1"

    Split by ; → section types (Lecture, Recitation, Lab, etc.)
    Split by , → first is type name, rest are slot options
    Each slot: Room/Days/SlotFlag/Time
    Days: M=Mon, T=Tue, W=Wed, R=Thu, F=Fri, S=Sat
    Time: "10" means 10-11, "2.30-4" means 2:30-4:00
"""

import json
import os
from itertools import product, combinations


# ─── Time Parsing ───────────────────────────────────────────

DAY_MAP = {
    "M": "Monday",
    "T": "Tuesday",
    "W": "Wednesday",
    "R": "Thursday",
    "F": "Friday",
    "S": "Saturday",
}


def parse_time_to_minutes(time_str):
    """
    Convert a time string to minutes from midnight.
    "10" → 600, "2.30" → 870, "9.30" → 570, "12" → 720
    MIT convention: times 1-6 are PM (13:00-18:00).
    """
    time_str = time_str.strip()
    if "." in time_str:
        parts = time_str.split(".")
        hours = int(parts[0])
        minutes = int(parts[1])
    else:
        hours = int(time_str)
        minutes = 0

    # MIT convention: 1-6 means PM
    if 1 <= hours <= 6:
        hours += 12

    return hours * 60 + minutes


def parse_time_range(time_str):
    """
    Parse a time range into (start_minutes, end_minutes).
    "9.30-11" → (570, 660)
    "10" → (600, 660)  (assumes 1 hour if no end time)
    "2.30-4" → (870, 960)
    "7-10 PM" → (1140, 1320)
    """
    time_str = time_str.strip()

    # Handle PM suffix
    is_pm = "PM" in time_str.upper()
    time_str = time_str.replace("PM", "").replace("pm", "").strip()

    if "-" in time_str:
        start_str, end_str = time_str.split("-", 1)
        start = parse_time_to_minutes(start_str)
        end = parse_time_to_minutes(end_str)
        if is_pm:
            if start < 720:
                start += 720
            if end < 720:
                end += 720
    else:
        start = parse_time_to_minutes(time_str)
        if is_pm and start < 720:
            start += 720
        end = start + 60  # assume 1 hour

    return start, end


def parse_days(days_str):
    """Parse "MW" → ["Monday", "Wednesday"], "TR" → ["Tuesday", "Thursday"]."""
    return [DAY_MAP[c] for c in days_str if c in DAY_MAP]


# ─── Schedule Parsing ───────────────────────────────────────

def parse_schedule_string(schedule_str):
    """
    Parse a FireRoad schedule string into structured data.

    Input: "Lecture,26-100/TR/0/2.30-4;Recitation,26-168/WF/0/10,38-166/WF/0/1"

    Output: {
        "Lecture": [
            {"room": "26-100", "days": [...], "start": 870, "end": 960, "time_str": "2.30-4"}
        ],
        "Recitation": [
            {"room": "26-168", "days": [...], "start": 600, "end": 660, "time_str": "10"},
            {"room": "38-166", "days": [...], "start": 780, "end": 840, "time_str": "1"},
        ]
    }
    """
    if not schedule_str:
        return {}

    sections = {}
    for section_block in schedule_str.split(";"):
        parts = section_block.split(",")
        if not parts:
            continue

        section_type = parts[0].strip()
        slots = []

        for slot_str in parts[1:]:
            slot_str = slot_str.strip()
            if not slot_str:
                continue

            pieces = slot_str.split("/")
            if len(pieces) < 4:
                continue

            room = pieces[0]
            days_str = pieces[1]
            time_str = pieces[3]

            try:
                days = parse_days(days_str)
                start, end = parse_time_range(time_str)
                slots.append({
                    "room": room,
                    "days": days,
                    "start": start,
                    "end": end,
                    "time_str": time_str,
                })
            except (ValueError, IndexError):
                continue

        if slots:
            sections[section_type] = slots

    return sections


def get_time_blocks(slot):
    """Convert a slot into list of (day, start, end) tuples."""
    return [(day, slot["start"], slot["end"]) for day in slot["days"]]


def blocks_conflict(blocks_a, blocks_b):
    """Check if any two time blocks overlap."""
    for day_a, start_a, end_a in blocks_a:
        for day_b, start_b, end_b in blocks_b:
            if day_a == day_b and start_a < end_b and start_b < end_a:
                return True
    return False


# ─── Scheduler ──────────────────────────────────────────────

class Scheduler:
    def __init__(self, courses_path=None):
        """Load courses from JSON file."""
        if courses_path is None:
            courses_path = os.path.join(os.path.dirname(__file__), "data", "courses.json")

        with open(courses_path, "r") as f:
            courses_list = json.load(f)

        self.courses = {c["subject_id"]: c for c in courses_list}

    def get_course(self, subject_id):
        """Look up a course by subject_id."""
        return self.courses.get(subject_id)

    def get_section_choices(self, subject_id):
        """
        For a course, return the sections and their slot options.
        Returns: list of (section_type, [slot_options])
        """
        course = self.courses.get(subject_id)
        if not course or not course.get("schedule"):
            return []

        sections = parse_schedule_string(course["schedule"])
        return list(sections.items())

    def get_all_possible_timeslots(self, subject_id):
        """
        Generate all possible time slot combinations for a course.
        Each combo is a list of time blocks: [(day, start, end), ...]
        """
        sections = self.get_section_choices(subject_id)
        if not sections:
            return []

        section_options = []
        for section_type, slots in sections:
            slot_blocks = [get_time_blocks(slot) for slot in slots]
            section_options.append(slot_blocks)

        combos = []
        for choice in product(*section_options):
            all_blocks = []
            for blocks in choice:
                all_blocks.extend(blocks)
            combos.append(all_blocks)

        return combos

    def find_schedules(self, candidate_courses, max_units=None,
                       max_hours=None, max_courses=None):
        """
        Find all valid (conflict-free) schedules from candidate courses.

        Args:
            candidate_courses: list of subject_ids the student is considering
            max_units: maximum total units
            max_hours: maximum hours per week
            max_courses: max number of courses (if None, tries all candidates)

        Returns: list of valid schedules, each is a dict with:
            courses, total_units, total_hours, time_blocks
        """
        # Validate candidates
        valid_candidates = []
        warnings = []
        for sid in candidate_courses:
            course = self.courses.get(sid)
            if not course:
                warnings.append(f"{sid}: not found in catalog")
                continue
            if not course.get("schedule"):
                warnings.append(f"{sid}: not offered this semester (no schedule data)")
                continue
            valid_candidates.append(sid)

        if warnings:
            print("Warnings:")
            for w in warnings:
                print(f"  {w}")

        # Get all timeslot combos for each candidate
        course_combos = {}
        for sid in valid_candidates:
            combos = self.get_all_possible_timeslots(sid)
            if combos:
                course_combos[sid] = combos
            else:
                warnings.append(f"{sid}: could not parse schedule")

        if not course_combos:
            return []

        if max_courses is None:
            max_courses = len(course_combos)

        course_ids = list(course_combos.keys())
        valid_schedules = []

        # Try all subsets from largest to smallest
        for num_courses in range(min(max_courses, len(course_ids)), 0, -1):
            for course_subset in combinations(course_ids, num_courses):
                # Check unit/hour limits
                total_units = sum(
                    self.courses[sid].get("total_units", 0) for sid in course_subset
                )
                if max_units and total_units > max_units:
                    continue

                total_hours = sum(
                    (self.courses[sid].get("in_class_hours", 0) or 0) +
                    (self.courses[sid].get("out_of_class_hours", 0) or 0)
                    for sid in course_subset
                )
                if max_hours and total_hours > max_hours:
                    continue

                # Try all schedule combos for this subset
                combo_lists = [course_combos[sid] for sid in course_subset]

                for schedule_choice in product(*combo_lists):
                    # Check pairwise conflicts
                    conflict = False
                    for i in range(len(schedule_choice)):
                        for j in range(i + 1, len(schedule_choice)):
                            if blocks_conflict(schedule_choice[i], schedule_choice[j]):
                                conflict = True
                                break
                        if conflict:
                            break

                    if not conflict:
                        time_blocks = {
                            sid: blocks
                            for sid, blocks in zip(course_subset, schedule_choice)
                        }
                        valid_schedules.append({
                            "courses": list(course_subset),
                            "total_units": total_units,
                            "total_hours": round(total_hours, 1),
                            "time_blocks": time_blocks,
                        })

                        # Cap results to avoid blowup
                        if len(valid_schedules) >= 50:
                            return valid_schedules

        return valid_schedules

    def print_schedule(self, schedule):
        """Pretty-print a schedule as a weekly grid."""
        print(f"\n{'─'*60}")
        courses_str = ", ".join(
            f"{sid} ({self.courses[sid]['title'][:30]})"
            for sid in schedule["courses"]
        )
        print(f"Courses: {courses_str}")
        print(f"Total units: {schedule['total_units']}  |  Est. hours/week: {schedule['total_hours']}")
        print()

        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        grid = {day: [] for day in days_order}

        for sid, blocks in schedule["time_blocks"].items():
            for day, start, end in blocks:
                if day in grid:
                    grid[day].append((start, end, sid))

        for day in days_order:
            slots = sorted(grid[day], key=lambda x: x[0])
            if slots:
                print(f"  {day}:")
                for start, end, sid in slots:
                    start_hr = start // 60
                    start_min = start % 60
                    end_hr = end // 60
                    end_min = end % 60
                    title = self.courses[sid]["title"][:35]
                    print(f"    {start_hr}:{start_min:02d}-{end_hr}:{end_min:02d}  {sid} {title}")

        print(f"{'─'*60}")


# ─── CLI ────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MIT Course Scheduler")
    parser.add_argument("courses", nargs="+",
                        help="Course IDs to schedule (e.g., 6.3900 6.1200 18.06)")
    parser.add_argument("--max-units", type=int, default=None,
                        help="Maximum total units")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Maximum hours per week")
    parser.add_argument("--max-results", type=int, default=5,
                        help="Maximum schedules to show (default: 5)")
    parser.add_argument("--data", default=None,
                        help="Path to courses.json")
    args = parser.parse_args()

    data_path = args.data or os.path.join(os.path.dirname(__file__), "data", "courses.json")
    s = Scheduler(data_path)

    print(f"Finding schedules for: {', '.join(args.courses)}")
    if args.max_units:
        print(f"Max units: {args.max_units}")
    if args.max_hours:
        print(f"Max hours/week: {args.max_hours}")
    print()

    results = s.find_schedules(
        candidate_courses=args.courses,
        max_units=args.max_units,
        max_hours=args.max_hours,
    )

    if not results:
        print("No valid schedules found.")
        return

    print(f"Found {len(results)} valid schedule(s). Showing top {min(args.max_results, len(results))}:")

    for schedule in results[:args.max_results]:
        s.print_schedule(schedule)


if __name__ == "__main__":
    main()