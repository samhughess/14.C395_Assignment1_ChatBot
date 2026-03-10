"""
FireRoad API Data Fetcher
=========================
Pulls MIT course catalog data from the FireRoad API and saves to data/courses.json.

Each course includes: subject_id, title, description, prerequisites, units,
schedule, terms offered, level, hours (from evals), rating, enrollment,
GIR/HASS/CI flags, instructors, and related subjects.

Usage:
    python fetch_fireroad.py                    # fetch all courses
    python fetch_fireroad.py --dept 6           # preview one department
    python fetch_fireroad.py --course 6.3900    # look up one course
"""

import json
import os
import argparse
import requests

API_BASE = "https://fireroad-dev.mit.edu"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def fetch_all_courses():
    """Fetch all courses with full details from FireRoad API."""
    url = f"{API_BASE}/courses/all"
    params = {"full": "true"}

    print(f"Fetching courses from {url}...")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    courses = resp.json()
    print(f"  Received {len(courses)} courses")
    return courses


def fetch_single_course(subject_id):
    """Fetch a single course for testing/debugging."""
    url = f"{API_BASE}/courses/lookup/{subject_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_department(dept):
    """Fetch all courses in a department."""
    url = f"{API_BASE}/courses/department/{dept}"
    params = {"full": "true"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def print_stats(courses):
    """Print summary of fetched data."""
    print(f"\n{'='*60}")
    print(f"Total courses: {len(courses)}")

    by_level = {}
    with_schedule = 0
    with_desc = 0
    with_rating = 0
    with_hours = 0
    gir_counts = {}
    hass_counts = {}
    ci_counts = {}

    for c in courses:
        lvl = c.get("level", "?")
        by_level[lvl] = by_level.get(lvl, 0) + 1

        if c.get("schedule"): with_schedule += 1
        if c.get("description"): with_desc += 1
        if c.get("rating"): with_rating += 1
        if c.get("in_class_hours") or c.get("out_of_class_hours"): with_hours += 1

        gir = c.get("gir_attribute", "")
        if gir: gir_counts[gir] = gir_counts.get(gir, 0) + 1

        hass = c.get("hass_attribute", "")
        if hass: hass_counts[hass] = hass_counts.get(hass, 0) + 1

        ci = c.get("communication_requirement", "")
        if ci: ci_counts[ci] = ci_counts.get(ci, 0) + 1

    print(f"\nBy level: {by_level}")
    print(f"With schedule:    {with_schedule}")
    print(f"With description: {with_desc}")
    print(f"With rating:      {with_rating}")
    print(f"With hours data:  {with_hours}")
    print(f"\nGIR attributes:   {gir_counts}")
    print(f"HASS attributes:  {hass_counts}")
    print(f"CI attributes:    {ci_counts}")

    print(f"\nSample courses:")
    samples = ["6.3900", "6.1200", "18.06", "14.01", "8.01", "21L.011"]
    by_id = {c["subject_id"]: c for c in courses}
    for s in samples:
        if s in by_id:
            c = by_id[s]
            print(f"  {c['subject_id']:10s} {c['title'][:40]:40s} "
                  f"[{c.get('level','?')}] {c.get('total_units',0):2d}u "
                  f"gir={c.get('gir_attribute','-'):5s} "
                  f"hass={c.get('hass_attribute','-'):6s} "
                  f"ci={c.get('communication_requirement','-'):4s} "
                  f"rating={c.get('rating','--')}")


def save_json(data, filename):
    """Save data to JSON file in data/ directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Fetch MIT course data from FireRoad API")
    parser.add_argument("--dept", type=str, default=None,
                        help="Preview a single department (e.g., 6)")
    parser.add_argument("--course", type=str, default=None,
                        help="Look up a single course (e.g., 6.3900)")
    args = parser.parse_args()

    print("=" * 60)
    print("FireRoad API Data Fetcher")
    print(f"Source: {API_BASE}")
    print("=" * 60)

    if args.course:
        course = fetch_single_course(args.course)
        print(json.dumps(course, indent=2))
        return

    if args.dept:
        courses = fetch_department(args.dept)
        print(f"Department {args.dept}: {len(courses)} courses")
        for c in courses[:5]:
            print(f"  {c['subject_id']:10s} {c.get('title','')[:50]}")
        return

    courses = fetch_all_courses()
    print_stats(courses)
    save_json(courses, "courses.json")


if __name__ == "__main__":
    main()