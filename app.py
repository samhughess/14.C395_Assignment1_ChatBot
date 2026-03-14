"""
MIT Course Advisor Chatbot — Gradio Interface
==============================================
Run with: python app.py
Then open http://localhost:7860

Features:
- Optional intake form (major, courses taken, year) in sidebar
- Chat interface with conversation history
- Example questions to get started
- Deployable to HuggingFace Spaces
"""

import gradio as gr
from src.chat import Chatbot

# ─── Create chatbot instance ────────────────────────────────

chatbot = Chatbot()


# ─── Chat function ──────────────────────────────────────────

def chat(message, history):
    """
    Generate a response for the current message.
    Gradio handles display and history management automatically.
    """
    return chatbot.get_response(message, history)


# ─── Profile update callback ───────────────────────────────

def update_profile(major, courses, year, semesters):
    """Called when the student updates their profile form."""
    # Extract major ID from dropdown (e.g., "6-3 — CS and Engineering" → "6-3")
    major_id = major.split(" —")[0].strip() if major and " —" in major else major

    chatbot.update_profile_from_form(
        major_id=major_id if major_id else None,
        courses_str=courses if courses else None,
        year=year if year else None,
        semesters_left=semesters if semesters else None,
    )

    parts = []
    if major:
        parts.append(f"Major: {major}")
    if courses:
        parts.append(f"Courses: {courses}")
    if year:
        parts.append(f"Year: {year}")
    if semesters:
        parts.append(f"Semesters left: {semesters}")

    if parts:
        return "✅ Profile updated!\n" + "\n".join(parts)
    return "No changes — fill in at least one field."


# ─── Build Gradio interface ─────────────────────────────────
def create_chatbot():
    """Create and configure the full Gradio app."""

    with gr.Blocks(title="MIT Course Advisor") as demo:

        gr.Markdown(
            """
            # 🎓 MIT Course Advisor
            **AI-powered course planning for EECS majors (6-3, 6-4, 6-7, 6-9, 6-14)**

            I can help with: 📋 Requirements · 📚 Course info · 📅 Semester planning · 🎯 Recommendations

            *Fill in your profile for personalized advising, or just start chatting!*
            """
        )

        with gr.Row():
            # ── Left column: Profile form ──
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Your Profile *(optional)*")

                major_input = gr.Dropdown(
                    choices=[
                        "",
                        "1-ENG — Civil/Environmental Eng.",
                        "1-12 — Climate System Sci. & Eng.",
                        "2 — Mechanical Engineering",
                        "2-A — Engineering (MechE Flexible)",
                        "2-OE — Mechanical & Ocean Eng.",
                        "3 — Materials Science & Eng.",
                        "3-A — Materials Science (Flexible)",
                        "3-C — Archaeology & Materials",
                        "4 — Architecture",
                        "4-B — Art and Design",
                        "5 — Chemistry",
                        "5-7 — Chemistry and Biology",
                        "6-3 — CS and Engineering",
                        "6-4 — AI and Decision Making",
                        "6-5 — EE with Computing",
                        "6-7 — CS and Molecular Biology",
                        "6-9 — Computation and Cognition",
                        "6-14 — CS, Econ, and Data Science",
                        "7 — Biology",
                        "8 — Physics",
                        "9 — Brain and Cognitive Sciences",
                        "10 — Chemical Engineering",
                        "10-B — Chemical-Biological Eng.",
                        "10-C — Chemical Eng. (Flexible)",
                        "10-ENG — Engineering (ChemE)",
                        "11 — Planning",
                        "11-6 — Urban Sci. & Planning w/ CS",
                        "12 — Earth, Atmos., & Planetary Sci.",
                        "14-1 — Economics",
                        "14-2 — Mathematical Economics",
                        "15-1 — Management",
                        "15-2 — Business Analytics",
                        "15-3 — Finance",
                        "16 — Aerospace Engineering",
                        "16-ENG — Engineering (Aero/Astro)",
                        "17 — Political Science",
                        "18 — Mathematics",
                        "18-C — Math with Computer Science",
                        "20 — Biological Engineering",
                        "21 — Humanities",
                        "21A — Anthropology",
                        "21E — Humanities and Engineering",
                        "21G — Global Studies & Languages",
                        "21H — History",
                        "21L — Literature",
                        "21M — Music",
                        "21S — Humanities and Science",
                        "21T — Theater Arts",
                        "21W — Writing",
                        "22 — Nuclear Science & Eng.",
                        "22-ENG — Engineering (Nuclear)",
                        "24-1 — Philosophy",
                        "24-2 — Linguistics & Philosophy",
                        "CMS — Comparative Media Studies",
                        "STS — Science, Tech, and Society",
                    ],
                    label="Major",
                    value="",
                    info="Select your major"
                )

                courses_input = gr.Textbox(
                    label="Courses Taken",
                    placeholder="e.g. 18.01, 18.02, 8.01, 8.02, 6.100A, 6.1010",
                    info="Comma-separated course numbers",
                    lines=3,
                )

                year_input = gr.Dropdown(
                    choices=["", "freshman", "sophomore", "junior", "senior"],
                    label="Year",
                    value="",
                )

                semesters_input = gr.Number(
                    label="Semesters Remaining",
                    value=None,
                    precision=0,
                    info="How many semesters until graduation?"
                )

                update_btn = gr.Button("Update Profile", variant="primary")
                profile_status = gr.Textbox(
                    label="Profile Status",
                    interactive=False,
                    lines=4,
                )

                update_btn.click(
                    fn=update_profile,
                    inputs=[major_input, courses_input, year_input, semesters_input],
                    outputs=[profile_status],
                )

            # ── Right column: Chat ──
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=chat,
                    title="Chat",
                    description="Ask me anything about MIT courses and degree planning!",
                    examples=[
                        "Tell me about 6.3900",
                        "What requirements do I still need for my major?",
                        "What courses should I take next semester?",
                        "I'm a 6-4 sophomore. I've taken 6.100A, 6.1010, 6.1200, 6.1210, 18.C06, 6.3700, and 6.3900. What should I focus on next?",
                        "Can I graduate in 4 more semesters?",
                        "What are the prereqs for 6.4200?",
                    ],
                )

    return demo


# ─── Launch ─────────────────────────────────────────────────

if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch()