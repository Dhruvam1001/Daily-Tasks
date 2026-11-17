import streamlit as st
from pathlib import Path

root = Path(__file__).resolve().parents[3]
current_file = Path(__file__).resolve()
pages = {}

for month_name in ["August", "October", "November"]:
    month = root / month_name

    if not month.is_dir():
        continue

    month_pages = []

    for date in sorted(month.iterdir()):
        if date.is_dir():

            for task in ["H1", "H2", "H3"]:
                files = [date / f"{task}.py", date / task / "app.py"]

                for file in files:
                    if (file.is_file() and file.stat().st_size > 0 and file.resolve() != current_file):
                        unique_path = f"{month_name.lower()}-{date.name.lower()}-{task.lower()}"
                        month_pages.append(st.Page(str(file), title = f"{date.name} - {task}", url_path = unique_path))
                        break

    if month_pages:
        pages[month_name] = month_pages

if pages:
    page = st.navigation(pages, position = "sidebar")
    page.run()
else:
    st.error("No application files found.")