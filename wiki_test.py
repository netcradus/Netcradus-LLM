# pyrefly: ignore [missing-import]
import wikipediaapi

print("Creating Wikipedia client...")

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="NetcradusLLM/1.0 (netcradusdeveloper@gmail.com)"
)

print("Requesting page...")

page = wiki.page("Artificial intelligence")

print("Page exists:", page.exists())

if page.exists():
    print("Title:", page.title)
    print("-" * 50)
    print(page.summary[:500])
else:
    print("Page not found.")