# pyrefly: ignore [missing-import]
import wikipediaapi
from pathlib import Path

print("Connecting to Wikipedia...")

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="NetcradusLLM/1.0 (netcradusdeveloper@gmail.com)"
)

print("Connected successfully!")

# Folder where articles will be saved
SAVE_FOLDER = Path("data/wikipedia")
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

# Articles to download
topics = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Python (programming language)"
]

for topic in topics:
    print(f"Downloading: {topic}")

    page = wiki.page(topic)

    if page.exists():
        filename = SAVE_FOLDER / f"{topic.replace(' ', '_')}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(page.text)

        print(f"Saved -> {filename}")
    else:
        print(f"Page not found: {topic}")

print("\nFinished downloading all articles!")