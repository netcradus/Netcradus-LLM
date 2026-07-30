from pathlib import Path

INPUT_FOLDER = Path("data/wikipedia")
OUTPUT_FOLDER = Path("data/processed")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_FOLDER / "training_data.txt"

print("Preparing dataset...")

count = 0

with open(output_file, "w", encoding="utf-8") as outfile:

    for file in INPUT_FOLDER.glob("*.txt"):

        print("Reading:", file.name)

        text = file.read_text(encoding="utf-8")

        outfile.write(text)
        outfile.write("\n\n")

        count += 1

print("--------------------------------")
print("Articles merged:", count)
print("Dataset saved to:", output_file)
print("Done!")