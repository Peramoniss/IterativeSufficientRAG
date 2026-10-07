from tqdm import tqdm
PARTS = 5

input_wiki_file = "./documents/wiki-18.jsonl"

print("Analyzing wiki file...")
total_lines = sum(1 for _ in open(input_wiki_file, "r"))
lines_per_part = total_lines // PARTS
print(f"Lines per part: {lines_per_part}")
output_file = "./documents/wiki-18_part1.jsonl"
outfile = open(output_file, "w", encoding="utf-8")
lines = 0
curr_part = 1

with open(input_wiki_file, "r") as infile:
    for line in tqdm(infile, total=total_lines, desc="Dividing Wikipedia dump", unit="article"):
            if not line.startswith('{'):
                try:
                    line = "{" + str(line.split('{', 1)[1])
                except:
                    print(f"Error in line: {line}.")
                    continue
            outfile.write(line)
            lines += 1
            if lines >= lines_per_part and curr_part != PARTS:
                lines = 0
                curr_part += 1
                outfile.close()
                output_file = f"./documents/wiki-18_part{curr_part}.jsonl"
                outfile = open(output_file, "w", encoding="utf-8")

    outfile.close()

print(f"Success! Wikipedia divided into {PARTS} parts.")