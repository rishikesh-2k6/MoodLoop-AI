"""
generate_upload_csv.py
──────────────────────
Reads metadata.csv and produces upload_info.csv containing:
  - video_name   (filename of the rendered video)
  - title        (YouTube / upload title)
  - description  (the narrative text)
  - caption      (the generated quote)
  - hashtags     (the hashtags string)

Only rows that have an actual video output file are included.
"""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METADATA_CSV = ROOT / "metadata.csv"
UPLOAD_CSV = ROOT / "upload_info.csv"

def generate():
    rows = []
    with METADATA_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            video_output = row.get("video_output", "").strip()

            # Skip runs that never rendered a video
            if not video_output:
                # Fall back to run_id-based filename if video exists in output/
                run_id = row.get("run_id", "").strip()
                candidate = ROOT / "output" / f"{run_id}.mp4"
                if candidate.exists():
                    video_output = str(candidate)
                else:
                    continue

            video_name = Path(video_output).name
            title = row.get("title", "").strip().strip('"')
            
            raw_caption = row.get("caption", "").strip().strip('"')
            quote = row.get("quote", "").strip().strip('"')
            
            # Split raw_caption into text and hashtags
            # Hashtags usually start on the last part of the text
            parts = raw_caption.split("\n\n")
            if len(parts) > 1 and "#" in parts[-1]:
                description = "\n\n".join(parts[:-1]).strip()
                hashtags = parts[-1].strip()
            else:
                # Try to extract hashtags via regex if \n\n split didn't work
                hash_matches = re.findall(r'#\w+', raw_caption)
                if hash_matches:
                    hashtags = " ".join(hash_matches)
                    description = re.sub(r'#\w+', '', raw_caption).strip()
                else:
                    description = raw_caption
                    hashtags = ""

            rows.append({
                "video_name": video_name,
                "title": title,
                "description": description,
                "caption": quote,  # The quote is the core caption text in video
                "hashtags": hashtags,
            })

    fieldnames = ["video_name", "title", "description", "caption", "hashtags"]
    with UPLOAD_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Created {UPLOAD_CSV}")
    print(f"  {len(rows)} video(s) included:\n")
    for r in rows:
        print(f"  • {r['video_name']}  —  {r['title']}")

if __name__ == "__main__":
    generate()
