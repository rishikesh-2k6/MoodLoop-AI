"""
generate_upload_csv.py
──────────────────────
Reads metadata.csv and produces upload_info.csv containing only:
  - video_name   (filename of the rendered video)
  - title        (YouTube / upload title)
  - caption      (description / caption text)

Only rows that have an actual video output file are included.
"""

import csv
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
            caption = row.get("caption", "").strip().strip('"')

            rows.append({
                "video_name": video_name,
                "title": title,
                "caption": caption,
            })

    with UPLOAD_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video_name", "title", "caption"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Created {UPLOAD_CSV}")
    print(f"  {len(rows)} video(s) included:\n")
    for r in rows:
        print(f"  • {r['video_name']}  —  {r['title']}")

if __name__ == "__main__":
    generate()
