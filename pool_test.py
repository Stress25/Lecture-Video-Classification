"""
pool_test_csvs.py
Pools all 15 test video aligned CSVs into a single file for SBERT encoding.
Run this after batch_pipeline.py finishes on the test videos.
"""
import csv
import os

TEST_VIDEOS = [
    "LectureMath_00000_000_002",
    "LectureMath_00000_000_005",
    "LectureMath_00000_001_002",
    "LectureMath_00000_001_007",
    "LectureMath_00000_001_008",
    "LectureMath_00000_001_010",
    "LectureMath_00000_001_015",
    "LectureMath_00005_002_009",
    "LectureMath_00005_011_010",
    "LectureMath_00008_001_020",
    "LectureMath_00009_007_074",
    "LectureMath_00010_002_004",
    "LectureMath_00011_031_003",
    "LectureMath_00012_008_004",
    "LectureMath_00012_018_002",
    "LectureMath_00013_001_026",
    "LectureMath_00013_001_088",
]

OUT_DIR = os.path.expanduser(
    "~/Documents/MastersResearch/Code/output_test"
)

for mode in ["contained", "overlap"]:
    rows, fieldnames = [], None
    missing = []

    for vid in TEST_VIDEOS:
        path = os.path.join(OUT_DIR, vid, f"aligned_{mode}.csv")
        if not os.path.exists(path):
            missing.append(vid)
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = ["video_id"] + reader.fieldnames
            for row in reader:
                row["video_id"] = vid
                rows.append(row)

    if missing:
        print(f"[{mode}] warning: missing files for: {missing}")

    out_path = os.path.join(OUT_DIR, f"all_test_aligned_{mode}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{mode}] Written {len(rows)} rows to {out_path}")