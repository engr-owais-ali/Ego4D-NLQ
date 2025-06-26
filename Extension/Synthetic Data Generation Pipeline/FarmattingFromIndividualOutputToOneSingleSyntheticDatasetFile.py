import json
import uuid
import glob
import os

# Constants: adjust these paths as needed
INPUT_DIR = "files"  # directory containing queries_part0.json ... queries_part19.json
OUTPUT_JSON = "synthetic_nlq_train.json"
DATE_STR = "250524"
MANIFEST_PATH = "s3://your-bucket/manifest.csv"

# 1) Merge all LLM output files
merged_raw = {}
for filepath in sorted(glob.glob(os.path.join(INPUT_DIR, "queries_part*.json"))):
    with open(filepath, "r") as f:
        part = json.load(f)
    # Merge; if the same video_uid appears, extend its narrations
    for vid, data in part.items():
        if vid not in merged_raw:
            merged_raw[vid] = data
        else:
            merged_raw[vid]["narrations"].extend(data.get("narrations", []))

# 2) Build the synthetic NLQ structure
out = {
    "version": "1",
    "date": DATE_STR,
    "description": "Synthetic NLQ pretrain (merged parts)",
    "manifest": MANIFEST_PATH,
    "videos": []
}

for vid, data in merged_raw.items():
    # Compute clip-level metadata
    narrs = data.get("narrations", [])
    if not narrs:
        continue
    clip_uid = str(uuid.uuid4())
    max_end_sec = max(n["video_end_sec"] for n in narrs)
    max_end_frame = max(n["video_end_frame"] for n in narrs)
    
    clip = {
        "clip_uid": clip_uid,
        "video_uid": vid,
        "video_start_sec": 0.0,
        "video_end_sec": max_end_sec,
        "video_start_frame": 0,
        "video_end_frame": max_end_frame,
        "clip_start_sec": 0.0,
        "clip_end_sec": max_end_sec,
        "clip_start_frame": 0,
        "clip_end_frame": max_end_frame,
        "source_clip_uid": clip_uid,
        "annotations": []
    }
    
    for narr in narrs:
        annotation = {
            "annotation_uid": str(uuid.uuid4()),
            "language_queries": [
                {
                    "clip_start_sec": narr["video_start_sec"],
                    "clip_end_sec": narr["video_end_sec"],
                    "video_start_sec": narr["video_start_sec"],
                    "video_end_sec": narr["video_end_sec"],
                    "video_start_frame": narr["video_start_frame"],
                    "video_end_frame": narr["video_end_frame"],
                    "template": narr.get("template", ""),
                    "query": narr["nlq_query"]
                }
            ]
        }
        clip["annotations"].append(annotation)
    
    out["videos"].append({
        "video_uid": vid,
        "clips": [clip]
    })

# 3) Write merged synthetic JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

print(f"Merged {len(merged_raw)} videos into {OUTPUT_JSON}")
