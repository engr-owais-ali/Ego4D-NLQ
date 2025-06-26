# Stage 1 of preprocessing
import json
import random
import os

ego4d_path = ""
narration_path = os.path.join(ego4d_path, "D:\Projects\ArashFIles\\narration.json")


with open(narration_path, 'r', encoding='utf-8') as f:
    ego4d_narrations = json.load(f)  # Load JSON data


import random

sampled_dataset = {}

def sample_consecutive_groups(clips, N, m, video_uid):
    total_required = m * N
    if len(clips) < N:
        # Not enough samples for one group, return all
        return

    max_start = len(clips) - N
    valid_starts = list(range(0, len(clips) - N + 1))
    random.shuffle(valid_starts)

    sampled_dataset[video_uid] = {}
    sampled_dataset[video_uid] = {"narrations": []}
    used_ranges = set()
    count = 0

    for start_idx in valid_starts:
        # Ensure non-overlapping: check if range [start_idx, start_idx+N) overlaps any previous
        if any(abs(start_idx - used) < N for used in used_ranges):
            continue
        group = clips[start_idx:start_idx + N]
        # for sample in group:
        #     sample.pop("_unmapped_timestamp_sec", None)
        narration_dict = {}
        narration_dict["texts"] = [sample["narration_text"] for sample in group]
        narration_dict["video_start_sec"] = group[0]["timestamp_sec"]
        narration_dict["video_end_sec"] = group[-1]["timestamp_sec"]
        narration_dict["video_start_frame"] = group[0]["timestamp_frame"]
        narration_dict["video_end_frame"] = group[-1]["timestamp_frame"]

        sampled_dataset[video_uid]["narrations"].append(narration_dict)

        used_ranges.add(start_idx)
        count += 1
        if count == m:
            break

    # If not enough non-overlapping groups were found, you could raise a warning or pad
    if count < m:
        print(f"Warning: Only {count} non-overlapping groups could be sampled (requested {m}).")

    return sampled_dataset



N = 5 # Cardinality of a sample
M = 7 # Number of samples
for video_uid in ego4d_narrations.keys():
    if 'narration_pass_2' in ego4d_narrations[video_uid]:

        # for each narration pass we have narrations and summarizations fields
        clips = ego4d_narrations[video_uid]['narration_pass_2']["narrations"]

        sample_consecutive_groups(clips, N, 7, video_uid)

with open("sampled_narrations.json", "w") as json_file:
    json.dump(sampled_dataset, json_file, indent=4)