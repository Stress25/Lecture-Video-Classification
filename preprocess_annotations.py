"""
preprocess_annotations.py
Parses a LectureMath annotation XML file and produces a clean CSV of
segment-level action labels, ready for use in multimodal fusion training.
 
Output columns:
    segment_id, start_frame, end_frame, start_time_ms, end_time_ms,
    start_time_sec, end_time_sec, label, secondary_label
"""
import xml.etree.ElementTree as ET
import csv
from collections import Counter
import os

fps = 29.97  # Frames per second for the videos

# step 1: Parse the XML file

def parse_xml(file_path):
    """Parses the LectureMath annotation XML file and extracts action keyframes 
    and video segments.
     Args:
        file_path (str): Path to the XML annotation file.
    Returns:
        action_keyframes (list): list of (frame,abs_time_ms,label), only from the speaker VideoObject (has <label> tag) 
        video_segments (list): list of (start,end)
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    action_keyframes = []
    # video_segments = []
    # keyframes = []
    # ===== id = speaker =====
    # Speaker action annotations are under the VideoObject with <label> tags
    speaker_objects = [
        video_object for video_object in root.findall(".//VideoObject")
        if (video_object.findtext("Id") or "").strip() == "speaker" or  (video_object.findtext("Name") or "").strip() == "speaker"
    ]

    if not speaker_objects:
        speaker_objects = [
            video_object for video_object in root.findall("./VideoObject")
            if video_object.find(".//VideoObjectLocation/Label") is not None
        ]
    
    for video_object in speaker_objects:
        for location in video_object.findall(".//VideoObjectLocation"):
            label_el = location.find("Label")
            frame_el = location.find("Frame")
            time_el = location.find("AbsTime")

            # only keep locations that are actually labelled keyframes
            if label_el is None or frame_el is None or time_el is None:
                continue
            if not (label_el.text and frame_el.text and time_el.text):
                continue

            frame = int(frame_el.text.strip())
            abs_time_ms = float(time_el.text.strip())
            label = label_el.text.strip()

            action_keyframes.append((frame,abs_time_ms,label))

    # sort by frame
    action_keyframes.sort(key=lambda x: x[0])

    # Total Frame Count
    seg_ends = [
        int(s.findtext("End").strip())
        for s in root.findall(".//VideoSegment")
        if s.findtext("End")
    ]

    kf_index = [
        int(k.findtext("Index").strip())
        for k in root.findall(".//VideoKeyFrame")
        if k.findtext("Index")
    ]

    total_frames = max(seg_ends) if seg_ends else (max(kf_index) if kf_index else 0)

    return action_keyframes, total_frames


# Assign labels to segments based on the frame-label mapping
def assign_labels_to_segments(action_keyframes, total_frames):
    """
    one segment per consqutive keyframe interval.
    Each speaker keyframe marks the start of an action that holds until the next keyframe
    """

    labelled = []

    for i, (start_frame,_,label) in enumerate(action_keyframes):
        end_frame = action_keyframes[i+1][0] - 1 if i+1 < len(action_keyframes) else total_frames - 1
        if end_frame < start_frame:
            continue  # Skip zero length or invalid segments

        start_time_ms = start_frame / fps * 1000
        end_time_ms = (end_frame + 1) / fps * 1000
        start_time_sec = start_frame / fps
        end_time_sec = (end_frame + 1) / fps
        duration_sec = end_time_sec - start_time_sec
        
        labelled.append({
            "segment_id": len(labelled),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_ms": f"{start_time_ms:.2f}",
            "end_time_ms": f"{end_time_ms:.2f}",
            "start_time_sec": f"{start_time_sec:.3f}",
            "end_time_sec": f"{end_time_sec:.3f}",
            "duration_sec": f"{duration_sec:.3f}",
            "label": label,
        })

    return labelled

# step 3: Write the labelled segments to a CSV file
def write_to_csv(labelled_segments, output_path):
    field_names = ["segment_id", "start_frame", "end_frame", "start_time_ms", "end_time_ms", "start_time_sec", "end_time_sec", "duration_sec","label"]

    with open(output_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(labelled_segments)
    
    print(f"Labelled segments written to {output_path}")



# main function to run the preprocessing
def preprocess_annotations(xml_path, output_csv_path):
    print(f"Parsing XML file: {xml_path}")

    action_keyframes, total_frames = parse_xml(xml_path)
    
    print(f" Action keyframes extracted: {len(action_keyframes)}")
    print(f" Total frames extracted: {total_frames}")
    
    if not action_keyframes:
        print("Warning: no speaker keyframes found. Check that the XML contains a VideoObject with Id='speaker' and <Label> tags inside its VideoObjectLocation.")
        return[]
    
    labelled_segments = assign_labels_to_segments(action_keyframes, total_frames)

    print(f" Labelled segments created: {len(labelled_segments)}")
    label_dist = Counter(seg["label"] for seg in labelled_segments)
    print("Label distribution across segments: ")
    for label,count in label_dist.most_common():
        print(f" {label}:{count} segments")

    write_to_csv(labelled_segments, output_csv_path)
    return labelled_segments

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1].endswith(".xml"):
        preprocess_annotations(sys.argv[1], sys.argv[2])

    # else len(sys.argv) == 3:
    else:
        print("Usage: python preprocess_annotations.py <input_xml_path> <output_csv_path>")
        print("Example: python preprocess_annotations.py annotations.xml labelled_segments.csv")    
