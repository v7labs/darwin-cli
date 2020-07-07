import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Generator

import darwin.datatypes as dt


def add_subelement_text(parent, name, value):
    sub = ET.SubElement(parent, name)
    sub.text = str(value)
    return sub


def export(annotation_files: Generator[dt.AnnotationFile, None, None], output_dir: Path):
    output = build_xml(list(annotation_files))
    # TODO, maybe an optional output name (like the dataset name if available)
    output_file_path = (output_dir / "output").with_suffix(".xml")
    with open(output_file_path, "wb") as f:
        f.write(ET.tostring(output))


def build_xml(annotation_files):
    label_lookup = build_label_lookup(annotation_files)
    root = ET.Element("annotations")
    add_subelement_text(root, "version", "1.1")
    build_meta(root, annotation_files, label_lookup)
    build_images(root, annotation_files, label_lookup)
    return root


def build_images(root, annotation_files, label_lookup):
    for id, annotation_file in enumerate(annotation_files, 1):
        image = ET.SubElement(root, "image")
        image.attrib["id"] = str(id)
        image.attrib["name"] = annotation_file.filename
        for annotation in annotation_file.annotations:
            build_annotation(image, annotation)


def build_annotation(image, annotation):
    if annotation.annotation_class.annotation_type == "bounding_box":
        box = ET.SubElement(image, "box")
        box.attrib["label"] = annotation.annotation_class.name
        print(annotation.data)
        box.attrib["xtl"] = str(annotation.data["x"])
        box.attrib["ytl"] = str(annotation.data["y"])
        box.attrib["xbr"] = str(annotation.data["x"] + annotation.data["w"])
        box.attrib["ybr"] = str(annotation.data["y"] + annotation.data["h"])
        box.attrib["occluded"] = "0"

        for sub in annotation.subs:
            if sub.annotation_type == 'attributes':
                for attribute_i in sub.data:
                    att = ET.SubElement(box, "attribute", attrib={"name": str(attribute_i)})
                    att.text = str(attribute_i) # darwin doesn't support key-valued attibutes
    else:
        print(f"[warning] skipping {annotation.annotation_class.annotation_type}")


def build_meta(root, annotation_files, label_lookup):
    meta = ET.SubElement(root, "meta")
    add_subelement_text(meta, "dumped", str(datetime.now()))

    task = ET.SubElement(meta, "task")
    add_subelement_text(task, "id", 1)
    add_subelement_text(task, "name", "exported_task_from_darwin")
    add_subelement_text(task, "size", len(annotation_files))
    add_subelement_text(task, "mode", "annotation")
    add_subelement_text(task, "overlapp", 0)
    add_subelement_text(task, "bugtracker", None)
    add_subelement_text(task, "flipped", False)
    add_subelement_text(meta, "created", str(datetime.now()))
    add_subelement_text(meta, "updated", str(datetime.now()))

    labels = ET.SubElement(task, "labels")
    build_labels(labels, label_lookup)

    segments = ET.SubElement(task, "segments")
    build_segments(segments, annotation_files)

    owner = ET.SubElement(task, "owner")
    add_subelement_text(owner, "username", "example_username")
    add_subelement_text(owner, "email", "user@example.com")


def build_segments(segments, annotation_files):
    segment = ET.SubElement(segments, "segment")
    add_subelement_text(segment, "id", 1)
    add_subelement_text(segment, "start", 1)
    add_subelement_text(segment, "end", len(annotation_files))
    add_subelement_text(segment, "url", "not applicable")


def build_labels(labels, label_lookup):
    for key in label_lookup.keys():
        label = ET.SubElement(labels, "label")
        add_subelement_text(label, "name", key)
        ET.SubElement(label, "attributes")


def build_label_lookup(annotation_files):
    labels = {}
    for annotation_file in annotation_files:
        for annotation_class in annotation_file.annotation_classes:
            if annotation_class.name not in labels and annotation_class.annotation_type == "bounding_box":
                labels[annotation_class.name] = len(labels)
    return labels
