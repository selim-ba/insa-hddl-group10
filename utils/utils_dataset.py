from pathlib import Path
import xml.etree.ElementTree as ET


def parse_xml(class_id,path_to_xml):
    path = Path(path_to_xml) / f"{class_id}.xml"
    if not path.exists():
        return {}

    root = ET.parse(path).getroot()

    size = root.find("size")
    obj = root.find("object")
    box = obj.find("bndbox")

    return {
        "width":     int(size.find("width").text),
        "height":    int(size.find("height").text),
        "depth":     int(size.find("depth").text),
        "segmented": int(root.find("segmented").text),
        #"xml_species": obj.find("name").text, we already have species info
        "pose":        obj.find("pose").text,
        "truncated":   int(obj.find("truncated").text),
        "occluded":    int(obj.find("occluded").text),
        "difficult":   int(obj.find("difficult").text),
        "bb_xmin": int(box.find("xmin").text),
        "bb_ymin": int(box.find("ymin").text),
        "bb_xmax": int(box.find("xmax").text),
        "bb_ymax": int(box.find("ymax").text),
        "xml_path": str(path),
    }

def count_files(path_to_img, path_to_xml, path_to_trimaps):
    trimaps_count = len(list(Path(path_to_trimaps).glob("*.png")))
    img_count = len(list(Path(path_to_img).glob("*.jpg")))
    xml_count = len(list(Path(path_to_xml).glob("*.xml")))
    return img_count, xml_count, trimaps_count
