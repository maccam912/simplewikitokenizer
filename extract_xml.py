import bz2

def extract_xml(bz2_file_path: str, xml_file_path: str):
    with bz2.open(bz2_file_path, "rt", errors="replace") as bz2_file, open(xml_file_path, "w", encoding="utf-8") as xml_file:
        for line in bz2_file:
            xml_file.write(line)