import xml.sax
import re

class WikiContentHandler(xml.sax.ContentHandler):
    def __init__(self, output_file):
        self.output_file = output_file
        self.current_tag = ""
        self.current_title = ""
        self.current_text = ""
        self.is_article = False

    def startElement(self, name, attrs):
        self.current_tag = name

    def endElement(self, name):
        if name == "page":
            if self.is_article:
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(self.current_text.strip() + "\n")
            self.current_title = ""
            self.current_text = ""
            self.is_article = False
        self.current_tag = ""

    def characters(self, content: str):
        if self.current_tag == "title":
            self.current_title += content
            if not re.match(r"(User|Talk|Wikipedia|File|MediaWiki|Template|Help|Category|Portal|File):", self.current_title):
                self.is_article = True
        elif self.current_tag == "text" and self.is_article and len(content) > 100:
            self.current_text += content.encode("ascii", errors="replace").decode("ascii", errors="replace").replace('[', '').replace(']', '')

def parse_xml(xml_file_path, text_file_path):
    parser = xml.sax.make_parser()
    parser.setContentHandler(WikiContentHandler(text_file_path))
    with open(xml_file_path, "r", encoding="utf-8") as f:
        parser.parse(f)
