from lxml import etree
from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup


NSMAP = {'fictionbook': 'http://www.gribuser.ru/xml/fictionbook/2.0'}


def fb2reader(fb2_filepath):
    tree = etree.parse(fb2_filepath)
    t_root = tree.getroot()

    bodies = t_root.xpath('//fictionbook:body', namespaces=NSMAP)

    book = []
    for body in bodies:
        paragraphs = body.xpath('.//fictionbook:p/text()', namespaces=NSMAP)
        for p in paragraphs:
            book.append(p.strip())
    return book


def txt_reader(txt_filepath, encoding):
    with open(txt_filepath, 'r', encoding=encoding) as f:
        return f.readlines()


def epub_reader(epub_filepath):
    try:
        book = epub.read_epub(epub_filepath)
        chapters = book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
        full_text = []

        for chapter in chapters:
            content = chapter.get_content().decode('utf-8')
            soup = BeautifulSoup(content, 'html.parser')
            chapter_text = soup.get_text().strip()
            full_text.append(chapter_text)
        return full_text
    except epub.EpubException:
        return '0'