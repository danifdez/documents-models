import unittest
from email.message import EmailMessage
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bs4 import BeautifulSoup
from odf import dc, teletype
from odf.opendocument import OpenDocumentText
from odf.style import Style
from odf.text import H, List, ListItem, P, Span

from tasks.extraction import extractor
from tasks.extraction.processors import (
    docling_processor,
    html_processor,
    media_processor,
    mhtml_processor,
)


class DocumentExtractionTests(unittest.TestCase):
    def extract(self, extension, body):
        return extractor.extract({"extension": extension, "_input_artifacts": {"document": body}})

    def test_extracts_local_html_and_preserves_original_metadata(self):
        html = """<html><head><title>Source</title>
        <meta name="dc.creator" content="Ada">
        <meta property="article:published_time" content="2026-11-20">
        </head><body><p>Unfiltered body</p></body></html>"""
        article = """<html><head><title>Ignore</title></head><body><div><div>
        <div class="copy" id="main" style="color:green"> Main text </div>
        <script>ignore()</script><p> </p><graphic src="summary.png"/>
        <table><row><cell role="head">Year</cell><cell>2026</cell></row><row/></table>
        </div></div></body></html>"""
        with patch.object(html_processor, "extract", return_value=article) as extract_article:
            result = self.extract(".htm", html.encode())

        self.assertEqual(extract_article.call_args.args[0], html)
        self.assertEqual((result["title"], result["author"], result["publication_date"]), (
            "Source", "Ada", "2026-11-20",
        ))
        content = result["content"]
        self.assertIn("<p> Main text </p>", content)
        self.assertEqual(BeautifulSoup(content, "html.parser").img["src"], "summary.png")
        self.assertIn("<table><tr><th>Year</th><td>2026</td></tr><tr><td></td></tr></table>", content)
        for markup in ("<html", "<head", "<script", "<div", "class=", "style=", "id="):
            self.assertNotIn(markup, content)

    def test_html_fragments_and_empty_extractions_keep_their_contract(self):
        with patch.object(html_processor, "extract", return_value="<div><p>Article</p></div>"):
            self.assertEqual(html_processor.process_html("<p>Source</p>")["content"], "<p>Article</p>")
        with patch.object(html_processor, "extract", return_value=None):
            self.assertEqual(html_processor.process_html("<p>Source</p>"), {
                "title": "No Title", "content": None, "author": None, "publication_date": None,
            })

    def test_prefers_the_mhtml_root_and_falls_back_to_the_largest_html_part(self):
        related = EmailMessage()
        related.make_related()
        related["Subject"] = "Archived page"
        for content_id, html in (("main", "<p>Café</p>"), ("second", "<p>Longer secondary page</p>")):
            part = EmailMessage()
            part.set_content(html, subtype="html", charset="iso-8859-1")
            part["Content-ID"] = f"<{content_id}>"
            related.attach(part)

        with patch.object(mhtml_processor, "process_html", side_effect=lambda body: {"content": body}):
            related.set_param("start", "<main>")
            chosen = self.extract(".mht", related.as_bytes())
            self.assertEqual(chosen["title"], "Archived page")
            self.assertEqual(chosen["content"].strip(), "<p>Café</p>")

            related.set_param("start", "<missing>")
            fallback = self.extract(".mhtml", related.as_bytes())
            self.assertEqual(fallback["content"].strip(), "<p>Longer secondary page</p>")

    def test_mhtml_decodes_a_standalone_html_document_with_an_unknown_charset(self):
        archive = b"Content-Type: text/html; charset=unknown\r\n\r\n<p>Caf\xc3\xa9</p>"
        with patch.object(mhtml_processor, "process_html", return_value={"title": "Page"}) as process:
            self.assertEqual(self.extract(".mht", archive), {"title": "Page"})
        process.assert_called_once_with("<p>Café</p>")

    def test_extracts_email_headers_and_html_without_attachment_bodies(self):
        message = EmailMessage()
        message["Subject"] = "Réunion"
        message["From"] = "Ada Example <ada@example.test>"
        message["To"] = "reader@example.test"
        message["Date"] = "Fri, 20 Nov 2026 08:30:00 +0200"
        message.set_content("Fallback text")
        message.add_alternative("<p>Meeting agenda</p>", subtype="html")
        message.add_attachment("<p>Attachment text</p>", subtype="html", filename="attachment.html")

        result = self.extract(".eml", message.as_bytes())
        self.assertEqual((result["title"], result["author"], result["publication_date"]), (
            "Réunion", "Ada Example", "2026-11-20",
        ))
        self.assertIn("reader@example.test", result["content"])
        self.assertIn("<p>Meeting agenda</p>", result["content"])
        self.assertNotIn("Fallback text", result["content"])
        self.assertNotIn("Attachment text", result["content"])

    def test_plain_email_uses_paragraphs_and_tolerates_an_invalid_date(self):
        message = (
            b"From: sender@example.test\r\nDate: not a date\r\n\r\n"
            b"First line\r\n\r\nSecond line"
        )
        result = self.extract(".eml", message)
        self.assertEqual(result["author"], "sender@example.test")
        self.assertIsNone(result["publication_date"])
        self.assertTrue(result["content"].endswith("<p>First line</p><p>Second line</p>"))
        self.assertEqual(self.extract(".eml", b"")["content"], "")

    def test_extracts_odt_metadata_headings_inline_text_and_list_items(self):
        document = OpenDocumentText()
        document.meta.addElement(dc.Title(text=" Quarterly report "))
        document.meta.addElement(dc.Creator(text="Ada"))
        document.meta.addElement(dc.Date(text="2026-11-20T08:30:00"))
        heading_style = Style(name="Heading 2", family="paragraph")
        document.styles.addElement(heading_style)
        document.text.addElement(H(outlinelevel=2, stylename=heading_style, text="Results"))
        document.text.addElement(P(text="   "))
        paragraph = P()
        teletype.addTextToElement(paragraph, "Total ")
        paragraph.addElement(Span(text="42"))
        document.text.addElement(paragraph)
        items = List()
        for text in ("First", "", "Second"):
            item = ListItem()
            item.addElement(P(text=text))
            items.addElement(item)
        document.text.addElement(items)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.odt"
            document.save(str(path))
            result = self.extract(".odt", path.read_bytes())

        self.assertEqual(result, {
            "title": "Quarterly report", "author": "Ada", "publication_date": "2026-11-20",
            "content": "<h2>Results</h2><p>Total 42</p><ul><li>First</li><li>Second</li></ul>",
        })

    def test_delegates_pdfs_and_word_files_and_removes_temporary_inputs(self):
        for extension, processor in ((".pdf", "process_pdf"), (".docx", "process_doc")):
            with self.subTest(extension=extension):
                def read_input(filename, expected_extension=extension):
                    path = Path(filename)
                    self.assertEqual(path.suffix, expected_extension)
                    self.assertEqual(path.read_bytes(), b"synthetic document")
                    return {"content": "Extracted text", "pages": 2}

                with patch.object(extractor, processor, side_effect=read_input) as process:
                    result = extractor.extract_indexed_file({
                        "extension": extension, "indexedFileId": 32,
                        "_input_artifacts": {"document": b"synthetic document"},
                    })
                self.assertEqual(result, {"content": "Extracted text", "pages": 2, "indexedFileId": 32})
                self.assertFalse(Path(process.call_args.args[0]).exists())

    def test_cleans_up_if_a_processor_fails_and_rejects_unsupported_types(self):
        with (
            patch.object(extractor, "process_doc", side_effect=RuntimeError("corrupt source")) as process,
            self.assertRaisesRegex(RuntimeError, "corrupt source"),
        ):
            self.extract(".doc", b"invalid")
        self.assertFalse(Path(process.call_args.args[0]).exists())
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            self.extract(".unknown", b"invalid")


class DoclingExtractionTests(unittest.TestCase):
    def convert(self, html, extension, pages=None):
        document = SimpleNamespace(export_to_html=Mock(return_value=html))
        if pages is not None:
            document.pages = pages
        with patch.object(docling_processor, "DocumentConverter") as converter:
            instance = converter.return_value
            instance.convert.return_value = SimpleNamespace(document=document)
            result = (extractor.process_pdf if extension == ".pdf" else extractor.process_doc)("report" + extension)
        instance.convert.assert_called_once_with("report" + extension)
        document.export_to_html.assert_called_once_with(image_mode="embedded")
        return result

    def test_converts_pdf_formulas_and_keeps_page_count(self):
        result = self.convert(
            '<html><body><div><div><div style="color:red" class="copy" id="body">Results</div>'
            '<math><annotation encoding="TeX">x^2</annotation></math><math/>'
            '<script>discard()</script><p> </p></div></div></body></html>', ".pdf", {1: {}, 2: {}},
        )
        self.assertEqual(result, {
            "content": '<p>Results</p><span data-formula="x^2">x^2</span>'
                       '<span data-formula-error="true">[Formula not extracted]</span>',
            "pages": 2,
        })

    def test_word_output_omits_page_count_if_docling_did_not_provide_it(self):
        self.assertEqual(self.convert("<html></html>", ".docx"), {"content": ""})


class MediaExtractionTests(unittest.TestCase):
    def test_common_audio_and_video_formats_expose_normalized_metadata(self):
        cases = (
            ("MP3", ".mp3", {"TIT2": [], "TIT1": "Clip", "TPE2": "Ada", "TDRC": "2026-11-20T08:30"}, "MP3"),
            ("MP4", ".mov", {"©nam": ["Clip"], "aART": ["Ada"], "©day": ["2026-11-20"]}, "29.97"),
            ("FLAC", ".flac", {"TITLE": ["Clip"], "ARTIST": ["Ada"], "DATE": ["2026-11-20"]}, "FLAC"),
            ("OggVorbis", ".ogg", {"title": ["Clip"], "artist": ["Ada"], "date": ["2026-11-20"]}, "OGG Vorbis"),
            ("WAVE", ".wav", {"TIT2": "Clip", "TPE1": "Ada", "TDRC": "2026-11-20"}, "WAV"),
        )
        for reader, extension, tags, expected_format in cases:
            with self.subTest(extension=extension):
                audio = SimpleNamespace(
                    tags=tags, info=SimpleNamespace(
                        length=3661.25, bitrate=192000, channels=2,
                        sample_rate=48000, bits_per_sample=24, video_fps=29.97,
                    ),
                )
                with patch.object(media_processor, reader, return_value=audio) as read:
                    result = extractor.extract({
                        "extension": extension, "_input_artifacts": {"document": b"synthetic media"},
                    })
                self.assertFalse(Path(read.call_args.args[0]).exists())
                self.assertEqual((result["title"], result["author"], result["publication_date"]), (
                    "Clip", "Ada", "2026-11-20",
                ))
                self.assertIn(expected_format, result["content"])
                self.assertIn("1:01:01", result["content"])

    def test_falls_back_when_no_format_specific_reader_exists(self):
        audio = SimpleNamespace(tags={"title": ["Clip"], "year": ["2026-11-20T00:00"]}, info=SimpleNamespace(length=65))
        with patch.object(media_processor.mutagen, "File", side_effect=[audio, None, SimpleNamespace(tags=None)]):
            result = media_processor.process_media("clip.webm")
            self.assertEqual((result["title"], result["publication_date"]), ("Clip", "2026-11-20"))
            self.assertIn("1:05", result["content"])
            self.assertEqual(media_processor.process_media("clip.bin")["content"], "<p>Audio</p>")
            self.assertIn("BIN", media_processor.process_media("clip.bin")["content"])

    def test_unreadable_media_returns_a_format_error_in_the_extracted_content(self):
        with patch.object(media_processor, "MP3", side_effect=ValueError("invalid header")):
            result = media_processor.process_media("clip.mp3")
        self.assertIsNone(result["title"])
        self.assertIn("invalid header", result["content"])
