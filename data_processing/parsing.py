# new fallback pipeline with llamaparser used as fallback parser
import os
import logging
import pdfplumber
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from llama_parse import LlamaParse
from utils.utils import PdfExtractionResult, PageText
from config import settings

logger = logging.getLogger(__name__)

LLAMAPARSE_API_KEY = settings.LLAMAPARSE_API_KEY
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract(filename: str) -> PdfExtractionResult:
    if os.path.splitext(filename)[1].lower() != ".pdf":
        raise ValueError("Unsupported file type")

    pages_data = []
    fallback_triggered = False

    try:
        with pdfplumber.open(filename) as pdf:
            logger.info(f"Total pages in PDF: {len(pdf.pages)}")

            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                text = ""
                method_used = ""

                # Step 1: Try direct text extraction
                try:
                    text = page.extract_text(layout=True)
                    if text and len(text.strip()) >= 50:
                        logger.info(f"✅ Parsed text from page {page_num} using pdfplumber.")
                        method_used = "pdfplumber"
                    else:
                        raise ValueError("Text too short or missing")
                except Exception as e:
                    logger.warning(f"⚠️ pdfplumber extract_text failed for page {page_num}: {e}")
                    text = ""

                # Step 2: Try OCR using page.to_image()
                if not text:
                    try:
                        logger.info(f"🔁 Attempting OCR with pdfplumber.to_image on page {page_num}...")
                        image = page.to_image(resolution=300).original
                        text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
                        if text and len(text.strip()) >= 50:
                            logger.info(f"✅ OCR succeeded for page {page_num} with pdfplumber.to_image.")
                            method_used = "ocr:plumber"
                        else:
                            raise ValueError("OCR result too short")
                    except Exception as e:
                        logger.warning(f"⚠️ OCR via pdfplumber.to_image failed for page {page_num}: {e}")
                        text = ""

                # Step 3: Final fallback — pdf2image + OCR
                if not text:
                    try:
                        logger.info(f"🧯 Final fallback: trying pdf2image + pytesseract on page {page_num}...")
                        fallback_images = convert_from_path(filename, dpi=300, first_page=page_num, last_page=page_num)
                        if fallback_images:
                            text = pytesseract.image_to_string(fallback_images[0], config='--oem 3 --psm 6')
                            logger.info(f"✅ OCR succeeded using pdf2image for page {page_num}.")
                            method_used = "ocr:pdf2image"
                    except Exception as e:
                        logger.error(f"❌ pdf2image fallback failed on page {page_num}: {e}")
                        text = ""

                # Extract tables (only from pdfplumber)
                table_text = ""
                try:
                    tables = page.extract_tables() or []
                    for table in tables:
                        table_text += "\n\nTable:\n" + "\n".join(
                            " | ".join(str(cell) if cell is not None else "" for cell in row)
                            for row in table
                        )
                except Exception as e:
                    logger.warning(f"⚠️ Table extraction failed on page {page_num}: {e}")

                # Clean and combine
                cleaned_text = "\n".join(line.strip() for line in (text or "").splitlines() if line.strip())
                full_text = cleaned_text + table_text

                if full_text.strip():
                    pages_data.append(PageText(
                        page_number=page_num,
                        text=full_text,
                        filename=os.path.basename(filename)
                    ))

    except Exception as e:
        logger.error(f"❌ Error using pdfplumber combo pipeline: {e}")
        fallback_triggered = True

    # If no valid pages parsed, fallback to LLaMAParse
    if not pages_data or fallback_triggered:
        logger.info("⚠️ Falling back to LLaMAParse due to insufficient content or failure.")
        try:
            parser = LlamaParse(api_key=LLAMAPARSE_API_KEY)
            job = parser.parse(filename)
            documents = job.get_text_documents()

            for i, doc in enumerate(documents):
                pages_data.append(PageText(
                    page_number=i + 1,
                    text=doc.text,
                    filename=os.path.basename(filename)
                ))

            logger.info("✅ Fallback to LLaMAParse succeeded.")
        except Exception as e:
            logger.error(f"❌ LLaMAParse fallback failed: {e}")

    return PdfExtractionResult(pages=pages_data)
