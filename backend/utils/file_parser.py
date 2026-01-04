"""
File parsing utilities for PDF and DOCX documents.
Extracts text with page number tracking.
Supports OCR for scanned/image-based PDFs.
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional
import pypdf
from docx import Document

# Optional OCR imports (for scanned PDFs)
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class FileParser:
    """Parser for PDF and DOCX files with page tracking."""
    
    @staticmethod
    def parse_pdf(file_path: Path) -> List[Tuple[str, int, bool]]:
        """
        Parse PDF file and extract text with page numbers.
        Falls back to OCR if text extraction fails (for scanned PDFs).
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of tuples: (text, page_number, is_ocr)
            is_ocr: True if OCR was used for this page, False otherwise
        """
        text_pages = []
        used_ocr = False
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                # Extract text from each page
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    # Only add non-empty pages
                    if text.strip():
                        text_pages.append((text, page_num + 1, False))  # 1-indexed pages, not OCR
                        
        except Exception as e:
            raise ValueError(f"Error parsing PDF {file_path}: {str(e)}")
        
        # If no text extracted, try OCR (for scanned/image-based PDFs)
        if not text_pages and OCR_AVAILABLE:
            try:
                ocr_pages = FileParser._parse_pdf_with_ocr(file_path)
                # Mark all OCR pages as using OCR
                text_pages = [(text, page_num, True) for text, page_num in ocr_pages]
                used_ocr = True
            except Exception as ocr_error:
                # If OCR fails, log warning but don't fail completely
                print(f"Warning: OCR extraction failed: {str(ocr_error)}")
                print("Note: Install Tesseract OCR for scanned PDF support.")
                print("Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
                print("Linux: sudo apt-get install tesseract-ocr")
                print("macOS: brew install tesseract")
        
        return text_pages
    
    @staticmethod
    def _parse_pdf_with_ocr(file_path: Path) -> List[Tuple[str, int]]:
        """
        Parse PDF using OCR (for scanned/image-based PDFs).
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of tuples: (text, page_number)
        """
        if not OCR_AVAILABLE:
            raise ImportError("OCR libraries not available. Install pytesseract, pdf2image, and Pillow.")
        
        text_pages = []
        
        try:
            # Convert PDF pages to images
            # Note: Requires poppler-utils to be installed on the system
            # Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases
            # Linux: sudo apt-get install poppler-utils
            # macOS: brew install poppler
            images = convert_from_path(str(file_path), dpi=300)
            # Higher DPI for better OCR accuracy
            
            # Extract text from each page using OCR
            for page_num, image in enumerate(images, start=1):
                try:
                    # Perform OCR on the image
                    # Note: Requires Tesseract OCR to be installed
                    # Windows: Set TESSDATA_PREFIX environment variable if needed
                    text = pytesseract.image_to_string(image, lang='eng')
                    # Default to English; can be extended for multilingual support
                    
                    # Clean up text
                    text = text.strip()
                    
                    # Only add non-empty pages
                    if text:
                        text_pages.append((text, page_num))
                except pytesseract.TesseractNotFoundError:
                    raise RuntimeError(
                        "Tesseract OCR not found. Please install Tesseract:\n"
                        "Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
                        "Linux: sudo apt-get install tesseract-ocr\n"
                        "macOS: brew install tesseract"
                    )
            
        except FileNotFoundError as e:
            if 'poppler' in str(e).lower() or 'pdftoppm' in str(e).lower():
                raise RuntimeError(
                    "Poppler not found. Required for PDF to image conversion:\n"
                    "Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases\n"
                    "Linux: sudo apt-get install poppler-utils\n"
                    "macOS: brew install poppler"
                )
            raise ValueError(f"Error performing OCR on PDF {file_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error performing OCR on PDF {file_path}: {str(e)}")
        
        return text_pages
    
    @staticmethod
    def parse_docx(file_path: Path) -> List[Tuple[str, int, bool]]:
        """
        Parse DOCX file and extract text.
        Note: DOCX doesn't have native page numbers, so we estimate based on content.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            List of tuples: (text, estimated_page_number, is_ocr)
            is_ocr: Always False for DOCX files
        """
        text_pages = []
        
        try:
            doc = Document(file_path)
            
            # Estimate pages: ~500 words per page
            words_per_page = 500
            current_text = ""
            current_page = 1
            word_count = 0
            
            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                para_text = paragraph.text.strip()
                if para_text:
                    current_text += para_text + "\n"
                    word_count += len(para_text.split())
                    
                    # Estimate new page every ~500 words
                    if word_count >= words_per_page:
                        text_pages.append((current_text.strip(), current_page, False))  # DOCX never uses OCR
                        current_text = ""
                        word_count = 0
                        current_page += 1
            
            # Add remaining text
            if current_text.strip():
                text_pages.append((current_text.strip(), current_page, False))
            
            # If no pages created, create at least one
            if not text_pages:
                full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                if full_text:
                    text_pages.append((full_text, 1, False))
                    
        except Exception as e:
            raise ValueError(f"Error parsing DOCX {file_path}: {str(e)}")
        
        return text_pages
    
    @staticmethod
    def parse_file(file_path: Path) -> List[Tuple[str, int, bool]]:
        """
        Parse file based on extension (PDF or DOCX).
        
        Args:
            file_path: Path to file
            
        Returns:
            List of tuples: (text, page_number, is_ocr)
            is_ocr: True if OCR was used, False otherwise
        """
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            return FileParser.parse_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return FileParser.parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported: .pdf, .docx")

