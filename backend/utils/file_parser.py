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
from backend.config import settings

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
                pages_needing_ocr = []
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    # Always add the page, even if text is minimal
                    # This ensures we don't lose any content
                    text = text.strip() if text else ""
                    
                    if text:
                        # Page has text, add it
                        text_pages.append((text, page_num + 1, False))  # 1-indexed pages, not OCR
                    else:
                        # Page has no text - might be image-based, mark for OCR
                        pages_needing_ocr.append(page_num + 1)
                        
        except Exception as e:
            raise ValueError(f"Error parsing PDF {file_path}: {str(e)}")
        
        # Try OCR for pages that had no text (per-page OCR fallback)
        if pages_needing_ocr and OCR_AVAILABLE:
            try:
                # Try OCR for specific pages that failed text extraction
                ocr_results = FileParser._parse_pdf_with_ocr_selective(file_path, pages_needing_ocr)
                for text, page_num in ocr_results:
                    if text:  # Only add if OCR found text
                        text_pages.append((text, page_num, True))  # Mark as OCR
            except Exception as ocr_error:
                # If selective OCR fails, try full document OCR
                if not text_pages:  # Only if we have NO text at all
                    try:
                        ocr_pages = FileParser._parse_pdf_with_ocr(file_path)
                        text_pages = [(text, page_num, True) for text, page_num in ocr_pages]
                    except Exception as full_ocr_error:
                        print(f"Warning: OCR extraction failed: {str(full_ocr_error)}")
                        print("Note: Install Tesseract OCR for scanned PDF support.")
        
        # Ensure we have at least some pages - if not, it's likely a problem
        if not text_pages:
            print(f"Warning: No text extracted from PDF {file_path}")
            print("This might indicate:")
            print("  1. PDF is image-based and OCR is not available")
            print("  2. PDF is corrupted or encrypted")
            print("  3. PDF has no extractable text content")
        
        return text_pages
    
    @staticmethod
    def _parse_pdf_with_ocr_selective(file_path: Path, page_numbers: List[int]) -> List[Tuple[str, int]]:
        """
        Parse specific pages of PDF using OCR (for pages that failed text extraction).
        
        Args:
            file_path: Path to PDF file
            page_numbers: List of page numbers (1-indexed) to process with OCR
            
        Returns:
            List of tuples: (text, page_number)
        """
        if not OCR_AVAILABLE:
            return []
        
        text_pages = []
        
        try:
            # Convert PDF pages to images
            images = convert_from_path(str(file_path), dpi=300, first_page=min(page_numbers), last_page=max(page_numbers))
            
            # Map image indices to actual page numbers
            page_map = {i: page_num for i, page_num in enumerate(range(min(page_numbers), max(page_numbers) + 1), start=0)}
            
            # Extract text from specified pages using OCR
            for img_idx, image in enumerate(images):
                actual_page = page_map.get(img_idx)
                if actual_page and actual_page in page_numbers:
                    try:
                        text = pytesseract.image_to_string(image, lang=settings.OCR_LANGUAGE)
                        text = text.strip()
                        if text:
                            text_pages.append((text, actual_page))
                    except Exception as e:
                        print(f"Warning: OCR failed for page {actual_page}: {str(e)}")
            
        except Exception as e:
            print(f"Warning: Selective OCR failed: {str(e)}")
        
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
                    text = pytesseract.image_to_string(image, lang=settings.OCR_LANGUAGE)
                    # Uses configurable OCR language (default: eng+ara for bilingual support)
                    
                    # Clean up text
                    text = text.strip()
                    
                    # Always add page, even if text is minimal (to ensure coverage)
                    # Empty text will be handled by chunking
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

