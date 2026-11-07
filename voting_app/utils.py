import os
import re
import logging
import cv2
import numpy as np
import pytesseract

from pdf2image import convert_from_path
from django.db import transaction
import uuid

from .models import VoterData


# DEBUG_DIR = os.path.join(
#     os.path.dirname(__file__),
#     "debug_ocr"
# )

# os.makedirs(DEBUG_DIR, exist_ok=True)


# =====================================================
# 1. CONFIGURATION
# =====================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler-26.09.0\Library\bin"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

DPI = 300
OCR_LANG = "hin+eng"

ROWS = 10
COLS = 3

TOP_MARGIN = 0.018    # Upar se margin adjust kiya hai taaki naam na kate
BOTTOM_MARGIN = 0.02
LEFT_MARGIN = 0.01
RIGHT_MARGIN = 0.01

MIN_AGE = 18
MAX_AGE = 100

DEBUG_OCR = True

logger = logging.getLogger(__name__)

DEVANAGARI_DIGITS = str.maketrans(
    "०१२३४५६७८९",
    "0123456789"
)


# =====================================================
# 2. TEXT CLEANING
# =====================================================

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\x0c", " ")
    text = text.translate(DEVANAGARI_DIGITS)
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


def clean_value(text):
    if not text:
        return ""

    text = clean_text(text)

    text = re.sub(
        r"^[\sः:;|,.\-_=]+",
        "",
        text
    )

    text = re.sub(
        r"[\sः:;|,.\-_=]+$",
        "",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =====================================================
# 3. IMAGE PREPROCESSING
# =====================================================

def preprocess_cell(cell):
    if cell is None or cell.size == 0:
        return None

    height, width = cell.shape

    # 1. Thoda bada karein clarity ke liye
    cell = cv2.resize(
        cell,
        (width * 2, height * 2),
        interpolation=cv2.INTER_CUBIC
    )

    # 2. Bilateral filter se noise hatayein par edges sharp rakhein
    denoised = cv2.bilateralFilter(cell, 5, 50, 50)

    # 3. Thresholding
    processed = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        15
    )

    # 4. Border Clearing: Cell ke kinaro par agar black box ki line hai toh use safed (white) kar dein
    border_pixels = 5
    processed[:border_pixels, :] = 255  # Top border
    processed[-border_pixels:, :] = 255 # Bottom border
    processed[:, :border_pixels] = 255  # Left border
    processed[:, -border_pixels:] = 255 # Right border

    # 5. Charon taraf safe white padding dein
    processed = cv2.copyMakeBorder(
        processed,
        15, 15, 15, 15,
        cv2.BORDER_CONSTANT,
        value=255
    )

    return processed


# =====================================================
# 4. OCR EXTRACTION
# =====================================================

def read_ocr(image):
    if image is None or image.size == 0:
        return ""

    configs = [
        "--oem 1 --psm 6",
        "--oem 1 --psm 4"
    ]

    best_text = ""
    max_score = -1

    for config in configs:
        try:
            text = pytesseract.image_to_string(
                image,
                lang=OCR_LANG,
                config=config,
                timeout=30
            )
            text = clean_text(text)
            
            if text:
                score = len(re.findall(r'[A-Za-z0-9\u0900-\u097F]', text))
                if score > max_score:
                    max_score = score
                    best_text = text
        except Exception as e:
            logger.warning("OCR failed (%s): %s", config, str(e))

    return best_text


# =====================================================
# 5. VOTER ID EXTRACTION (ROBUST & OCR ERROR CORRECTED)
# =====================================================

def extract_voter_id(text):
    if not text:
        return ""
        
    text_clean = re.sub(r"\s+", "", text).upper()
    
    # Helper to normalize common OCR confusions in number/ID segments
    def fix_ocr_chars(val):
        return (val.replace('I', '1')
                   .replace('L', '1')
                   .replace('O', '0')
                   .replace('S', '5')
                   .replace('Z', '2')
                   .replace('B', '8'))

    # 1. Slash format (Yahan \d+ ki jagah [A-Z0-9]+ kiya hai taaki beech ke OCR errors bhi pakde jayein)
    match_slash = re.search(r"([A-Z]{2,5}/[A-Z0-9]+/[A-Z0-9]+/[A-Z0-9]+)", text_clean)
    if match_slash:
        vid = match_slash.group(1)
        parts = vid.split('/')
        normalized_parts = [parts[0]]
        for p in parts[1:]:
            normalized_parts.append(fix_ocr_chars(p))
        return "/".join(normalized_parts)
        
    # 2. Partial slash format
    match_partial_slash = re.search(r"(/[A-Z0-9]+/[A-Z0-9]+/[A-Z0-9]+)", text_clean)
    if match_partial_slash:
        sub = match_partial_slash.group(1)
        parts = sub.split('/')
        normalized_parts = ['UP']
        for p in parts[1:]:
            normalized_parts.append(fix_ocr_chars(p))
        return "/".join(normalized_parts)
        
    # 3. Standard 3 letters + 7 characters (jaise NXR0423038)
    match_std = re.search(r"\b([A-Z]{3}[A-Z0-9]{7})\b", text_clean)
    if match_std:
        vid = match_std.group(1)
        prefix = vid[:3]
        digits_part = fix_ocr_chars(vid[3:])
        return prefix + digits_part
        
    # 4. General fallback
    match_gen = re.search(r"\b([A-Z]{2,5}[A-Z0-9]{6,12})\b", text_clean)
    if match_gen:
        return fix_ocr_chars(match_gen.group(1))

    return ""


# =====================================================
# 6. SERIAL NUMBER EXTRACTION
# =====================================================

def extract_serial(text):
    text = clean_text(text)
    if not text:
        return ""
    
    matches = re.findall(r"\b(\d{1,3})\b", text)
    for num_str in matches:
        try:
            num = int(num_str)
            if 1 <= num <= 999:
                return str(num)
        except ValueError:
            continue
            
    return ""


# =====================================================
# 7. NAME EXTRACTION
# =====================================================

def extract_name(text):
    text = clean_text(text)
    
    if "अनुभाग" in text or "बरवा" in text:
        match_named = re.search(r"नाम\s*[:：|=-]?\s*([^\n]+)", text, flags=re.IGNORECASE)
        if match_named:
            val = clean_value(match_named.group(1))
            val = re.split(r"(?:पिता|पति|माता|आयु|लिंग|Age|Father|Husband)", val, flags=re.IGNORECASE)[0]
            val = clean_value(val)
            if len(val) > 2 and "अनुभाग" not in val:
                return val
        return ""

    patterns = [
        r"(?:निर्वाचक|मतदाता)\s*(?:का)?\s*नाम\s*[:：|=-]?\s*([^\n]+)",
        r"\bनाम\s*[:：|=-]?\s*([^\n]+)",
        r"\bName\b\s*[:：|=-]?\s*([^\n]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            val = clean_value(match.group(1))
            val = re.split(r"(?:पिता|पति|माता|आयु|लिंग|Age|Father|Husband)", val, flags=re.IGNORECASE)[0]
            val = clean_value(val)
            if len(val) > 2:
                return val

    lines = [clean_value(l) for l in text.splitlines() if clean_value(l)]
    for line in lines:
        if not re.search(r"(\d{7}|[A-Z]{3}\d{7}|आयु|पति|पिता|लिंग|मकान|अनुभाग)", line) and len(line) > 3:
            return line

    return ""


# =====================================================
# 8. RELATION NAME EXTRACTION
# =====================================================

def extract_relation(text):
    if not text:
        return ""
    text = clean_text(text)
    
    # Agar cell me sirf anubhaag/section hai toh return empty
    if re.search(r"^(?:अनुभाग|Section)", text, re.IGNORECASE) and not re.search(r"पिता|पति|माता|प्रति", text):
        return ""
    
    lines = [clean_value(l) for l in text.splitlines() if clean_value(l)]
    
    for i, line in enumerate(lines):
        # Agar line me pita, pati, mata, prati ya 'के नाम' jaisa shabd hai
        if re.search(r"(पिता|पति|माता|प्रति|के\s*नाम)", line, re.IGNORECASE):
            
            # 1. Check karein ki kya usi line me colon/semicolon ke baad naam hai
            # Jaise: "पिता का नाम: हदीश" ya "पिता का नाम; नगीना"
            parts = re.split(r"[:：;\-–\|\=]", line, maxsplit=1)
            if len(parts) > 1:
                val = clean_value(parts[1])
                val = re.split(r"(?:मकान|आयु|लिंग|Age|House|Gender|फोटो|photo|उपलब्ध)", val, flags=re.IGNORECASE)[0]
                val = clean_value(val)
                if val and len(val) > 1 and not re.search(r"पिता|पति|माता|प्रति", val):
                    return val
            
            # 2. Agar usi line me nahi mila, toh theek agli line (next line) me check karein
            if i + 1 < len(lines):
                next_line = clean_value(lines[i + 1])
                # Agli line se aage ke label words hata dein
                val = re.sub(r"^(?:का\s*नाम|नाम|के\s*नाम)\s*[:：;\-–\|\=]?\s*", "", next_line, flags=re.IGNORECASE)
                val = re.split(r"(?:मकान|आयु|लिंग|Age|House|Gender|फोटो|photo|उपलब्ध)", val, flags=re.IGNORECASE)[0]
                val = clean_value(val)
                if val and len(val) > 1 and not re.search(r"पिता|पति|माता|प्रति|मकान", val):
                    return val

    return ""


# =====================================================
# 9. HOUSE NUMBER EXTRACTION
# =====================================================

def extract_house_no(text):
    if not text:
        return ""
    text = clean_text(text)
    
    # 1. Look for line with 'मकान' or 'House'
    lines = [clean_value(l) for l in text.splitlines() if clean_value(l)]
    for line in lines:
        if re.search(r"(मकान|House|संख?्या)", line, flags=re.IGNORECASE):
            match = re.search(r"\b(\d{1,4}|[A-Z0-9]+)\b", line)
            if match:
                num = match.group(1)
                
                # --- OCR ERROR FIXING FOR HOUSE NO ---
                # 1. Tesseract reads '01' as '04' or '0I'
                if num in ["04", "0I", "0i", "O4", "0A"]:
                    num = "01"
                # 2. Tesseract reads '1' as '7' or 'I' or 'l'
                elif num in ["7", "I", "l", "T"]:
                    num = "1"
                # --------------------------------------

                if not re.search(r"आयु|Age|लिंग|Gender", line, flags=re.IGNORECASE):
                    return num

    return ""


# =====================================================
# 10. AGE EXTRACTION
# =====================================================

def extract_age(text):
    text = clean_text(text)
    if not text:
        return None

    match = re.search(r"(?:आयु|Age|उम्र)\s*[:：|=-]?\s*(\d{1,3})", text, re.IGNORECASE)
    if match:
        try:
            age = int(match.group(1))
            if MIN_AGE <= age <= MAX_AGE:
                return age
        except ValueError:
            pass

    numbers = re.findall(r"\b(\d{2})\b", text)
    for num_str in numbers:
        try:
            age = int(num_str)
            if 18 <= age <= 100:
                return age
        except ValueError:
            continue

    return None


# =====================================================
# 11. GENDER EXTRACTION
# =====================================================

def extract_gender(text):
    text = clean_text(text)

    if re.search(r"महिला|स्त्री|औरत|Female", text, re.IGNORECASE):
        return "महिला"

    if re.search(r"पुरुष|पुरूष|Male", text, re.IGNORECASE):
        return "पुरुष"

    return ""


# =====================================================
# 12. EXTRACT COMPLETE VOTER DATA
# =====================================================

def extract_voter_data(cell_text):
    return {
        "voter_id": extract_voter_id(cell_text),
        "serial_no": extract_serial(cell_text),
        "name": extract_name(cell_text),
        "relation_name": extract_relation(cell_text),
        "house_no": extract_house_no(cell_text),
        "age": extract_age(cell_text),
        "gender": extract_gender(cell_text),
    }


# =====================================================
# 13. PROCESS ONE PAGE
# =====================================================

def process_page(image, page_no):
    total_boxes = ROWS * COLS
    empty_ocr = 0
    detected_records = 0
    missing_voter_id = 0

    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape

    top = int(height * TOP_MARGIN)
    bottom = int(height * (1 - BOTTOM_MARGIN))
    left = int(width * LEFT_MARGIN)
    right = int(width * (1 - RIGHT_MARGIN))

    grid_height = bottom - top
    grid_width = right - left

    page_records = []

    for row in range(ROWS):
        for col in range(COLS):
            y1 = top + int(row * (grid_height / ROWS))
            y2 = top + int((row + 1) * (grid_height / ROWS))
            
            x1 = left + int(col * (grid_width / COLS))
            x2 = left + int((col + 1) * (grid_width / COLS))

            crop_y1 = max(0, y1 + 1)
            crop_y2 = min(height, y2 - 2)
            crop_x1 = max(0, x1 + 1)
            crop_x2 = min(width, x2 - 1)

            cell = gray[crop_y1:crop_y2, crop_x1:crop_x2]

            if cell is None or cell.size == 0:
                continue

            processed = preprocess_cell(cell)

            debug_name = f"page_{page_no}_row_{row+1}_col_{col+1}"

            cv2.imwrite(
                os.path.join(DEBUG_DIR, f"{debug_name}_original.png"),
                cell
            )
            if processed is not None and processed.size > 0:
                cv2.imwrite(
                    os.path.join(DEBUG_DIR, f"{debug_name}_processed.png"),
                    processed
                )

            if processed is None or processed.size == 0:
                continue

            cell_text = read_ocr(processed)
            print(f"--- Cell Text [Row {row+1}, Col {col+1}] ---\n{cell_text}\n-------------------------")

            if not cell_text:
                empty_ocr += 1
                continue

            data = extract_voter_data(cell_text)

            if not data["name"]:
                continue

            data["page_no"] = page_no
            data["row_no"] = row + 1
            data["column_no"] = col + 1
            data["raw_text"] = cell_text

            if not data["voter_id"]:
                missing_voter_id += 1

            page_records.append(data)
            detected_records += 1

    summary = {
        "total_boxes": total_boxes,
        "empty_ocr": empty_ocr,
        "detected_records": detected_records,
        "missing_voter_id": missing_voter_id,
    }

    return page_records, summary


# =====================================================
# 14. DATABASE SAVE
# =====================================================

def save_voter_records(records):
    saved = 0
    updated = 0
    skipped = 0
    pending_id = 0

    for data in records:
        voter_id = (data.get("voter_id") or "").strip().upper()

        defaults = {
            "serial_no": data.get("serial_no") or "",
            "name": data.get("name") or "",
            "relation_name": data.get("relation_name") or "",
            "house_no": data.get("house_no") or "",
            "age": data.get("age"),
            "gender": data.get("gender") or "",
        }

        try:
            with transaction.atomic():
                if not voter_id:
                    voter_id = f"TEMP_{uuid.uuid4().hex[:10].upper()}"
                    pending_id += 1

                obj, created = VoterData.objects.get_or_create(
                    voter_id=voter_id,
                    defaults=defaults
                )

                if created:
                    saved += 1
                    continue

                changed_fields = []

                for field, value in defaults.items():
                    if value is None or value == "":
                        continue

                    if getattr(obj, field) != value:
                        setattr(obj, field, value)
                        changed_fields.append(field)

                if changed_fields:
                    obj.save(update_fields=changed_fields)
                    updated += 1

        except Exception as e:
            skipped += 1
            print(f"--- DB ERROR: {e} ---")
            logger.exception(
                "Database error: voter_id=%s, page=%s, row=%s, col=%s",
                voter_id,
                data.get("page_no"),
                data.get("row_no"),
                data.get("column_no")
            )

    return {
        "saved": saved,
        "updated": updated,
        "skipped": skipped,
        "pending_id": pending_id
    }


# =====================================================
# 15. MAIN PDF PROCESSING
# =====================================================

def parse_voter_pdf(file_path):
    print("\n" + "=" * 55)
    print("VOTER PDF OCR STARTED")
    print("=" * 55)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if not os.path.exists(TESSERACT_PATH):
        raise FileNotFoundError(f"Tesseract not found: {TESSERACT_PATH}")

    if not os.path.isdir(POPPLER_PATH):
        raise FileNotFoundError(f"Poppler directory not found: {POPPLER_PATH}")

    try:
        images = convert_from_path(
            file_path,
            dpi=DPI,
            poppler_path=POPPLER_PATH
        )
    except Exception as e:
        logger.exception("PDF conversion failed")
        raise RuntimeError(f"PDF conversion error: {e}") from e

    total_pages = len(images)
    all_records = []

    total_boxes = 0
    empty_ocr = 0
    detected_records = 0
    missing_voter_id = 0

    print("Total PDF Pages:", total_pages)

    for page_index, image in enumerate(images):
        page_no = page_index + 1
        print(f"\nProcessing Page {page_no}/{total_pages}")

        try:
            records, summary = process_page(image, page_no)
            all_records.extend(records)

            total_boxes += summary["total_boxes"]
            empty_ocr += summary["empty_ocr"]
            detected_records += summary["detected_records"]
            missing_voter_id += summary["missing_voter_id"]

            print(f"Page {page_no}: {len(records)} records detected")

        except Exception:
            logger.exception("Page %s processing failed", page_no)

    result = save_voter_records(all_records)

    print("\n" + "=" * 55)
    print("VOTER OCR COMPLETED")
    print("=" * 55)

    print("Total Pages:", total_pages)
    print("Total Boxes:", total_boxes)
    print("Empty OCR:", empty_ocr)
    print("Detected Records:", detected_records)
    print("Missing Voter ID:", missing_voter_id)
    print("Not Detected:", total_boxes - detected_records)

    print("New Records:", result["saved"])
    print("Updated Records:", result["updated"])
    print("Skipped Records:", result["skipped"])
    print("Pending ID:", result["pending_id"])
    print("=" * 55)

    return {
        "total_pages": total_pages,
        "total_boxes": total_boxes,
        "empty_ocr": empty_ocr,
        "total_detected": len(all_records),
        "missing_voter_id": missing_voter_id,
        **result
    }