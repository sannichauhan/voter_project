import glob
import os
import re
import cv2
import easyocr
import numpy as np

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from pdf2image import convert_from_path

from .forms import UploadFileForm

# Initialize EasyOCR reader (Module level loading is performant)
reader = easyocr.Reader(["hi", "en"], gpu=False)


# -------------------------------------------------------------------
# Helper Utilities
# -------------------------------------------------------------------


def convert_hindi_num_to_english(text):
    hindi_digits = "०१२३४५६७८९"
    english_digits = "0123456789"
    translation_table = str.maketrans(hindi_digits, english_digits)
    return text.translate(translation_table)


def natural_sort_key(s):
    return [
        int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)
    ]


def crop_exact_voter_boxes(pdf_path, output_folder="media/voter_blocks_temp"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = convert_from_path(pdf_path, dpi=300)
    total_saved = 0

    for page_num, pillow_img in enumerate(images, start=1):
        img_bgr = cv2.cvtColor(np.array(pillow_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        img_h, img_w = gray.shape
        boxes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (0.25 * img_w < w < 0.35 * img_w) and (
                0.06 * img_h < h < 0.12 * img_h
            ):
                boxes.append((x, y, w, h))

        boxes.sort(key=lambda b: (round(b[1] / 40), b[0]))

        for idx, (x, y, w, h) in enumerate(boxes, start=1):
            cropped_box = img_bgr[y : y + h, x : x + w]
            total_saved += 1
            file_name = f"page_{page_num}_box_{idx}.png"
            file_path = os.path.join(output_folder, file_name)
            cv2.imwrite(file_path, cropped_box)

    return output_folder


def clean_voter_card_ocr(ocr_list):
    raw_text = " \n ".join(ocr_list)
    clean_text = convert_hindi_num_to_english(raw_text)

    serial_no = "N/A"
    epic_id = "N/A"
    name = "N/A"
    relation_type = "N/A"
    relation_name = "N/A"
    house_no = "N/A"
    age = "N/A"
    gender = "N/A"

    epic_match = re.search(
        r"(?:NXR\d+|UP/\d+/\d+/\d+)", clean_text, re.IGNORECASE
    )
    if epic_match:
        epic_id = epic_match.group(0)

    sr_match = re.search(r"^\s*(\d{1,4})\b", clean_text)
    if sr_match:
        serial_no = sr_match.group(1)

    name_match = re.search(
        r"नाम\s*[:ः]?\s*([^\n\r:,]+)", clean_text, re.IGNORECASE
    )
    if name_match:
        name = name_match.group(1).strip()
    else:
        for i, item in enumerate(ocr_list):
            if item.strip() == "नाम" and i + 1 < len(ocr_list):
                name = convert_hindi_num_to_english(
                    ocr_list[i + 1].strip()
                )
                break

    rel_match = re.search(
        r"(पति|पिता)\s*का\s*नाम\s*[:ः]?\s*([^\n\r:,]+)",
        clean_text,
        re.IGNORECASE,
    )
    if rel_match:
        relation_type = rel_match.group(1).strip()
        relation_name = rel_match.group(2).strip()

    house_match = re.search(
        r"मकान\s*संख्या\s*[:ः]?\s*([^\n\r:,]+)", clean_text, re.IGNORECASE
    )
    if house_match:
        house_no = house_match.group(1).strip()

    age_match = re.search(r"आयु\s*[:ः]?\s*(\d+)", clean_text, re.IGNORECASE)
    if age_match:
        age = age_match.group(1).strip()
    else:
        age_alt = re.search(r"(\d+)\s*लिंग", clean_text, re.IGNORECASE)
        if age_alt:
            age = age_alt.group(1).strip()

    if "महिला" in clean_text:
        gender = "महिला"
    elif "पुरुष" in clean_text:
        gender = "पुरुष"

    return {
        "क्रम संख्या": serial_no,
        "EPIC ID": epic_id,
        "नाम": name,
        "संबंध": relation_type,
        "पिता/पति का नाम": relation_name,
        "मकान संख्या": house_no,
        "आयु": age,
        "लिंग": gender,
    }


# -------------------------------------------------------------------
# Views / API Endpoints
# -------------------------------------------------------------------


def home_page(request):
    return render(request, "index.html")


@csrf_exempt  # Optional: API client / Postman hit karne ke liye
def upload_pdf_view(request):
    if request.method == "POST":
        pdf_file = request.FILES.get("pdf_file")

        if not pdf_file:
            return JsonResponse(
                {"status": "error", "message": "No PDF file provided"},
                status=400,
            )

        # 1. Media folder me PDF Save karein
        media_dir = os.path.join("media", "uploaded_pdfs")
        os.makedirs(media_dir, exist_ok=True)
        file_path = os.path.join(media_dir, pdf_file.name)

        with open(file_path, "wb+") as destination:
            for chunk in pdf_file.chunks():
                destination.write(chunk)

        # 2. PDF Se Blocks Crop Karein
        output_folder = os.path.join(
            "media", "voter_blocks_temp", pdf_file.name.replace(".pdf", "")
        )
        crop_exact_voter_boxes(file_path, output_folder)

        # 3. EasyOCR Processing on Cropped Images
        image_paths = glob.glob(os.path.join(output_folder, "*.png"))
        image_paths.sort(key=natural_sort_key)

        json_obj = []
        for img_path in image_paths:
            results = reader.readtext(img_path, detail=0)
            parsed_record = clean_voter_card_ocr(results)
            json_obj.append(parsed_record)

        # 4. JSON Response Return Karein
        return JsonResponse(
            {
                "status": "success",
                "total_records": len(json_obj),
                "data": json_obj,
            },
            json_dumps_params={"ensure_ascii": False},
            safe=False,
        )

    return JsonResponse(
        {"status": "error", "message": "Only POST requests are allowed"},
        status=405,
    )