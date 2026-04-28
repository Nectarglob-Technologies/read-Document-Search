from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
import fitz  # PyMuPDF
import os
import io

app = FastAPI()

PDF_FOLDER = "pdfs"  # your pdf directory


@app.get("/pdf")
def serve_pdf(file: str = Query(...), page: int = 1, text: str = ""):

    file_path = os.path.join(PDF_FOLDER, file)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    doc = fitz.open(file_path)

    # ----------------------------
    # 🔥 HIGHLIGHT TEXT
    # ----------------------------
    if text:
        for p in doc:
            matches = p.search_for(text)

            for inst in matches:
                highlight = p.add_highlight_annot(inst)
                highlight.update()

    # ----------------------------
    # SAVE TO MEMORY
    # ----------------------------
    buffer = io.BytesIO()
    doc.save(buffer)
    doc.close()

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{file}"'
        }
    )
