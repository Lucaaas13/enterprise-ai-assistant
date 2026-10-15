from io import BytesIO

from pypdf import PdfReader


def extract_pages_from_pdf(
    file_content: bytes,
) -> list[tuple[int, str]]:
    reader = PdfReader(BytesIO(file_content))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append(
                (
                    page_number,
                    text.strip(),
                )
            )

    return pages
