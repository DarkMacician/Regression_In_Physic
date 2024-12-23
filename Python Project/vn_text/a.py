from PyPDF2 import PdfReader, PdfWriter

def extract_first_three_pages(input_pdf_path, output_pdf_path):
    # Open the input PDF
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    # Add the first three pages to the writer
    for page_number in range(49, 53):
        writer.add_page(reader.pages[page_number])

    # Write the output to a new PDF file
    with open(output_pdf_path, 'wb') as output_pdf:
        writer.write(output_pdf)

# Example usage
input_pdf_path = "D:\Python Project/vn_text/10-de-on-tap-cuoi-hoc-ki-1-toan-11-ctst-cau-truc-trac-nghiem-moi.pdf"  # Replace with your input PDF file path
output_pdf_path = "de04.pdf"  # Replace with desired output file path
extract_first_three_pages(input_pdf_path, output_pdf_path)
