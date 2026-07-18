"""Generate a few synthetic HR PDFs to test the renamer (no real data)."""
import os
import fitz

HERE = os.path.dirname(os.path.abspath(__file__))

DOCS = {
    "scan_0001.pdf": [
        "QATAR AVIATION SERVICES",
        "",
        "ANNUAL LEAVE APPLICATION FORM",
        "",
        "Employee Name:  Abdul Karim Molla",
        "Employee No.:   5018",
        "Department:     Ground Operations",
        "Leave Type:     Annual Leave",
        "Leave Year:     2025",
        "From: 12/07/2025   To: 05/08/2025",
        "",
        "Employee Signature: ______________",
        "Approved by Manager: ______________",
    ],
    "scan_0002.pdf": [
        "LETTER OF OFFER",
        "",
        "Date: 03 June 2025",
        "",
        "Dear Mr. Abdul Karim Molla,",
        "",
        "We are pleased to offer you the position of",
        "Ramp Agent with Qatar Aviation Services.",
        "Employee Number: 5018",
        "",
        "Yours sincerely,",
        "Human Resources Department",
    ],
    "scan_0003.pdf": [
        "EMPLOYMENT CONTRACT",
        "",
        "This employment contract is made between",
        "the Company and the Employee below:",
        "",
        "Employee Name: Abdul Karim Molla",
        "Staff No.: 5018",
        "Position: Ramp Agent",
        "Contract Duration: 2 years",
    ],
}


def main():
    for name, lines in DOCS.items():
        doc = fitz.open()
        page = doc.new_page()
        y = 80
        for i, line in enumerate(lines):
            size = 18 if i == 2 or (name != "scan_0001.pdf" and i == 0) else 12
            page.insert_text((72, y), line, fontsize=size, fontname="helv")
            y += 26 if line else 14
        doc.save(os.path.join(HERE, name))
        doc.close()
        print("wrote", name)


if __name__ == "__main__":
    main()
