"""
Convert all Markdown documentation files in docs/ to professional PDF.
Uses fpdf2 (pure Python, no system dependencies).

Usage:
    python docs/generate_pdfs.py                 # Convert all .md files in docs/
    python docs/generate_pdfs.py Phase_1         # Convert a specific file
"""
import sys
import os
import re
import glob
from fpdf import FPDF

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))


def sanitize_text(text):
    """Replace Unicode characters with ASCII equivalents for fpdf2 compatibility."""
    replacements = {
        "\u2014": "-",      # em-dash
        "\u2013": "-",      # en-dash
        "\u2019": "'",      # right single quote
        "\u2018": "'",      # left single quote
        "\u201c": '"',      # left double quote
        "\u201d": '"',      # right double quote
        "\u2026": "...",    # ellipsis
        "\u2192": "->",     # right arrow
        "\u2190": "<-",     # left arrow
        "\u2193": "v",      # down arrow
        "\u2191": "^",      # up arrow
        "\u2022": "-",      # bullet
        "\u25b6": ">",      # play
        "\u2713": "[OK]",   # checkmark
        "\u2714": "[OK]",   # heavy checkmark
        "\u2717": "[X]",    # cross
        "\u2718": "[X]",    # heavy cross
        "\u00d7": "x",      # multiplication sign
        "\u2265": ">=",     # >=
        "\u2264": "<=",     # <=
        "\u2260": "!=",     # !=
        "\u03b2": "beta",   # beta
        "\u2080": "0",      # subscript 0
        "\u2081": "1",      # subscript 1
        "\u2085": "5",      # subscript 5
        "\u00b2": "2",      # superscript 2
        "\u00b3": "3",      # superscript 3
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove any remaining non-latin-1 characters
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text



class DocPDF(FPDF):
    """Custom PDF with header, footer, and Markdown rendering."""

    def __init__(self, title=""):
        super().__init__()
        self.doc_title = title
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, "Business Entity Resolution - Documentation", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title, level=1):
        """Render a heading."""
        title = sanitize_text(title)
        if level == 1:
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(26, 26, 46)
            self.ln(6)
            self.multi_cell(0, 10, title)
            # Red underline
            self.set_draw_color(233, 69, 96)
            self.set_line_width(0.8)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(6)
        elif level == 2:
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(22, 33, 62)
            self.ln(5)
            self.multi_cell(0, 8, title)
            # Light underline
            self.set_draw_color(220, 220, 220)
            self.set_line_width(0.3)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)
        elif level == 3:
            self.set_font("Helvetica", "B", 12)
            self.set_text_color(15, 52, 96)
            self.ln(4)
            self.multi_cell(0, 7, title)
            self.ln(2)
        elif level == 4:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(50, 50, 50)
            self.ln(3)
            self.multi_cell(0, 7, title)
            self.ln(2)

    def body_text(self, text):
        """Render body paragraph text with inline formatting."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(34, 34, 34)
        text = sanitize_text(text)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def code_block(self, code):
        """Render a code block with dark background."""
        self.set_fill_color(45, 45, 45)
        self.set_text_color(248, 248, 242)
        self.set_font("Courier", "", 8)
        
        lines = code.strip().split("\n")
        # Calculate height
        block_height = len(lines) * 5 + 6
        
        # Check if we need a new page
        if self.get_y() + block_height > self.h - 25:
            self.add_page()
        
        start_y = self.get_y()
        x = self.l_margin
        w = self.w - self.l_margin - self.r_margin
        
        # Draw background
        self.rect(x, start_y, w, block_height, style="F")
        
        self.set_xy(x + 4, start_y + 3)
        for i, line in enumerate(lines):
            line = sanitize_text(line)
            self.cell(0, 5, line)
            if i < len(lines) - 1:
                self.ln(5)
                self.set_x(x + 4)
        
        self.set_y(start_y + block_height + 4)
        self.set_text_color(34, 34, 34)

    def render_table(self, headers, rows):
        """Render a table with styled headers."""
        self.set_font("Helvetica", "", 9)
        
        # Calculate column widths
        num_cols = len(headers)
        available_width = self.w - self.l_margin - self.r_margin
        col_width = available_width / num_cols
        
        # Adjust widths based on content
        col_widths = []
        for i in range(num_cols):
            max_len = len(headers[i])
            for row in rows:
                if i < len(row):
                    max_len = max(max_len, len(str(row[i])))
            col_widths.append(max(max_len, 5))
        
        # Normalize widths
        total = sum(col_widths)
        col_widths = [w / total * available_width for w in col_widths]
        # Cap at reasonable max
        col_widths = [min(w, available_width * 0.5) for w in col_widths]
        # Redistribute remaining
        remaining = available_width - sum(col_widths)
        if remaining > 0:
            col_widths = [w + remaining / num_cols for w in col_widths]
        
        row_height = 7
        
        # Check if table fits on page
        table_height = (len(rows) + 1) * row_height + 4
        if self.get_y() + table_height > self.h - 25:
            self.add_page()
        
        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(26, 26, 46)
        self.set_text_color(255, 255, 255)
        for i, header in enumerate(headers):
            header = sanitize_text(header)
            self.cell(col_widths[i], row_height, header, border=1, fill=True)
        self.ln()
        
        # Data rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(34, 34, 34)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 0:
                self.set_fill_color(248, 249, 250)
            else:
                self.set_fill_color(255, 255, 255)
            
            for i in range(num_cols):
                val = str(row[i]) if i < len(row) else ""
                val = sanitize_text(val)
                # Truncate if too long
                max_chars = int(col_widths[i] / 2)
                if len(val) > max_chars:
                    val = val[:max_chars - 2] + ".."
                self.cell(col_widths[i], row_height, val, border=1, fill=True)
            self.ln()
        
        self.ln(4)

    def bullet_item(self, text, indent=0):
        """Render a bullet point."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(34, 34, 34)
        x = self.l_margin + indent * 8
        self.set_x(x)
        text = sanitize_text(text)
        self.cell(5, 6, chr(149))  # bullet char
        self.multi_cell(self.w - self.r_margin - x - 5, 6, " " + text)
        self.ln(1)

    def hr(self):
        """Horizontal rule."""
        self.set_draw_color(230, 230, 230)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(6)

    def bold_text(self, text):
        """Render bold inline text."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(26, 26, 46)
        text = sanitize_text(text)
        self.multi_cell(0, 6, text)
        self.set_font("Helvetica", "", 10)
        self.ln(1)


def parse_table(lines, start_idx):
    """Parse a Markdown table starting at start_idx, return (headers, rows, end_idx)."""
    headers = [h.strip() for h in lines[start_idx].strip("|").split("|")]
    rows = []
    idx = start_idx + 2  # skip separator line
    while idx < len(lines) and "|" in lines[idx] and lines[idx].strip():
        row = [c.strip() for c in lines[idx].strip("|").split("|")]
        rows.append(row)
        idx += 1
    return headers, rows, idx


def md_to_pdf(md_path, pdf_path):
    """Convert a Markdown file to PDF."""
    print(f"  Converting: {os.path.basename(md_path)}")
    
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = content.split("\n")
    
    # Extract title from first H1
    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line.lstrip("# ").strip()
            break
    
    pdf = DocPDF(title=title)
    pdf.alias_nb_pages()
    pdf.add_page()
    
    i = 0
    in_code_block = False
    code_lines = []
    
    while i < len(lines):
        line = lines[i]
        
        # Code block (fenced)
        if line.strip().startswith("```"):
            if in_code_block:
                # End code block
                pdf.code_block("\n".join(code_lines))
                code_lines = []
                in_code_block = False
            else:
                # Start code block
                in_code_block = True
            i += 1
            continue
        
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
        
        # Headings
        if line.startswith("#### "):
            pdf.chapter_title(line[5:].strip(), level=4)
            i += 1
            continue
        elif line.startswith("### "):
            pdf.chapter_title(line[4:].strip(), level=3)
            i += 1
            continue
        elif line.startswith("## "):
            pdf.chapter_title(line[3:].strip(), level=2)
            i += 1
            continue
        elif line.startswith("# "):
            pdf.chapter_title(line[2:].strip(), level=1)
            i += 1
            continue
        
        # Horizontal rule
        if line.strip() in ("---", "***", "___"):
            pdf.hr()
            i += 1
            continue
        
        # Table
        if "|" in line and i + 1 < len(lines) and re.match(r"^\|?[\s\-:|]+\|", lines[i + 1]):
            headers, rows, end_idx = parse_table(lines, i)
            pdf.render_table(headers, rows)
            i = end_idx
            continue
        
        # Bullet points
        if re.match(r"^\s*[-*]\s+", line):
            indent = len(line) - len(line.lstrip())
            text = re.sub(r"^\s*[-*]\s+", "", line)
            # Handle **bold** in bullets
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            # Handle `code` in bullets
            text = re.sub(r"`(.+?)`", r"\1", text)
            pdf.bullet_item(text, indent=indent // 2)
            i += 1
            continue
        
        # Numbered list
        if re.match(r"^\s*\d+\.\s+", line):
            text = re.sub(r"^\s*\d+\.\s+", "", line)
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            text = re.sub(r"`(.+?)`", r"\1", text)
            pdf.bullet_item(text)
            i += 1
            continue
        
        # Bold line (starts with **)
        if line.strip().startswith("**") and line.strip().endswith("**"):
            text = line.strip().strip("*")
            pdf.bold_text(text)
            i += 1
            continue
        
        # Empty line
        if not line.strip():
            pdf.ln(3)
            i += 1
            continue
        
        # Regular paragraph
        # Clean up inline markdown
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"`(.+?)`", r"\1", text)
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)  # links
        
        # Accumulate multi-line paragraphs
        para = text
        while i + 1 < len(lines) and lines[i + 1].strip() and \
              not lines[i + 1].startswith("#") and \
              not lines[i + 1].startswith("|") and \
              not lines[i + 1].startswith("```") and \
              not lines[i + 1].strip().startswith("-") and \
              not lines[i + 1].strip().startswith("*") and \
              not lines[i + 1].strip() in ("---", "***"):
            i += 1
            next_text = re.sub(r"\*\*(.+?)\*\*", r"\1", lines[i])
            next_text = re.sub(r"\*(.+?)\*", r"\1", next_text)
            next_text = re.sub(r"`(.+?)`", r"\1", next_text)
            next_text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", next_text)
            para += " " + next_text
        
        if para.strip():
            pdf.body_text(para.strip())
        
        i += 1
    
    pdf.output(pdf_path)
    file_size = os.path.getsize(pdf_path) / 1024
    print(f"    -> Generated: {os.path.basename(pdf_path)} ({file_size:.1f} KB)")


def main():
    print("=" * 60)
    print("  Generating PDF Documentation")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
        md_files = glob.glob(os.path.join(DOCS_DIR, f"*{pattern}*.md"))
    else:
        md_files = [f for f in glob.glob(os.path.join(DOCS_DIR, "*.md"))
                     if os.path.basename(f) != "generate_pdfs.py"]
    
    if not md_files:
        print("  No markdown files found!")
        return
    
    print(f"\n  Found {len(md_files)} file(s):\n")
    
    for md_path in sorted(md_files):
        pdf_path = md_path.replace(".md", ".pdf")
        try:
            md_to_pdf(md_path, pdf_path)
        except Exception as e:
            print(f"    ERROR on {os.path.basename(md_path)}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n  PDFs saved in: {DOCS_DIR}")


if __name__ == "__main__":
    main()
