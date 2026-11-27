"""Test pages for the OCRia LayoutNet port (docs/ocr-nano.md): tables (ruled, unruled, merged cells), columns, headings, lists, a form, a 300 dpi JPEG, a sideways scan. Usage: python ocria_layout_pages.py OUT_DIR (OUT_DIR/page.png, the skewed page, comes from the OCR checks)."""
import sys, textwrap
from PIL import Image, ImageDraw, ImageFont

out = sys.argv[1]
HEL = "/System/Library/Fonts/Helvetica.ttc"
TIM = "/System/Library/Fonts/Times.ttc"
def F(path, size, index=0):
    return ImageFont.truetype(path, size, index=index)
HB = lambda s: F(HEL, s, 1)  # bold
H = lambda s: F(HEL, s)
T = lambda s: F(TIM, s)

LOREM = ("Our team reviewed the results of the last quarter and found that demand for the new "
         "service grew faster than expected in every region. Costs rose more slowly than revenue, "
         "which left room to invest in support and training. The board approved the budget for "
         "next year after a short discussion of the risks described in the appendix. Customers "
         "asked for better reporting tools, and the product group has started work on a first "
         "version that will ship in the spring. Hiring remains the main constraint on growth.")

def para(d, x, y, width_chars, text, font, lh):
    for line in textwrap.wrap(text, width_chars):
        d.text((x, y), line, fill="black", font=font)
        y += lh
    return y

def grid_table(d, x, y, colw, rowh, rows, font, hfont, ruled=True, pad=12):
    for r, row in enumerate(rows):
        cx = x
        for c, cell in enumerate(row):
            f = hfont if r == 0 else font
            d.text((cx + pad, y + r * rowh + (rowh - f.size) // 2 - 2), cell, fill="black", font=f)
            cx += colw[c]
    if ruled:
        tw = sum(colw)
        for r in range(len(rows) + 1):
            d.line([(x, y + r * rowh), (x + tw, y + r * rowh)], fill="black", width=2)
        cx = x
        for c in range(len(colw) + 1):
            d.line([(cx, y), (cx, y + len(rows) * rowh)], fill="black", width=2)
            if c < len(colw):
                cx += colw[c]
    return y + len(rows) * rowh

# 1: invoice, ruled table
img = Image.new("RGB", (1700, 2200), "white"); d = ImageDraw.Draw(img)
d.text((120, 100), "INVOICE", fill="black", font=HB(72))
d.text((120, 210), "Northwind Supplies Ltd.", fill="black", font=HB(34))
for i, l in enumerate(["42 Harbor Road", "Portland, OR 97201", "billing@example.com"]):
    d.text((120, 260 + i * 42), l, fill="black", font=H(30))
for i, (k, v) in enumerate([("Invoice no.", "2026-0412"), ("Date", "November 27, 2026"), ("Due date", "December 27, 2026")]):
    d.text((1000, 220 + i * 42), k, fill="black", font=HB(30)); d.text((1240, 220 + i * 42), v, fill="black", font=H(30))
rows = [["Description", "Quantity", "Unit price", "Amount"],
        ["Steel brackets, type A", "120", "$2.40", "$288.00"],
        ["Hex bolts M8 x 40", "500", "$0.12", "$60.00"],
        ["Mounting plates", "40", "$7.85", "$314.00"],
        ["Rubber washers", "1000", "$0.03", "$30.00"],
        ["Delivery and handling", "1", "$45.00", "$45.00"]]
y = grid_table(d, 120, 520, [640, 220, 280, 320], 70, rows, H(30), HB(30))
for i, (k, v) in enumerate([("Subtotal", "$737.00"), ("Tax (8%)", "$58.96"), ("Total", "$795.96")]):
    d.text((1020, y + 40 + i * 50), k, fill="black", font=HB(32) if k == "Total" else H(32))
    d.text((1300, y + 40 + i * 50), v, fill="black", font=HB(32) if k == "Total" else H(32))
y = para(d, 120, y + 260, 80, "Payment terms: net 30 days. Please include the invoice number with your payment. "
         "Late payments are subject to a fee of 1.5% per month.", T(30), 40)
img.save(out + "/invoice.png")

# 2: report, two columns, headings, bullet list, unruled table
img = Image.new("RGB", (1700, 2200), "white"); d = ImageDraw.Draw(img)
d.text((120, 100), "Annual Operations Review", fill="black", font=HB(64))
d.text((120, 190), "Prepared by the planning office, November 2026", fill="black", font=T(32))
for col, x0 in enumerate((120, 890)):
    y = 290
    d.text((x0, y), ["1. Overview", "2. Outlook"][col], fill="black", font=HB(40)); y += 70
    y = para(d, x0, y, 42, LOREM, T(28), 36)
    y += 30
    d.text((x0, y), ["1.1 Highlights", "2.1 Priorities"][col], fill="black", font=HB(34)); y += 56
    items = [["Revenue up 14% on the year", "Two new offices opened", "Support response time halved", "Staff turnover at a record low"],
             ["Ship the reporting tools", "Expand training for partners", "Reduce delivery times further", "Review pricing in March"]][col]
    for it in items:
        d.text((x0 + 10, y), "•", fill="black", font=T(28)); d.text((x0 + 40, y), it, fill="black", font=T(28)); y += 40
    y += 30
    y = para(d, x0, y, 42, LOREM[:400], T(28), 36)
y = 1500
d.text((120, y), "Table 1. Results by region", fill="black", font=HB(32)); y += 60
rows = [["Region", "Revenue", "Costs", "Margin"], ["North", "4,210", "3,100", "26.4%"], ["South", "3,980", "2,950", "25.9%"],
        ["East", "5,020", "3,640", "27.5%"], ["West", "2,870", "2,300", "19.9%"]]
grid_table(d, 120, y, [460, 360, 360, 300], 52, rows, T(30), HB(30), ruled=False)
d.text((120, 2080), "Page 3", fill="black", font=T(24))
img.save(out + "/report.png")

# 3: numbered list, key-value form, table with merged header
img = Image.new("RGB", (1700, 2200), "white"); d = ImageDraw.Draw(img)
d.text((120, 100), "Application for Membership", fill="black", font=HB(56))
y = 220
for k, v in [("Name:", "Maria Lopez"), ("Member ID:", "A-77120"), ("Address:", "18 Elm Street, Springfield"), ("Phone:", "(555) 014-2290"), ("Start date:", "01/02/2027")]:
    d.text((120, y), k, fill="black", font=HB(32)); d.text((420, y), v, fill="black", font=H(32)); y += 54
y += 40
d.text((120, y), "Instructions", fill="black", font=HB(40)); y += 70
for i, s in enumerate(["Fill in every field of the form in block capitals.",
                       "Attach a copy of a photo identity document.",
                       "Sign and date the declaration at the bottom of the page.",
                       "Return the form to the front desk or send it by post."]):
    d.text((140, y), f"{i + 1}.", fill="black", font=T(30)); d.text((190, y), s, fill="black", font=T(30)); y += 46
y += 50
d.text((120, y), "Membership fees", fill="black", font=HB(40)); y += 80
x0 = 120; colw = [420, 280, 280, 280, 280]
d.line([(x0, y), (x0 + sum(colw), y)], fill="black", width=2)
d.text((x0 + 12, y + 30), "Plan", fill="black", font=HB(30))
d.text((x0 + 420 + 150, y + 8), "Monthly", fill="black", font=HB(30))
d.text((x0 + 980 + 150, y + 8), "Yearly", fill="black", font=HB(30))
d.line([(x0 + 420, y + 50), (x0 + sum(colw), y + 50)], fill="black", width=1)
for i, t in enumerate(["Adult", "Student", "Adult", "Student"]):
    d.text((x0 + 420 + i * 280 + 12, y + 58), t, fill="black", font=HB(28))
y += 100
d.line([(x0, y), (x0 + sum(colw), y)], fill="black", width=2)
for r, row in enumerate([["Basic", "$20", "$12", "$200", "$120"], ["Standard", "$35", "$20", "$350", "$200"], ["Premium", "$60", "$35", "$600", "$350"]]):
    cx = x0
    for c, t in enumerate(row):
        d.text((cx + 12, y + 14 + r * 56), t, fill="black", font=H(30)); cx += colw[c]
y += 3 * 56 + 10
d.line([(x0, y), (x0 + sum(colw), y)], fill="black", width=2)
y += 80
y = para(d, 120, y, 85, "Declaration: I confirm that the information given on this form is correct and complete, "
         "and I agree to follow the rules of the club.", T(30), 40)
d.text((120, y + 80), "Signature: ____________________        Date: ____________", fill="black", font=T(30))
img.save(out + "/form.png")

# 4: financial statement, dense numeric table, footnotes
img = Image.new("RGB", (1700, 2200), "white"); d = ImageDraw.Draw(img)
d.text((120, 100), "Consolidated Statement of Income", fill="black", font=HB(50))
d.text((120, 170), "(in thousands of dollars, except per share data)", fill="black", font=T(28))
rows = [["", "2026", "2025", "2024"], ["Net sales", "812,400", "754,210", "701,880"], ["Cost of sales", "498,120", "470,300", "441,050"],
        ["Gross profit", "314,280", "283,910", "260,830"], ["Research and development", "61,200", "58,400", "52,770"],
        ["Selling, general and administrative", "120,450", "112,030", "104,600"], ["Operating income", "132,630", "113,480", "103,460"],
        ["Interest expense", "(8,210)", "(9,050)", "(9,870)"], ["Income before taxes", "124,420", "104,430", "93,590"],
        ["Income taxes", "26,130", "21,930", "19,650"], ["Net income", "98,290", "82,500", "73,940"],
        ["Earnings per share", "$2.41", "$2.03", "$1.82"]]
y = 260
for r, row in enumerate(rows):
    f = HB(30) if r == 0 or row[0] in ("Gross profit", "Operating income", "Net income") else H(30)
    d.text((120, y), row[0], fill="black", font=f)
    for c in range(1, 4):
        w = d.textlength(row[c], font=f)
        d.text((1000 + c * 200 - w, y), row[c], fill="black", font=f)
    if r == 0:
        d.line([(900, y + 44), (1600, y + 44)], fill="black", width=2)
    y += 56
y += 60
for i, s in enumerate(["(1) Figures for 2024 were restated to reflect the sale of the coatings business.",
                       "(2) Earnings per share are computed on the weighted average number of shares outstanding.",
                       "See the accompanying notes, which are an integral part of these statements."]):
    d.text((120, y + i * 38), s, fill="black", font=T(26))
img.save(out + "/statement.png")

import sys, textwrap
from PIL import Image, ImageDraw, ImageFont

out = sys.argv[1]
HEL = "/System/Library/Fonts/Helvetica.ttc"
TIM = "/System/Library/Fonts/Times.ttc"
F = lambda p, s, i=0: ImageFont.truetype(p, s, index=i)
HB = lambda s: F(HEL, s, 1)
H = lambda s: F(HEL, s)
T = lambda s: F(TIM, s)

# a: the report page as a 300 dpi JPEG (twice the pixels)
im = Image.open(out + "/report.png").convert("RGB")
im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
im.save(out + "/report300.jpg", quality=92, dpi=(300, 300))

# b: ruled table with merged cells and wrapped text
img = Image.new("RGB", (1700, 2200), "white"); d = ImageDraw.Draw(img)
d.text((120, 100), "Project Schedule 2027", fill="black", font=HB(54))
d.text((120, 180), "Milestones by quarter, with owners and status.", fill="black", font=T(30))
x0, y0 = 120, 280
cols = [120, 520, 820, 1120, 1580]
rows = [y0, y0 + 70, y0 + 140, y0 + 260, y0 + 380, y0 + 450, y0 + 570, y0 + 640]
for y in rows:
    d.line([(cols[0], y), (cols[-1], y)], fill="black", width=2)
for i, x in enumerate(cols):
    d.line([(x, rows[0]), (x, rows[-1])], fill="black", width=2)
def put(c, r, t, f, dy=18):
    for k, line in enumerate(t.split("\n")):
        d.text((cols[c] + 12, rows[r] + dy + k * 38), line, fill="black", font=f)
put(0, 0, "Phase", HB(30)); put(1, 0, "Milestone", HB(30)); put(2, 0, "Owner", HB(30)); put(3, 0, "Status", HB(30))
put(0, 1, "Design", H(28)); put(1, 1, "Requirements signed", H(28)); put(2, 1, "A. Chen", H(28)); put(3, 1, "Done", H(28))
put(1, 2, "Architecture review\nwith external panel", H(28)); put(2, 2, "B. Okafor", H(28)); put(3, 2, "Done", H(28))
put(0, 3, "Build", H(28)); put(1, 3, "First prototype\nof the controller", H(28)); put(2, 3, "C. Duarte", H(28)); put(3, 3, "In progress", H(28))
put(1, 4, "Field tests", H(28)); put(2, 4, "C. Duarte", H(28)); put(3, 4, "Planned", H(28))
put(0, 5, "Launch", H(28)); put(1, 5, "Pilot customers\nin two regions", H(28)); put(2, 5, "D. Novak", H(28)); put(3, 5, "Planned", H(28))
put(1, 6, "General release", H(28)); put(2, 6, "D. Novak", H(28)); put(3, 6, "Planned", H(28))
# merge the phase cells: white-out the horizontal lines inside column 0
for y in (rows[2], rows[4], rows[6]):
    d.line([(cols[0] + 2, y), (cols[1] - 2, y)], fill="white", width=4)
y = rows[-1] + 80
for line in textwrap.wrap("Dates are indicative and depend on the results of the field tests. The steering "
                          "committee reviews the schedule every month and publishes changes on the project site.", 85):
    d.text((120, y), line, fill="black", font=T(30)); y += 40
img.save(out + "/schedule.png")

# c: three columns with running header and footer
img = Image.new("RGB", (1700, 2200), "white"); d = ImageDraw.Draw(img)
d.text((120, 60), "THE VALLEY GAZETTE", fill="black", font=HB(26)); d.text((1250, 60), "Spring 2027 Edition", fill="black", font=H(26))
d.line([(120, 100), (1580, 100)], fill="black", width=2)
d.text((120, 150), "Town Approves New Library", fill="black", font=HB(60))
text = ("The council voted on Tuesday to build a new public library on the site of the old market hall. "
        "Construction will begin in the autumn and should take about eighteen months. The building will "
        "include a children's wing, a study hall with long opening hours, and a small cafe run by local "
        "volunteers. Residents who spoke at the meeting welcomed the plan, though several asked the council "
        "to keep the historic facade of the hall. The architect said the design keeps the front wall and "
        "the clock tower, which will be restored. Funding comes from a regional grant and from the sale "
        "of two smaller buildings the town no longer uses. The old branch library will stay open until the "
        "new one is finished.")
for c, x in enumerate((120, 620, 1120)):
    y = 280
    if c == 1:
        d.text((x, y), "Opening hours", fill="black", font=HB(34)); y += 56
    for line in textwrap.wrap(text if c != 2 else text[:520], 30):
        d.text((x, y), line, fill="black", font=T(28)); y += 36
d.line([(120, 2080), (1580, 2080)], fill="black", width=2)
d.text((120, 2100), "valleygazette.example", fill="black", font=H(24)); d.text((1520, 2100), "4", fill="black", font=H(24))
img.save(out + "/gazette.png")

# d: the invoice scanned sideways (turned a quarter turn)
Image.open(out + "/invoice.png").rotate(90, expand=True).save(out + "/invoice_sideways.png")
