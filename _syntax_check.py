import json, ast, sys
nb = json.load(open("25C15052.ipynb", encoding="utf-8"))
errs = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    try:
        ast.parse("".join(cell["source"]))
    except SyntaxError as e:
        errs += 1
        print(f"CELL [{i}] line {e.lineno}: {e.msg}")
em = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown" and "—" in "".join(c["source"]))
print(f"Cells: {len(nb['cells'])} | Syntax errors: {errs} | em dash: {em}")
sys.exit(1 if errs else 0)
