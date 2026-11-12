import html
import re
import shutil
import subprocess
from pathlib import Path

DOC_DIR = Path(__file__).resolve().parent
CHROME_EXE = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

LATEX_SYMBOLS = {
    r"\alpha": "&alpha;",
    r"\sigma": "&sigma;",
    r"\eta": "&eta;",
    r"\pi": "&pi;",
    r"\Delta": "&Delta;",
    r"\rho": "&rho;",
    r"\mu": "&mu;",
    r"\pm": "&plusmn;",
    r"\le": "&le;",
    r"\ge": "&ge;",
    r"\neq": "&ne;",
    r"\approx": "&approx;",
    r"\sim": "~",
    r"\cdot": "&middot;",
    r"\times": "&times;",
    r"\in": "&isin;",
    r"\cup": "&cup;",
    r"\setminus": "\\",
    r"\mapsto": "&rarr;",
    r"\mathbb{C}": "<b>C</b>",
    r"\mathbb{Z}": "<b>Z</b>",
    r"\mathbb{I}": "<b>I</b>",
    r"\mathbb{E}": "<b>E</b>",
    r"\mathbb{P}": "<b>P</b>",
    r"\mathcal{N}": "<i>N</i>",
    r"\mathcal{R}": "<i>R</i>",
    r"\operatorname{Re}": "Re",
    r"\operatorname{Im}": "Im",
    r"\operatorname{clip}": "clip",
    r"\operatorname{round}": "round",
    r"\operatorname{median}": "median",
    r"\mathrm{PRF}": "PRF",
    r"\mathrm{dB}": "dB",
    r"\mathrm{PG}": "PG",
    r"\mathrm{eff}": "eff",
    r"\dots": "&hellip;",
    r"\qquad": "&emsp;&emsp;",
    r"\quad": "&emsp;",
    r"\;": "&nbsp;",
    r"\,": "",
    r"\!": "",
    r"\bmod": " mod ",
    r"\%": "%",
}


def _strip_braces(s: str) -> str:
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and i < len(s) - 1:
                    ok = False
                    break
        if ok:
            return s[1:-1].strip()
    return s


def _extract_brace_arg(s: str, pos: int) -> tuple[str, int]:
    """Extract {...} starting at pos (pointing to '{'). Returns (inner, next_pos)."""
    depth = 0
    start = pos + 1
    for i in range(pos, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i], i + 1
    return s[start:], len(s)


def _tex_to_html(tex: str) -> str:
    s = tex.strip()
    # Delimiters and accents FIRST (before \le matches inside \left!)
    for d_cmd, d_rep in (
        (r"\left(", "("),
        (r"\right)", ")"),
        (r"\left[", "["),
        (r"\right]", "]"),
        (r"\left\{", "{"),
        (r"\right\}", "}"),
        (r"\left|", "|"),
        (r"\right|", "|"),
        (r"\lvert", "|"),
        (r"\rvert", "|"),
    ):
        s = s.replace(d_cmd, d_rep)
    s = re.sub(r"\\hat\{\\mu\}", "&mu;&#770;", s)
    s = re.sub(r"\\hat\{([A-Za-z])\}", r"\1&#770;", s)
    # Clean spacing commands first
    s = s.replace(r"\stackrel{\text{i.i.d.}}{\sim}", " ~<sub>i.i.d.</sub> ")
    s = s.replace(r"\_", "&#95;").replace(r"\mathcal{R}", "<i>R</i>")
    s = re.sub(r"\\text\{([^}]*)\}", r'<span class="mtxt">\1</span>', s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r'<span class="mrm">\1</span>', s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"<b>\1</b>", s)

    # Convert \begin{cases} ... \end{cases}
    if r"\begin{cases}" in s:
        m = re.search(r"\\begin\{cases\}(.*?)\\end\{cases\}", s, flags=re.S)
        if m:
            rows = []
            for row in m.group(1).strip().split(r"\\"):
                parts = [ _tex_to_html(p.strip()) for p in row.split("&") ]
                if len(parts) == 2:
                    rows.append(f"<tr><td>{parts[0]}</td><td class='case-cond'>{parts[1]}</td></tr>")
                else:
                    rows.append(f"<tr><td colspan='2'>{parts[0]}</td></tr>")
            tbl = f"<span class='cases-wrap'><span class='brace'>&#123;</span><table class='cases'>{''.join(rows)}</table></span>"
            s = s[:m.start()] + tbl + s[m.end():]

    # Convert \dfrac / \frac iteratively from innermost
    for frac_cmd in (r"\dfrac", r"\frac"):
        while frac_cmd in s:
            idx = s.find(frac_cmd)
            p = idx + len(frac_cmd)
            while p < len(s) and s[p] == " ":
                p += 1
            if p < len(s) and s[p] == "{":
                num, p2 = _extract_brace_arg(s, p)
                while p2 < len(s) and s[p2] == " ":
                    p2 += 1
                if p2 < len(s) and s[p2] == "{":
                    den, p3 = _extract_brace_arg(s, p2)
                    frac_html = f"<span class='frac'><span class='num'>{_tex_to_html(num)}</span><span class='den'>{_tex_to_html(den)}</span></span>"
                    s = s[:idx] + frac_html + s[p3:]
                    continue
            break

    # Convert \sum_{...}^{...} and \operatorname{median}_{...}
    def repl_sum(m):
        sub = _tex_to_html(m.group(1))
        sup = _tex_to_html(m.group(2))
        return f"<span class='opstack'><span class='opsup'>{sup}</span><span class='opsym'>&Sigma;</span><span class='opsub'>{sub}</span></span>"
    s = re.sub(r"\\sum_\{([^}]*)\}\^\{([^}]*)\}", repl_sum, s)
    s = re.sub(r"\\sum_\{([^}]*)\}", lambda m: f"<span class='opstack'><span class='opsym'>&Sigma;</span><span class='opsub'>{_tex_to_html(m.group(1))}</span></span>", s)
    s = re.sub(r"\\max_\{([^}]*)\}", lambda m: f"<span class='opname'>max</span><sub>{_tex_to_html(m.group(1))}</sub>", s)
    s = re.sub(r"\\operatorname\{median\}_\{([^}]*)\}", lambda m: f"<span class='opname'>median</span><sub>{_tex_to_html(m.group(1))}</sub>", s)

    # Named standard functions
    for fn in ("exp", "log", "max", "min", "ln", "cos"):
        s = s.replace("\\" + fn, f"<span class='opname'>{fn}</span>")

    s = s.replace(r"\sqrt{-1}", "&radic;<span style='text-decoration:overline'>&minus;1</span>")

    # Apply symbol table longest-first
    for k in sorted(LATEX_SYMBOLS, key=len, reverse=True):
        s = s.replace(k, LATEX_SYMBOLS[k])

    # Convert _{...} and _x, ^{...} and ^x
    while True:
        m = re.search(r"_\{([^{}]*)\}", s)
        if not m:
            break
        s = s[:m.start()] + f"<sub>{m.group(1)}</sub>" + s[m.end():]
    while True:
        m = re.search(r"\^\{([^{}]*)\}", s)
        if not m:
            break
        s = s[:m.start()] + f"<sup>{m.group(1)}</sup>" + s[m.end():]

    s = re.sub(r"_([A-Za-z0-9]+)", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([A-Za-z0-9\-\*]+)", r"<sup>\1</sup>", s)
    s = s.replace(r"\{", "{").replace(r"\}", "}").replace(r"\|", "|")
    s = re.sub(r"\{([^{}<>]+)\}", r"\1", s)
    s = s.replace(" - ", " &minus; ")
    return f"<span class='math'>{s}</span>"


def _inline_md_to_html(text: str, base_dir: Path) -> str:
    # Protect inline math $...$ first
    math_tokens: list[str] = []

    def stash_math(m):
        math_tokens.append(_tex_to_html(m.group(1)))
        return f"@@MATH{len(math_tokens)-1}@@"

    text = re.sub(r"(?<!\\)\$([^$\n]+?)\$", stash_math, text)
    text = text.replace(r"\$", "$")

    # Protect inline code `...`
    code_tokens: list[str] = []

    def stash_code(m):
        code_tokens.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"@@CODE{len(code_tokens)-1}@@"

    text = re.sub(r"`([^`]+?)`", stash_code, text)

    # Links [label](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: f"<a href='{html.escape(m.group(2))}'>{html.escape(m.group(1))}</a>", text)

    out = text.replace(r"\*", "*").replace(r"\_", "_")
    out = re.sub(r"\*\*([^*]+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", out)

    for i, c in enumerate(code_tokens):
        out = out.replace(f"@@CODE{i}@@", c)
    for i, mh in enumerate(math_tokens):
        out = out.replace(f"@@MATH{i}@@", mh)
    return out


def md_to_html_doc(md_text: str, base_dir: Path, title: str, one_page: bool = False) -> str:
    lines = md_text.splitlines()
    html_parts: list[str] = []
    i = 0
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            html_parts.append("</ul>")
            in_ul = False
        if in_ol:
            html_parts.append("</ol>")
            in_ol = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            close_lists()
            i += 1
            continue

        if stripped == "---":
            close_lists()
            html_parts.append("<hr/>")
            i += 1
            continue

        # Multi-line or single-line $$ ... $$ display equation
        if stripped.startswith("$$"):
            close_lists()
            if stripped.endswith("$$") and len(stripped) > 4:
                eq = stripped[2:-2].strip()
                i += 1
            else:
                eq_lines = [stripped[2:]]
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    eq_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    eq_lines.append(lines[i].strip()[:-2])
                    i += 1
                eq = "\n".join(eq_lines).strip()
            html_parts.append(f"<div class='eq-block'>{_tex_to_html(eq)}</div>")
            continue

        # Headings
        hm = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if hm:
            close_lists()
            lvl = len(hm.group(1))
            cls = "doc-title" if lvl == 1 else ""
            html_parts.append(f"<h{lvl} class='{cls}'>{_inline_md_to_html(hm.group(2), base_dir)}</h{lvl}>")
            i += 1
            continue

        # Images ![alt](src)
        im = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if im:
            close_lists()
            alt = im.group(1)
            rel = im.group(2)
            img_path = (base_dir / rel).resolve()
            uri = img_path.as_uri()
            caption = ""
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("*Figure"):
                caption = lines[i + 1].strip().strip("*")
                i += 1
            cap_html = f"<figcaption>{_inline_md_to_html(caption or alt, base_dir)}</figcaption>"
            html_parts.append(f"<figure class='fig-card'><img src='{uri}' alt='{html.escape(alt)}'/>{cap_html}</figure>")
            i += 1
            continue

        # Markdown Table
        if "|" in stripped and i + 1 < len(lines) and re.match(r"^\|?[\s:\-|]+\|?$", lines[i + 1].strip()):
            close_lists()
            headers = [c.strip() for c in stripped.strip("|").split("|")]
            aligns_raw = [c.strip() for c in lines[i + 1].strip().strip("|").split("|")]
            aligns = []
            for a in aligns_raw:
                if a.startswith(":") and a.endswith(":"):
                    aligns.append("center")
                elif a.endswith(":"):
                    aligns.append("right")
                else:
                    aligns.append("left")
            i += 2
            rows_html = []
            while i < len(lines) and "|" in lines[i].strip() and lines[i].strip():
                cols = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                tds = []
                for idx_c, col in enumerate(cols):
                    al = aligns[idx_c] if idx_c < len(aligns) else "left"
                    tds.append(f"<td style='text-align:{al}'>{_inline_md_to_html(col, base_dir)}</td>")
                rows_html.append("<tr>" + "".join(tds) + "</tr>")
                i += 1
            ths = "".join(
                f"<th style='text-align:{aligns[idx_h] if idx_h < len(aligns) else 'left'}'>{_inline_md_to_html(h, base_dir)}</th>"
                for idx_h, h in enumerate(headers)
            )
            html_parts.append(f"<table><thead><tr>{ths}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>")
            continue

        # Bullet list
        bm = re.match(r"^(\s*)[-\*]\s+(.*)$", line)
        if bm:
            indent = len(bm.group(1))
            if in_ol:
                html_parts.append("</ol>")
                in_ol = False
            if not in_ul:
                html_parts.append("<ul>")
                in_ul = True
            cls = "sub-li" if indent >= 2 else ""
            html_parts.append(f"<li class='{cls}'>{_inline_md_to_html(bm.group(2), base_dir)}</li>")
            i += 1
            continue

        # Numbered list (preserve exact <li value="N">)
        nm = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if nm:
            if in_ul:
                html_parts.append("</ul>")
                in_ul = False
            if not in_ol:
                html_parts.append("<ol>")
                in_ol = True
            item_num = int(nm.group(1))
            content = nm.group(2)
            while i + 1 < len(lines) and (lines[i + 1].startswith("   ") or lines[i + 1].startswith("  ")) and not lines[i + 1].strip().startswith(("-", "*")) and not re.match(r"^\d+\.", lines[i + 1].strip()):
                if lines[i + 1].strip().startswith("$$"):
                    break
                i += 1
                content += " " + lines[i].strip()
            html_parts.append(f"<li value='{item_num}'>{_inline_md_to_html(content, base_dir)}</li>")
            i += 1
            continue

        # Paragraph
        close_lists()
        p_lines = [stripped]
        while i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if not nxt or nxt.startswith(("#", "-", "*", "$$", "![", "---")) or re.match(r"^\d+\.", nxt) or ("|" in nxt and i + 2 < len(lines) and "---" in lines[i + 2]):
                break
            i += 1
            p_lines.append(nxt)
        p_text = " ".join(p_lines)
        cls = "footnote" if p_text.startswith("*Note") or (p_text.startswith("*") and p_text.endswith("*")) else ""
        html_parts.append(f"<p class='{cls}'>{_inline_md_to_html(p_text, base_dir)}</p>")
        i += 1

    close_lists()

    if one_page:
        page_css = """
        @page { size: A4; margin: 8.5mm 11mm 8.5mm 11mm; }
        body { font-size: 8.1pt; line-height: 1.25; }
        h1.doc-title { font-size: 13.0pt; margin: 0 0 3px 0; padding-bottom: 3px; }
        h2 { font-size: 9.4pt; margin: 5px 0 2px 0; }
        h3 { font-size: 8.6pt; margin: 4px 0 2px 0; }
        table { font-size: 7.1pt; margin: 3px 0 4px 0; }
        th, td { padding: 2px 3.5px; }
        p, li { margin: 1.5px 0; }
        """
    else:
        page_css = """
        @page { size: A4; margin: 16mm 16mm 17mm 16mm; }
        body { font-size: 9.6pt; line-height: 1.46; }
        h1.doc-title { font-size: 17pt; margin: 0 0 8px 0; padding-bottom: 6px; }
        h2 { font-size: 12.2pt; margin: 16px 0 6px 0; page-break-after: avoid; }
        h3 { font-size: 10.6pt; margin: 11px 0 4px 0; page-break-after: avoid; }
        table { font-size: 8.1pt; margin: 7px 0 11px 0; }
        th, td { padding: 4px 6px; }
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>
{page_css}
body {{
  font-family: "Segoe UI", "Calibri", Arial, sans-serif;
  color: #1c2430;
  margin: 0;
  padding: 0;
}}
h1.doc-title {{
  color: #0f2c59;
  border-bottom: 2px solid #0f2c59;
  font-weight: 700;
}}
h2 {{
  color: #0f2c59;
  border-bottom: 1px solid #d0d7e2;
  padding-bottom: 2px;
  font-weight: 700;
}}
h3 {{
  color: #1f4068;
  font-weight: 600;
}}
hr {{
  border: none;
  border-top: 1px solid #d9e0ea;
  margin: 8px 0;
}}
code {{
  font-family: "Consolas", "Courier New", monospace;
  font-size: 0.9em;
  background: #f3f6fa;
  padding: 0.5px 3px;
  border-radius: 3px;
  color: #1b365d;
}}
.math {{
  font-family: "Cambria Math", "Times New Roman", Georgia, serif;
  font-style: normal;
}}
.opname, .mrm, .mtxt {{
  font-style: normal;
  font-family: "Cambria Math", "Times New Roman", serif;
}}
.eq-block {{
  background: #f8fafd;
  border-left: 3px solid #2b5b84;
  padding: 6px 12px;
  margin: 6px 0;
  text-align: center;
  font-size: 1.03em;
  page-break-inside: avoid;
}}
.frac {{
  display: inline-flex;
  flex-direction: column;
  vertical-align: middle;
  text-align: center;
  margin: 0 2px;
  font-size: 0.92em;
}}
.frac .num {{
  border-bottom: 1px solid #1c2430;
  padding: 0 3px 1px 3px;
}}
.frac .den {{
  padding: 1px 3px 0 3px;
}}
.opstack {{
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  vertical-align: middle;
  line-height: 1.0;
  margin: 0 2px;
}}
.opstack .opsym {{ font-size: 1.25em; }}
.opstack .opsub, .opstack .opsup {{ font-size: 0.68em; }}
.cases-wrap {{
  display: inline-flex;
  align-items: center;
  vertical-align: middle;
}}
.cases-wrap .brace {{
  font-size: 2.3em;
  font-weight: 300;
  margin-right: 3px;
}}
table.cases {{
  border: none;
  width: auto;
  margin: 0;
}}
table.cases td {{
  border: none;
  padding: 1px 6px;
  background: transparent;
  text-align: left;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  page-break-inside: avoid;
}}
th {{
  background: #edf2f8;
  color: #0f2c59;
  font-weight: 700;
  border-top: 1.5px solid #2b5b84;
  border-bottom: 1.5px solid #2b5b84;
}}
td {{
  border-bottom: 1px solid #dce3ec;
}}
tr:nth-child(even) td {{
  background: #fafcfe;
}}
ul, ol {{
  margin: 4px 0 6px 0;
  padding-left: 18px;
}}
li.sub-li {{
  margin-left: 14px;
  list-style-type: circle;
}}
figure.fig-card {{
  margin: 8px 0 12px 0;
  text-align: center;
  page-break-inside: avoid;
}}
figure.fig-card img {{
  max-width: 97%;
  max-height: 295px;
  object-fit: contain;
  border: 1px solid #d5dce6;
  border-radius: 3px;
}}
figcaption {{
  margin-top: 4px;
  font-size: 8.2pt;
  color: #465366;
  font-style: italic;
}}
p.footnote {{
  font-size: 8.1pt;
  color: #445063;
  background: #f9fbfd;
  border-left: 2px solid #9aa8bd;
  padding: 4px 8px;
}}
</style>
</head>
<body>
{''.join(html_parts)}
</body>
</html>"""


def render_pair(md_name: str, pdf_name: str, title: str, one_page: bool = False) -> None:
    md_path = DOC_DIR / md_name
    html_path = DOC_DIR / (md_path.stem + "_tmp.html")
    pdf_path = DOC_DIR / pdf_name
    profile_dir = DOC_DIR / "_chrome_tmp"

    html_text = md_to_html_doc(md_path.read_text(encoding="utf-8"), DOC_DIR, title, one_page=one_page)
    html_path.write_text(html_text, encoding="utf-8")

    cmd = [
        str(CHROME_EXE),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--user-data-dir={profile_dir}",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True)
    html_path.unlink(missing_ok=True)
    shutil.rmtree(profile_dir, ignore_errors=True)
    print(f"Generated: {pdf_path} ({pdf_path.stat().st_size:,} bytes)")


def prep_html(md_name: str, title: str, one_page: bool = False) -> Path:
    md_path = DOC_DIR / md_name
    html_path = DOC_DIR / (md_path.stem + "_tmp.html")
    html_text = md_to_html_doc(md_path.read_text(encoding="utf-8"), DOC_DIR, title, one_page=one_page)
    html_path.write_text(html_text, encoding="utf-8")
    print(f"Prepared: {html_path.name}")
    return html_path


if __name__ == "__main__":
    import sys
    if "--prep" in sys.argv:
        prep_html("objective_1_final_report.md", "Objective I Final Validation Report", one_page=False)
        prep_html("objective_1_one_page_summary.md", "Objective I One-Page Executive Summary", one_page=True)
        prep_html("objective_2_unresolved_limitations.md", "Objective II Technical Limitations Handoff", one_page=True)
    elif "--cleanup" in sys.argv:
        for p in DOC_DIR.glob("*_tmp.html"):
            p.unlink(missing_ok=True)
        shutil.rmtree(DOC_DIR / "_chrome_tmp", ignore_errors=True)
        print("Cleaned temporary HTML and Chrome profile files.")
    else:
        render_pair("objective_1_final_report.md", "objective_1_final_report.pdf", "Objective I Final Validation Report", one_page=False)
        render_pair("objective_1_one_page_summary.md", "objective_1_one_page_summary.pdf", "Objective I One-Page Executive Summary", one_page=True)
        render_pair("objective_2_unresolved_limitations.md", "objective_2_unresolved_limitations.pdf", "Objective II Technical Limitations Handoff", one_page=True)
        render_pair("objective_1_final_audit.md", "objective_1_final_audit.pdf", "Objective I Comprehensive Audit Report", one_page=False)
