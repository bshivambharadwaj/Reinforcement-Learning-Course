"""Render editable equations.json to self-contained SVGs (requires latex + dvisvgm).

The README needs no runtime math engine. Fonts are outlined in the SVG files.
Run: python3 assets/equations/generate_equations.py
"""
from pathlib import Path
import json, subprocess, tempfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
NS='http://www.w3.org/2000/svg'
ET.register_namespace('',NS)
ET.register_namespace('xlink','http://www.w3.org/1999/xlink')

def generate():
    equations=json.loads((ROOT/'equations.json').read_text())
    with tempfile.TemporaryDirectory(prefix='rl-equations-') as work:
        tmp=Path(work)
        tex=r'''\documentclass{article}
\usepackage{amsmath,amssymb}
\usepackage[active,tightpage]{preview}
\PreviewEnvironment{equation*}
\begin{document}
'''
        for eq in equations: tex+='\\begin{equation*}\n'+eq['latex']+'\n\\end{equation*}\n'
        tex+='\\end{document}\n'
        (tmp/'equations.tex').write_text(tex)
        run=subprocess.run(['latex','-interaction=nonstopmode','-halt-on-error','equations.tex'],cwd=tmp,capture_output=True,text=True)
        if run.returncode:raise RuntimeError(run.stdout[-5000:])
        run=subprocess.run(['dvisvgm','--no-fonts','--exact-bbox','--page=1-','--output=raw-%p.svg','equations.dvi'],cwd=tmp,capture_output=True,text=True)
        if run.returncode:raise RuntimeError(run.stderr)
        pages=sorted(tmp.glob('raw-*.svg'),key=lambda p:int(p.stem.split('-')[-1]))
        if len(pages)!=len(equations):raise RuntimeError(f'Expected {len(equations)} equation pages, got {len(pages)}')
        for eq,page in zip(equations,pages):
            raw=ET.parse(page).getroot()
            x,y,w,h=map(float,raw.attrib['viewBox'].split())
            scale=min(1.9,704/w)
            if scale<1.65: print(f"Small type: {eq['id']}, {eq['section']}, scale={scale:.2f}")
            height=max(62,round(h*scale+32))
            svg=ET.Element(f'{{{NS}}}svg',{'width':'760','height':str(height),'viewBox':f'0 0 760 {height}','role':'img','aria-labelledby':'title desc'})
            ET.SubElement(svg,f'{{{NS}}}title',{'id':'title'}).text=eq['section']
            ET.SubElement(svg,f'{{{NS}}}desc',{'id':'desc'}).text=eq['latex']
            ET.SubElement(svg,f'{{{NS}}}rect',{'width':'760','height':str(height),'rx':'10','fill':'#F5F1E8'})
            ET.SubElement(svg,f'{{{NS}}}rect',{'x':'0','y':'12','width':'3','height':str(height-24),'rx':'1.5','fill':'#AF824A'})
            group=ET.SubElement(svg,f'{{{NS}}}g',{'fill':'#292332','transform':f'translate({(760-w*scale)/2-x*scale:.4f},{(height-h*scale)/2-y*scale:.4f}) scale({scale:.5f})'})
            for child in raw: group.append(child)
            ET.ElementTree(svg).write(ROOT/(eq['id']+'.svg'),encoding='utf-8',xml_declaration=True)
    print(f'Rendered {len(equations)} equations with outlined fonts.')

if __name__=='__main__':generate()
