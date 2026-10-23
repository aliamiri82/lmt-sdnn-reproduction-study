"""
Rebuild Fig. 1 of the LMT-SDNN paper as an editable vector diagram.

One layout description is written out in two formats:

  LMT-SDNN_framework.vsdx    Visio package (a zip of OOXML parts).  Hand-built,
                             because no Visio install is available here to drive.
  LMT-SDNN_framework.drawio  draw.io / diagrams.net native XML.  Guaranteed to
                             open cleanly, and the safety net if a viewer is
                             fussy about the hand-authored .vsdx.

Both carry real shapes and connectors -- nothing is a flattened image, so every
box, label and arrow stays editable.

Coordinates below are a neutral pixel space with the origin at the TOP-LEFT.
The Visio writer flips the Y axis and converts to inches; Visio's origin is
bottom-left and its unit is the inch.
"""
from __future__ import annotations

import pathlib
import zipfile
from xml.sax.saxutils import escape

OUT = pathlib.Path(__file__).parent.parent / "diagram"
OUT.mkdir(exist_ok=True)

CANVAS_W, CANVAS_H = 1580, 760
DPI = 96.0                      # pixels per inch for the Visio conversion

# --------------------------------------------------------------------------- #
#  palette (matches the figure in the paper)
# --------------------------------------------------------------------------- #
BLUE_D = "#2E5B8B"      # cylinders / KD_loss
BLUE_L = "#BDD7EE"      # conv blocks
BLUE_M = "#9DC3E6"      # soft prediction
YELLOW = "#FFE699"      # flow segmentation / hard prediction
ORANGE = "#F4B183"      # soft label / hard label / results
PINK = "#E6CCE6"        # hidden-state pills
GREEN = "#C5E0B4"       # loss terms
GREY = "#F2F2F2"        # background panel
WHITE = "#FFFFFF"
LINE = "#404040"

# --------------------------------------------------------------------------- #
#  shapes:  id -> (x, y, w, h, label, kind, fill)
#  kind: rect | round | cyl | cloud | panel | group | trap | text
# --------------------------------------------------------------------------- #
N: dict[str, tuple] = {}
E: list[tuple] = []


def node(nid, x, y, w, h, label="", kind="rect", fill=WHITE, fontsize=11,
         bold=False):
    N[nid] = (x, y, w, h, label, kind, fill, fontsize, bold)


def edge(a, b, label="", dashed=False, points=None):
    E.append((a, b, label, dashed, points or []))


# ---- background panel ------------------------------------------------------
node("panel", 210, 28, 980, 712, "", "panel", GREY)

# ---- left input column -----------------------------------------------------
node("train_l", 42, 52, 120, 62, "Train Data", "cyl", BLUE_L, 12, True)
node("flow_l", 30, 330, 152, 34, "Flow Segmentation", "rect", YELLOW, 11, True)
node("feat_l", 30, 364, 152, 34, "Feature Scaling", "rect", YELLOW, 11, True)
node("net_l", 32, 500, 150, 86, "LMT-SDNN\nNetwork", "cloud", WHITE, 12, True)

# ---- the three networks ----------------------------------------------------
node("t1box", 232, 46, 402, 196, "", "group", WHITE)
node("t1title", 232, 50, 402, 22, "Teacher1(Residual Inception)", "text", "", 13)
node("t2box", 232, 262, 402, 186, "", "group", WHITE)
node("t2title", 232, 266, 402, 22, "Teacher2(1DCNN-BiLstm)", "text", "", 13)
node("stbox", 232, 470, 402, 214, "", "group", WHITE)
node("sttitle", 232, 474, 402, 22, "Student(DSC-BiGRU)", "text", "", 13)

# teacher 1: an input stack feeding three residual branches, then a merge
node("t1_in", 252, 128, 34, 40, "", "rect", BLUE_L)
for r, yy in enumerate((96, 148, 200)):
    for c, xx in enumerate((330, 410, 490)):
        node(f"t1_{r}{c}", xx, yy, 30, 30, "", "rect", BLUE_L)
node("t1_sum", 552, 138, 22, 22, "+", "round", WHITE, 13)
node("t1_out", 588, 128, 34, 40, "", "rect", BLUE_D)

# teacher 2: standard conv then a BiLSTM grid
node("t2_in", 252, 344, 34, 40, "", "rect", BLUE_L)
node("t2_k", 300, 320, 40, 40, "", "rect", WHITE)
node("t2_c", 352, 344, 34, 40, "", "rect", BLUE_D)
for r, yy in enumerate((312, 350, 388)):
    for c, xx in enumerate((410, 476)):
        node(f"t2_l{r}{c}", xx, yy, 54, 22, "LSTM", "rect", YELLOW, 9)
node("t2_sum", 548, 348, 22, 22, "+", "round", WHITE, 13)
node("t2_out", 580, 316, 22, 92, "", "rect", WHITE)
node("t2_lab1", 240, 412, 160, 18, "1D-Standard-Convolutin", "text", "", 10)
node("t2_lab2", 420, 412, 110, 18, "BiLSTM", "text", "", 10)

# student: depthwise-separable conv then a BiGRU grid
node("st_in", 252, 556, 34, 40, "", "rect", BLUE_L)
node("st_k", 300, 540, 40, 40, "", "rect", WHITE)
node("st_c", 352, 556, 34, 40, "", "rect", BLUE_D)
for r, yy in enumerate((524, 562, 600)):
    for c, xx in enumerate((410, 476)):
        node(f"st_g{r}{c}", xx, yy, 54, 22, "GRU", "rect", YELLOW, 9)
node("st_sum", 548, 560, 22, 22, "+", "round", WHITE, 13)
node("st_out", 580, 528, 22, 92, "", "rect", WHITE)
node("st_lab1", 240, 630, 160, 18, "1D-Separabel-Convolutin", "text", "", 10)
node("st_lab2", 420, 630, 110, 18, "BiGRU", "text", "", 10)

# ---- distillation plumbing -------------------------------------------------
node("sm_t1", 664, 66, 112, 28, "Softmax(T=t1)", "round", WHITE, 10)
node("soft1", 812, 66, 88, 28, "Soft label", "rect", ORANGE, 10)
node("h_t1", 664, 122, 124, 28, "Teacher1_hidden", "round", PINK, 10)
node("dkl1", 902, 118, 148, 28, "DKL(soft_loss_tea1)", "rect", GREEN, 10)

node("plus", 966, 168, 26, 26, "+", "text", "", 20, True)

node("sm_t2", 664, 232, 112, 28, "Softmax(T=t2)", "round", WHITE, 10)
node("soft2", 796, 232, 84, 28, "Soft label", "rect", ORANGE, 10)
node("dkl2", 902, 232, 148, 28, "DKL(soft_loss_tea2)", "rect", GREEN, 10)
node("h_t2", 664, 296, 124, 28, "Teacher2_hidden", "round", PINK, 10)

node("tsw", 796, 280, 68, 106, "TSW_loss", "rect", GREEN, 9)
node("h_st", 664, 390, 124, 28, "Student_hidden", "round", PINK, 10)

node("sm_s2", 664, 452, 112, 28, "Softmax(T=t2)", "round", WHITE, 10)
node("sp1", 810, 446, 100, 40, "Soft\nprediction", "trap", BLUE_M, 10)
node("sm_s1", 664, 518, 112, 28, "Softmax(T=t1)", "round", WHITE, 10)
node("sp2", 810, 512, 100, 40, "Soft\nprediction", "trap", BLUE_M, 10)
node("sm_s0", 664, 588, 112, 28, "Softmax(T=1)", "round", WHITE, 10)
node("hardp", 812, 588, 108, 28, "Hard prediction", "rect", YELLOW, 10)
node("cs", 938, 588, 84, 28, "CS _loss", "rect", GREEN, 10)
node("hardl", 812, 664, 96, 28, "Hard label", "rect", ORANGE, 10)

node("kd", 1044, 366, 116, 58, "KD_loss", "rect", BLUE_D, 15, True)

# ---- right output column ---------------------------------------------------
node("train_r", 1222, 52, 120, 62, "Train Data", "cyl", BLUE_L, 12, True)
node("flow_r", 1210, 330, 152, 34, "Flow Segmentation", "rect", YELLOW, 11, True)
node("feat_r", 1210, 364, 152, 34, "Feature Scaling", "rect", YELLOW, 11, True)
node("mdl_r", 1212, 500, 150, 86, "LMT-SDNN\nModel", "cloud", WHITE, 12, True)
node("cls_r", 1396, 330, 132, 58, "Classfication\nResults", "rect", ORANGE, 11, True)
node("cm_r", 1396, 52, 132, 62, "Confusion\nMatrix", "rect", BLUE_D, 12, True)

# ---- connectors ------------------------------------------------------------
edge("train_l", "flow_l")
edge("feat_l", "net_l")

edge("t1_out", "sm_t1")
edge("t1_out", "h_t1")
edge("sm_t1", "soft1")
edge("soft1", "dkl1")
edge("t2_out", "sm_t2")
edge("t2_out", "h_t2")
edge("sm_t2", "soft2")
edge("soft2", "dkl2")

edge("h_t1", "tsw")
edge("h_t2", "tsw")
edge("h_st", "tsw")
edge("st_out", "h_st")

edge("st_out", "sm_s2")
edge("st_out", "sm_s1")
edge("st_out", "sm_s0")
edge("sm_s2", "sp1")
edge("sm_s1", "sp2")
edge("sm_s0", "hardp")
edge("hardl", "hardp")
edge("hardp", "cs")

edge("dkl1", "kd")
edge("dkl2", "kd")
edge("tsw", "kd")
edge("sp1", "kd")
edge("sp2", "kd")
edge("cs", "kd")

edge("train_r", "flow_r")
edge("feat_r", "mdl_r")
edge("mdl_r", "cls_r")
edge("cls_r", "cm_r")
# both run along the bottom edge in the paper figure, not across the middle
edge("kd", "stbox", points=[(1102, 716), (433, 716)])
edge("net_l", "mdl_r", points=[(107, 748), (1287, 748)])


# =========================================================================== #
#  draw.io writer
# =========================================================================== #
DRAWIO_STYLE = {
    "rect":  "rounded=0;whiteSpace=wrap;html=1;",
    "round": "rounded=1;arcSize=40;whiteSpace=wrap;html=1;",
    "cyl":   "shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=10;"
             "whiteSpace=wrap;html=1;",
    "cloud": "ellipse;shape=cloud;whiteSpace=wrap;html=1;",
    "trap":  "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;"
             "html=1;fixedSize=1;",
    "panel": "rounded=1;arcSize=3;whiteSpace=wrap;html=1;",
    "group": "rounded=1;arcSize=8;whiteSpace=wrap;html=1;verticalAlign=top;",
    "text":  "text;html=1;align=center;verticalAlign=middle;strokeColor=none;"
             "fillColor=none;",
}


def _pts_xml(pts) -> str:
    if not pts:
        return ""
    inner = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts)
    return f'<Array as="points">{inner}</Array>'


def write_drawio(path: pathlib.Path) -> None:
    out = [
        '<mxfile host="app.diagrams.net" type="device">',
        '  <diagram name="LMT-SDNN framework">',
        f'    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" '
        f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
        f'pageScale="1" pageWidth="{CANVAS_W}" pageHeight="{CANVAS_H}" '
        f'math="0" shadow="0">',
        '      <root>',
        '        <mxCell id="0" />',
        '        <mxCell id="1" parent="0" />',
    ]
    for nid, (x, y, w, h, label, kind, fill, fs, bold) in N.items():
        style = DRAWIO_STYLE.get(kind, DRAWIO_STYLE["rect"])
        if kind != "text":
            style += f"fillColor={fill};strokeColor={LINE};"
        style += f"fontSize={fs};"
        if bold:
            style += "fontStyle=1;"
        out.append(
            f'        <mxCell id="{nid}" value="{escape(label)}" '
            f'style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry"/></mxCell>')
    for i, (a, b, label, dashed, pts) in enumerate(E):
        st = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;"
              f"strokeColor={LINE};endArrow=block;endFill=1;")
        if dashed:
            st += "dashed=1;"
        out.append(
            f'        <mxCell id="e{i}" value="{escape(label)}" style="{st}" '
            f'edge="1" parent="1" source="{a}" target="{b}">'
            f'<mxGeometry relative="1" as="geometry">{_pts_xml(pts)}'
            f'</mxGeometry></mxCell>')
    out += ['      </root>', '    </mxGraphModel>', '  </diagram>', '</mxfile>']
    path.write_text("\n".join(out), encoding="utf-8")


# =========================================================================== #
#  Visio (.vsdx) writer
# =========================================================================== #
VNS = "http://schemas.microsoft.com/office/visio/2012/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def px2in(v: float) -> float:
    return round(v / DPI, 4)


def _geom_rect() -> str:
    """Unit rectangle; Visio scales it by Width/Height."""
    return (
        "<Section N='Geometry' IX='0'>"
        "<Cell N='NoFill' V='0'/><Cell N='NoLine' V='0'/>"
        "<Row T='RelMoveTo' IX='1'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
        "<Row T='RelLineTo' IX='2'><Cell N='X' V='1'/><Cell N='Y' V='0'/></Row>"
        "<Row T='RelLineTo' IX='3'><Cell N='X' V='1'/><Cell N='Y' V='1'/></Row>"
        "<Row T='RelLineTo' IX='4'><Cell N='X' V='0'/><Cell N='Y' V='1'/></Row>"
        "<Row T='RelLineTo' IX='5'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
        "</Section>")


def _geom_ellipse() -> str:
    return (
        "<Section N='Geometry' IX='0'>"
        "<Cell N='NoFill' V='0'/><Cell N='NoLine' V='0'/>"
        "<Row T='Ellipse' IX='1'>"
        "<Cell N='X' V='0.5' F='Width*0.5'/><Cell N='Y' V='0.5' F='Height*0.5'/>"
        "<Cell N='A' V='1'   F='Width*1'/>  <Cell N='B' V='0.5' F='Height*0.5'/>"
        "<Cell N='C' V='0.5' F='Width*0.5'/><Cell N='D' V='1'   F='Height*1'/>"
        "</Row></Section>")


def _shape_xml(sid: int, x, y, w, h, label, kind, fill, fs, bold) -> str:
    """One Visio Shape.  Visio's Y axis points up, so flip it."""
    pin_x = px2in(x + w / 2)
    pin_y = px2in(CANVAS_H - (y + h / 2))
    wid, hgt = px2in(w), px2in(h)

    cells = [
        f"<Cell N='PinX' V='{pin_x}'/>",
        f"<Cell N='PinY' V='{pin_y}'/>",
        f"<Cell N='Width' V='{wid}'/>",
        f"<Cell N='Height' V='{hgt}'/>",
        f"<Cell N='LocPinX' V='{wid/2}' F='Width*0.5'/>",
        f"<Cell N='LocPinY' V='{hgt/2}' F='Height*0.5'/>",
        "<Cell N='Angle' V='0'/>",
    ]
    if kind == "text":
        cells += ["<Cell N='FillPattern' V='0'/>", "<Cell N='LinePattern' V='0'/>"]
    else:
        cells += [
            f"<Cell N='FillForegnd' V='{fill}'/>",
            "<Cell N='FillPattern' V='1'/>",
            f"<Cell N='LineColor' V='{LINE}'/>",
            "<Cell N='LineWeight' V='0.01'/>",
        ]
        if kind in ("round", "panel", "group"):
            cells.append("<Cell N='Rounding' V='0.08'/>")
    cells += [
        f"<Cell N='Char.Size' V='{round(fs/72.0, 4)}'/>",
        f"<Cell N='Char.Style' V='{1 if bold else 0}'/>",
        "<Cell N='Para.HorzAlign' V='1'/>",
        "<Cell N='VerticalAlign' V='1'/>",
    ]
    geom = _geom_ellipse() if kind in ("cloud",) else _geom_rect()
    text = f"<Text>{escape(label)}</Text>" if label else "<Text/>"
    return (f"<Shape ID='{sid}' NameU='S{sid}' Type='Shape' LineStyle='0' "
            f"FillStyle='0' TextStyle='0'>"
            + "".join(cells) + geom + text + "</Shape>")


def _connector_xml(sid: int, a: str, b: str, ids: dict) -> str:
    """A straight one-dimensional connector between two shape centres."""
    ax, ay, aw, ah = N[a][:4]
    bx, by, bw, bh = N[b][:4]
    x0, y0 = px2in(ax + aw / 2), px2in(CANVAS_H - (ay + ah / 2))
    x1, y1 = px2in(bx + bw / 2), px2in(CANVAS_H - (by + bh / 2))
    return (
        f"<Shape ID='{sid}' NameU='C{sid}' Type='Shape' LineStyle='0' "
        f"FillStyle='0' TextStyle='0'>"
        f"<Cell N='PinX' V='{(x0+x1)/2}'/><Cell N='PinY' V='{(y0+y1)/2}'/>"
        f"<Cell N='Width' V='{abs(x1-x0)}'/><Cell N='Height' V='{abs(y1-y0)}'/>"
        f"<Cell N='LocPinX' V='{abs(x1-x0)/2}' F='Width*0.5'/>"
        f"<Cell N='LocPinY' V='{abs(y1-y0)/2}' F='Height*0.5'/>"
        f"<Cell N='BeginX' V='{x0}'/><Cell N='BeginY' V='{y0}'/>"
        f"<Cell N='EndX' V='{x1}'/><Cell N='EndY' V='{y1}'/>"
        f"<Cell N='LineColor' V='{LINE}'/><Cell N='LineWeight' V='0.01'/>"
        f"<Cell N='EndArrow' V='4'/>"
        f"<Cell N='ObjType' V='2'/>"
        "<Section N='Geometry' IX='0'>"
        "<Cell N='NoFill' V='1'/><Cell N='NoLine' V='0'/>"
        "<Row T='MoveTo' IX='1'>"
        f"<Cell N='X' V='0' F='Width*0'/><Cell N='Y' V='0' F='Height*0'/></Row>"
        "<Row T='LineTo' IX='2'>"
        f"<Cell N='X' V='{abs(x1-x0)}' F='Width*1'/>"
        f"<Cell N='Y' V='{abs(y1-y0)}' F='Height*1'/></Row>"
        "</Section><Text/></Shape>")


def write_vsdx(path: pathlib.Path) -> None:
    ids = {}
    shapes = []
    sid = 1
    for nid, spec in N.items():
        ids[nid] = sid
        shapes.append(_shape_xml(sid, *spec))
        sid += 1
    for a, b, _lab, _d, _p in E:
        shapes.append(_connector_xml(sid, a, b, ids))
        sid += 1

    page1 = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        f"<PageContents xmlns='{VNS}' xmlns:r='{R_NS}' xml:space='preserve'>"
        f"<Shapes>{''.join(shapes)}</Shapes></PageContents>")

    pages = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        f"<Pages xmlns='{VNS}' xmlns:r='{R_NS}' xml:space='preserve'>"
        "<Page ID='0' NameU='Framework' Name='Framework' ViewScale='-1' "
        "ViewCenterX='0' ViewCenterY='0'>"
        "<PageSheet LineStyle='0' FillStyle='0' TextStyle='0'>"
        f"<Cell N='PageWidth' V='{px2in(CANVAS_W)}'/>"
        f"<Cell N='PageHeight' V='{px2in(CANVAS_H)}'/>"
        "<Cell N='PageScale' V='1'/><Cell N='DrawingScale' V='1'/>"
        "</PageSheet><Rel r:id='rId1'/></Page></Pages>")

    document = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        f"<VisioDocument xmlns='{VNS}' xmlns:r='{R_NS}' xml:space='preserve'>"
        "<DocumentSettings TopPage='0' DefaultTextStyle='0' "
        "DefaultLineStyle='0' DefaultFillStyle='0'>"
        "<GlyphSettings>0</GlyphSettings></DocumentSettings>"
        "<Colors/><FaceNames/><StyleSheets/></VisioDocument>")

    ct = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
        "<Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>"
        "<Default Extension='xml' ContentType='application/xml'/>"
        "<Override PartName='/visio/document.xml' ContentType='application/vnd.ms-visio.drawing.main+xml'/>"
        "<Override PartName='/visio/pages/pages.xml' ContentType='application/vnd.ms-visio.pages+xml'/>"
        "<Override PartName='/visio/pages/page1.xml' ContentType='application/vnd.ms-visio.page+xml'/>"
        "<Override PartName='/visio/windows.xml' ContentType='application/vnd.ms-visio.windows+xml'/>"
        "<Override PartName='/docProps/core.xml' ContentType='application/vnd.openxmlformats-package.core-properties+xml'/>"
        "<Override PartName='/docProps/app.xml' ContentType='application/vnd.openxmlformats-officedocument.extended-properties+xml'/>"
        "</Types>")

    root_rels = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
        "<Relationship Id='rId1' Type='http://schemas.microsoft.com/visio/2010/relationships/document' Target='visio/document.xml'/>"
        "<Relationship Id='rId2' Type='http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties' Target='docProps/core.xml'/>"
        "<Relationship Id='rId3' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties' Target='docProps/app.xml'/>"
        "</Relationships>")

    doc_rels = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
        "<Relationship Id='rId1' Type='http://schemas.microsoft.com/visio/2010/relationships/pages' Target='pages/pages.xml'/>"
        "<Relationship Id='rId2' Type='http://schemas.microsoft.com/visio/2010/relationships/windows' Target='windows.xml'/>"
        "</Relationships>")

    pages_rels = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
        "<Relationship Id='rId1' Type='http://schemas.microsoft.com/visio/2010/relationships/page' Target='page1.xml'/>"
        "</Relationships>")

    windows = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        f"<Windows xmlns='{VNS}' xmlns:r='{R_NS}' ClientWidth='1600' "
        "ClientHeight='900'><Window ID='0' WindowType='Drawing' "
        "WindowState='1073741824' Document='visio/document.xml' Page='0' "
        "ViewScale='-1' ViewCenterX='0' ViewCenterY='0'/></Windows>")

    core = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<cp:coreProperties "
        "xmlns:cp='http://schemas.openxmlformats.org/package/2006/metadata/core-properties' "
        "xmlns:dc='http://purl.org/dc/elements/1.1/'>"
        "<dc:title>LMT-SDNN framework</dc:title>"
        "<dc:creator>phase2/src/make_diagram.py</dc:creator>"
        "</cp:coreProperties>")

    app = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<Properties xmlns='http://schemas.openxmlformats.org/officeDocument/2006/extended-properties'>"
        "<Application>Microsoft Visio</Application></Properties>")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
        z.writestr("visio/document.xml", document)
        z.writestr("visio/_rels/document.xml.rels", doc_rels)
        z.writestr("visio/pages/pages.xml", pages)
        z.writestr("visio/pages/_rels/pages.xml.rels", pages_rels)
        z.writestr("visio/pages/page1.xml", page1)
        z.writestr("visio/windows.xml", windows)


def main() -> None:
    d = OUT / "LMT-SDNN_framework.drawio"
    v = OUT / "LMT-SDNN_framework.vsdx"
    write_drawio(d)
    write_vsdx(v)
    print(f"  shapes: {len(N)}   connectors: {len(E)}")
    print(f"  {d.name}: {d.stat().st_size/1024:.1f} KB")
    print(f"  {v.name}: {v.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
