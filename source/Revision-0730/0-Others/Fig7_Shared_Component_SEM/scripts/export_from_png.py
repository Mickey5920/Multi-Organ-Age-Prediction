#!/usr/bin/env python3
"""Create font-stable SVG, PDF and TIFF files directly from approved PNGs.

The SVG contains one embedded PNG image and no text nodes. The PDF contains
one losslessly compressed RGB image and no font resources. No plotting code is
imported or executed by this module.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import zlib

from PIL import Image


BASE = Path(__file__).resolve().parent.parent
DEFAULT_DIRECTORY = BASE / "outputs" / "Figure6_split_exports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_dpi(image: Image.Image) -> float:
    dpi = image.info.get("dpi", (600.0, 600.0))
    if isinstance(dpi, tuple):
        values = [float(value) for value in dpi if float(value) > 0]
        return sum(values) / len(values) if values else 600.0
    value = float(dpi)
    return value if value > 0 else 600.0


def rgb_image(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def write_svg_from_png(png_path: Path, svg_path: Path) -> None:
    with Image.open(png_path) as image:
        width, height = image.size
        dpi = image_dpi(image)
    width_mm = width / dpi * 25.4
    height_mm = height / dpi * 25.4
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_mm:.6f}mm" height="{height_mm:.6f}mm" '
        f'viewBox="0 0 {width} {height}">\n'
        f'  <image x="0" y="0" width="{width}" height="{height}" '
        'preserveAspectRatio="none" '
        f'href="data:image/png;base64,{encoded}"/>\n'
        '</svg>\n'
    )
    svg_path.write_text(svg, encoding="utf-8")


def pdf_object(number: int, body: bytes) -> bytes:
    return f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"


def write_pdf_from_png(png_path: Path, pdf_path: Path) -> None:
    with Image.open(png_path) as source:
        dpi = image_dpi(source)
        image = rgb_image(source)
    width, height = image.size
    width_points = width / dpi * 72.0
    height_points = height / dpi * 72.0
    pixels = image.tobytes()
    compressed = zlib.compress(pixels, level=9)
    content = (
        "q\n"
        f"{width_points:.6f} 0 0 {height_points:.6f} 0 0 cm\n"
        "/Im0 Do\n"
        "Q\n"
    ).encode("ascii")

    objects = [
        pdf_object(1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        pdf_object(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        pdf_object(
            3,
            (
                "<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {width_points:.6f} {height_points:.6f}] "
                "/Resources << /ProcSet [/PDF /ImageC] "
                "/XObject << /Im0 5 0 R >> >> "
                "/Contents 4 0 R >>"
            ).encode("ascii"),
        ),
        pdf_object(
            4,
            f"<< /Length {len(content)} >>\nstream\n".encode("ascii")
            + content
            + b"endstream",
        ),
        pdf_object(
            5,
            (
                "<< /Type /XObject /Subtype /Image "
                f"/Width {width} /Height {height} "
                "/ColorSpace /DeviceRGB /BitsPerComponent 8 "
                "/Interpolate false /Filter /FlateDecode "
                f"/Length {len(compressed)} >>\nstream\n"
            ).encode("ascii")
            + compressed
            + b"\nendstream",
        ),
    ]

    document = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(document))
        document.extend(obj)
    xref_offset = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    pdf_path.write_bytes(document)


def write_tiff_from_png(png_path: Path, tiff_path: Path) -> None:
    with Image.open(png_path) as source:
        dpi = image_dpi(source)
        image = rgb_image(source)
    image.save(
        tiff_path,
        format="TIFF",
        compression="tiff_lzw",
        dpi=(dpi, dpi),
    )


def create_derived_exports(
    png_path: Path,
    svg_path: Path | None = None,
    pdf_path: Path | None = None,
    tiff_path: Path | None = None,
) -> dict[str, object]:
    png_path = Path(png_path)
    svg_path = Path(svg_path) if svg_path is not None else png_path.with_suffix(".svg")
    pdf_path = Path(pdf_path) if pdf_path is not None else png_path.with_suffix(".pdf")
    tiff_path = Path(tiff_path) if tiff_path is not None else png_path.with_suffix(".tiff")
    write_svg_from_png(png_path, svg_path)
    write_pdf_from_png(png_path, pdf_path)
    write_tiff_from_png(png_path, tiff_path)
    with Image.open(png_path) as image:
        width, height = image.size
        dpi = image_dpi(image)
    svg_text = svg_path.read_text(encoding="utf-8")
    pdf_bytes = pdf_path.read_bytes()
    return {
        "png": png_path.name,
        "width_pixels": width,
        "height_pixels": height,
        "dpi": dpi,
        "svg": svg_path.name,
        "pdf": pdf_path.name,
        "tiff": tiff_path.name,
        "svg_text_elements": svg_text.count("<text"),
        "svg_embedded_images": svg_text.count("<image"),
        "pdf_font_resources": pdf_bytes.count(b"/Font"),
        "pdf_image_resources": pdf_bytes.count(b"/Subtype /Image"),
        "sha256": {
            png_path.name: sha256(png_path),
            svg_path.name: sha256(svg_path),
            pdf_path.name: sha256(pdf_path),
            tiff_path.name: sha256(tiff_path),
        },
    }


def main() -> int:
    directory = DEFAULT_DIRECTORY
    png_paths = sorted(directory.glob("Figure6_*.png"))
    if not png_paths:
        raise RuntimeError(f"no Figure 6 PNG files found in {directory}")
    rows = [create_derived_exports(path) for path in png_paths]
    report = {
        "status": "FONT_STABLE_PNG_DERIVED_EXPORTS",
        "plotting_code_executed": False,
        "source_format": "PNG",
        "svg_method": "single embedded PNG image with no text nodes",
        "pdf_method": "single losslessly compressed RGB image with no font resources",
        "converted_file_count": len(rows),
        "files": rows,
    }
    report_path = directory / "PNG_DERIVED_EXPORT_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    split_report_path = directory / "SPLIT_EXPORT_REPORT.json"
    if split_report_path.exists():
        split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
        split_report["svg_pdf_export_source"] = "approved PNG files"
        split_report["svg_text_editability"] = False
        split_report["svg_pdf_font_substitution_risk"] = False
        split_report_path.write_text(
            json.dumps(split_report, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
