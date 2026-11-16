"""Rasterize the official SVG as an albedo decal; no changes to robot geometry."""
from pathlib import Path
import gi, cairo
import re
gi.require_version('Rsvg','2.0')
from gi.repository import Rsvg
root=Path(__file__).resolve().parents[1]/'materials'/'lm3_reference'
s=(root/'lebai_official_logo.svg').read_text()
s=re.sub(r'<!--.*?-->','',s,flags=re.S)
s=s.replace('viewBox="0 0 340 156"','viewBox="24 41 292 62"')
# Match the black/red printed product marking, rather than the lighter website header palette.
s=s.replace('#FF5A59','#BE242C').replace('#595656','#27282B')
(root/'lebai_print_decal.svg').write_text(s)
surface=cairo.ImageSurface(cairo.FORMAT_ARGB32,1752,372)
ctx=cairo.Context(surface)
handle=Rsvg.Handle.new_from_data(s.encode())
viewport=Rsvg.Rectangle(); viewport.x=0; viewport.y=0; viewport.width=1752; viewport.height=372
handle.render_document(ctx,viewport)
surface.write_to_png(str(root/'lebai_marking.png'))
print('OFFICIAL_LOGO_RASTERIZED')
