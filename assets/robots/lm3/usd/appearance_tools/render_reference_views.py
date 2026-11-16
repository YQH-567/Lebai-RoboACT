from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting'})
from pathlib import Path
import omni.usd, omni.replicator.core as rep
from pxr import UsdLux,UsdGeom,Gf
from PIL import Image
import numpy as np
root=Path(__file__).resolve().parents[1]
assert omni.usd.get_context().open_stage(str(root/'lebai_lm3_lmg90_color_new1114.usd'))
for _ in range(8): app.update()
s=omni.usd.get_context().get_stage()
# Runtime-only verification lights/cameras. The USD is never saved by this script.
with rep.new_layer():
    fill=UsdLux.DomeLight.Define(s,'/MaterialVerificationFill')
    fill.CreateIntensityAttr(450)
    fill.CreateColorAttr(Gf.Vec3f(1,1,1))
    def capture(name,eye,target):
        camera=rep.create.camera(position=eye,look_at=target,focal_length=35,clipping_range=(0.01,100.0))
        rp=rep.create.render_product(camera,(1440,1080))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach(rp)
        rep.orchestrator.step(rt_subframes=16)
        for _ in range(12):app.update()
        data=rgb.get_data(); assert data.size>0 and data[:,:,:3].std()>5
        Image.fromarray(data[:,:,:3]).save(root/'appearance_tools'/name)
        print('VIEW_SAVED',name,data.shape,float(data[:,:,:3].std()),flush=True)
        rgb.detach(rp)
    capture('reference_appearance_overview.png',(.95,-1.2,.90),(-.25,-.04,.15))
    capture('reference_appearance_reverse.png',(-.9,1.05,.78),(-.25,-.06,.17))
    capture('reference_appearance_gripper.png',(-.19,-.85,.43),(-.54,-.31,.13))
app.close()
