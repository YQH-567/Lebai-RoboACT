"""Apply reference LM3 paint using USD material edits only. Run with Isaac Sim Python."""
from pathlib import Path
import hashlib, json, shutil
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'lebai_lm3_lmg90_color_new1114.usd'
BACKUP = ROOT / 'lebai_lm3_lmg90_color_new1114.before_projection.usd'
if not BACKUP.exists():
    shutil.copy2(TARGET, BACKUP)
stage = Usd.Stage.Open(str(TARGET))
arm = '/World/lebai_lm3'
gripper = '/World/lmg90_gripper_inertial_stable/gripper'
links = [arm + '/base_link'] + [arm + '/link_' + str(i) for i in range(1,7)]
glinks = [gripper + '/gripper_base_link'] + [gripper + '/gripper_' + side + '_' + suffix for side in ('l','r') for suffix in ('link1','link2','link_finger')]
visual_roots = [p + '/visuals' for p in links+glinks]

# Compare original authored content after removing only permitted material edits.
def non_material_content(layer):
    clean = Sdf.Layer.CreateAnonymous('.usda')
    clean.TransferContent(layer)
    looks = clean.GetPrimAtPath('/World/Looks')
    for child in list(looks.nameChildren.values()):
        if child.name.startswith('Industrial_'):
            e=Sdf.BatchNamespaceEdit(); e.Add(child.path,Sdf.Path.emptyPath); assert clean.Apply(e)
    for path in visual_roots:
        spec=clean.GetPrimAtPath(path)
        if spec:
            for prop in list(spec.properties.values()):
                if prop.name.startswith('material:binding'):
                    spec.RemoveProperty(prop)
    return clean.ExportToString()

def allprims(s):
    return list(Usd.PrimRange(s.GetPseudoRoot(),Usd.TraverseInstanceProxies()))

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

before = non_material_content(stage.GetRootLayer())
assert before == non_material_content(Sdf.Layer.FindOrOpen(str(BACKUP))), "Non-material content differs from the preserved pre-projection USD"
external_hashes = {x.realPath:digest(x.realPath) for x in stage.GetUsedLayers() if x.realPath and x!=stage.GetRootLayer()}
oldpaths = {str(p.GetPath()): p.GetTypeName() for p in allprims(stage)}
collision_bindings = {str(p.GetPath()):str(UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0].GetPath()) for p in allprims(stage) if p.IsA(UsdGeom.Mesh) and '/collisions/' in str(p.GetPath())}
rows=[]
for part_id,link in enumerate(links):
    matpath='/World/Looks/Industrial_LM3_Reference_' + ('Base' if part_id==0 else 'Link'+str(part_id))
    mat=UsdShade.Material.Define(stage,matpath)
    shader=UsdShade.Shader.Define(stage,matpath+'/ProjectedPaint')
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath('./materials/lm3_reference/lm3_reference.mdl'),'mdl')
    shader.SetSourceAssetSubIdentifier('LM3Reference','mdl')
    shader.CreateInput('part_id',Sdf.ValueTypeNames.Int).Set(part_id)
    inp=shader.CreateInput('logo_texture',Sdf.ValueTypeNames.Asset)
    inp.Set(Sdf.AssetPath('./materials/lm3_reference/lebai_marking.png'))
    inp.GetAttr().SetColorSpace('sRGB')
    shader.CreateInput('shell_color',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.58,.60,.63))
    shader.CreateInput('trim_color',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.28,.34,.38))
    shader.CreateInput('shell_roughness',Sdf.ValueTypeNames.Float).Set(.47)
    shader.CreateInput('shell_metallic',Sdf.ValueTypeNames.Float).Set(.18)
    shader.CreateOutput('out',Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
    # Basic non-MDL viewer fallback; RTX uses the projected MDL material above.
    fallback=UsdShade.Shader.Define(stage,matpath+'/PreviewSurface')
    fallback.CreateIdAttr('UsdPreviewSurface')
    fallback.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.58,.60,.63))
    fallback.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.47)
    fallback.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(.18)
    mat.CreateSurfaceOutput().ConnectToSource(fallback.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(link+'/visuals')).Bind(mat,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    rows.append({'part':link,'material':matpath})

red = UsdShade.Material(stage.GetPrimAtPath('/World/Looks/Industrial_GraphiteGray'))
red.GetPrim().SetDisplayName('LMG90 Reference Red — All Moving Fingers')
redshader=UsdShade.Shader(stage.GetPrimAtPath(str(red.GetPath())+'/PreviewSurface'))
redshader.GetInput('diffuseColor').Set(Gf.Vec3f(.55,.018,.025))
redshader.GetInput('roughness').Set(.43)
redshader.GetInput('metallic').Set(.28)
white=UsdShade.Material(stage.GetPrimAtPath('/World/Looks/Industrial_SatinLightGray'))
whiteshader=UsdShade.Shader(stage.GetPrimAtPath(str(white.GetPath())+'/PreviewSurface'))
whiteshader.GetInput('diffuseColor').Set(Gf.Vec3f(.58,.60,.63))
whiteshader.GetInput('roughness').Set(.47)
whiteshader.GetInput('metallic').Set(.18)
for link in glinks:
    mat=white if link.endswith('/gripper_base_link') else red
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(link+'/visuals')).Bind(mat,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    rows.append({'part':link,'material':str(mat.GetPath())})

assert non_material_content(stage.GetRootLayer()) == before, 'Unexpected non-material change: refusing to save'
for p,h in external_hashes.items():
    assert digest(p)==h, 'Referenced asset changed'
for p in allprims(stage):
    if p.IsA(UsdGeom.Mesh) and '/visuals/' in str(p.GetPath()):
        row=next(r for r in rows if str(p.GetPath()).startswith(r['part']+'/visuals/'))
        row['visual_mesh']=str(p.GetPath())
        row['collision_mesh']=str(p.GetPath()).replace('/visuals/','/collisions/')
        assert str(UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0].GetPath())==row['material']
for p,mat in collision_bindings.items():
    assert str(UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(p)).ComputeBoundMaterial()[0].GetPath())==mat
newpaths={str(p.GetPath()):p.GetTypeName() for p in allprims(stage)}
assert all(newpaths.get(p)==t for p,t in oldpaths.items())
assert all(p.startswith('/World/Looks/Industrial_') for p in set(newpaths)-set(oldpaths))
joints=[str(p.GetPath()) for p in allprims(stage) if p.GetTypeName()=='PhysicsRevoluteJoint']
fixed=[str(p.GetPath()) for p in allprims(stage) if p.GetTypeName()=='PhysicsFixedJoint']
assert len(joints)==12
assert stage.GetRootLayer().Save()
# Disk reload, not just cached composition, also passes the authored-content comparison.
saved=Sdf.Layer.OpenAsAnonymous(str(TARGET))
assert non_material_content(saved)==before
report={'target':str(TARGET),'backup':str(BACKUP),'joint_count':len(joints),'fixed_joint_count':len(fixed),'visual_count':len(rows),'collision_count':len(collision_bindings),'root_non_material_content_unchanged':True,'referenced_asset_hashes_unchanged':True,'original_prim_paths_and_types_unchanged':True,'rows':rows,'revolute_joints':joints,'fixed_joints':fixed,'reference_layer_hashes':external_hashes,'target_sha256':digest(TARGET),'material_dependencies':{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'materials'/'lm3_reference').iterdir() if p.is_file()}}
(ROOT/'appearance_tools'/'reference_appearance_validation.json').write_text(json.dumps(report,indent=2))
print('APPEARANCE_SAVED',json.dumps({k:v for k,v in report.items() if k not in ('rows','reference_layer_hashes','revolute_joints','fixed_joints')}),flush=True)
app.close()
