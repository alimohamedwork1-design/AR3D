import sys,json,math
from pathlib import Path
import numpy as np,torch,pycolmap
from PIL import Image
from plyfile import PlyData
root=Path('/workspace/experiment');sys.path.insert(0,str(root))
from gsplat_backend import load_scene,train_arrays
scene=root/'gsplat-baseline/input_undist';points,colors,views=load_scene(scene,max_side=640)
recon=pycolmap.Reconstruction(scene/'sparse');ordered=[image.name for _,image in sorted(recon.images.items())]
refs=sorted((root/'spirula-output/model').glob('eval-gt-*.png'))
originals=sorted((root/'fixture/images').glob('*.png'));matches=[]
for ref in refs:
 a=np.asarray(Image.open(ref).convert('RGB'),dtype=float)
 errors=[]
 for p in originals:
  b=np.asarray(Image.open(p).convert('RGB'),dtype=float);errors.append(np.mean((a-b)**2) if a.shape==b.shape else float('inf'))
 idx=int(np.argmin(errors));assert errors[idx]<1,matches
 matches.append('frame_'+str(idx).zfill(4)+'.png')
# gsplat used photo_XX names; map exact original fixture names to native prepared indices.
held=[i for i,n in enumerate(ordered) if 'frame_'+str(int(Path(n).stem.split('_')[-1])).zfill(4)+'.png' in matches]
assert len(held)==3 and len(set(matches))==3
stats=train_arrays(points,colors,[v for i,v in enumerate(views) if i not in held],root/'gsplat-heldout',iterations=3000,deadline_seconds=300)
p=next((root/'gsplat-heldout').rglob('point_cloud.ply'));v=PlyData.read(p)['vertex'].data
T=lambda names:torch.tensor(np.stack([v[n] for n in names],axis=1).copy(),device='cuda')
means=T(['x','y','z']);scales=T([f'scale_{i}' for i in range(3)]).exp();quats=T([f'rot_{i}' for i in range(4)]);opacities=T(['opacity']).squeeze(-1).sigmoid();dc=T([f'f_dc_{i}' for i in range(3)])[:,None,:];rest=T([f'f_rest_{i}' for i in range(45)]).reshape(-1,3,15).transpose(1,2);sh=torch.cat([dc,rest],dim=1)
from gsplat import rasterization
scores=[]
with torch.no_grad():
 for i in held:
  pixels,K,pose=views[i]
  rendered,_,_=rasterization(means=means,quats=quats,scales=scales,opacities=opacities,colors=sh,viewmats=torch.tensor(pose,device='cuda')[None],Ks=torch.tensor(K,device='cuda')[None],width=pixels.shape[1],height=pixels.shape[0],sh_degree=2,packed=False)
  mse=float(torch.mean((rendered[0]-torch.tensor(pixels,device='cuda'))**2));scores.append(-10*math.log10(max(mse,1e-12)))
  Image.fromarray(np.round(rendered[0].clamp(0,1).cpu().numpy()*255).astype('uint8')).save(root/'gsplat-heldout'/('eval-'+str(i)+'.png'))
stats.update(held_out_names=[ordered[i] for i in held],held_out_psnr_db=scores,avg_held_out_psnr_db=sum(scores)/len(scores),scope='same source frames; same three held-out image identities; independent camera recovery and training recipes; no representative-property claim')
(root/'gsplat-heldout/comparison.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)
