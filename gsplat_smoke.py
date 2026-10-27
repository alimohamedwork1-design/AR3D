"""Owned synthetic scene; real training/rendering, no customer inputs or downloads."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile


def fixture(root, side=96, span=None):
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(20261027)
    points=rng.normal(size=(400,3)).astype(np.float32)
    points/=np.linalg.norm(points,axis=1,keepdims=True)
    colors=rng.uniform(0.1,0.95,(len(points),3)).astype(np.float32)
    root=Path(root);(root/'images').mkdir(parents=True);(root/'sparse/0').mkdir(parents=True)
    intrinsic=np.array([[side,0,side/2],[0,side,side/2],[0,0,1]],dtype=np.float32)
    image_rows=[]
    for index in range(12):
        angle=index*2*np.pi/12 if span is None else (index/11-0.5)*span
        center=np.array([3*np.cos(angle),0.5,3*np.sin(angle)],dtype=np.float32)
        forward=-center/np.linalg.norm(center)
        right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        rotation=np.stack([right,down,forward]);translation=-rotation@center
        view=np.eye(4,dtype=np.float32);view[:3,:3]=rotation;view[:3,3]=translation
        camera=points@rotation.T+translation
        uv=(camera@intrinsic.T);uv=uv[:,:2]/uv[:,2:]
        image=np.zeros((side,side,3),dtype=np.float32)
        for j in np.argsort(camera[:,2])[::-1]:
            u,v=uv[j];sigma=max(0.6,side*0.035/camera[j,2])
            x0,x1=max(0,int(u-3*sigma)),min(side,int(u+3*sigma)+1)
            y0,y1=max(0,int(v-3*sigma)),min(side,int(v+3*sigma)+1)
            if x1<=x0 or y1<=y0:continue
            yy,xx=np.mgrid[y0:y1,x0:x1]
            alpha=(0.8*np.exp(-((xx-u)**2+(yy-v)**2)/(2*sigma*sigma)))[:,:,None]
            image[y0:y1,x0:x1]=image[y0:y1,x0:x1]*(1-alpha)+colors[j]*alpha
        name=f'view_{index:02d}.png'
        Image.fromarray(np.round(image*255).astype('uint8')).save(root/'images'/name)
        q=Rotation.from_matrix(rotation).as_quat()
        image_rows.append(' '.join(map(str,[index+1,q[3],q[0],q[1],q[2],*translation,1,name]))+'\n\n')
    (root/'sparse/0/cameras.txt').write_text(f'1 PINHOLE {side} {side} {side} {side} {side/2} {side/2}\n')
    (root/'sparse/0/images.txt').write_text(''.join(image_rows))
    (root/'sparse/0/points3D.txt').write_text('\n'.join(' '.join(map(str,[i+1,*p,*np.round(colors[i]*255).astype(int),0])) for i,p in enumerate(points))+'\n')
    return root


def run():
    from gsplat_backend import load_scene,train_arrays
    with tempfile.TemporaryDirectory(prefix='arqary-gsplat-smoke-') as directory:
        root=fixture(Path(directory)/'input')
        points,colors,views=load_scene(root,max_side=96)
        stats=train_arrays(points,colors,views,Path(directory)/'output',iterations=120,deadline_seconds=240)
        ply=next((Path(directory)/'output').rglob('point_cloud.ply'))
        data=ply.read_bytes()
        stats.update(ply_bytes=len(data),ply_sha256=hashlib.sha256(data).hexdigest(),
            fixture_kind='generated_sphere_known_poses',camera_estimation_verified=False,
            source_scope='owned_synthetic_only',ply_base64=base64.b64encode(data).decode())
        return stats


def photogrammetry_run():
    from photogrammetry_fixture import generate
    from gsplat_backend import prepare_scene,load_scene,train_arrays
    with tempfile.TemporaryDirectory(prefix='arqary-photo-smoke-') as directory:
        root=Path(directory)
        generate(root/'input/images')
        scene=prepare_scene(root,max_image_size=640)
        points,colors,views=load_scene(scene,max_side=640)
        stats=train_arrays(points,colors,views,root/'output',iterations=120,deadline_seconds=300)
        data=next((root/'output').rglob('point_cloud.ply')).read_bytes()
        stats.update(fixture_kind='original_raytraced_cube_images_only',camera_estimation_verified=True,
            sparse_points=len(points),source_scope='owned_synthetic_only',ply_bytes=len(data),
            ply_sha256=hashlib.sha256(data).hexdigest(),ply_base64=base64.b64encode(data).decode())
        return stats


if __name__=='__main__':
    print(json.dumps(run()))
