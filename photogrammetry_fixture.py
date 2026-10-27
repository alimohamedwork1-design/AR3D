"""Original textured cube photos rendered by ray/plane intersections.

The reconstruction receives only PNG files; ground-truth cameras are not passed.
This is a digital fixture, not evidence of physical-device acceptance.
"""
from pathlib import Path


def generate(directory, side=640):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261027)
    textures=[]
    for face in range(7):
        image=Image.new('RGB',(1024,1024),'white');draw=ImageDraw.Draw(image)
        font=ImageFont.load_default(size=20)
        for row in range(16):
            for col in range(16):
                x,y=col*64,row*64
                color=tuple(int(v) for v in rng.integers(40,235,3))
                draw.rectangle((x,y,x+63,y+63),fill=color,outline='black',width=2)
                draw.ellipse((x+8,y+7,x+40,y+39),fill='white',outline='black',width=3)
                draw.text((x+4,y+41),f'{face}{row:x}{col:x}',fill='black',font=font)
                for _ in range(4):
                    a,b=rng.integers(3,58,2);draw.rectangle((x+a,y+b,x+a+4,y+b+4),fill='black')
        textures.append(np.asarray(image))
    yy,xx=np.mgrid[0:side,0:side]
    rays=np.stack([(xx-side/2)/side,(yy-side/2)/side,np.ones_like(xx)],axis=-1).reshape(-1,3)
    for index,angle in enumerate(np.linspace(-0.45,0.45,12)):
        center=np.array([3*np.sin(angle),0.8,3*np.cos(angle)])
        forward=-center/np.linalg.norm(center);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right)
        rotation=np.stack([right,np.cross(forward,right),forward]);direction=rays@rotation
        depth=np.full(side*side,np.inf);pixels=np.full((side*side,3),180,dtype=np.uint8)
        for face in range(7):
            axis=face//2 if face<6 else 1
            plane=(-1 if face%2==0 else 1) if face<6 else -1.3
            denominator=direction[:,axis]
            t=(plane-center[axis])/np.where(np.abs(denominator)>1e-9,denominator,1e-9)
            point=center+direction*t[:,None]
            axes=[i for i in range(3) if i!=axis]
            valid=(t>0)&(t<depth)
            if face<6:valid&=(np.abs(point[:,axes])<=1.001).all(axis=1)
            else:valid&=(np.abs(point[:,[0,2]])<5).all(axis=1)
            u,v=point[:,axes].T
            extent=1 if face<6 else 5
            u=np.clip(((u/extent+1)*511.5).astype(int),0,1023)
            v=np.clip(((v/extent+1)*511.5).astype(int),0,1023)
            pixels[valid]=textures[face][v[valid],u[valid]];depth[valid]=t[valid]
        Image.fromarray(pixels.reshape(side,side,3)).save(directory/f'photo_{index:02d}.png')
    return directory
