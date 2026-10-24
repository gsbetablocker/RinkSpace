"""Draw exact sampled SceneModel geometry with the existing Pillow runtime."""
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG='#fafbf9'
INK='#24464f'
FONT=Path('C:/Windows/Fonts/msyh.ttc')


class Canvas:
    def __init__(self,width,height):
        self.image=Image.new('RGB',(width,height),BG)
        self.draw=ImageDraw.Draw(self.image)
        self.fonts={}

    def text(self,xy,text,size=23,fill=INK,anchor='mm',box=False):
        font=self.fonts.setdefault(size,ImageFont.truetype(str(FONT),size))
        if box:
            b=self.draw.textbbox(xy,text,font=font,anchor=anchor)
            self.draw.rectangle((b[0]-5,b[1]-3,b[2]+5,b[3]+3),fill=BG)
        self.draw.text(xy,text,font=font,fill=fill,anchor=anchor)

    def line(self,points,fill=INK,width=2):
        self.draw.line([tuple(map(float,p)) for p in points],fill=fill,width=width)

    def save(self,path):self.image.save(path)


class Chart:
    def __init__(self,c,rect,xlim,ylim):
        self.c=c;self.rect=rect;self.xlim=xlim;self.ylim=ylim

    def xy(self,q):
        q=np.asarray(q,dtype=float)
        x,y,w,h=self.rect
        return np.stack((x+(q[...,0]-self.xlim[0])/(self.xlim[1]-self.xlim[0])*w,
                         y+h-(q[...,1]-self.ylim[0])/(self.ylim[1]-self.ylim[0])*h),axis=-1)

    def path(self,q,fill=INK,width=2,clip=False):
        p=self.xy(q)
        if not clip:self.c.line(p,fill,width);return
        # Clip pixels after mapping. No endpoints or source geometry are changed.
        x,y,w,h=map(int,self.rect)
        layer=Image.new('RGBA',(w+1,h+1))
        d=ImageDraw.Draw(layer)
        d.line([tuple(v) for v in p-[x,y]],fill=fill,width=width)
        self.c.image.paste(layer,(x,y),layer)

    def text(self,q,text,size=23,fill=INK,offset=(0,0),box=True):
        self.c.text(self.xy(q)+offset,text,size,fill,box=box)


def color_for(p):
    if p['ad'] is not None:return '#a66a31'
    if re.match(r'ice\.[FR][1-4]:',p['label']):return '#087d80'
    if any(s in p['label'] for s in ['lane','house_connection','curling_centerline']):return '#809497'
    return '#477faf'


def ad_anchors(paths,board):
    groups={}
    for p in paths:
        if p['ad'] is not None and p['unwrapped'] is not None:groups.setdefault(p['ad'],[]).append(p['unwrapped'])
    result={}
    for n,parts in groups.items():
        q=np.concatenate(parts)
        angle=np.angle(np.mean(np.exp(2j*np.pi*q[:,0]/board.perimeter)))
        station=angle/(2*np.pi)*board.perimeter%board.perimeter
        result[n]=board.xyz(np.array([station]),np.array([board.height]))[0,:2]
    return result


def top_view(c,paths,board,x,title):
    c.text((x,135),title,34,anchor='lm')
    ax=Chart(c,(x+40,205,1500,900),(-.625,.625),(-.375,.375))
    boundary=board.xyz(np.linspace(0,board.perimeter,1025),0)[:,:2]
    c.draw.polygon([tuple(v) for v in ax.xy(boundary)],fill='#eef4f4')
    for t in np.arange(-.5,.51,.1):
        ax.path([[t,-.27],[t,.27]],'#dbe5e4',1)
        ax.text([t,-.34],f'{t:.1f}',18,fill='#6a8389')
    ax.path(boundary,'#355e65',3)
    for p in paths:
        if p['kind']=='board_boundary':continue
        ax.path(p['xyz'][:,:2],color_for(p),2)
    lookup={p['label']:p for p in paths}
    for name in [*[f'{s}{i}' for s in ['F','R'] for i in range(1,5)],'C0',*[f'A{i}' for i in range(1,5)]]:
        p=lookup.get(f'ice.{name}:outer')
        if not p:continue
        center=np.asarray(p['definition']['center_xy']);radius=p['definition']['radius']
        q=ax.xy(center);c.draw.ellipse((q[0]-3,q[1]-3,q[0]+3,q[1]+3),fill=color_for(p))
        ax.text(center+[0,radius],name,27,offset=(0,-18))
    for name in ['B1','C','B2','goal_line_left','goal_line_right']:
        p=lookup.get(f'ice.{name}:segment0')
        if p:
            ends=np.asarray(p['definition']['endpoints_xy']).reshape(2,2)
            ax.text(ends[np.argmax(ends[:,1])],{'goal_line_left':'G1','goal_line_right':'G2'}.get(name,name),25,fill='#477faf',offset=(0,-20))
    for row in range(1,5):
        p=lookup.get(f'ice.lane{row}:edge0')
        if p:
            ends=np.asarray(p['definition']['endpoints_xy']).reshape(2,2)
            ax.text(ends.mean(axis=0),f'Lane {row}',20,fill='#697f84',offset=(0,17))
    q=ax.xy([0,0]);c.line([q+[-8,0],q+[8,0]],'#d67832',3);c.line([q+[0,-8],q+[0,8]],'#d67832',3)
    groups={k:[] for k in ['top','bottom','left','right']}
    for n,p in ad_anchors(paths,board).items():
        side=('right' if p[0]>0 else 'left') if abs(p[0])/.5>abs(p[1])/(board.width/2) else ('top' if p[1]>0 else 'bottom')
        groups[side].append((n,p))
    for side,items in groups.items():
        dim=0 if side in ('top','bottom') else 1
        items.sort(key=lambda v:v[1][dim])
        if not items:continue
        targets=np.array([p[dim] for _,p in items]);gap=.067 if dim==0 else .041
        for i in range(1,len(targets)):targets[i]=max(targets[i],targets[i-1]+gap)
        targets-=targets.mean()-np.mean([p[dim] for _,p in items])
        for (n,p),t in zip(items,targets):
            q=[t,(1 if side=='top' else -1)*(board.width/2+.067)] if dim==0 else [(1 if side=='right' else -1)*.565,t]
            ax.path([p,q],'#b0987b',1);ax.text(q,f'Ad{n}',19,fill='#8a592c')
    c.text((x+790,1130),'长边方向 x / L；两图同一坐标、同一比例',22,fill='#536e75')


def plot_top(old_paths,new_paths,board,out):
    c=Canvas(3340,1390)
    c.text((60,55),'实际模型的新旧俯视对照',45,anchor='lm')
    top_view(c,old_paths,board,60,'旧 · d78 原版')
    top_view(c,new_paths,board,1710,'新 · 共同冰场结构候选')
    lines=[
        '青绿：F/R 靶圈　蓝色：冰球标线　灰色：Lane 与连线　棕色：围挡广告　橙色 +：场地中心',
        'G1 / G2 为两条球门线。L = 1；米制未验证，现有数值只是初值。本步没有拟合、没有重放。',
        '旧版缺 R1 / R2；新图补足部分来自已登记 8a28 几何库存与共享结构，不代表新增原片观测。',
        '所有线条由实际模型求值。广告 x / x+13 共用模板与尺寸；Ad25 位置缺失，未虚构摆放。']
    for i,s in enumerate(lines):c.text((60,1210+i*40),s,25,fill='#536e75',anchor='lm')
    c.save(out/'top_comparison.png')


def plot_board(old_paths,new_paths,board,out):
    c=Canvas(3520,2150)
    c.text((60,55),'整圈围挡剪开后的展开对照 · 全部已存广告轮廓与编号',43,anchor='lm')
    c.text((60,115),'上下实线为围挡顶部 / 冰面交界。纵轴放大，浅蓝色为弯角；四段沿弧长连续，广告可以跨段。',25,fill='#536e75',anchor='lm')
    P=board.perimeter
    allq=[p['unwrapped'] for paths in [old_paths,new_paths] for p in paths if p['ad'] is not None and p['unwrapped'] is not None]
    low=min(0.,min(float(q[:,1].min()) for q in allq));high=max(board.height,max(float(q[:,1].max()) for q in allq));h=high-low
    names=['下长边','右下弯角','右短边','右上弯角','上长边','左上弯角','左短边','左下弯角']
    for row in range(4):
        a,b=P*row/4,P*(row+1)/4
        for col,paths in enumerate([old_paths,new_paths]):
            x=60+col*1750;y=200+row*440
            c.text((x,y),f'{"旧 d78" if col==0 else "新共同场地"} · 第 {row+1} / 4 段',30,anchor='lm')
            ax=Chart(c,(x+55,y+70,1570,220),(a,b),(low-.05*h,high+.05*h))
            c.draw.rectangle((x+55,y+70,x+1625,y+290),fill='#eff4f3')
            for seg in range(8):
                left,right=board.breaks[seg:seg+2]
                if right<a or left>b:continue
                left=max(left,a);right=min(right,b)
                q=ax.xy([[left,high+.05*h],[right,low-.05*h]])
                if seg%2:c.draw.rectangle(tuple(q.reshape(-1)),fill='#d5e7ec')
                ax.path([[left,low],[left,high]],'#9bb0b7',1)
                c.text((ax.xy([(left+right)/2,low])[0],y+325),names[seg],21,fill='#5e747c')
            for z in [0,board.height]:ax.path([[a,z],[b,z]],'#355e65',3)
            for t in np.arange(0,P,.05):
                if a<=t<=b:ax.path([[t,0],[t,board.height]],'#c4d1d0',1)
            groups={}
            for p in paths:
                if p['ad'] is None or p['unwrapped'] is None:continue
                q=p['unwrapped'].copy();q[:,0]=np.unwrap(q[:,0]/P*2*np.pi)/(2*np.pi)*P
                q[:,0]-=np.floor(q[:,0].mean()/P)*P
                for shift in [-P,0,P]:
                    v=q+[shift,0]
                    if v[:,0].max()<a or v[:,0].min()>b:continue
                    ax.path(v,'#95602f',2,clip=True)
                    hit=(v[:,0]>=a)&(v[:,0]<=b)
                    if hit.any():groups.setdefault(p['ad'],[]).extend(v[hit,0].tolist())
            items=sorted(((n,float(np.mean(v))) for n,v in groups.items()),key=lambda v:v[1])
            original=np.array([ax.xy([t,0])[0] for _,t in items]);labels=original.copy()
            for i in range(1,len(labels)):labels[i]=max(labels[i],labels[i-1]+85)
            if len(labels):labels-=labels.mean()-original.mean()
            for (n,t),label_x,target_x in zip(items,labels,original):
                c.line([[target_x,y+70],[label_x,y+52]],'#b0987b',1)
                c.text((label_x,y+40),f'Ad{n}',22,fill='#845426',box=True)
            if not groups:c.text((x+840,y+180),'d78 此段没有已存广告几何',26,fill='#899a9c')
            for t in np.linspace(a,b,5):c.text((ax.xy([t,0])[0],y+363),f'{t:.3f}',20,fill='#6a8389')
            c.text((x+840,y+399),'连续弧长 s / L',20,fill='#536e75')
    c.text((60,2020),'广告 x / x+13 共用同一静态模板与宽高，安装位置独立。数字仍为初值，本步未拟合。',25,fill='#536e75',anchor='lm')
    c.text((60,2070),'Ad25：已有 Ad12 配对图案，缺少独立安装位置。字形、缺失及外溢照实保留；未补造外框。',25,fill='#536e75',anchor='lm')
    c.save(out/'board_comparison.png')
