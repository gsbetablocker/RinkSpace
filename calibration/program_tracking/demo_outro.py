"""Continuous ice-plane unfolding for demo endings (no video I/O).

This is a presentation camera, not a fitted or physically verified camera.
Brown terms are ignored by both the main-shot and ending plane warp. Use the
shared plane_homography for the main shot too, to keep the first ending frame
identical with no layer reset.

Coordinates: bounds=(xmin,ymin,xmax,ymax), ice z=0. Texture pixel (0,0) is
(xmin,ymax), bottom-right is (xmax,ymin). Sizes are (width,height). BGRA is
straight alpha, uint8 0..255 or floating point 0..1. frame() takes the raw last
native frame, before trajectory/heat overlays.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Sequence
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _camera(value):
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (25,) or not np.isfinite(result).all():
        raise ValueError("camera25 must contain 25 finite values")
    return result


def _bounds(value):
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if (result.shape != (4,) or not np.isfinite(result).all()
            or result[2] <= result[0] or result[3] <= result[1]):
        raise ValueError("bounds_xy must be (xmin,ymin,xmax,ymax)")
    return result


def _size(value):
    if len(value) != 2 or any(int(v) != v or v < 2 for v in value):
        raise ValueError("image dimensions must be two integers >=2")
    return int(value[0]), int(value[1])


def _ease(value):
    t = float(np.clip(value, 0.0, 1.0))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def plane_homography(camera25, bounds_xy, texture_wh):
    """Texture-pixel -> native-pixel planar homography, without Brown.

    Camera intrinsics must already be in the output/native pixel domain.
    This shared function is the continuity contract with the main renderer.
    """
    c, b = _camera(camera25), _bounds(bounds_xy)
    tw, th = _size(texture_wh)
    r = Rotation.from_rotvec(c[3:6]).as_matrix()
    fx, fy = np.exp(c[6:8])
    k = np.array([[fx, c[10], c[8]], [0.0, fy, c[9]], [0.0, 0.0, 1.0]])
    tex_world = np.array([
        [(b[2] - b[0]) / (tw - 1), 0.0, b[0]],
        [0.0, -(b[3] - b[1]) / (th - 1), b[3]],
        [0.0, 0.0, 1.0]])
    h = k @ np.column_stack((r[:, 0], r[:, 1], -r @ c[:3])) @ tex_world
    if not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("ice plane has a singular camera projection")
    scale = h[2, 2] if abs(h[2, 2]) > 1e-12 else np.linalg.norm(h)
    return h / scale


@dataclass(frozen=True)
class PlaneTexture:
    """Prepared premultiplied texture reusable across all ending frames."""
    premultiplied: np.ndarray

    @property
    def wh(self):
        return self.premultiplied.shape[1], self.premultiplied.shape[0]


def prepare_texture(texture_bgra):
    if isinstance(texture_bgra, PlaneTexture):
        return texture_bgra
    texture = np.asarray(texture_bgra)
    if texture.ndim != 3 or texture.shape[2] != 4:
        raise ValueError("texture must be an HxWx4 straight-alpha BGRA image")
    _size((texture.shape[1], texture.shape[0]))
    scale = 255.0 if texture.dtype == np.uint8 else 1.0
    value = texture.astype(np.float32) / scale
    if not np.isfinite(value).all() or value.min() < 0.0 or value.max() > 1.0:
        raise ValueError("float textures must lie in 0..1")
    value[..., :3] *= value[..., 3:4]
    return PlaneTexture(np.ascontiguousarray(value))


def _warp_prepared(texture, homography, output_wh, camera25=None):
    warped = cv2.warpPerspective(
        texture.premultiplied, np.asarray(homography, np.float64),
        _size(output_wh), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0, 0.0))
    if camera25 is not None:
        # A planar homography alone also maps the plane behind the camera.
        # Reject those rays, including near-horizon rays in low broadcast views.
        c = _camera(camera25)
        r = Rotation.from_rotvec(c[3:6]).as_matrix()
        w, h = _size(output_wh)
        fx, fy = np.exp(c[6:8])
        xx = np.arange(w, dtype=np.float64)[None, :]
        yy = np.arange(h, dtype=np.float64)[:, None]
        ny = (yy - c[9]) / fy
        world_ray_z = (r[0, 2] * (xx - c[8]) / fx
                       + (r[1, 2] - r[0, 2] * c[10] / fx) * ny + r[2, 2])
        valid = ((-c[2] * world_ray_z > 0.0)
                 & (abs(c[2]) > .05 * np.abs(world_ray_z)))
        warped *= valid[..., None]
    return warped


def warp_plane(texture_bgra, camera25, bounds_xy, output_wh=(1920,1080)):
    """Warp full-plane layer to uint8 straight-alpha BGRA.

    Premultiplied interpolation prevents fringes at transparent edges.
    The input may be prepare_texture(...) for reuse.
    """
    texture = prepare_texture(texture_bgra)
    h = plane_homography(camera25, bounds_xy, texture.wh)
    warped = _warp_prepared(texture, h, output_wh, camera25)
    alpha = warped[..., 3:4]
    color = np.divide(warped[..., :3], alpha,
                      out=np.zeros_like(warped[..., :3]), where=alpha > 1e-7)
    return np.clip(np.concatenate((color,alpha),axis=2)*255.0+.5,0,255).astype(np.uint8)


class OutroTransform:
    """Unfold the same ice texture from broadcast perspective to top view.

    Uses the shortest SO(3) path to the nearest horizontal top-view direction.
    The optical-axis ice point moves to rink center as camera distance changes
    smoothly: one pan/tilt/pullback, without going around the ice backside.
    No text, statistics, gauge or inset video is drawn.
    """
    def __init__(self,last_camera25,bounds_xy,output_wh=(1920,1080),
                 turn_seconds=3.0,fade_seconds=.55,hold_seconds=1.0,visibility_gain=2.3):
        self.last_camera25 = _camera(last_camera25).copy()
        self.bounds_xy = _bounds(bounds_xy).copy()
        self.output_wh = _size(output_wh)
        self.turn_seconds = float(turn_seconds)
        self.fade_seconds = float(fade_seconds)
        self.hold_seconds = float(hold_seconds)
        self.visibility_gain = float(visibility_gain)
        if not np.isfinite(self.visibility_gain) or self.visibility_gain < 1.0:
            raise ValueError("visibility_gain must be finite and >=1")
        if (not np.isfinite([self.turn_seconds,self.fade_seconds,self.hold_seconds]).all()
                or self.turn_seconds<=0 or self.fade_seconds<=0 or self.hold_seconds<0):
            raise ValueError("turn/fade must be positive and hold nonnegative")
        self.total_seconds = max(self.turn_seconds,self.fade_seconds)+self.hold_seconds
        self.fit_fraction = .82
        self._r0 = Rotation.from_rotvec(self.last_camera25[3:6])
        candidates = [Rotation.from_matrix(np.diag([s,-s,-1.0])) for s in (1.0,-1.0)]
        end_rot = min(candidates,key=lambda r:(r*self._r0.inv()).magnitude())
        self.rotation_radians = float((end_rot*self._r0.inv()).magnitude())
        self._slerp = Slerp([0.,1.],Rotation.from_quat([
            self._r0.as_quat(),end_rot.as_quat()]))
        b,c = self.bounds_xy,self.last_camera25
        self._center = np.array([(b[0]+b[2])*.5,(b[1]+b[3])*.5,0.])
        forward = self._r0.as_matrix()[2]
        axis_d = -c[2]/forward[2] if abs(forward[2])>1e-9 else -1.
        self._distance0 = float(axis_d if axis_d>1e-6 else np.linalg.norm(c[:3]-self._center))
        if self._distance0<1e-6:
            raise ValueError("last camera is coincident with the ice center")
        self._anchor0 = c[:3]+forward*self._distance0
        w,h = self.output_wh
        focal = float(np.exp(np.mean(c[6:8])))
        self._distance1 = max(focal*(b[2]-b[0])/((w-1)*self.fit_fraction),
                              focal*(b[3]-b[1])/((h-1)*self.fit_fraction))
        end = c.copy()
        end[:3] = self._center+[0.,0.,self._distance1]
        end[3:6] = end_rot.as_rotvec()
        end[6:8] = np.log(focal)
        end[8:11] = [(w-1)*.5,(h-1)*.5,0.]
        end[11:25] = 0.
        self.end_camera25 = end
        yy,xx = np.mgrid[0:h,0:w].astype(np.float32)
        vignette = np.clip(1.-(((xx/(w-1)-.5)/.8)**2+((yy/(h-1)-.5)/.85)**2),0.,1.)
        self._background = np.empty((h,w,3),np.float32)
        for channel,(low,high) in enumerate(((13.,26.),(9.,17.),(7.,12.))):
            self._background[...,channel] = low+(high-low)*vignette

    def camera_at(self,elapsed_seconds):
        """Return the presentation camera, exact original at elapsed<=0."""
        elapsed = float(elapsed_seconds)
        if not np.isfinite(elapsed):
            raise ValueError("elapsed_seconds must be finite")
        if elapsed<=0:
            return self.last_camera25.copy()
        if elapsed>=self.turn_seconds:
            return self.end_camera25.copy()
        u = _ease(elapsed/self.turn_seconds)
        rotation = self._slerp(u)
        forward = rotation.as_matrix()[2]
        anchor = self._anchor0*(1.-u)+self._center*u
        distance = np.exp((1.-u)*np.log(self._distance0)+u*np.log(self._distance1))
        camera = self.last_camera25*(1.-u)+self.end_camera25*u
        camera[:3] = anchor-forward*distance
        camera[3:6] = rotation.as_rotvec()
        camera[11:25] = 0.
        return camera

    def homography_at(self,elapsed_seconds,texture_wh):
        return plane_homography(self.camera_at(elapsed_seconds),self.bounds_xy,texture_wh)

    def precompute(self,elapsed_seconds:Sequence[float],texture_wh):
        """Precompute small H matrices without copying frames or textures."""
        return [self.homography_at(t,texture_wh) for t in elapsed_seconds]

    def frame(self,last_native_bgr,texture_bgra,elapsed_seconds,*,
              homography=None,foreground_mask=None):
        """Render BGR. Last native frame has no trajectory/heat overlay.

        Optional output-sized foreground_mask is 1/255 on athlete, zero on
        ice. Athlete occlusion fades with broadcast, revealing the full plane.
        Prepared texture and precomputed H may be reused for all ending frames.
        """
        native = np.asarray(last_native_bgr)
        w,h = self.output_wh
        if native.shape!=(h,w,3) or native.dtype!=np.uint8:
            raise ValueError("last_native_bgr must be output-sized uint8 BGR")
        texture = prepare_texture(texture_bgra)
        hmat = (self.homography_at(elapsed_seconds,texture.wh) if homography is None
                else np.asarray(homography,np.float64))
        warped = _warp_prepared(texture,hmat,self.output_wh,self.camera_at(elapsed_seconds))
        fade = _ease(float(elapsed_seconds)/self.fade_seconds)
        if fade > 0.0 and self.visibility_gain > 1.0:
            # The broadcast needs subtle overlays; the dark ending needs more
            # visibility. Raise opacity smoothly without changing straight RGB.
            # Keeping this branch inactive at t=0 preserves byte-exact continuity.
            alpha = warped[...,3:4].copy()
            target_alpha = 1.0 - np.power(1.0 - alpha, self.visibility_gain)
            target_alpha = np.maximum(target_alpha, alpha)
            new_alpha = alpha + fade * (target_alpha - alpha)
            ratio = np.divide(new_alpha, alpha, out=np.ones_like(alpha),
                              where=alpha > 0.0)
            warped *= ratio
            warped[...,3:4] = new_alpha
        if foreground_mask is not None:
            mask = np.asarray(foreground_mask)
            if mask.shape==(h,w,1):
                mask = mask[...,0]
            if mask.shape!=(h,w):
                raise ValueError("foreground_mask must match output height/width")
            mask = mask.astype(np.float32)/(255. if mask.dtype==np.uint8 else 1.)
            if not np.isfinite(mask).all() or mask.min()<0 or mask.max()>1:
                raise ValueError("mask must lie in 0..1 or uint8 0..255")
            visibility = 1.-mask[...,None]*(1.-fade)
            warped *= visibility
        base = native.astype(np.float32)*(1.-fade)+self._background*fade
        result = warped[...,:3]*255.+base*(1.-warped[...,3:4])
        return np.clip(result+.5,0,255).astype(np.uint8)

    def selfcheck(self,texture_wh=(1600,800)):
        """Check start continuity, top-view fit and path positive depth."""
        b = self.bounds_xy
        corners = np.array([[b[0],b[1],0.],[b[2],b[1],0.],
                            [b[2],b[3],0.],[b[0],b[3],0.]])
        h0 = plane_homography(self.last_camera25,b,texture_wh)
        error = float(np.max(np.abs(h0-self.homography_at(0.,texture_wh))))
        c = self.end_camera25
        r = Rotation.from_rotvec(c[3:6]).as_matrix()
        xyz = (corners-c[:3])@r.T
        uv = xyz[:,:2]/xyz[:,2:3]*np.exp(c[6:8])+c[8:10]
        w,h = self.output_wh
        fits = bool(((uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)).all())
        depths = []
        for t in np.linspace(0.,self.turn_seconds,101):
            camera = self.camera_at(t)
            rot = Rotation.from_rotvec(camera[3:6]).as_matrix()
            depths.append(float(np.min((corners-camera[:3])@rot[2])))
        report = {
            "start_homography_max_abs_error":error,
            "end_rink_bounds_inside_frame":fits,
            "end_screen_bounds_xyxy":[*uv.min(axis=0).tolist(),*uv.max(axis=0).tolist()],
            "end_longest_fill_fraction":float(max(np.ptp(uv[:,0])/(w-1),np.ptp(uv[:,1])/(h-1))),
            "end_view_is_downward":bool(np.allclose(r[2],[0.,0.,-1.],atol=1e-10)),
            "shortest_turn_degrees":float(np.degrees(self.rotation_radians)),
            "path_min_corner_depth":min(depths)}
        if error>1e-12 or not fits or not report["end_view_is_downward"]:
            raise AssertionError(report)
        return report


def selfcheck():
    """Small synthetic check, including first-frame compositing continuity."""
    camera = np.zeros(25,np.float64)
    camera[:3] = [0.,-110.,85.]
    forward = -camera[:3]/np.linalg.norm(camera[:3])
    right = np.array([1.,0.,0.])
    down = np.cross(forward,right)
    camera[3:6] = Rotation.from_matrix(np.stack((right,down,forward))).as_rotvec()
    camera[6:8] = np.log(700.)
    camera[8:10] = [319.5,179.5]
    outro = OutroTransform(camera,[-97.,-49.,97.,49.],(640,360))
    report = outro.selfcheck((400,200))
    image = np.full((360,640,3),150,np.uint8)
    texture = np.zeros((200,400,4),np.uint8)
    texture[60:140,80:320] = [255,130,20,170]
    warped = warp_plane(texture,camera,outro.bounds_xy,(640,360))
    alpha = warped[...,3:4].astype(np.float32)/255.
    reference = np.clip(warped[...,:3]*alpha+image*(1.-alpha)+.5,0,255).astype(np.uint8)
    rendered = outro.frame(image,prepare_texture(texture),0.)
    report["start_composite_max_byte_error"] = int(np.abs(reference.astype(int)-rendered.astype(int)).max())
    if report["start_composite_max_byte_error"]>1:
        raise AssertionError(report)
    return report


if __name__=="__main__":
    print(json.dumps(selfcheck(),indent=2))
