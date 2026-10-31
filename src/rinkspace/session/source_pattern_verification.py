"""Source-image print agreement after removing slowly varying brightness.

These checks neither establish physical identity alone nor create observations.
The warp and validity mask match the registered-print caller. Correlation policy
is explicit; weak texture remains pending rather than passing a normalized score.
"""
from __future__ import annotations

import cv2
import numpy as np


def _roi(roi, shape):
    a = np.asarray(roi)
    if a.shape != (4,) or a.dtype.kind not in 'iu':
        raise ValueError('An inclusive integer reference ROI is required')
    x0, y0, x1, y1 = map(int, a)
    if not (0 <= x0 <= x1 < shape[1] and 0 <= y0 <= y1 < shape[0]):
        raise ValueError('Reference ROI is outside the original source image')
    return x0, y0, x1, y1


def _homography(value):
    h = np.asarray(value, float)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError('A finite source-image homography is required')
    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError as error:
        raise ValueError('Source-image homography is singular') from error
    return h, inverse


def _correlation(a, b, valid):
    left, right = a[valid].astype(float).ravel(), b[valid].astype(float).ravel()
    if not len(left):
        return None
    left -= left.mean(); right -= right.mean()
    denominator = float(np.linalg.norm(left)*np.linalg.norm(right))
    if denominator <= 1e-12:
        return None
    return float(np.clip(left@right/denominator, -1., 1.))


def source_pattern_scores(image, exclusion, meta, reference, h, *,
                          fine_sigma=1.5, background_sigma=8., minimum_valid_pixels=500):
    """Measure whole-gray, highpass and signed-gradient NCC on the same pixels.

    ``image`` and ``reference['source_image']`` are full native BGR images;
    ``reference['reference_mask']`` is the original full-image search mask.
    No correlation cutoff or physical-instance assignment is inferred here.
    """
    image, source = np.asarray(image), np.asarray(reference['source_image'])
    mask = np.asarray(exclusion, bool)
    reference_mask = np.asarray(reference['reference_mask'], bool)
    for rgb in (image, source):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Pattern checks require original uint8 native BGR images')
    if mask.shape != image.shape[:2] or reference_mask.shape != source.shape[:2]:
        raise ValueError('Pattern masks must match their original image domains')
    if (not np.isfinite([fine_sigma, background_sigma]).all() or fine_sigma <= 0
            or background_sigma <= fine_sigma):
        raise ValueError('Background blur must be wider than the positive fine blur')
    if isinstance(minimum_valid_pixels, bool) or int(minimum_valid_pixels) != minimum_valid_pixels or minimum_valid_pixels < 1:
        raise ValueError('minimum_valid_pixels must be a positive integer')
    x0, y0, x1, y1 = _roi(meta['roi_xyxy_inclusive'], source.shape[:2])
    _, inverse = _homography(h)
    shift = np.array([[1., 0., -x0], [0., 1., -y0], [0., 0., 1.]])
    transform = shift@inverse
    size = (x1-x0+1, y1-y0+1)
    warped = cv2.warpPerspective(image, transform, size, flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective((~mask).astype('uint8'), transform, size, flags=cv2.INTER_NEAREST) > 0
    reference_valid = reference_mask[y0:y1+1, x0:x1+1]
    valid &= reference_valid
    a = cv2.cvtColor(source[y0:y1+1, x0:x1+1], cv2.COLOR_BGR2GRAY).astype(float)
    b = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(float)
    af = cv2.GaussianBlur(a, (0, 0), fine_sigma)
    bf = cv2.GaussianBlur(b, (0, 0), fine_sigma)
    ah = af-cv2.GaussianBlur(a, (0, 0), background_sigma)
    bh = bf-cv2.GaussianBlur(b, (0, 0), background_sigma)
    ga = np.stack([cv2.Sobel(af, cv2.CV_64F, 1, 0, ksize=3),
                   cv2.Sobel(af, cv2.CV_64F, 0, 1, ksize=3)], axis=-1)
    gb = np.stack([cv2.Sobel(bf, cv2.CV_64F, 1, 0, ksize=3),
                   cv2.Sobel(bf, cv2.CV_64F, 0, 1, ksize=3)], axis=-1)
    count, total = int(valid.sum()), int(reference_valid.sum())
    def rms(array):
        return float(np.std(array[valid])) if count else None
    return dict(status='scores_only' if count >= minimum_valid_pixels else 'insufficient_current_pattern_overlap_pending',
        valid_pixels=count, available_reference_pixels=total,
        valid_reference_fraction=count/max(1, total), minimum_valid_pixels=int(minimum_valid_pixels),
        whole_visible_pattern_correlation=_correlation(af, bf, valid),
        highpass_pattern_correlation=_correlation(ah, bh, valid),
        signed_gradient_pattern_correlation=_correlation(ga, gb, valid),
        gradient_magnitude_correlation=_correlation(np.linalg.norm(ga, axis=-1), np.linalg.norm(gb, axis=-1), valid),
        reference_highpass_rms=rms(ah), current_highpass_rms=rms(bh),
        reference_gray_rms=rms(af), current_gray_rms=rms(bf),
        fine_sigma=float(fine_sigma), background_sigma=float(background_sigma),
        background_removal='Gaussian(fine_sigma)-Gaussian(background_sigma); slowly varying brightness only',
        warp_and_validity='same reference ROI inverse homography and native exclusion intersection',
        physical_identity_verified_by_this_score=False, supported=None, fit_eligible=False,
        source_pixels_or_observations_generated=False)


def check_source_pattern(image, exclusion, meta, reference, h, *, minimum_highpass_ncc,
                         minimum_texture_rms, minimum_signed_gradient_ncc=None,
                         minimum_valid_pixels=500, fine_sigma=1.5, background_sigma=8.):
    """Apply a caller-owned pattern policy; unresolved texture always stays pending.

    Cutoffs are deliberately required, rather than chosen from one false match.
    A pass is pattern evidence only; the caller still owns instance identity,
    overlap conflicts and observation eligibility.
    """
    thresholds = [minimum_highpass_ncc, minimum_texture_rms]
    if minimum_signed_gradient_ncc is not None: thresholds.append(minimum_signed_gradient_ncc)
    if (not np.isfinite(thresholds).all() or not -1 <= minimum_highpass_ncc <= 1
            or minimum_texture_rms <= 0
            or (minimum_signed_gradient_ncc is not None and not -1 <= minimum_signed_gradient_ncc <= 1)):
        raise ValueError('Explicit finite correlation and positive source-texture thresholds required')
    result = source_pattern_scores(image, exclusion, meta, reference, h,
        minimum_valid_pixels=minimum_valid_pixels, fine_sigma=fine_sigma, background_sigma=background_sigma)
    result.update(supported=False, policy=dict(minimum_highpass_ncc=float(minimum_highpass_ncc),
        minimum_texture_rms=float(minimum_texture_rms),
        minimum_signed_gradient_ncc=minimum_signed_gradient_ncc))
    if result['valid_pixels'] < minimum_valid_pixels:
        return result
    texture = [result['reference_highpass_rms'], result['current_highpass_rms']]
    if min(texture) < minimum_texture_rms or result['highpass_pattern_correlation'] is None:
        result.update(status='weak_texture_pending', reason='insufficient_background_removed_source_texture')
        return result
    supported = result['highpass_pattern_correlation'] >= minimum_highpass_ncc
    if minimum_signed_gradient_ncc is not None:
        supported &= (result['signed_gradient_pattern_correlation'] is not None
                      and result['signed_gradient_pattern_correlation'] >= minimum_signed_gradient_ncc)
    result.update(supported=bool(supported),
        status='source_pattern_supported' if supported else 'source_pattern_agreement_pending')
    if not supported:
        result['reason'] = 'low_pattern_agreement_can_be_mismatch_or_unresolved_blur'
    return result


def warped_reference_support(reference_mask, roi_xyxy_inclusive, h, image_hw, exclusion=None):
    """Rasterize the actual usable reference ROI into current native source pixels."""
    reference_mask = np.asarray(reference_mask, bool)
    if reference_mask.ndim != 2: raise ValueError('Reference support must be a source-image mask')
    x0, y0, x1, y1 = _roi(roi_xyxy_inclusive, reference_mask.shape)
    height, width = map(int, image_hw)
    if min(height, width) < 1: raise ValueError('Current source dimensions must be positive')
    h, _ = _homography(h)
    shift = np.array([[1., 0., x0], [0., 1., y0], [0., 0., 1.]])
    support = cv2.warpPerspective(reference_mask[y0:y1+1, x0:x1+1].astype('uint8'),
        h@shift, (width, height), flags=cv2.INTER_NEAREST) > 0
    if exclusion is not None:
        mask = np.asarray(exclusion, bool)
        if mask.shape != support.shape: raise ValueError('Current support/exclusion domains differ')
        support &= ~mask
    return support


def source_pattern_overlap(first_support, second_support):
    """Measure source-pixel overlap, including containment missed by ordinary IoU."""
    first, second = np.asarray(first_support, bool), np.asarray(second_support, bool)
    if first.ndim != 2 or first.shape != second.shape:
        raise ValueError('Competing pattern supports must share the current source-image domain')
    first_area, second_area = int(first.sum()), int(second.sum())
    intersection = int(np.count_nonzero(first & second))
    union = first_area+second_area-intersection
    return dict(first_source_pixels=first_area, second_source_pixels=second_area,
        intersection_source_pixels=intersection, union_source_pixels=union,
        intersection_over_union=intersection/max(1, union),
        intersection_over_smaller=intersection/max(1, min(first_area, second_area)),
        overlap_of_first=intersection/max(1, first_area), overlap_of_second=intersection/max(1, second_area),
        overlap_alone_assigns_no_identity=True)
