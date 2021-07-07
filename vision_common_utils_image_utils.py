import cv2
import math
import numpy as np
try:
    import skfmm
except ImportError as _:
    print("Could not import 'skfmm'. Fast Marching Method related functions will not be usable.")


def create_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def overlay_mask_on_image(image, mask, mask_opacity=0.6, mask_color=(0, 255, 0)):
    if mask.ndim == 3:
        assert mask.shape[2] == 1
        _mask = mask.squeeze(axis=2)
    else:
        _mask = mask
    mask_bgr = np.stack((_mask, _mask, _mask), axis=2)
    masked_image = np.where(mask_bgr > 0, mask_color, image)
    return ((mask_opacity * masked_image) + ((1. - mask_opacity) * image)).astype(np.uint8)


def overlay_heatmap_on_image(image, heatmap, opacity=0.6, cutoff=0.0, colormap=cv2.COLORMAP_JET):
    """
    Overlays a heat-map on top of an image
    :param image: Numpy array (H, W, C)
    :param heatmap: Numpy array (H, W) either of type float in [0, 1], or of type uint8 in [0, 255]
    :param opacity: float in [0, 1]
    :param cutoff: float in [0, 1]. Values smaller than cutoff will not be overlayed.
    :param colormap: color map to use for the heatmap
    :return: Numpy array (H, W, C)
    """
    if heatmap.dtype == np.float32:
        heatmap = (heatmap * 255).astype(np.uint8)
    elif heatmap.dtype != np.uint8:
        raise ValueError("heatmap must be of type float32 or uint8")

    color_heatmap = cv2.applyColorMap(heatmap, colormap)
    masked_image = np.where(heatmap[:, :, None] > int(cutoff * 255.), color_heatmap, image)

    return ((opacity * masked_image) + ((1. - opacity) * image)).astype(np.uint8)


def expand_mask(mask, instance_ids):
    """
    Takes a single channel mask where N instance IDs are encoded into the pixel values and returns an N-channel array
    containing binary masks for each instance IDs
    :param mask: The single channel mask.
    :param instance_ids: Iterable of instance IDs (excluding background)
    :return: N-channel array with binary masks
    """
    if not instance_ids:  # guard against no instance masks
        return np.zeros((0,) + mask.shape, dtype=mask.dtype)
    return np.stack([np.squeeze(mask == i) for i in instance_ids], axis=0)


def bbox_from_mask(mask, order='Y1Y2X1X2', return_none_if_invalid=False):
    reduced_y = np.any(mask, axis=0)
    reduced_x = np.any(mask, axis=1)

    x_min = reduced_y.argmax()
    if x_min == 0 and reduced_y[0] == 0:  # mask is all zeros
        if return_none_if_invalid:
            return None
        else:
            return -1, -1, -1, -1

    x_max = len(reduced_y) - np.flip(reduced_y, 0).argmax()

    y_min = reduced_x.argmax()
    y_max = len(reduced_x) - np.flip(reduced_x, 0).argmax()

    if order == 'Y1Y2X1X2':
        return y_min, y_max, x_min, x_max
    elif order == 'X1X2Y1Y2':
        return x_min, x_max, y_min, y_max
    elif order == 'X1Y1X2Y2':
        return x_min, y_min, x_max, y_max
    elif order == 'Y1X1Y2X2':
        return y_min, x_min, y_max, x_max
    else:
        raise ValueError("Invalid order argument: %s" % order)


def bbox_intersection(bbox_1, bbox_2, return_coords=False):
    ymin_1, ymax_1, xmin_1, xmax_1 = bbox_1
    ymin_2, ymax_2, xmin_2, xmax_2 = bbox_2

    ymin_int = max(ymin_1, ymin_2)
    ymax_int = min(ymax_1, ymax_2)

    xmin_int = max(xmin_1, xmin_2)
    xmax_int = min(xmax_1, xmax_2)

    if ymin_int >= ymax_int or xmin_int >= xmax_int:
        bbox_int = [-1, -1, -1, -1]
        intersection = 0.
    else:
        bbox_int = [ymin_int, ymax_int, xmin_int, xmax_int]
        intersection = float(ymax_int - ymin_int) * float(xmax_int - xmin_int)

    return intersection, bbox_int if return_coords else intersection


def bbox_union(bbox_1, bbox_2, return_coords=False):
    ymin_1, ymax_1, xmin_1, xmax_1 = bbox_1
    ymin_2, ymax_2, xmin_2, xmax_2 = bbox_2

    ymin_union = min(ymin_1, ymin_2)
    ymax_union = max(ymax_1, ymax_2)

    xmin_union = min(xmin_1, xmin_2)
    xmax_union = max(xmax_1, xmax_2)

    bbox_int = [ymin_union, ymax_union, xmin_union, xmax_union]
    union = float(ymax_union - ymin_union) * float(xmax_union - xmin_union)

    return union, bbox_int if return_coords else union


def bbox_iou(bbox_1, bbox_2):
    return bbox_intersection(bbox_1, bbox_2, False) / bbox_union(bbox_1, bbox_2, False)


def draw_bbox_on_image(image, bbox, color=(0, 255, 0), thickness=2, mode="yyxx"):
    # bbox in format [ymin, ymax, xmin, xmax]

    if mode == "xyxy":
        xmin, ymin, xmax, ymax = bbox
    elif mode == "yyxx":
        ymin, ymax, xmin, xmax = bbox
    else:
        raise NotImplementedError()

    assert 0 <= ymin < ymax <= image.shape[0], "(ymin, ymax) = ({}, {}), image height = {}".format(ymin, ymax, image.shape[0])
    assert 0 <= xmin < xmax <= image.shape[1]

    image_copy = np.copy(image)
    cv2.rectangle(cv2.UMat(image_copy), (xmin, ymin), (xmax, ymax), color, thickness)
    return image_copy


def resize_image(image, resize_width, resize_height, interpolation=cv2.INTER_AREA, ar_tolerance=0.1):
    if (image.shape[0], image.shape[1]) == (resize_height, resize_width):
        return image

    ar_orig = float(image.shape[1]) / image.shape[0]  # width / height
    ar_new = float(resize_width / resize_height)

    if abs(ar_orig - ar_new) < ar_tolerance:
        return cv2.resize(image, (resize_width, resize_height), interpolation=interpolation)

    if ar_orig > ar_new:
        # input image is wider than resized dims --> crop along x-axis
        adjusted_width = int(round(ar_new * image.shape[0]))
        crop_start = int(round((image.shape[1] - adjusted_width) / 2.))
        cropped_image = image[:, crop_start:crop_start + adjusted_width]
    else:
        adjusted_height = int(round(image.shape[1] / ar_new))
        crop_start = int(round((image.shape[0] - adjusted_height) / 2.))
        cropped_image = image[crop_start:crop_start + adjusted_height]

    return cv2.resize(cropped_image, (resize_width, resize_height), interpolation=interpolation)


def resize_and_pad(image, new_size=None, fx=0, fy=0, interpolation=cv2.INTER_AREA, multiple_of=32):
    resized = cv2.resize(image, new_size, None, fx, fy, interpolation)
    h, w = resized.shape[:2]
    padded_height = multiple_of * int(math.ceil(h / float(multiple_of)))
    padded_width = multiple_of * int(math.ceil(w / float(multiple_of)))
    pad_bottom = padded_height - h
    pad_right = padded_width - w
    return cv2.copyMakeBorder(resized, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, None, 0)


def resize_min_max(image, min_size, max_size, interpolation=cv2.INTER_LINEAR):
    lower_size = float(min(image.shape[:2]))
    higher_size = float(max(image.shape[:2]))

    scale_factor = min_size / lower_size
    if higher_size * scale_factor > max_size:
        scale_factor = max_size / higher_size

    resized = cv2.resize(image, None, None, fx=scale_factor, fy=scale_factor, interpolation=interpolation)
    # print(resized.shape)
    return resized


def pad_image(image, size_multiple_of=32):
    h, w = image.shape[:2]
    new_h = int(math.ceil(float(h) / size_multiple_of)) * size_multiple_of
    new_w = int(math.ceil(float(w) / size_multiple_of)) * size_multiple_of
    return cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, None, 0)


def compute_grayscale_gradient(image):
    im = image.astype(np.float32)
    gx = np.abs(cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)) / 4.
    gy = np.abs(cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)) / 4.
    return np.sqrt(gx*gx + gy*gy) / 2.


def compute_binary_gradient(image, ksize):
    assert image.ndim == 2
    dilated = cv2.dilate(image, np.ones((ksize, ksize)))
    eroded = cv2.erode(image, np.ones((ksize, ksize)))
    return dilated - eroded


def run_fmm(mask, normalized_distance=True, normalize_directions=True, return_nearest_contour=True):
    if not np.any(mask):
        if return_nearest_contour:
            return np.zeros_like(mask, np.float32), None
        else:
            return np.zeros_like(mask, np.float32)

    phi = np.ones_like(mask, np.float32)
    phi[mask > 0] = 0
    dx = np.ones((2,), np.float32) / np.array(mask.shape[:2], np.float32) if normalized_distance else 1.0
    result = skfmm.distance(phi, dx, order=2, return_nearest_contour=return_nearest_contour)
    if return_nearest_contour:
        distance, nearest_mask_pts = result
        coords = np.indices(mask.shape, np.int).transpose(1, 2, 0)
        directions = nearest_mask_pts - coords

        if normalize_directions:
            h, w = mask.shape
            directions = directions.astype(np.float32)
            directions[:, :, 0] /= float(h)
            directions[:, :, 1] /= float(w)

        return distance, directions

    else:
        return result


def visualize_fmm(mask, distance, directions, normalized_directions=True):
    distance_map = cv2.applyColorMap(cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                                     cv2.COLORMAP_JET)
    mask_grad = compute_grayscale_gradient(mask)
    mask_bgr = np.stack((mask_grad, mask_grad, mask_grad), axis=2)
    distance_map[mask_bgr > 0] = 0

    if normalized_directions:
        directions[:, :, 0] *= mask.shape[0]
        directions[:, :, 1] *= mask.shape[1]
        directions = directions.astype(np.int)

    def mouseCallback(event, x, y, flags, params):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        np_y, np_x = directions[y, x]
        np_y += y
        np_x += x
        print("Click at %d, %d, Nearest point: %d, %d" % (x, y, np_x, np_y))
        cv2.circle(distance_map, (np_x, np_y), 2, (0, 200, 0), -1)
        cv2.imshow('Distance', distance_map)

    cv2.namedWindow('Distance', cv2.WINDOW_NORMAL)
    cv2.imshow('Distance', distance_map)
    cv2.setMouseCallback('Distance', mouseCallback)
    cv2.waitKey()


def imshow(**kwargs):
    for window_name in kwargs:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    for window_name, img in kwargs.items():
        cv2.imshow(window_name, img)
    cv2.waitKey(0)


if __name__ == '__main__':
    from PIL import Image
    im1 = np.array(Image.open('/globalwork/data/youtube-vos/train/Annotations/0c26bc77ac/00000.png'), np.uint8)
    im2 = np.array(Image.open('/globalwork/data/youtube-vos/train/Annotations/0c26bc77ac/00005.png'), np.uint8)
    im1 = (im1 == 1).astype(np.uint8)
    cv2.imwrite('/tmp/mask.png', im1)
    # im2 = (im2 == 1).astype(np.uint8)

    # dist, direc = run_fmm(im1)
    # visualize_fmm(im1, dist, direc)

    # from PIL import Image
    # im = np.array(Image.open('/globalwork/data/youtube-vos/train/CleanedAnnotations/0e9f2785ec/00020.png'), np.uint8) / 2.
    # grad = compute_grayscale_gradient(im)
    # print(grad.min(), grad.max())
    # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    # cv2.imshow('Image', im * 255)
    # cv2.imshow('Mask', grad * 255)
    # cv2.waitKey()
    # parse_colored_instance_mask('/globalwork/data/youtube-vos/train/CleanedAnnotations/0e9f2785ec/00020.png')
