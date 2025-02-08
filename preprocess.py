import cv2, numpy as np, re

def parse_layout(l):
    return l.strip("[]").replace(" ", "").split(",")

def needs_transpose(l):
    return parse_layout(l)[1] == "C"

def get_target_size(shape, l):
    return (shape[3], shape[2]) if parse_layout(l)[1] == "C" else (shape[2], shape[1])

def parse_scale_value(s):
    m = re.search(r'\[(.*)\]', s)
    return float(m.group(1)) if m else None

def preprocess_image(img, layout, target_size, scale=None):
    img = cv2.resize(img, target_size)
    if scale is not None:
        img = img.astype(np.float32) / scale
    img = np.transpose(img, (2, 0, 1)) if needs_transpose(layout) else img
    return np.expand_dims(img, axis=0)

input_shape = [1, 3, 62, 62]  # obtained from input.get_shape()
rt_layout = "[N,C,H,W]"         # obtained from rt_info
target_size = get_target_size(input_shape, rt_layout)
scale_str = "data[255.0]"       # from conversion parameters
scale_value = parse_scale_value(scale_str)
image = cv2.imread("path/to/your/image.jpg")
preprocessed = preprocess_image(image, rt_layout, target_size, scale=scale_value)
print("Preprocessed image shape:", preprocessed.shape)
