import PySimpleGUI as sg
import cv2
import numpy as np
from scanner import scan
from datetime import datetime


def reduce_bit_depth(img):
    height, width = img.shape[0], img.shape[1]
    for row in range(0, height):
        for col in range(0, width):
            img[row][col] = img[row][col] & 0b11110000
    return img


def get_mode(img):
    unq, count = np.unique(img.reshape(img.shape[0] * img.shape[1], 3), axis=0, return_counts=True)
    return unq[count.argmax()]


def mouseRGB(event, image, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
        bgr_pixel = image[y][x]
        temp1 = np.uint8([[[bgr_pixel[0], bgr_pixel[1], bgr_pixel[2]]]])
        temp2 = cv2.cvtColor(temp1, cv2.COLOR_BGR2HSV)
        hsv_pixel = temp2[0][0]
        print("hsv pixel: ", hsv_pixel)
        print("bgr pixel: ", bgr_pixel)
        print("Coordinates of pixel: X: ", x, "Y: ", y)


def sample_pixels(img, sample_fraction):
    pixels = img.reshape((-1, 3))
    num_pixels = pixels.shape[0]
    num_samples = int(num_pixels * sample_fraction)
    idx = np.arange(num_pixels)
    np.random.shuffle(idx)
    temp = pixels[idx[:num_samples]]
    return temp.reshape(img.shape)


def enhance_roi(img):
    height, width = img.shape[0], img.shape[1]
    # copy_of_image = img.copy()
    copy_of_image = sample_pixels(img, 5)
    image_with_reduced_bit_depth = reduce_bit_depth(copy_of_image)
    background_color = get_mode(image_with_reduced_bit_depth)
    hsv_image_with_reduced_bit_depth = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    temp1 = np.uint8([[[background_color[0], background_color[1], background_color[2]]]])
    temp2 = cv2.cvtColor(temp1, cv2.COLOR_BGR2HSV)
    hsv_background_color = temp2[0][0]

    result = np.zeros((height, width, 3), np.uint8)
    result.fill(255)
    for row in range(0, height):
        for col in range(0, width):
            value = hsv_image_with_reduced_bit_depth[row][col][2]
            saturation = hsv_image_with_reduced_bit_depth[row][col][1]
            # if not (abs(int(value) - int(hsv_background_color[2])) > 20 or abs(
            #         int(saturation) - int(hsv_background_color[1])) > 19):
            if abs(int(np.round(value)) - int(np.round(hsv_background_color[2]))) > 20 or \
                    abs(int(np.round(saturation)) - int(np.round(hsv_background_color[1]))) > 19:
                result[row][col] = img[row][col]

    # print("Background color in RGB: ", cxz.shape)
    # print("Background color in HSV: ", hsv_background_color)
    # print("Number of pixels of background's color: ", count)
    # print("Image size: ", width * height)
    # print("Percentage of background color in image: ", (count * 100) / (width * height))

    # cv2.imshow("initial image", image)
    # cv2.imshow("denoised image", denoised_image)
    # cv2.imshow("RGB image with reduced bit depth", image_with_reduced_bit_depth)
    # cv2.imshow("HSV image with reduced bit depth", hsv_image_with_reduced_bit_depth)
    # cv2.imshow("Result image", result)
    # cv2.waitKey(0)
    # return result
    return result


def enhance_image(img):
    result = np.zeros(img.shape, img.dtype)
    denoised_image = cv2.fastNlMeansDenoisingColored(img, None, 3, 10, 7, 21)
    height, width = img.shape[0], img.shape[1]
    cols_incr = [0, int(width / 5), int(2 * width / 5), int(3 * width / 5), int(4 * width / 5), width]
    rows_incr = [0, int(height / 4), int(height / 2), int(3 * height / 4), height]
    # rows_incr = [0, int(height / 8), int(height / 4), int(3 * height / 8), int(height / 2), int(5 * height / 8), int(3 * height / 4), int(7 * height / 8), height]
    for i in range(0, len(rows_incr) - 1):
        for j in range(0, len(cols_incr) - 1):
            rows_s = rows_incr[i]
            rows_e = rows_incr[i + 1]
            cols_s = cols_incr[j]
            cols_e = cols_incr[j + 1]
            img_temp = denoised_image[rows_s:rows_e, cols_s:cols_e]
            if img_temp.any():
                roi = enhance_roi(img_temp)
                result[rows_s:rows_e, cols_s:cols_e] = roi
                # cv2.rectangle(maskImg, (y, x), (y + cols_incr - 1, x + rows_incr - 1), (0, 0, 255), 1)
    return result


def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


ok = False
if __name__ == "__main__":

    sg.theme('DarkTeal1')

    col1 = [
        [sg.Button('Start', size=(29, 1), pad=((5, 5), (5, 1)), font='Helvetica 14')],
        [sg.Button('Save', size=(29, 1), pad=((5, 5), (0, 1)), font='Helvetica 14')],
        [sg.Button('Stop', size=(29, 1), pad=((5, 5), (0, 1)), font='Helvetica 14')],
        [sg.Text(text=' ' * 45)],
        [sg.Radio(key='Alb-negru', group_id="RADIO1", default=True, text="Black & White", font='Helvetica 14'),
            sg.Radio(key='Original', group_id="RADIO1", text="Original", font='Helvetica 14', pad=((80, 0), (0, 0)))],
        [sg.Text(key="contrast_separator", text='_' * 45)],
        [sg.Text(text=' ' * 45)],
        [sg.Text('Contrast:', key='contrast_label', font='Helvetica 14', pad=((0, 0), (0, 1)), size=(14, 1))],
        [sg.Slider(key="contrast_slider", range=(1, 100), default_value=1, size=(29, 15), orientation='horizontal', font=('Helvetica', 12))],
        [sg.Text(text=' ' * 45)],
        [sg.Text('Brightness:', key='bright_label', font='Helvetica 14', justification="left")],
        [sg.Slider(key="bright_slider", range=(1, 100), default_value=1, orientation='horizontal', size=(29, 15), font=('Helvetica', 12))],
        # [sg.Text(text=' ' * 45)],
        # [sg.Text(text=' ' * 45)],
        [sg.Text(text=' ' * 45, font=('Helvetica', 1))]
        # [sg.Image(filename='', key='-empty-')],
    ]

    col3 = [
        [sg.Image(filename='', key='-image-')]
    ]

    col = sg.Col([
        [sg.Frame(title="", layout=col1, pad=((5, 0), (2, 1)))],
        [sg.Frame(title="", layout=col3, pad=((5, 0), (80, 0)))]
    ])

    col2 = [[sg.Image(filename='', key='-scan-')]]

    layout = [[col, sg.Frame(title="", layout=col2)]]
    window = sg.Window('Real time document scanner', layout, no_titlebar=False, location=(0, 0))

    image_elem = window['-image-']
    scan_elem = window['-scan-']
    # empty_elem = window['-empty-']

    show_sliders = False
    cap = cv2.VideoCapture(1)
    while True:
        button, values = window.read(timeout=0)

        if button is 'Stop' or values is None:
            # sys.exit(0)
            ok = False
        elif button is 'Start':
            ok = True

        if values['Original']:
            window.Element('contrast_slider').Update(visible=True)
            window.Element('bright_slider').Update(visible=True)
            window.Element('contrast_label').Update(visible=True)
            window.Element('bright_label').Update(visible=True)
            window.Element('contrast_separator').Update(visible=True)
        else:
            window.Element('contrast_slider').Update(visible=False)
            window.Element('bright_slider').Update(visible=False)
            window.Element('contrast_label').Update(visible=False)
            window.Element('bright_label').Update(visible=False)
            window.Element('contrast_separator').Update(visible=False)

        ret, frame = cap.read()

        temp = frame.copy()
        width = int(frame.shape[1] * 0.5)
        height = int(frame.shape[0] * 0.5)

        black_image = np.zeros((735, 1050, 3), np.uint8)
        black_image[:, :, :] = (98, 99, 57)
        if ok:
            if values['Original']:
                contrast = values['contrast_slider']
                bright = values['bright_slider']
                image = scan(frame, 1, contrast, bright)
            elif values['Alb-negru']:
                try:
                    image = scan(frame)
                except Exception:
                    sg.SystemTray.notify('Error', icon=sg.SYSTEM_TRAY_MESSAGE_ICON_CRITICAL,
                                         message='Something went wrong')
            if button is 'Save':
                dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                path = "Scans\\" + dt_string + ".png"
                print(path)
                status = cv2.imwrite(path, image)
                print("Image written to file-system : ", status)
                sg.SystemTray.notify('Success', 'Scan saved')
            imgbytes = cv2.imencode('.png', image)[1].tobytes()  # ditto
            scan_elem.update(data=imgbytes)
        else:
            imgbytes = cv2.imencode('.png', black_image)[1].tobytes()  # ditto
            scan_elem.update(data=imgbytes)
            if button is 'Save':
                sg.SystemTray.notify('Error', icon=sg.SYSTEM_TRAY_MESSAGE_ICON_CRITICAL, message='Cannot save scanned image')

        dim = (width, height)

        resized = cv2.resize(temp, dim, interpolation=cv2.INTER_AREA)
        imgbytes = cv2.imencode('.png', resized)[1].tobytes()  # ditto
        image_elem.update(data=imgbytes)


# def change_brightness(img, value=30):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     v = cv2.add(v, value)
#     v[v > 255] = 255
#     v[v < 0] = 0
#     final_hsv = cv2.merge((h, s, v))
#     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     return img


# if __name__ == "__main__":
#     image = cv2.imread("TrainedModel/Capture2.JPG")
#     cv2.imshow("image", image)
#     e1 = cv2.getTickCount()
#     # enhanced_image = enhance_image(image)
#     # enhanced_image, alpha, beta = automatic_brightness_and_contrast(image)
#
#     lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     lab_planes = cv2.split(lab_img)
#     lab_img[:, :, 0] = 0
#
#     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     # dst = clahe.apply(lab_planes[0])
#     # cv2.copyTo(dst, lab_planes[0])
#     # final = cv2.merge(lab_planes, lab_img)
#     # image_clahe = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)
#     # cv2.imshow("CLAHE", image_clahe)
#
#     cal = cv2.convertScaleAbs(image, 1, 1.6)
#     cv2.imshow("contrast_fixed", cal)
#     # result = change_brightness(image, 90)
#
#     lookUpTable = np.empty((1, 256), np.uint8)
#     for i in range(256):
#         lookUpTable[0, i] = np.clip(pow(i / 255.0, 0.8) * 255.0, 0, 255)
#     res = cv2.LUT(image, lookUpTable)
#     cv2.imshow("res", res)
#
#     e2 = cv2.getTickCount()
#     time = (e2 - e1) / cv2.getTickFrequency()
#     print(time)
#     cv2.waitKey(0)
