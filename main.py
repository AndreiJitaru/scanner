from datetime import datetime
import cv2
import numpy as np
import PySimpleGUI as sG
from scanner import scan


ok = False
if __name__ == "__main__":

    sG.theme('DarkTeal1')

    col1 = [
        [sG.Button('Start', size=(29, 1), pad=((5, 5), (5, 1)), font='Helvetica 14')],
        [sG.Button('Save', size=(29, 1), pad=((5, 5), (0, 1)), font='Helvetica 14')],
        [sG.Button('Stop', size=(29, 1), pad=((5, 5), (0, 1)), font='Helvetica 14')],
        [sG.Text(text=' ' * 45)],
        [sG.Radio(key='Alb-negru', group_id="RADIO1", default=True, text="Black & White", font='Helvetica 14'),
            sG.Radio(key='Original', group_id="RADIO1", text="Original", font='Helvetica 14', pad=((80, 0), (0, 0)))],
        [sG.Text(key="contrast_separator", text='_' * 45)],
        [sG.Text(text=' ' * 45)],
        [sG.Text('Contrast:', key='contrast_label', font='Helvetica 14', pad=((0, 0), (0, 1)), size=(14, 1))],
        [sG.Slider(key="contrast_slider", range=(1, 100), default_value=1, size=(29, 15), orientation='horizontal', font=('Helvetica', 12))],
        [sG.Text(text=' ' * 45)],
        [sG.Text('Brightness:', key='bright_label', font='Helvetica 14', justification="left")],
        [sG.Slider(key="bright_slider", range=(1, 100), default_value=1, orientation='horizontal', size=(29, 15), font=('Helvetica', 12))],
        # [sg.Text(text=' ' * 45)],
        # [sg.Text(text=' ' * 45)],
        [sG.Text(text=' ' * 45, font=('Helvetica', 1))]
        # [sg.Image(filename='', key='-empty-')],
    ]

    col3 = [
        [sG.Image(filename='', key='-image-')]
    ]

    col = sG.Col([
        [sG.Frame(title="", layout=col1, pad=((5, 0), (2, 1)))],
        [sG.Frame(title="", layout=col3, pad=((5, 0), (80, 0)))]
    ])

    col2 = [[sG.Image(filename='', key='-scan-')]]

    layout = [[col, sG.Frame(title="", layout=col2)]]
    window = sG.Window('Real time document scanner', layout, no_titlebar=False, location=(0, 0))

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
                    sG.SystemTray.notify('Error', icon=sG.SYSTEM_TRAY_MESSAGE_ICON_CRITICAL,
                                         message='Something went wrong')
            if button is 'Save':
                dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                path = "Scans\\" + dt_string + ".png"
                print(path)
                status = cv2.imwrite(path, image)
                print("Image written to file-system : ", status)
                sG.SystemTray.notify('Success', 'Scan saved')
            imgbytes = cv2.imencode('.png', image)[1].tobytes()  # ditto
            scan_elem.update(data=imgbytes)
        else:
            imgbytes = cv2.imencode('.png', black_image)[1].tobytes()  # ditto
            scan_elem.update(data=imgbytes)
            if button is 'Save':
                sG.SystemTray.notify('Error', icon=sG.SYSTEM_TRAY_MESSAGE_ICON_CRITICAL, message='Cannot save scanned image')

        dim = (width, height)

        resized = cv2.resize(temp, dim, interpolation=cv2.INTER_AREA)
        imgbytes = cv2.imencode('.png', resized)[1].tobytes()  # ditto
        image_elem.update(data=imgbytes)


# if __name__ == "__main__":
#
#     cap = cv2.VideoCapture(1)
#     start_scan = False
#     frame_counter = 0
#     while True:
#         ret, frame = cap.read()
#         if cv2.waitKey(1) & 0xFF == ord('e'):
#             start_scan = True
#             cv2.destroyAllWindows()
#         if start_scan:
#             cv2.imshow("initial_image", frame)
#             image = scan(frame)
#             # if frame_counter == 1:
#             #     print(frame_counter)
#             # if frame_counter % 5 == 0:
#             #     print(frame_counter)
#             frame_counter += 1
#             print("Frame: ", frame_counter)
#             path = "Scans\\frame" + str(frame_counter) + ".png"
#             status = cv2.imwrite(path, frame)
#             path = path = "Scans\\scan" + str(frame_counter) + ".png"
#             status = cv2.imwrite(path, image)
#
#             # cv2.waitKey(0)
#
#             cv2.imshow("scanned_image", image)
#         else:
#             cv2.imshow("initial_image", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

    # image = cv2.imread("TrainedModel/13.jpg")
    # graph, x, y = load_graph(arguments.cornerModel, "Corner/inputTensor", "Corner/outputTensor")
    # graphCorners, xCorners, yCorners = load_graph(arguments.documentModel, "Input/inputTensor", "FCLayers/outputTensor")
    # initial_image_corners = find_corners_in_image(image, graph, graphCorners, xCorners, yCorners, x, y, arguments)
    #
    # projection_matrix = compute_projection_matrix(image, initial_image_corners)
    # initial_image_corners_as_arrays = convert_to_list_of_arrays(initial_image_corners)
    # ipm_image, initial_ipm_corners_as_arrays = \
    #     compute_ipm_fast_numba(image, projection_matrix, np.asarray(initial_image_corners_as_arrays))
    # initial_ipm_corners = convert_to_list_of_tuples(initial_ipm_corners_as_arrays)
    # initial_ipm_corners.reverse()
    #
    # for a in range(0, len(initial_image_corners)):
    #     point_a = initial_image_corners[a % 4]
    #     point_b = initial_image_corners[(a + 1) % 4]
    #     cv2.line(image, point_a, point_b, (0, 255, 255), 2)
    #     if a == 0:
    #         cv2.circle(image, point_a, 5, (0, 0, 255), -1)
    #         temp = point_a
    #     elif a == 1:
    #         cv2.circle(image, point_a, 5, (0, 255, 0), -1)
    #     elif a == 2:
    #         cv2.circle(image, point_a, 5, (255, 0, 0), -1)
    #     else:
    #         cv2.circle(image, point_a, 5, (255, 0, 255), -1)
    #         cv2.circle(image, temp, 5, (0, 0, 255), -1)
    #
    # for a in range(0, len(initial_ipm_corners)):
    #     point_a = initial_ipm_corners[a % 4]
    #     point_b = initial_ipm_corners[(a + 1) % 4]
    #     cv2.line(ipm_image, point_a, point_b, (0, 255, 255), 2)
    #     if a == 0:
    #         cv2.circle(ipm_image, point_a, 5, (0, 0, 255), -1)
    #         temp = point_a
    #     elif a == 1:
    #         cv2.circle(ipm_image, point_a, 5, (0, 255, 0), -1)
    #     elif a == 2:
    #         cv2.circle(ipm_image, point_a, 5, (255, 0, 0), -1)
    #     else:
    #         cv2.circle(ipm_image, point_a, 5, (255, 0, 255), -1)
    #         cv2.circle(ipm_image, temp, 5, (0, 0, 255), -1)
    #
    # cv2.imshow("Ipm", ipm_image)
    # cv2.imshow("img", image)
    # cv2.waitKey(0)
