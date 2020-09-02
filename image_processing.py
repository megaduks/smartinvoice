import cv2
import numpy as np
import pytesseract

from pathlib import Path
from scipy.ndimage import interpolation as inter
from imutils.object_detection import non_max_suppression
from typing import List

from settings import OCR_MIN_CONFIDENCE as MIN_CONFIDENCE
from settings import OCR_PADDING as PADDING
from settings import OCR_TESSERACT_CONFIG as CONFIG


def correct_skew(image: np.ndarray, delta=0.05, limit=5) -> np.ndarray:
    '''Corrects skew in test using the Projection Profile method, limited in maximum angle of skew,
    delta determines step between angles checked'''

    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def group_boxes(boxes: List[np.ndarray]) -> List[np.ndarray]:
    # sorting bounding boxes into lines of text, returns a list of a list of np.arrays
    # first, sorting from top to bottom

    boxes = sorted(boxes, key=lambda r: r[1])
    heights = []
    for box in boxes:
        heights.append(abs(box[1] - box[3]))
    # getting the average height of the bounding boxes to approximate the height of single line of text
    avgHeight = int(sum(heights) / len(heights))
    # vertical_threshold to determine if a bounding box belongs to the same line
    vertical_threshold = avgHeight / 2
    horizontal_threshold = avgHeight * 10
    groups = []
    grouped_boxes = []
    idx1 = 0
    # clustering boxes of similar Y-value into subgroups
    while idx1 < len(boxes) - 1:
        sub_group = []
        sub_group.append(boxes[idx1])
        idx2 = idx1 + 1
        while idx2 <= len(boxes) - 1:
            if abs(boxes[idx2][1] - boxes[idx1][1]) <= vertical_threshold and \
                    abs(boxes[idx2][3] - boxes[idx1][3]) <= vertical_threshold:
                sub_group.append(boxes[idx2])
                idx2 += 1
            else:
                break
        idx1 = idx2
        groups.append(sub_group)

    # startX, startY, endX, endY
    # creating a new box that contains the ones in the subgroup
    for sub in groups:
        startX = min([x[0] for x in sub])
        startY = min([x[1] for x in sub])
        endX = max([x[2] for x in sub])
        endY = max([x[3] for x in sub])
        grouped_boxes.append(np.asarray([startX, startY, endX, endY]))

    return grouped_boxes


def decode_predictions(scores, geometry):
    '''Process EAST output into relevant ROIs and their confidences'''

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rectangles = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scores_data[x] < MIN_CONFIDENCE:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offset_x, offset_y) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rectangles.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    # return a tuple of the bounding boxes and associated confidences
    return rectangles, confidences


def clean_output(text : str) -> str:
    # Remove non-ASCII characters from the input string
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    return text


class InvoiceOCR:

    def __init__(self, model_path: str, img_width=1056, img_height=1920, ):
        self.img_width = img_width
        self.img_height = img_height
        self.layer_names = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        self.net = cv2.dnn.readNet(model_path)

    def process_image(self, image: np.ndarray):

        # rotate the image to correct skew
        image = correct_skew(image)
        orig = image.copy()
        (orig_img_height, orig_img_width) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (new_img_width, new_img_height) = (self.img_width, self.img_height)
        ratio_w = orig_img_width / float(new_img_width)
        ratio_h = orig_img_height / float(new_img_height)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (new_img_width, new_img_height))

        (img_height, image_width) = image.shape[:2]

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (image_width, img_height),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(self.layer_names)

        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rectangles, confidences) = decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rectangles), probs=confidences)
        boxes = group_boxes(boxes)
        # initialize the list of results
        results = []

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * ratio_w)
            startY = int(startY * ratio_h)
            endX = int(endX * ratio_w)
            endY = int(endY * ratio_h)

            # in order to obtain a better OCR of the text we can potentially
            # apply a bit of padding surrounding the bounding box -- here we
            # are computing the deltas in both the x and y directions
            dX = int((endX - startX) * PADDING)
            dY = int((endY - startY) * PADDING)

            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(orig_img_width, endX + (dX * 2))
            endY = min(orig_img_height, endY + (dY * 2))

            # extract the actual padded ROI
            roi = orig[startY:endY, startX:endX]

            # in order to apply Tesseract v4 to OCR text we must supply
            # (1) a language, (2) an OEM flag of 4, indicating that the we
            # wish to use the LSTM neural net model for OCR, and finally
            # (3) an OEM value, in this case, 7 which implies that we are
            # treating the ROI as a single line of text
            text = pytesseract.image_to_string(roi, config=CONFIG)
            # adding space to each output from tesseract
            text += " "

            # add the bounding box coordinates and OCR'd text to the list
            # of results
            #results.append(((startX, startY, endX, endY), text))
            results.append(((startX, startY, endX, endY), text))

        # sort the results bounding box coordinates from top to bottom:
        results = sorted(results, key=lambda r: r[0][1])

        OCR_text_output = ""
        for _, text in results:
            OCR_text_output += text

        return OCR_text_output
        # save results to a .txt file:

        # with open("test.txt", "w") as file:
        # for _, text in results:
         #     file.write(text)


if __name__ == '__main__':
    image = cv2.imread("1.png")
    OCR = InvoiceOCR(model_path="/home/oliver/Documents/smartinvoice/models/frozen_east_text_detection.pb")
    output = OCR.process_image(image)
    print(output)

