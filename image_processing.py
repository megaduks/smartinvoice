import cv2
import numpy as np
import pytesseract
import argparse
import os

from tqdm import tqdm
from scipy.ndimage import interpolation as inter
from imutils.object_detection import non_max_suppression
from typing import List, Union

from settings import BC_OVERLAP_THRESHOLD as OVERLAP_THRESHOLD
from settings import OCR_MIN_CONFIDENCE as MIN_CONFIDENCE
from settings import BC_PADDING as BC_PADDING
from settings import OCR_PADDING as OCR_PADDING
from settings import OCR_TESSERACT_CONFIG as CONFIG
from settings import INVOICE_EAST_MODEL as MODEL_PATH


class Graph:
    """
    Graph object for handling bounding box relations.
    """

    def __init__(self, V: int):
        self.V = V
        self.adj = [[] for _ in range(V)]

    def DFSUtil(self, temp, vertex, visited):
        """
        Deep-first search algorithm for finding connected components.
        """
        visited[vertex] = True
        temp.append(vertex)

        for v in self.adj[vertex]:
            if not visited[v]:
                temp = self.DFSUtil(temp, v, visited)

        return temp

    def addEdge(self, v, w):
        """
        Method for adding undirected edges to the graph.
        """
        self.adj[v].append(w)
        self.adj[w].append(v)

    def connectedComponents(self) -> List[int]:

        """
        Method for finding connected components in an undirected graph.
        :return List of indices of connected components.
        """
        visited = []
        connected_components = []

        for i in range(self.V):
            visited.append(False)

        for v in range(self.V):
            if not visited[v]:
                temp = []
                connected_components.append(self.DFSUtil(temp, v, visited))

        return connected_components


def region_overlap_ratio(boxA: Union[np.ndarray], boxB: Union[np.ndarray]) -> float:
    """
    For calculating intersection of union of two bounding boxes.
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def is_inline(boxA: Union, boxB: Union, vertical_threshold: float) -> bool:
    if abs(boxA[1] - boxB[1]) <= vertical_threshold and abs(boxA[3] - boxB[3]) <= vertical_threshold:
        return True
    else:
        return False


def correct_skew(image: np.ndarray, delta=0.5, limit=5) -> np.ndarray:
    """
    Corrects skew in text using the Projection Profile method, limited in maximum angle of skew,
    delta determines step between angles checked.
    """

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
    heights = []
    for box in boxes:
        heights.append(abs(box[3] - box[1]))
    avgHeight = int(sum(heights) / len(heights))

    # verticalThreshold to determine if a bounding box belongs to the same line
    verticalThreshold = avgHeight / 4

    # graph representation of overlap between bounding boxes
    g = Graph(len(boxes))

    # filling edges where overlap is sufficient
    for boxA in range(len(boxes)):
        for boxB in range(boxA + 1, len(boxes)):
            if region_overlap_ratio(boxes[boxA], boxes[boxB]) > 0:
                g.addEdge(boxA, boxB)

    connected_components_idxs = g.connectedComponents()

    grouped_boxes = []
    for group in connected_components_idxs:
        temp = []
        for idx in group:
            temp.append(boxes[idx])
        grouped_boxes.append(temp)

    final = []
    for group in grouped_boxes:
        temp_g = Graph(len(group))
        for boxA in range(len(group)):
            for boxB in range(boxA + 1, len(group)):
                if is_inline(group[boxA], group[boxB], verticalThreshold):
                    temp_g.addEdge(boxA, boxB)
        final_idx = temp_g.connectedComponents()
        temp1 = []
        for sub in final_idx:
            temp = []
            for idx in sub:
                temp.append(group[idx])
            temp1.append(temp)

        final.append(temp1)

    final_merged = []
    for block in final:
        para = []
        for line in block:
            startX = min([x[0] for x in line])
            startY = min([x[1] for x in line])
            endX = max([x[2] for x in line])
            endY = max([x[3] for x in line])
            para.append((startX, startY, endX, endY))
        para = sorted(para, key=lambda r: r[1])
        final_merged.append(para)

    final_merged = sorted(final_merged, key=lambda r: r[0][1])

    flat = []
    for box in final_merged:
        for line in box:
            flat.append(line)

    return flat


def group_boxes_blocks(boxes: List[np.ndarray]) -> List[List[np.ndarray]]:
    heights = []
    for box in boxes:
        heights.append(abs(box[3] - box[1]))
    avgHeight = int(sum(heights) / len(heights))

    # verticalThreshold to determine if a bounding box belongs to the same line
    verticalThreshold = avgHeight / 4

    # graph representation of overlap between bounding boxes
    g = Graph(len(boxes))

    # filling edges where overlap is sufficient
    for boxA in range(len(boxes)):
        for boxB in range(boxA + 1, len(boxes)):
            if region_overlap_ratio(boxes[boxA], boxes[boxB]) > 0:
                g.addEdge(boxA, boxB)

    connected_components_idxs = g.connectedComponents()

    grouped_boxes = []
    for group in connected_components_idxs:
        temp = []
        for idx in group:
            temp.append(boxes[idx])
        grouped_boxes.append(temp)


    merged_boxes = []

    for box in grouped_boxes:
        startX = min([x[0] for x in box])
        startY = min([x[1] for x in box])
        endX = max([x[2] for x in box])
        endY = max([x[3] for x in box])

        merged_boxes.append(np.asarray([startX, startY, endX, endY]))

    return merged_boxes


def resize_image():
    pass

def pad_boxes(boxes: List[np.ndarray], img_width, img_height) -> List[np.ndarray]:

    padded_boxes = []
    for (startX, startY, endX, endY) in boxes:

        dX = int((endX - startX) * BC_PADDING)
        dY = int((endY - startY) * BC_PADDING)

        # apply padding to each side of the bounding box, respectively

        padded_startX = max(0, startX - dX)
        padded_startY = max(0, startY - dY)
        padded_endX = min(img_width, endX + (dX * 2))
        padded_endY = min(img_height, endY + (dY * 2))

        padded_boxes.append((padded_startX, padded_startY, padded_endX, padded_endY))

    return padded_boxes



def decode_predictions(scores: List, geometry: List) -> Union:
    """
    Process EAST output into relevant ROIs and their confidences
    """

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


def clean_output(text: str) -> str:
    """
    Remove non-ASCII characters from the input string.
    """
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    return text


class InvoiceOCR:

    def __init__(self, model_path: str, img_width=1056, img_height=1920):
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
        resized_image = cv2.resize(image, (new_img_width, new_img_height))

        (img_height, image_width) = resized_image.shape[:2]

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(resized_image, 1.0, (image_width, img_height),
                                     (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(self.layer_names)

        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rectangles, confidences) = decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rectangles), probs=confidences)
        boxes = group_boxes(boxes)
        padded_boxes = []

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
            dX = int((endX - startX) * OCR_PADDING)
            dY = int((endY - startY) * OCR_PADDING)

            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(orig_img_width, endX + (dX * 2))
            endY = min(orig_img_height, endY + (dY * 2))
            padded_boxes.append((startX, startY, endX, endY))

        padded_boxes = group_boxes(padded_boxes)
        results = []

        for (startX, startY, endX, endY) in tqdm(padded_boxes):

            roi = orig[startY:endY, startX:endX]

            # in order to apply Tesseract v4 to OCR text we must supply
            # (1) a language, (2) an OEM flag of 4, indicating that the we
            # wish to use the LSTM neural net model for OCR, and finally
            # (3) an OEM value, in this case, 7 which implies that we are
            # treating the ROI as a single line of text

            text = pytesseract.image_to_string(roi, config=CONFIG)

            text += " \n"
            results.append(((startX, startY, endX, endY), text))

        return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", nargs='+', default=[], type=str,
                    help="path to input image")
    ap.add_argument("-d", "--display", type=bool, default=False,
                    help="whenever to display the test with bounding boxes")
    args = vars(ap.parse_args())

    merge_early = True

    for pathToImg in tqdm(args['image']):

        image = cv2.imread(pathToImg)
        OCR = InvoiceOCR(model_path=MODEL_PATH.as_posix())
        results = OCR.process_image(image)

        base = os.path.basename(pathToImg)
        imgName, ext = base.split(".")
        with open(f"{imgName}.txt", "w") as file:
            for _, text in results:
                file.write(text)




        output = image.copy()

        if args['display']:
            for ((startX, startY, endX, endY), text) in results:
                # using OpenCV, then draw the text and a bounding box surrounding
                # the text region of the input image

                cv2.rectangle(output, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(output, clean_output(text), (startX, endY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)

                # show the output image
            cv2.imwrite("out.jpg", output)

