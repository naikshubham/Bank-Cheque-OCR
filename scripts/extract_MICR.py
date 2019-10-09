# import the necessary packages
from imports import *
charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "T", "U", "A", "D"]
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))


# initialize the list of reference character names, in the same
# order as they appear in the reference image where the digits
# their names and:
# T = Transit (delimit bank branch routing transit)
# U = On-us (delimit customer account number)
# A = Amount (delimit transaction amount)
# D = Dash (delimit parts of numbers, such as routing or account)


def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):
    # grab the internal Python iterator for the list of character
    # contours, then  initialize the character ROI and location
    # lists, respectively
    charIter = charCnts.__iter__()
    rois = []
    locs = []
    # keep looping over the character contours until we reach the end
    # of the list
    while True:
        try:
            # grab the next character clea from the list, compute
            # its bounding box, and initialize the ROI
            c = next(charIter)
            (cX, cY, cW, cH) = cv2.boundingRect(c)
            roi = None
            # check to see if the width and height are sufficiently
            # large, indicating that we have found a digit
            if cW >= minW and cH >= minH:
                # extract the ROI
                roi = image[cY:cY + cH, cX:cX + cW]
                rois.append(roi)
                locs.append((cX, cY, cX + cW, cY + cH))
            # otherwise, we are examining one of the special symbols
            else:
                # MICR symbols include three separate parts, so we
                # need to grab the next two parts from our iterator,
                # followed by initializing the bounding box
                # coordinates for the symbol
                parts = [c, next(charIter), next(charIter)]
                (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf, -np.inf)

                # loop over the parts
                for p in parts:
                    # compute the bounding box for the part, then
                    # update our bookkeeping variables
                    (pX, pY, pW, pH) = cv2.boundingRect(p)
                    sXA = min(sXA, pX)
                    sYA = min(sYA, pY)
                    sXB = max(sXB, pX + pW)
                    sYB = max(sYB, pY + pH)

                # extract the ROI
                roi = image[sYA:sYB, sXA:sXB]
                rois.append(roi)
                locs.append((sXA, sYA, sXB, sYB))
        # we have reached the end of the iterator; gracefully break from the loop
        except StopIteration:
            break
    # return a tuple of the ROIs and locations
    return rois, locs


def find_ref_micr_contours(image):
    ref = imutils.resize(image, width=400)
    ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refCnts = imutils.grab_contours(refCnts)
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
    return ref, refCnts


def find_ref_micr_data():
    directory = './'
    image = cv2.imread(directory + 'reference_micr.png', 0)
    ref, refCnts = find_ref_micr_contours(image)
    refROIs = extract_digits_and_symbols(ref, refCnts, minW=10, minH=20)[0]
    chars = {}
    for (name, roi) in zip(charNames, refROIs):
        roi = cv2.resize(roi, (30, 30))
        chars[name] = roi
    return chars


def extract_blackhat(image):  # Here we want cheque image
    (h, w,) = image.shape[:2]
    delta = int(h - (h * 0.15))
    bottom = image[delta:h, 0:w]
    gray = np.copy(bottom)
    cv2.imwrite('bottom.jpg', gray)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    return blackhat, gray, delta


def find_group_contours(image):
    blackhat = extract_blackhat(image=image)[0]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8");
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    groupCnts = imutils.grab_contours(groupCnts)
    return groupCnts


def group_locations(image):
    groupCnts = find_group_contours(image=image)
    groupLocs = []
    for (i, c) in enumerate(groupCnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 50 and h > 15:
            groupLocs.append((x, y, w, h))
    groupLocs = sorted(groupLocs, key=lambda x: x[0])
    return groupLocs


def extract_micr(image):
    blackhat, gray, delta = extract_blackhat(image=image)
    groupLocs = group_locations(image=image)
    chars = find_ref_micr_data()
    output = []
    # loop over the group locations
    for (gX, gY, gW, gH) in groupLocs:
        # initialize the group output of characters
        groupOutput = []
        group = gray[gY - 2:gY + gH + 2, gX - 2:gX + gW + 2]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # cv2.imshow("Group", group)
        # cv2.waitKey(0)

        charCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        charCnts = imutils.grab_contours(charCnts)
        charCnts = contours.sort_contours(charCnts, method="left-to-right")[0]
        (rois, locs) = extract_digits_and_symbols(group, charCnts)
        for roi in rois:
            scores = []
            roi = cv2.resize(roi, (36, 36))
            for charName in charNames:
                result = cv2.matchTemplate(roi, chars[charName], cv2.TM_CCORR)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)
            groupOutput.append(charNames[np.argmax(scores)])
        cv2.rectangle(image, (gX - 10, gY + delta - 10), (gX + gW + 10, gY + gY + delta), (0, 0, 255), 2)
        cv2.putText(image, "".join(groupOutput),
                    (gX - 10, gY + delta - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 2)
        output.append("".join(groupOutput))
    output = " ".join(output)
    return output, image
