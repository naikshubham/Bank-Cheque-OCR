from imports import *
from preprocess import pad_img
from vision import vision_api

def ext_amount(image, template):

	amount_path = './feilds/Amount/'
	if os.path.exists(amount_path) and os.path.isdir(amount_path):
		shutil.rmtree(amount_path)
	img_bkp = np.copy(image)
	if img_bkp.ndim == 2:
		img_bkp = cv2.cvtColor(img_bkp, cv2.COLOR_GRAY2RGB)
	if not os.path.exists(amount_path):
		os.mkdir(amount_path)
	#image = cv2.imread('./preprocessed_img.jpg', 0)
	#template = cv2.imread('rupee_template.jpg', 0)
	#image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]
	template = cv2.threshold(template, 0, 255, cv2.THRESH_OTSU)[1]
	
	template = cv2.Canny(template, 50, 200)
	(tH, tW) = template.shape[:2]
	
	# loop over the scales of the image
	found = None
	count = 0
	for scale in np.linspace(0.2, 1.0, 20)[::-1]:
		resized = imutils.resize(image, width = int(image.shape[1] * scale))
		r = image.shape[1] / float(resized.shape[1])
 
		# if the resized image is smaller than the template, then break
		# from the loop
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break
			
		# detect edges in the resized, grayscale image and apply template
		# matching to find the template in the image
		edged = cv2.Canny(resized, 50, 200)
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
 
		# check to see if the iteration should be visualized
		# draw a bounding box around the detected region
		clone = np.dstack([edged, edged, edged])
		cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
			(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
		#cv2.imwrite('template_'+str(count)+'.jpg', clone)
		count += 1
		# if we have found a new maximum correlation value, then update
		# the bookkeeping variable
		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)
 
	# unpack the bookkeeping variable and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
 
	# draw a bounding box around the detected result and display the image
	cv2.rectangle(img_bkp, (startX, startY), (endX, endY), (0, 0, 255), 2)
	cv2.imwrite('final_templ.jpg', img_bkp)
	# print('startX->', startX, 'startY->', startY, 'endX->', endX, 'endY->', endY)
	"""
	w, h = template.shape[::-1]
	method	= cv2.TM_CCOEFF #cv2.TM_CCOEFF_NORMED # cv2.TM_CCOEFF_NORMED #
	# Apply template Matching
	res = cv2.matchTemplate(image,template,method)
	#print(res)
	#threshold = 0.5
	#loc = np.where( res >= threshold)
	#for pt in zip(*loc[::-1]):
	#	cv2.rectangle(img_bkp, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	#cv2.imwrite('temps_matched.jpg', img_bkp)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)
	if image.ndim == 2:
		cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	cv2.rectangle(image,top_left, bottom_right,(0,0, 255), 2)
	print(top_left, bottom_right)
#		  (1722, 408) (1785, 481)	
	#"""
	amt_x1 = endX + 10 # bottom_right[0] + 10
	amt_y1 = startY - 10 # top_left[1] - 10
	amt_x2 = endX + 30 # bottom_right[0] + 10
	amt_y2 = endY + 10 # bottom_right[1] + 10
	h = abs(amt_y2 - amt_y1)
	y = amt_y1
	x = amt_x2
	w = abs(x - image.shape[1])
	#print('y->', y, 'y+h->', y+h, 'x->', x, 'x+w->', x+w)
	amount = image[y:y+h, x:x+w]
	padded_img, padded_img_bkp = pad_img(amount, amount)
	padded_img = cv2.copyMakeBorder(padded_img, top=5, bottom=5, left=5, right=5, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
	padded_img_bkp = cv2.copyMakeBorder(padded_img_bkp, top=5, bottom=5, left=5, right=5, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
	cv2.imwrite('./feilds/Amount/padded_amount.jpg', padded_img)
	amount = vision_api('./feilds/Amount/padded_amount.jpg')
	# print('amount->', "".join(amount))
	amount = "".join(amount)
	return amount
	# new_mask = np.ones_like(padded_img) * 255
	# im, amountContours, hier = cv2.findContours(padded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# hierarchy = hier[0]
	# count = 0
	# #finalAmountContours = contours.sort_contours(amountContours, method = 'left-to-right')[0]
	# for contour_hier in zip(amountContours, hierarchy):
	# 	# Returns the location and width,height for every contour
	# 	contour = contour_hier[0]
	# 	currentHierarchy = contour_hier[1]
	# # for contour in amountContours:
	# 	if not currentHierarchy[3] > 0:
	# 		x, y, w, h = cv2.boundingRect(contour)
	# 		if w < 100 and h > 10:	 
	# 			# if not currentHierarchy[3] > 0:
	# 			cv2.rectangle(padded_img_bkp, (x, y), (x + w, y + h), (0, 0, 255), 1)
	# 			new_img = padded_img[y:y+h, x:x+w]
	# 			cv2.imwrite('./feilds/Amount/amount_img_'+str(count)+'.jpg', new_img)
	# 			count += 1
	# 			#cv2.rectangle(new_mask, (x, y), (x + w, y + h), (0, 0, 255), -1)
	# cv2.imwrite('amount_1.jpg', padded_img_bkp)
	# """
	# finalAmountContours = cv2.findContours(new_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
	# finalAmountContours = contours.sort_contours(finalAmountContours, method = 'left-to-right')[0]
	# count = 0
	# for c in finalAmountContours:
	# 	# Returns the location and width,height for every contour
	# 	x, y, w, h = cv2.boundingRect(c)
	# 	if w < 100 and h > 10 and h < new_mask.shape[0] - 20:
	# 		cv2.rectangle(padded_img_bkp, (x, y), (x + w, y + h), (0, 255, 255), 1)
	# 		new_img = padded_img[y:y+h, x:x+w]
	# 		cv2.imwrite('./feilds/Amount/amount_img_'+str(count)+'.jpg', new_img)
	# 		count += 1
	# cv2.imwrite('amount_img.jpg', padded_img_bkp)
	# cv2.imwrite('amount_mask.jpg', new_mask)
	#"""