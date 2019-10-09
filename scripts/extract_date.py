from imports import *
from preprocess import pad_img
from vision import vision_api

mnist_model = load_model('./mnist_GC_v1.h5')
#model = load_model('./mnist_model.h5')

def ext_date(img, mask):

	date_path = './feilds/Date/'
	if os.path.exists(date_path) and os.path.isdir(date_path):
		shutil.rmtree(date_path)

	if not os.path.exists(date_path):
 		os.mkdir(date_path)
	rows, cols = img.shape
	x = 0 # row
	y = cols - 700 # col
	w = 250
	h = 700

	date_img_bkp = img[x:x+w, y:y+h]
	date_img_mask = mask[x:x+w, y:y+h]
	padded_img, padded_img_bkp = pad_img(date_img_mask, date_img_bkp)
	if padded_img_bkp.ndim == 2:
		date_img_color = cv2.cvtColor(padded_img_bkp, cv2.COLOR_GRAY2RGB)
	cv2.imwrite('init_date_mask.jpg', padded_img)
	new_mask = np.ones_like(padded_img) * 255
	im, dateContours, hier = cv2.findContours(padded_img_bkp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#dateContours = contours.sort_contours(dateContours, method="left-to-right")[0]
	# hierarchy = hier[0]
	count = 0
	# for contour_hier in zip(dateContours, hierarchy):
	# 	# Returns the location and width,height for every contour
	# 	currentContour = contour_hier[0]
	# 	currentHierarchy = contour_hier[1]
	for contour in dateContours:
		x, y, w, h = cv2.boundingRect(contour)
		if w < 100:  
			# if not currentHierarchy[3] > 0:
			cv2.rectangle(date_img_color, (x, y), (x + w, y + h), (0, 0, 255), 1)
			cv2.rectangle(new_mask, (x, y), (x + w, y + h), (0, 0, 255), -1)

	finalDateContours = cv2.findContours(new_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
	finalDateContours = contours.sort_contours(finalDateContours, method = 'right-to-left')[0]
	contour_counts = 0
	date = []
	for cont in finalDateContours:
		x, y, w, h = cv2.boundingRect(cont)
		if (7 < w < 100) and h > 10 and contour_counts < 8:
			new_img = padded_img_bkp[y - 5:y+h+5, x-5:x+w+5]
			# new_img = cv2.copyMakeBorder(new_img, top=0, bottom=0, left=0, right=40, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
			# new_img = cv2.resize(new_img, (28, 28))
			date.append(new_img)
			# pil_img = Image.fromarray(new_img)
			cv2.imwrite('./feilds/Date/date_img_'+str(count)+'.jpg', new_img)
			count += 1
			contour_counts += 1
			# print(tr.image_to_text(pil_img))
			#new_img = cv2.bitwise_not(new_img)
			#new_img = cv2.resize(new_img, (28, 28))
			#new_img = cv2.threshold(new_img, 0, 255, cv2.THRESH_OTSU)[1]
			#new_img = new_img.astype('float32')
			#new_img /= 255
			#data = new_img.reshape(-1, 28, 28, 1) # keras
			#model_out = model.predict([data]) # keras
			#if np.argmax(model_out) == 1:
			#print(np.argmax(model_out))
	f_date = []
	digits = [file for file in os.listdir('./feilds/Date/')]
	digits = digits[::-1]
	for d in digits:
		# print(d)
		img = cv2.imread('./feilds/Date/'+d, 0)
		img = cv2.resize(img, (28, 28))
		img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
		img = cv2.bitwise_not(img)
		img = img.reshape(-1, 28, 28, 1)
		model_out = mnist_model.predict_classes(img)
		f_date.append(str(model_out[0]))
	# print(f_date ,'date')
	f_date.insert(2,'-')
	f_date.insert(5,'-')
	date__ = "".join(f_date)
	# hstack = np.hstack(date)
	# cv2.imwrite('./feilds/Date/date.jpg', hstack)
	# date_ = vision_api('./feilds/Date/date.jpg')
	# print(date_)
	# new_date = []
	# for d in date_:
	# 	if len(d) > 1:
	# 		for d_ in range(len(d)):
	# 			new_date.append(d[d_])
	# 	else:
	# 		new_date.append(d)
	# print('new_date->', new_date)
	# date_ = new_date[::-1]
	# 

	# cv2.imwrite('date_img_color.jpg', date_img_color)
	# cv2.imwrite('date_mask.jpg', new_mask)
	return date__