from imports import *
from vision import vision_api
from preprocess import pad_img, detect_horizontal_line, correct_line

def ext_ocr_details(img):	
	# ------ connected components ---- ###
	edge_image = cv2.Canny(img, 50, 150)
	cv_img_bkp = np.copy(img)
	kernel = np.ones((2, 7), np.uint8)  # <-- Optimizer
	img_dilation = cv2.dilate(edge_image, kernel, iterations=1)
	th, img_th = cv2.threshold(img_dilation, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	f_fill_img = np.copy(img_th)
	h, w = img_th.shape[:2]
	f_fill_img_inv = cv2.bitwise_not(f_fill_img)
	cv2.imwrite('./../dilated.jpg', f_fill_img_inv)
	pil_img = Image.fromarray(img)
	cord_dict = {}
	re1 = 'ifs'
	re2 = 'sign'
	re3 = '\bpay\b'
	re4 = 'bearer'
	re5 = 'no.'
	re7 = 'lfs'
	re8 = 'alc'
	re9 = 'a/c'
	generic_regex = re.compile(("(%s|%s|%s|%s|%s|%s|%s|%s)" % (re1,re2,re3,re4,re5,re7,re8,re9)), re.IGNORECASE)
	re6 = re.compile(r"\b(pay)\b", re.I)

	connectivity = 8
	retval, labels, stats, centroids = cv2.connectedComponentsWithStats(f_fill_img_inv, connectivity)
	label_range = range(1, retval)
	stats = sorted(stats, key=lambda a: a[0])
	bb_img = np.copy(img)

	for label in label_range:
		x, y, w, h, size = stats[label]
		if h>10 and w < 300:

			if y - 5 >= 0:
				y = y - 5
			else:
				y = y
			if x - 5 >= 0:
				x = x - 5
			else:
				x = x
			if (x + w + 10) <= img.shape[1]:
				w = w + 10
			else:
				w = w 
			if (y + h + 10) <= img.shape[0]:
				h = h + 10
			else:
				h = h 	

			cv2.rectangle(cv_img_bkp, (x, y), (x + w, y + h), (0, 0, 255), 1)
			pil_word = pil_img.crop((x, y, x+w, y+h))
			pil_word.save('pil_word.jpg')
			text = vision_api('./pil_word.jpg')
			# print('ac_no ->', "".join(ac_no))
			text = "".join(text)
			# text = tr.image_to_string(pil_word, config='--psm 6')
			# print(text)
			filtered_text = text.strip().lower()
			matches = generic_regex.findall(filtered_text)
			matches_2 = re.search(re6, filtered_text)
			if matches != [] or matches_2 is not None:
				# print('Detected text->', text)
				# print('matches ->', matches)
				if filtered_text not in cord_dict.keys():
					cord_dict[filtered_text] = [(x, y), (x+w, y+h)]
	cv2.imwrite('./../check_cont_.jpg', cv_img_bkp)
			#word = img[y:y+h, x:x+w]
			#word = cv2.resize(word, (100, 32))
			#word = cv2.threshold(word, 0, 255, cv2.THRESH_OTSU)[1]
			#word = word.astype('float32')
			#word /= 255
			#data = word.reshape(-1, 100, 32, 1) # keras
			#model_out = model.predict([data])  #

			#if np.argmax(model_out) == 1:
			#	str_label = 'printed'
			#	img[y:y+h, x:x+w] = 255
			#else:
			#	str_label = 'handwritten'
	ac_no = ''
	for key,value in cord_dict.items():
		if 'no' in key.strip().lower():
			x = value[1][0] + 100 
			y = value[0][1] - 20
			w = 1000
			h = 90
			cv2.imwrite('./../feilds/ac_no.jpg', img[y:y+h, x:x+w])
			ac_no = vision_api('./../feilds/ac_no.jpg')
			# print('ac_no ->', "".join(ac_no))
			ac_no = "".join(ac_no)

		if 'alc' in key.strip().lower():
			x = value[1][0] + 100 
			y = value[0][1] - 20
			w = 1000
			h = 90
			cv2.imwrite('./../feilds/ac_no.jpg', img[y:y+h, x:x+w])
			ac_no = vision_api('./../feilds/ac_no.jpg')
			# print('ac_no ->', "".join(ac_no))
			ac_no = "".join(ac_no)

		if 'a/c' in key.strip().lower():
			x = value[1][0] + 100 
			y = value[0][1] - 20
			w = 1000
			h = 90
			cv2.imwrite('./../feilds/ac_no.jpg', img[y:y+h, x:x+w])
			ac_no = vision_api('./../feilds/ac_no.jpg')
			# print('ac_no ->', "".join(ac_no))
			ac_no = "".join(ac_no)

		if 'ifs' in key.strip().lower():
			x = value[1][0] + 10
			y = value[0][1]
			w = 250
			h = 30
			#print('ifsc ->', (x, y, w, h))
			ifsc_pt = y+h 
			cv2.imwrite('./../feilds/ifsc.jpg', img[y:y+h, x:x+w])
			ifsc = vision_api('./../feilds/ifsc.jpg')
			# print('ifsc->', "".join(ifsc))
			ifsc = "".join(ifsc)

		if 'lfs' in key.strip().lower():
			x = value[1][0] + 10
			y = value[0][1]
			w = 250
			h = 30
			ifsc_pt = y+h
			#print('ifsc ->', (x, y, w, h))
			cv2.imwrite('./../feilds/ifsc.jpg', img[y:y+h, x:x+w])
			ifsc = vision_api('./../feilds/ifsc.jpg')
			# print('ifsc->', "".join(ifsc))
			ifsc = "".join(ifsc)

		if 'sign' in key.strip().lower():
			x = value[0][0] - 200
			y = value[0][1] - 260
			w = abs(x - img.shape[1])
			h = 600
			sign = img[y:y+h, x:x+w]
			sign = pad_img(sign, sign)[0]
			# sign = sign.flatten()
			# sign = Image.fromarray(sign)
			cv2.imwrite('./../feilds/org_signature.jpg', sign)
			sign = cv2.resize(sign, (200, 100))
			cv2.imwrite('./../feilds/signature.jpg', sign)

		if 'bearer' in key.strip().lower():
			w = value[0][0] - 285
			y = value[0][1] - 40
			# print('ifsc pt->', ifsc_pt)
			# print('payee->', y)
			if y < ifsc_pt:
				y = ifsc_pt
			x = 200
			h = value[1][1] + 20
			bearer = img[y:h, x:w]
			bearer = detect_horizontal_line(bearer)
			bearer = correct_line(bearer)[0]
			bearer = pad_img(bearer, bearer)[0]
			bearer = pad_bearer(bearer)
			cv2.imwrite('./../feilds/payee.jpg', bearer)
			bearer = vision_api('./../feilds/payee.jpg')
			# print('payee name->', " ".join(bearer).strip())
			bearer = " ".join(bearer)

	return [bearer, ac_no, ifsc, sign]


def pad_bearer(img):
	col =0
	b_ratios = []
	for row in range(0, img.shape[1]):
	    black_len = 0
	    row_ = img[:, row]
	    total_pixels = (row_.shape[0])
	    black_len = black_len + len(row_[row_ == 0])
	    black_rate = black_len / total_pixels
	#     print('col->',col,'val->',round(black_rate, 4))
	    b_ratios.append(round(black_rate, 4))
	    col += 1

	for r in range(len(b_ratios)):
	    if b_ratios[r] > 0.1:
	        start = r
	        break
	for r in range(len(b_ratios)-1, 0, -1):
	    if b_ratios[r] > 0.1:
	        stop = r
	        break
	if start - 20 >=0:
	    start = start -20
	else:
	    start = 0
	if stop + 20 <= img.shape[1]:
	    stop = stop + 20
	else:
	    stop = img.shape[1]
	    
	new_img = img[:,start:stop]
	return new_img