from imports import *

def start_pos(image, restart):
    """Function to find start point of the Non - White pixel"""
    for i in range(restart, image.shape[0]):
        if 0 in image[i]:
            return i


def stop_pos(image, start):
    """Function to find Stop point of the Non - White row"""
    stop = None
    for i in range(start, image.shape[0]):
        if 0 not in image[i]:
            stop = i
            break
    if stop is None:
        stop = image.shape[0]
        return stop
    else:
        return stop

		
def strt_stp_pos_image(bw_image):
    """this function is used for padding for tracking start and stop points"""
#     bw_image = np.array(bw_image)
    restart = 0
    start_pos_arr = []
    stop_pos_arr = []
    image_last_row = 0

    for i in range(bw_image.shape[0] - 1, 0, -1):
        if 0 in bw_image[i]:
            image_last_row = i
            break

    while restart < image_last_row + 1:
        start_position = start_pos(bw_image, restart)  
        stop_position = stop_pos(bw_image, start_position)  
        if abs(start_position - stop_position) > 10:
            start_pos_arr.append(start_position)
            stop_pos_arr.append(stop_position)
        # Added below if condition to handle image which have signature till last row
        if stop_position is None:
            stop_position = image_last_row
        restart = stop_position
        if restart == image_last_row:
            break
    return start_pos_arr, stop_pos_arr

	
def start(image):
	"""Function to find start point of the Non - White pixel"""
	for i in range(image.shape[0]):
		if 0 in image[i]:
			return i


def stop(image):
	"""Function to find Stop point of the Non - White row"""
	for i in range(image.shape[0]-1, 0, -1):
		if 0 in image[i]:
			return i
			
def detect_horizontal_line(image):
    edges = cv2.Canny(image,50,150,apertureSize = 3) 
    lines = cv2.HoughLines(edges,1,np.pi/2, int(image.shape[1] * 0.5))
    if lines is not None:
        print("Horizontal line detected")
        horizontal_line_indicator = True
#         for r,theta in lines[0]: 
        for i in  range(0,lines.shape[0]):
            r,theta = lines[i,0]
#            print('line 1')
            a = np.cos(theta) 
            b = np.sin(theta) 
            x0 = a*r 
            y0 = b*r 
            x1 = int(x0 + 1000*(-b)) 
            y1 = int(y0 + 1000*(a)) 
#             x2 = int(x0 - 1000*(-b)) 
#             y2 = int(y0 - 1000*(a)) 
#             x2 = image.shape[1]
            x2 = image.shape[1]
            y2 = y1
    # --Draws a complete horizontal line ------ #
            cv2.line(image,(x1,y1), (x2,y2), (0,0,255),3)
    return image

def correct_line(image):
	"""this function preprocesses the form, removes all the lines across the form"""	
	binary_img = cv2.bitwise_not(image)
	binary_image = np.copy(binary_img)

	col = binary_image.shape[1]

	kernel_size = int(col / 80)
#	  v_kernel_size = int(col / 40)
	v_kernel_size = 20
#	  kernel_size = 20
#	  kernel_size = 10
#	  print('kernel size->', kernel_size)
	# ----Create horizontal and vertical line mask -- #

	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
	vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
	horizontal_temp = cv2.erode(binary_image, horizontal_kernel, iterations = 3)
	horizontal_mask = cv2.dilate(horizontal_temp, horizontal_kernel, iterations = 3)
	vertical_temp = cv2.erode(binary_image, vertical_kernel, iterations =3)
	vertical_mask = cv2.dilate(vertical_temp, vertical_kernel, iterations =3)
#	  cv2.imwrite('init_hor_mask.jpg', horizontal_mask)
#	  cv2.imwrite('init_ver_mask.jpg', vertical_mask)
	horizontal_line = np.copy(horizontal_mask)
	vertical_line = np.copy(vertical_mask)

	horizontal_line = cv2.threshold(horizontal_line, 0, 255, cv2.THRESH_OTSU)[1]
	vertical_line = cv2.threshold(vertical_line, 0, 255, cv2.THRESH_OTSU)[1]

	# -- Store final horizontal & vertical mask -- #
	# -- in the below numpy arrays				-- #

	horizontal_line_cpy = np.zeros_like(horizontal_line)
	vertical_line_cpy = np.zeros_like(vertical_line)

	horizontal_contours = cv2.findContours(horizontal_line, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
	vertical_contours = cv2.findContours(vertical_line, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

	try:
		cv2.drawContours(horizontal_line_cpy, horizontal_contours, -1, (255, 255, 255), 2)
		cv2.fillPoly(horizontal_line_cpy, pts = horizontal_contours, color = (255, 255, 255))
	except Exception:
		print("No horizontal contours detected ")
	
	try:
		cv2.drawContours(vertical_line_cpy, vertical_contours, -1, (255, 255, 255), 2)
		cv2.fillPoly(vertical_line_cpy, pts = vertical_contours, color = (255, 255, 255))
	except Exception:
		print("No verticals contours detected ")
		
	horizontal_line = np.copy(horizontal_line_cpy)
	vertical_line = np.copy(vertical_line_cpy)
	horizontal_line_mask = np.copy(horizontal_line_cpy)
	vertical_line_mask = np.copy(vertical_line_cpy)
#	  cv2.imwrite('hor_mask.jpg', horizontal_line_cpy)
#	  cv2.imwrite('ver_mask.jpg', vertical_line_cpy)
	horizontal_line = cv2.bitwise_not(horizontal_line)
	vertical_line = cv2.bitwise_not(vertical_line)
	
	final_mask = cv2.bitwise_or(vertical_line, vertical_line, mask = horizontal_line)
#	  final_mask = draw_horizontal_line(final_mask)

	cv2.imwrite('./../final_mask.jpg', final_mask)
	final_mask = cv2.threshold(final_mask, 0, 255, cv2.THRESH_OTSU)[1]

	res = cv2.bitwise_and(binary_image, binary_image, mask = final_mask)
	res = cv2.bitwise_not(cv2.threshold(res, 0, 255, cv2.THRESH_OTSU)[1])

	# -- For filling lost features -- #
	temp = cv2.inpaint(res, horizontal_line_mask, 3, cv2.INPAINT_TELEA)
	line_corrected_img = cv2.inpaint(temp, vertical_line_mask, 3, cv2.INPAINT_TELEA)
	line_corrected_img = cv2.threshold(line_corrected_img, 0, 255, cv2.THRESH_OTSU)[1]

	return line_corrected_img, final_mask
	
	
def pad_img(img, img_bkp):
	start_pos = start(img)
	stop_pos = stop(img)
	if (start_pos - 10) >= 0:
		start_pos = start_pos - 10
	else:
		start_pos = start_pos
	if (stop_pos + 10) <= img.shape[0]:
		stop_pos = stop_pos + 10
	else:
		stop_pos = stop_pos
	
	pad_img = img[start_pos:stop_pos]
	pad_img_bkp = img_bkp[start_pos:stop_pos]
	
	transpose = cv2.transpose(pad_img)
	transpose_bkp = cv2.transpose(pad_img_bkp)
	start_pos = start(transpose)
	stop_pos = stop(transpose)
	if (start_pos - 10) >= 0:
		start_pos = start_pos - 10
	else:
		start_pos = start_pos
	if (stop_pos + 10) <= img.shape[0]:
		stop_pos = stop_pos + 10
	else:
		stop_pos = stop_pos
	pad_img = transpose[start_pos:stop_pos]
	pad_img_bkp = transpose_bkp[start_pos:stop_pos]
	padded_img = cv2.transpose(pad_img)
	padded_img_bkp = cv2.transpose(pad_img_bkp)
	return padded_img, padded_img_bkp