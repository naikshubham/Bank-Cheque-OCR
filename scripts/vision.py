from imports import *
  
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'D:/Projects/AS_Hackathon_bank_cheque/hackathon-bf39302e7ac7.json'

def vision_api(f):

	client = vision.ImageAnnotatorClient() 
	# f = 'amount_img.jpg'
	with io.open(f, 'rb') as image: 
	    content = image.read()
	      
	image = vision.types.Image(content = content)
	response = client.document_text_detection(image = image)
	  
	a = plt.imread(f)
	plt.imshow(a)
	  
	txt = []
	for page in response.full_text_annotation.pages: 
	        for block in page.blocks: 
	            # print('\nConfidence: {}%\n'.format(block.confidence * 100)) 
	            for paragraph in block.paragraphs: 
	  
	                for word in paragraph.words: 
	                    word_text = ''.join([symbol.text for symbol in word.symbols]) 
	                    txt.append(word_text) 
	                      
	return txt