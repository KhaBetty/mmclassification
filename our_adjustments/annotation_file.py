import glob
import csv

# RUN IN LINUX!

def build_annotation_dict(dir: str, prefix_img: str, classes_dir):
	annot_dict = {}
	for index, class_dir in enumerate(classes_dir):
		images_paths = glob.glob(dir + '/' + class_dir + '/*' + prefix_img)
		annot_dict.update(dict.fromkeys(images_paths, index))
	return annot_dict

if __name__ == "__main__":
	csv_file = 'annotation_file.csv' #specify for train/ val/ test/
	dir = ''
	prefix_img = ''
	classes_dir = []
	annot_dict = build_annotation_dict(dir, prefix_img, classes_dir)
	try:
		with open(csv_file, 'w') as csvfile:
			writer = csv.writer(csv_file)
			for key, value in annot_dict.items():
				writer.writerow([key, value])
	except IOError:
		print("I/O error")