import numpy as np
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--rating_path',type=str,default='./SCUT-FBP5500_v2/All_Ratings.xlsx')
parser.add_argument('--cross_val_path',type=str, default='./SCUT-FBP5500_v2/train_test_files/5_folders_cross_validations_files/cross_validation_')
args = parser.parse_args()

def main():
	readfile = pd.read_excel(args.rating_path)
	rater = readfile.iloc[:,0].values
	filename = readfile.iloc[:,1].values
	rating = readfile.iloc[:,2].values
	print('image count: '.format(len(filename)))
	img_dict = {}
	for i, name in enumerate(filename):
		if name not in img_dict:
			img_dict[name] = np.zeros(5)
			img_dict[name][int(rating[i])-1] += 1
		else:
			img_dict[name][int(rating[i])-1] += 1
		print('\rfile: {} | rater {}'.format(i+1,np.sum(img_dict[name])), end='')
	print('')

	for i in range(1,6):
		trainpath = args.cross_val_path+str(i)+'/train_'+str(i)+'.txt'
		testpath = args.cross_val_path+str(i)+'/test_'+str(i)+'.txt'
		with open(trainpath,'r') as trainfile:
			img_file = get_annotation(img_dict,trainfile)
			write_annotation(img_file, trainpath[:-3]+'csv')
		with open(testpath,'r') as testfile:
			img_file = get_annotation(img_dict,testfile)
			write_annotation(img_file, testpath[:-3]+'csv')

def write_annotation(img_file, filename):
	writefile = open(filename,'w')
	for i, line in enumerate(img_file):
		writefile.write('{},{}'.format(str(i),line[0]))
		for j in line[1]:
			writefile.write(',{}'.format(str(j)))
		writefile.write('\n')
	writefile.close()	

def get_annotation(img_dict,file):
	img_file = []
	for line in file:
		img = line.split(' ')[0]
		img_file.append([img,img_dict[img]])
	return img_file


if __name__ == '__main__':
    main()

