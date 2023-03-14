# cocotoyolo

Reformats coco .odgt to yolo .txt 

This program allows you to convert annotations from the .odgt format to YOLO format, generating .txt files for each image. To use the program, follow the same steps as before, including downloading the images and annotation_train.odgt file, and placing the annotation file in the project folder.

Once you have completed these steps, run the program, which will create YOLO .txt labels for each image based on the annotations in the .odgt file. These labels will be saved in the dataset folder alongside the images.

If you find this program helpful, please consider giving it a star.

CrowdHuman usage: Download the CrowdHuman dataset from the official website or from a third-party source. The dataset consists of images and annotations in JSON format.

Extract the images from the downloaded file and save them to a folder named "Images".

Download the annotation file, which is named "annotation_train.odgt", and save it to the project folder.

Run the main.py, which will read the annotation file and create the labels for the images.

The output images and labels will be saved under the dataset folder.
