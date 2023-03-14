import cv2
import os

target_class = ["head", "vbody"]   #[] --> all
#target_class = []
annotations_path = "annotation_train.odgt"
crowdHuman_path = "Images"

xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

#output
datasetPath = "dataset"
imgPath = "images/"
labelPath = "labels/"
imgType = "jpg"  # jpg, png

def check_env():
    if not os.path.exists(os.path.join(datasetPath, imgPath)):
        os.makedirs(os.path.join(datasetPath, imgPath))

    if not os.path.exists(os.path.join(datasetPath, labelPath)):
        os.makedirs(os.path.join(datasetPath, labelPath))

def writeObjects(label, bbox,w,h):
    with open(object_xml_file) as file:
        file_content = file.read()

    file_updated = file_content.replace("{NAME}", label)
    file_updated = file_updated.replace("{XMIN}", str(round((bbox[0]+bbox[2]/2)/w, 6)))
    file_updated = file_updated.replace("{YMIN}", str(round((bbox[1]+bbox[3]/2)/h, 6)))
    file_updated = file_updated.replace("{XMAX}", str(round(bbox[2]/w, 6)))
    file_updated = file_updated.replace("{YMAX}", str(round(bbox[3]/h, 6))+"\n")

    return file_updated

def generateXML(imgfile, filename, fullpath, bboxes, imgfilename):
    xmlObject = ""
    img = cv2.imread(imgfile)
    (h, w, ch) = img.shape
    for (labelName, bbox) in bboxes:
        if (labelName == "person_head" or labelName == "person_vbox"):
            if (labelName == "person_head"):
                name = "0"
            else:
                name = "1"
            xmlObject = xmlObject + writeObjects(name, bbox,w,h)

    with open(xml_file) as file:
        xmlfile = file.read()


    cv2.imwrite(os.path.join(datasetPath, imgPath, imgfilename), img)


    xmlfile = xmlfile.replace( "{WIDTH}", str(w) )
    xmlfile = xmlfile.replace( "{HEIGHT}", str(h) )
    xmlfile = xmlfile.replace( "{FILENAME}", filename )
    xmlfile = xmlfile.replace( "{PATH}", fullpath + filename )
    xmlfile = xmlfile.replace( "{OBJECTS}", "&"+xmlObject )

    return xmlfile

def makeLabelFile(filename, bboxes, imgfile):
    jpgFilename = filename + "." + imgType
    xmlFilename = filename + ".txt"

    xmlContent = generateXML(imgfile, xmlFilename, os.path.join(datasetPath ,labelPath, xmlFilename), bboxes, jpgFilename)

    file = open(os.path.join(datasetPath, labelPath, xmlFilename), "w")
    x = xmlContent.split("&")
    file.write(x[1])
    file.close

if __name__ == "__main__":
    check_env()
    img_filename = {}
    img_bboxes = {}

    f = open(annotations_path)
    lines = f.readlines()
    total_lines = len(lines)
    for lineID, line in enumerate(lines):
        data = eval(line)
        img_id = data["ID"]
        total_iid = len(data["gtboxes"])

        for iid, infoBody in enumerate(data["gtboxes"]):
            tag = infoBody["tag"]
            hbbox = infoBody["hbox"]
            head_attr = infoBody["head_attr"]
            fbbox = infoBody["fbox"]
            vbbox = infoBody["vbox"]
            extra = infoBody["extra"]

            if(len(target_class)==0 or ("head" in target_class)):
                if(img_id in img_bboxes):
                    last_bbox_data = img_bboxes[img_id]
                    last_bbox_data.append((tag+"_head", hbbox))
                    img_bboxes.update( {img_id:last_bbox_data} )
                else:
                    img_bboxes.update( {img_id:[(tag+"_head", hbbox)]} )

            if(len(target_class)==0 or ("fbody" in target_class)):
                if(img_id in img_bboxes):
                    last_bbox_data = img_bboxes[img_id]
                    last_bbox_data.append((tag+"_fbox", fbbox))
                    img_bboxes.update( {img_id:last_bbox_data} )
                else:
                    img_bboxes.update( {img_id:[(tag+"_fbox", fbbox)]} )

            if(len(target_class)==0 or ("vbody" in target_class)):
                if(img_id in img_bboxes):
                    last_bbox_data = img_bboxes[img_id]
                    last_bbox_data.append((tag+"_vbox", vbbox))
                    img_bboxes.update( {img_id:last_bbox_data} )
                else:
                    img_bboxes.update( {img_id:[(tag+"_vbox", vbbox)]} )

            filename = img_id + ".jpg"
            img_filename.update( { filename:img_id } )

            print("[Left over] {}/{} , {}: head:{} full:{} visible:{}".format(total_lines-lineID, total_iid-iid, img_id,\
                hbbox, fbbox, vbbox))

        for file in os.listdir(crowdHuman_path):
            file_name, file_extension = os.path.splitext(file)

            if(file in img_filename):
                bbox_objects = {}
                if(file in img_filename):
                    img_id = img_filename[file]
                    if(img_id in img_bboxes):
                        bboxes = img_bboxes[img_id]
                        makeLabelFile(file_name, bboxes, os.path.join(crowdHuman_path, file))
