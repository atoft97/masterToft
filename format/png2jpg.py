from PIL import Image
import json
import os

#for s in ["train", "test", "validation"]:
for s in ["train"]:
    with open('annotations/instances_'+s+'.json') as json_file:
        data = json.load(json_file)
        for json_img in data['images']:
            fileName = json_img['file_name']
            print(fileName)
            if (fileName.endswith('.png') or fileName.endswith('.PNG')):
                img = Image.open("images"+"/"+fileName)
                if (fileName.endswith('.png')):
                    newFIleName = fileName.replace(".png", ".jpg")
                elif (fileName.endswith('.PNG')):
                    newFIleName = fileName.replace(".PNG", ".jpg")
                img.save("/lhome/asbjotof/work/2022/masterToft/data/dataset/"+s+"/images"+"/"+newFIleName)

                json_img['file_name'] = newFIleName

    with open('/lhome/asbjotof/work/2022/masterToft/data/dataset/'+s+'/annotations/instances_'+s+'.json', 'w') as outfile:
        json.dump(data, outfile)


