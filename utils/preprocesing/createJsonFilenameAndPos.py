import csv
import os
import json

startPath = "../data/lidarFilename"
#startPath = "../data/rgbFileName"
csvFiles =os.listdir(startPath)
print(csvFiles)

posPath = "../data/pos"
directionPath = "../data/direction"

namePosDirection = {}
#from tqdm import tqdm



for csvFile in csvFiles:
    fullPath = startPath + "/" + csvFile

    print(csvFile)

    #put pos in list
    fullPosPath = posPath +  "/" + csvFile
    pos = []
    with open(fullPosPath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            #print(row)
            if (line_count > 0):
                timeStamp = row[0]
                latitude = row[1]
                longitude = row[2]
                altitude = row[3]
                
                posElement = {'time': timeStamp,'latitude': latitude, 'longitude': longitude, 'altitude': altitude}
                pos.append(posElement)
            line_count += 1
    #print(pos)


    fullDirectionPath = directionPath +  "/" + csvFile
    directions = []
    with open(fullDirectionPath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if (line_count > 0):
                timeStamp = row[0]
                heading = row[1]
                
                directionElement = {'time': timeStamp,'direction': heading}
                directions.append(directionElement)
            line_count += 1
    #print(directions)

    filenames = []
    with open(fullPath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if (line_count > 0):
                timeStamp = row[0]
                filename = row[1]
                
                filenameElement = {'time': timeStamp,'filename': filename}
                filenames.append(filenameElement)

            line_count += 1

    
    directionCounter = 0
    posCounter = 0
    for filenameTime in tqdm(filenames):
        #print(filenameTime)

        timeStamp = filenameTime['time']
        #print(timeStamp)

        posDirElement = {}

        #get direction
        for direction in directions[directionCounter:]: #need som logic for if the image time is before the sensor time, take the 0 element from sensor time
            dirTime = direction['time']
            #print(dirTime)
            #print(timeStamp)
            if (dirTime > timeStamp):
                #hent forrige element
                posDirElement['direction'] = lastDir['direction']
                break
            lastDir = direction

        #print(posDirElement)

        #get pos
        #print(pos)
        #lastPosition = pos[posCounter:][0]
        #print(lastPosition)

        #print(pos[posCounter:][0]['time'])
        #print(pos[posCounter:][-1]['time'])
        #add number of iterations to posCounter, so that is can start the search from where it left of when getting new image time
        for position in pos[posCounter:]: #need som logic for if the image time is before the sensor time, take the 0 element from sensor time
            
            #print(position)
            posTime = position['time']
            #print(posTime)
            #print(timeStamp)

            if (posTime > timeStamp):
                #hent forrige element
                posDirElement['latitude'] = lastPosition['latitude']
                posDirElement['longitude'] = lastPosition['longitude']
                posDirElement['altitude'] = lastPosition['altitude']
                break
            lastPosition = position

        #print(posDirElement['latitude'], posDirElement['longitude'])
        posDirElement['time'] = timeStamp

        namePosDirection[filenameTime['filename']] = posDirElement 
        #print(namePosDirection)

#with open("../data/RGBnamePosDir.json", "w") as outfile:
#    json.dump(namePosDirection, outfile)

with open("../data/LiDARnamePosDir.json", "w") as outfile:
    json.dump(namePosDirection, outfile)


        


    