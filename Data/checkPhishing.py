import os
import pandas as pandas

# ----------------Config----------------------------
tagLinkPercentageUpperLevel1 = 17
tagLinkPercentageLowerLevel2 = 17
tagLinkPercentageUpperLevel2 = 81

reqLinkPercenatgeUpperLevel1 = 22
reqLinkPercenatgeUpperLevel2 = 61

titleNumOfWordsUpperLimit1 = 24

filePath = 'data.csv'

titleStatusHeading = "Evaluated status for title"
favIconStatusHeading = "Evaluated status for favicon"
tagLinkStatusHeading = "Evaluated status for tag link percentage"
reqUrlStatusHeading = "Evaluated status for req url percenatge"
symbolUrlStatusHeading = "Evaluated status for having url symbol"
lastPositionStatusHeading = "Evaluated status for Last position"
ipAddressStatusHeading = "Evaluated status for IP Address"
lengthofURLStatusHeading = "Evaluated status for Length of URL"
dotCountStatusHeading = "Evaluated status for Dot count"

columnHeadingList = [titleStatusHeading, favIconStatusHeading,
                     tagLinkStatusHeading, reqUrlStatusHeading,
                     symbolUrlStatusHeading, lastPositionStatusHeading, 
                     ipAddressStatusHeading, lengthofURLStatusHeading, 
                     dotCountStatusHeading]
# -------------------------------------------------

dataFrame = pandas.read_csv(filePath)


urlList = []

titleList = []
favIconUrlList = []
tagLinkPercentageList = []
reqUrlPercentageList = []
symbolUrlList = []
lastPositionList = []
ipAddressList = []
lengthofURLList = []
dotCountList = []

titleStatusList = []  # id-1
favIconStatusList = []  # id-2
tagLinkStatusList = []  # id-3
reqUrlStatusList = []  # id-4
symbolUrlStatusList = []  # id-5
lastPositionStatusList = []  # id-6
ipAddressStatusList = []  # id-7
lengthofURLStatusList = []  # id-8
dotCountStatusList = []  # id-9


def setStatus(status: bool,column: int):
    strStatus = str(status)
    print()
    print(column)
    if (column == 1):
        titleStatusList.append(strStatus)
    elif (column == 2):
        favIconStatusList.append(strStatus)
    elif (column == 3):
        tagLinkStatusList.append(strStatus)
    elif (column == 4):
        reqUrlStatusList.append(strStatus)
    elif (column == 5):
        symbolUrlStatusList.append(strStatus)
    elif (column == 6):
        lastPositionStatusList.append(strStatus)
    elif (column == 7):
        ipAddressStatusList.append(strStatus)
    elif (column == 8):
        lengthofURLStatusList.append(strStatus)
    elif (column == 9):
        dotCountStatusList.append(strStatus)
    else:
        print("Invalid array id")
    # True if phishing, flase if legit
    # if (len(statusList)-1 >= rowNum):
    #     statusList[rowNum] = strStatus
    # else:
    #     statusList.append(strStatus)


def main():
    trueCount = 0
    for val in dataFrame.values:
        urlList.append(str(val[4]).lower().split('/')[2])
        titleList.append(val[0])
        favIconUrlList.append(val[1])
        tagLinkPercentageList.append(val[2])
        reqUrlPercentageList.append(val[3])
        symbolUrlList.append(val[4])
        lastPositionList.append(val[7])
        ipAddressList.append(val[11])
        lengthofURLList.append(val[5])
        dotCountList.append(val[10])
    numRecords = len(urlList)

    for index in range(numRecords):
        # os.system('cls')
        print("Processing URL:"+str(urlList[index]))

        # Check favIcon URL
        if (str(favIconUrlList[index]).lower().find('null') > -1):
            setStatus(False, 1)
        elif ((str(favIconUrlList[index]).find('http') > -1) and (not str(favIconUrlList[index]).find(str(urlList[index])) > -1)):
            setStatus(False, 1)
        else:
            setStatus(True, 1)

        # Check tag link percentage
        if (float(tagLinkPercentageList[index]) < tagLinkPercentageUpperLevel1):
            setStatus(False, 2)
        elif (float(tagLinkPercentageList[index]) <= tagLinkPercentageUpperLevel2):
            setStatus(True, 2)
        else:
            # Not given in requirements
            setStatus(False, 2)

        # Check req URL percentage
        if (float(reqUrlPercentageList[index]) < reqLinkPercenatgeUpperLevel1):
            setStatus(False, 3)
        elif (float(reqUrlPercentageList[index]) < reqLinkPercenatgeUpperLevel2):
            trueCount = trueCount+1
            setStatus(True, 3)
        else:
            # Not given in requirements
            setStatus(False, 3)

        # Check no of words in title
        if (len(str(titleList[index]).split(' ')) < titleNumOfWordsUpperLimit1):
            setStatus(True, 4)
        else:
            setStatus(False, 4)

        # Check @ symble in URL
        if (str(symbolUrlList[index]).find('@') > -1):
            setStatus(True, 5)
        else:
            setStatus(False, 5)

        # Check // Last position of URL
        if (int(lastPositionList[index]) > 7):
            setStatus(True, 6)
        else:
            setStatus(False, 6)

        # Check IP address in URL
        if (str(ipAddressList[index]).find('null') > -1):
            setStatus(False, 7)
        else:
            setStatus(True, 7)
        
        # Check length of URL
        if (int(lengthofURLList[index]) < 54):
            setStatus(False, 8)
        else:
            setStatus(True, 8)

        # Check dot count in URL
        if (int(dotCountList[index]) < 5):
            setStatus(True, 9)
        else:
            setStatus(False, 9)

    # os.system('cls')
    print("Processed "+str(len(titleList))+" records")
    print("Appending evaluated status to <"+filePath+">")

    # Convert True->1, False->0
    index = 0

    convertedStatusList = []
    statusArryList = [titleStatusList, favIconStatusList,
                      tagLinkStatusList, reqUrlStatusList, 
                      symbolUrlStatusList, lastPositionStatusList, 
                      ipAddressStatusList, lengthofURLStatusList, 
                      dotCountStatusList]
    for col in statusArryList:
        temparray = []
        for rec in col:
            # print(str(rec).lower())
            if (str(rec).lower() == 'false'):
                print("FALSE")
                temparray.append('0')
            else:
                temparray.append('1')
                print("TRUE")
        print(temparray)
        convertedStatusList.append(temparray)

    # print(convertedStatusList)

    evaluatedDataFrame = pandas.DataFrame({titleStatusHeading: convertedStatusList[0], favIconStatusHeading: convertedStatusList[1],
                                          tagLinkStatusHeading: convertedStatusList[2], reqUrlStatusHeading: convertedStatusList[3],       
                                          symbolUrlStatusHeading: convertedStatusList[4], lastPositionStatusHeading: convertedStatusList[5], 
                                          ipAddressStatusHeading: convertedStatusList[6], lengthofURLStatusHeading: convertedStatusList[7], 
                                          dotCountStatusHeading: convertedStatusList[8]
                                          })
    for heading in range(8):
        dataFrame[columnHeadingList[heading]]=evaluatedDataFrame[columnHeadingList[heading]]
    dataFrame.to_csv(filePath,index=False,header=True)

    print("Done appending data")
    print(trueCount)


if __name__ == "__main__":
    main()
