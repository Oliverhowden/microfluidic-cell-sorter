# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class Tracker():
    def init(self, bounds):
        self.bounds = bounds
        self.nextcellID = 0
        self.cells = OrderedDict()
        self.roiImages = OrderedDict()
        self.classificationResults = OrderedDict()
        self.hasBeenClassified = OrderedDict()
        self.disappearedCells = OrderedDict()
        self.maxdisappearedCells = 1

    def returnAllKnownCellIDs(self):
        return self.cells

    def returnAllCellResults(self, id):
        cellResults = {
            'roiImage': self.roiImages.get(id, None),
            'location': self.cells.get(id, None),
            'hasBeenClassified': self.hasBeenClassified.get(id, None),
            'classificationResult': self.classificationResults.get(
                id, None),
            'dissapearedFramesCount': self.disappearedCells.get(
                id, None)}

        return cellResults

    def setRoiForID(self, img, id):
        self.roiImages[id] = img

    def getROIFromID(self, id):
        return self.roiImages.get(id, None)

    def hasThisCellBeenClassified(self, id):
        return self.hasBeenClassified.get(id, False)

    def setClassificationBoolTrue(self, id):
        self.hasBeenClassified[id] = True

    def setClassificationResult(self, id, result):
        self.classificationResults[id] = result
        self.setClassificationBoolTrue(id)

    def getClassificationResult(self, id):
        # return self.classificationResults[id]
        return self.classificationResults.get(id, None)

    def returnCurrentYFromCellID(self, id):
        self.x, self.y = self.cells[id]
        return self.y

    def registerNewcell(self, middlePoint):
        self.cells[self.nextcellID] = middlePoint
        self.disappearedCells[self.nextcellID] = 0
        self.nextcellID += 1

    def deregisterOldcell(self, cellID):
        del self.cells[cellID]
        del self.disappearedCells[cellID]

    def updateTracker(self, rects):
        # make sure we have rects to compute

        if len(rects) == 0:
            # mark all objects as dissapeared, if this is incorrect
            # the next step will reappear them
            for cellID in list(self.disappearedCells.keys()):
                self.disappearedCells[cellID] += 1

                # unregister
                if self.disappearedCells[cellID] > self.maxdisappearedCells:
                    self.deregisterOldcell(cellID)
                # TODO Maybe remove this, its causing problems.
                elif self.returnCurrentYFromCellID(cellID) > self.bounds:
                    self.deregisterOldcell(cellID)

            # nothing to update, therefore return
            return self.cells

        # init middlePoints for this frame
        inputmiddlePoints = np.zeros((len(rects), 2), dtype="int")

        for (i, (sX, sY, eX, eY)) in enumerate(rects):
            # derive middlepoint
            mX = int((sX + eX) / 2.0)
            mY = int((sY + eY) / 2.0)
            inputmiddlePoints[i] = (mX, mY)

        # register new untracked cells
        if len(self.cells) == 0:
            for i in range(0, len(inputmiddlePoints)):
                self.registerNewcell(inputmiddlePoints[i])

        # if they already exist, then we need to computer the distances and match them
        else:
            # get cell IDs and middlePoints
            cellIDs = list(self.cells.keys())
            cellmiddlePoints = list(self.cells.values())

            # compare all input middlepoints to existing middlepointsself.
            # first step is to match to existing ones
            D = dist.cdist(np.array(cellmiddlePoints), inputmiddlePoints)

            # therefor, find the smallest values present in the distances array
            # sort the array in this min->max order
            rows = D.min(axis=1).argsort()

            # same as above for cols
            cols = D.argmin(axis=1)[rows]

            # decide what we're going to do based on whether its been done before
            # ie register, deregister or update. So mark what we've checked already
            alreadyUsedRows = set()
            alreadyUsedCols = set()

            # zip them and iterate over at the same time
            for (row, col) in zip(rows, cols):
                # if we've checked it, jump to the next one
                if row in alreadyUsedRows or col in alreadyUsedCols:
                    continue

                # if we haven't, get the cell id and set its middlepoint,
                # at the same time reset the disappearedCounter we added to in step 1

                cellID = cellIDs[row]
                self.cells[cellID] = inputmiddlePoints[col]
                self.disappearedCells[cellID] = 0

                # Add it to the alreadyUsedRows/Cols array so we know we've worked on this already.
                alreadyUsedRows.add(row)
                alreadyUsedCols.add(col)

            # compute the indexes we haven't looked at yet.
            notUsedRows = set(range(0, D.shape[0])).difference(alreadyUsedRows)
            notUsedCols = set(range(0, D.shape[1])).difference(alreadyUsedCols)

            # if we have more middlepoints than inputs, we may have lost an object somewere along the way
            # this could be getting stuck behind dust on the lense etc
            if D.shape[0] >= D.shape[1]:
                for row in notUsedRows:
                    # therefor, find the culprit and increment the dissapeared counter
                    cellID = cellIDs[row]
                    self.disappearedCells[cellID] += 1

                    # check if its greater than our max allowed set in the init method
                    if self.disappearedCells[cellID] > self.maxdisappearedCells:
                        self.deregisterOldcell(cellID)

            # last option to exhaust is if we have more objects than we are tracking,
            # if thats the case we have new objects appearing, and we need to register it/them
            else:
                for col in notUsedCols:
                    self.registerNewcell(inputmiddlePoints[col])

        # respond with all the cells that are trackable
        return self.cells
