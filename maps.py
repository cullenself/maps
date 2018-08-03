#imports
from osgeo import gdal, gdal_array
import numpy as np
import math
import json
from geojson import LineString

dataset = gdal.Open('map', gdal.GA_ReadOnly)
if not dataset:
    raise IOError('GDAL failed to open map')
srcArray = gdal_array.DatasetReadAsArray(dataset).T # Must transpose so that can access [X,Y]
GT = dataset.GetGeoTransform()
cartPositions = {}
start = (-105.81171870231627,40.42018594917153)
end = (-105.77877581119537,40.372110495658944)

def GT2Transform(GT):
    # GT is GDAL GeoTransform
    # Put into Linear Algebra Notation for ease of manipulation
    offset = np.matrix([[GT[0]],[GT[3]]])
    Q = np.matrix([[GT[1], GT[2]],[GT[4],GT[5]]])
    return offset, Q

offset, Q = GT2Transform(GT)

def pos2coord(pos):
    # position should be tuple (X, Y)
    # output is (E, N) tuple
    return tuple((offset + Q*np.matrix(pos).T).A1)

def coord2pos(coord):
    # coordinate is tuple (E, N)
    # output is (X, Y) tuple
    return tuple((np.linalg.inv(Q)*(np.matrix(coord).T - offset)).A1)
#gui

#pathfinding
def path(start, end):
    print('Starting Pathfinding')
    print('Beginning:')
    print(start)
    print('Goal:')
    print(end)
    # start and end are both (X, Y) tuples
    # output is a list of positions
    closedSet   = set()
    openSet     = set()
    gCost       = {}
    fCost       = {}
    cameFrom    = {}

    # Initialize Open Set
    openSet.add(start) # If constructed with set(start), Python adds X and Y rather than (X,Y)
    gCost[start] = 0
    fCost[start] = estimate(start,end)

    while end not in closedSet: # might get stuck in infinite loop TODO: change to openSet not empty
        openScores = {pos: fCost[pos] for pos in fCost if pos in openSet} # This line is slow TODO
        current = min(openScores, key=openScores.get)
        closedSet.add(current)
        openSet.remove(current)
        print('Current: (%d,%d)' % current, end='       \r')

        for adjacent in getAdjacent(current):
            if adjacent in closedSet: # Already been visited
                continue # maybe address changing gCosts
            posGCost = gCost[current] + cost(current,adjacent) # Cost Function
            if adjacent not in openSet: # Never seen before
                openSet.add(adjacent)
                gCost[adjacent] = posGCost
            if posGCost <= gCost[adjacent]: # This is the current most efficient route
                cameFrom[adjacent] = current
                gCost[adjacent] = posGCost
                fCost[adjacent] = posGCost + estimate(adjacent, end)
    print('\nDone')
    path_taken = list()
    path_taken.append(current)
    while current in cameFrom:
        current = cameFrom[current]
        path_taken.append(current)
    path_taken.reverse()
    return path_taken

def estimate(curr, end):
    # Generate Estimate of Cost
    # Current method: be lazy and just pass straight to cost()
    return cost(curr, end)

def findDistance(a, b):
    # Straight line distance ||a-b|| (including elevation)
    return np.linalg.norm(pos2cart(a) - pos2cart(b))

def cost(a, b):
    # Weighted Score between straight line distance and elevation change
    distWeight = 1
    elevationWeight = 100
    return distWeight * findDistance(a,b)**2 + elevationWeight * findElevation(a,b)**2

def getAdjacent(pos):
    adj = [[1,0], [-1,0], [0,1], [0,-1]]
    options = [np.array(pos) + a for a in adj]
    options = [tuple(pos) for pos in options if all(pos>=0) and all(pos<np.array(srcArray.shape))]
    return options

def pos2cart(pos):
    if pos not in cartPositions:
        coord = pos2coord(pos)
        alt = srcArray[pos]
        longitude = coord[0]*math.pi/180
        latitude  = coord[1]*math.pi/180

        a = 6378137.0 # (m) from GRS80 (not WGS84)
        e2 = 0.006694380022903415749574948586289306212443890 # GRS80
        x = (a * math.cos(longitude) / math.sqrt(1 + (1-e2) * math.tan(latitude)**2)) + alt * math.cos(longitude)*math.cos(latitude)
        y = (a * math.sin(longitude) / math.sqrt(1 + (1-e2) * math.tan(latitude)**2)) + alt * math.sin(longitude)*math.cos(latitude)
        z = (a * (1-e2) * math.sin(latitude) / math.sqrt(1-e2 * math.sin(latitude)**2)) + alt * math.cos(latitude)
        cartPositions[pos] = np.array([x,y,z])
    return cartPositions[pos]

def lines(start,end):
    # Start and end are coordinate tuples
    start_pos = tuple(map(lambda x: int(round(x)),coord2pos(start)))
    end_pos = tuple(map(lambda x: int(round(x)),coord2pos(end)))
    
    foundPath = path(start_pos, end_pos)
    path_coord = [pos2coord(pos) for pos in foundPath]
    
    # Create GeoJSON object
    obj = {'type':'FeatureCollection','features':[{}]}
    obj['features'][0] = {'type': 'Feature', 'properties': {}}
    obj['features'][0]['geometry'] = LineString(path_coord)
    with open('output.json', 'w') as fp:
        json.dump(obj, fp)

    return pathStats(foundPath)

def pathStats(path):
    # Ideal Distance
    print('Ideal Distance: %.2f' % (findDistance(path[0],path[-1])))
    # Ideal Elevation Gain
    print('Ideal Elevation Gain: %.2f' % (findElevation(path[0],path[-1])))
    # Actual Distance
    print('Actual Distance: %.2f' % (sum([findDistance(path[t],path[t+1]) for t in range(len(path)-1)])))
    # Actual Elevation Gain
    elevationChange = [findElevation(path[t],path[t+1]) for t in range(len(path)-1)]
    print('Actual Elevation: %.2f' % (sum(elevationChange)))
    elevationProfile = [srcArray[pos] for pos in path]
    return elevationProfile

def findElevation(a, b):
    dEl = srcArray[b] - srcArray[a]
    return dEl if dEl > 0 else 0
