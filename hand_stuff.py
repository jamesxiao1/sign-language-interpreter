import numpy as np

def normalize_hand(row):
    # a row is 63 nums for one hand
    points=row.reshape(21,3) # make double matrix 21 rows of 3
    points=points-points[0] # no wrist
    size=np.linalg.norm(points,axis=1).max() # normalize
    points=points/size # finish normalize
    return points.reshape(-1) # put it back to 63 flat