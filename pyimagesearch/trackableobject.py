class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False

		##ADDED###
		# initialize a boolean to indicate if the vehicle was detected
		# above the line at the entrance

		self.entered = False

		self.already_in_lot = False
