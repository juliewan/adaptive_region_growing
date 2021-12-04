import random
import numpy as np
import cv2
from PIL import Image

# resize, grayscale, gaussian blur
def preprocess_image(image_path, rescale):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # grayscale
	width = int(image.shape[1] * rescale / 100)
	height = int(image.shape[0] * rescale / 100)
	dim = (width, height)
	image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) # resize
	image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT) # Gaussian smoothing
	return image

# user interacted seed selection using mouse click
def left_click(event, x, y, flags, param):
	global seeds
	if event == cv2.EVENT_LBUTTONDOWN:
		seeds.append((x, y))

def pick_seeds(image):
	window_title = "Propose Seeds"
	image = np.array(image, dtype = np.uint8)
	cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
	cv2.setMouseCallback(window_title, left_click)
	cv2.imshow(window_title, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# make sure region exploration is contained within image
def pixel_within_image(pixel_coord, image_shape):
	return (0 <= pixel_coord[0] < image_shape[0]) and (0 <= pixel_coord[1] < image_shape[1])

# return neighbor pixel intensity
def get_neighbor_intensity(image, pixel_coord, orientation, image_shape):
	neighbor_coord = (pixel_coord[0] + orientation[0], pixel_coord[1] + orientation[1]) # neighbor's coordinate
	if pixel_within_image(neighbor_coord, image_shape): # check that neighbor is within image
		neighbor_intensity = image[neighbor_coord[0], neighbor_coord[1]]
		return neighbor_intensity
	else:
		return None

# determine mean and std of 3 x 3 neighborhood around seed
def neighborhood_mean_std(image, pixel_coord, orientations):
	neighborhood = []
	neighborhood.append(image[pixel_coord[0], pixel_coord[1]]) # add center pixel
	for orientation in orientations: # add 8-conn neighbors
		neighbor_intensity = get_neighbor_intensity(image, pixel_coord, orientation, image.shape)
		if neighbor_intensity is None:
			continue
		neighborhood.append(neighbor_intensity)
	neighborhood_mean = np.mean(neighborhood)
	neighborhood_std = np.std(neighborhood)

	return neighborhood_mean, neighborhood_std

# return neighbor's coordinate
def get_neighbor_coord(pixel_coord, orientation, image_shape):
	neighbor_coord = (pixel_coord[0] + orientation[0], pixel_coord[1] + orientation[1])
	if pixel_within_image(neighbor_coord, image_shape): # check that neighbor is within image
		return neighbor_coord
	else:
		return None

# recruit unexamined pixels to active front
def recruit_neighbors(image, pixel_coord, active_front, segmented_region, orientations):
	for orientation in orientations:
		neighbor = get_neighbor_coord(pixel_coord, orientation, image.shape)
		if neighbor is None:
			continue
		if segmented_region[neighbor[0], neighbor[1]] == 0:
			active_front.append(neighbor)
			segmented_region[neighbor[0], neighbor[1]] = 150 # color active front
	return active_front

# randomly absorb any active front pixel to region
# region growing termination maintained by variation thresholds
def random_walk(image, active_front, running_mean):
	af_dist = [abs(image[af_pixel[0], af_pixel[1]] - running_mean) for af_pixel in active_front]
	if len(af_dist) == 0:
		return -1, -1000, 1000
	smallest_distance = running_mean - np.amin(af_dist)
	largest_distance = np.amax(af_dist) + running_mean
	random_af = random.choice(list(enumerate(af_dist)))[0]
	return random_af, smallest_distance, largest_distance

# determine the most homogeneous active front pixel and its intensity value
# and additionally the most heterogeneous active front value
# in other words, ensure least perturbance to segmented region mean
def distance(image, active_front, running_mean):
	af_dist = [abs(image[af_pixel[0], af_pixel[1]] - running_mean) for af_pixel in active_front]
	if len(af_dist) == 0:
		return -1, -1000, 1000
	smallest_distance = running_mean - np.amin(af_dist)
	largest_distance = np.amax(af_dist) + running_mean
	closest_af = np.argmin(af_dist)
	return closest_af, smallest_distance, largest_distance

# update mean, std, and thresholds as region grows
def adaptive_growing(new_x, running_mean, std, region_size, weight):
	prev_mu = running_mean
	new_mu = (prev_mu * (region_size - 1) + new_x) / region_size # update segmented region mean
	new_std = np.sqrt((1 / region_size) * ((region_size - 1) * (std ** 2) + (new_x - new_mu) * (new_x - prev_mu)))  # update std
	c = 20.0 / np.sqrt(region_size)  # decrease c
	upper_threshold = new_mu + (new_std * weight + c)
	lower_threshold = new_mu - (new_std * weight + c)
	return new_mu, new_std, upper_threshold, lower_threshold

#  plain jane region growing with user specified seeds and
#  thresholds chosen for Marie, my predominantly white Ragdoll.
#  homogeneity criterion semi-automatically initialized through seed(s)
def nonadaptive_region_growing(seeds, image):
	print("Performing Non-Adaptive Region Growing: ")
	global nonadaptive_grown_region
	for seed in seeds:
		print("Growing from seed ({}, {})...".format(seed[1], seed[0]))
		pixel_coord = [seed[1], seed[0]]  # seed pixel from which region grows
		if nonadaptive_grown_region[pixel_coord[0], pixel_coord[1]] == 255:
			print("Seed ({}, {}) was previously segmented.".format(seed[1], seed[0]))
			continue  # has already been segmented into a region
		# initialize mean using 3 x 3 seed neighborhood
		mean, std = neighborhood_mean_std(image, pixel_coord, orientations)
		# some arbitrarily defined thresholds
		upper_threshold = mean * 1.49
		lower_threshold = mean * 0.50
		smallest_distance = mean
		largest_distance = mean
		active_front = []

		while (smallest_distance > lower_threshold and largest_distance < upper_threshold):
			nonadaptive_grown_region[pixel_coord[0], pixel_coord[1]] = 255 # segment current pixel
			active_front = recruit_neighbors(image, pixel_coord, active_front, nonadaptive_grown_region, orientations)
			closest_af, smallest_distance, largest_distance = distance(image, active_front, mean)
			if closest_af == -1:
				break  # move onto next seed
			pixel_coord = active_front[closest_af]  # reassign boundary representative
			del active_front[closest_af]  # remove segmented pixel from active front
		print("Non-Adaptive Mean for seed ({}, {}): {}".format(seed[1], seed[0], mean))

#  first exploratory seeded region growing for estimating homogeneity parameter
#  homogeneity criterion semi-automatically initialized through seed(s)
def first_region_growing(seeds, image):
	print("Performing First Adaptive Region Growing: ")
	global explored_grown_region
	weight = 1.5
	for seed in seeds:
		print("Growing from seed ({}, {})...".format(seed[1], seed[0]))
		pixel_coord = [seed[1], seed[0]]  # seed pixel from which region grows
		if explored_grown_region[pixel_coord[0], pixel_coord[1]] == 255:
			print("Seed ({}, {}) was previously segmented.".format(seed[1], seed[0]))
			continue  # has already been segmented into a region
		region_size = 1
		# initialize mean using 3 x 3 seed neighborhood
		running_mean, std = neighborhood_mean_std(image, pixel_coord, orientations)
		# thresholds asserted through Pohle and Toennies (2002) equations
		c = 20.0/np.sqrt(region_size)
		upper_threshold = running_mean + (std * weight + c)
		lower_threshold = running_mean - (std * weight + c)
		smallest_distance = running_mean
		largest_distance = running_mean
		active_front = []

		while (smallest_distance > lower_threshold and largest_distance < upper_threshold):
			explored_grown_region[pixel_coord[0], pixel_coord[1]] = 255 # segment current pixel
			active_front = recruit_neighbors(image, pixel_coord, active_front, explored_grown_region, orientations)
			random_af, smallest_distance, largest_distance = random_walk(image, active_front, running_mean)
			if random_af == -1:
				break  # move onto next seed
			pixel_coord = active_front[random_af]  # reassign boundary representative
			new_x = float(image[pixel_coord[0], pixel_coord[1]])  # newly segmented pixel's intensity
			region_size += 1  # increment segmented region size
			# adaptive growing method so update homogeneity parameters
			running_mean, std, upper_threshold, lower_threshold = adaptive_growing(new_x, running_mean, std, region_size, weight)
			del active_front[random_af]  # remove segmented pixel from active front
		print("Estimated Adaptive Region Growing mean and standard deviation for seed ({}, {}): {}, {}".format(
			seed[1], seed[0], running_mean, std))
	return running_mean, std, region_size

#  second region growing reuses same seed(s) and is parameterized using
#  first grown region's supplied homogeneity estimate
#  weight increased to 2.58 to encompass 99% of region if noise is gaussian
def second_region_growing(seeds, image, learned_mean, learned_std, region_size):
	print("Performing Second Adaptive Region Growing: ")
	global final_grown_region
	weight = 2.58
	for seed in seeds:
		print("Growing from seed ({}, {})...".format(seed[1], seed[0]))
		pixel_coord = [seed[1], seed[0]]
		if final_grown_region[pixel_coord[0], pixel_coord[1]] == 255:
			print("Seed ({}, {}) was previously segmented.".format(seed[1], seed[0]))
			continue  # has already been segmented into a region
		region_size = region_size  # previous denominator to not skew our precious learned mean
		# assert learned homogeneity estimates from first region growing
		running_mean, std = learned_mean, learned_std
		# thresholds asserted through Pohle and Toennies (2002) equations
		upper_threshold = running_mean + (std * weight)
		lower_threshold = running_mean - (std * weight)
		smallest_distance = running_mean
		largest_distance = running_mean
		active_front = []

		while (smallest_distance > lower_threshold and largest_distance < upper_threshold):
			final_grown_region[pixel_coord[0], pixel_coord[1]] = 255  # segment current pixel
			active_front = recruit_neighbors(image, pixel_coord, active_front, final_grown_region, orientations)
			closest_af, smallest_distance, largest_distance = distance(image, active_front, learned_mean)
			if closest_af == -1:
				break  # move onto next seed
			pixel_coord = active_front[closest_af]  # reassign boundary representative
			new_x = float(image[pixel_coord[0], pixel_coord[1]])  # obtain segmented pixel's intensity
			region_size += 1  # increment segmented region size
			# adaptive growing method so still finetune homogeneity parameters
			running_mean, std, upper_threshold, lower_threshold = adaptive_growing(new_x, running_mean, std, region_size, weight)
			del active_front[closest_af]  # remove segmented pixel from active front
		print("Fine-tuned Adaptive Region Growing mean and standard deviation for seed ({}, {}): {}, {}".format(
			seed[1], seed[0], running_mean, std))

# overlay segmentation over image and save
def grown_region_result(original_image, segmented_region, save_name):
	overlay_segmentation = np.maximum(original_image, segmented_region)
	overlay_segmentation = np.asarray(overlay_segmentation, dtype = np.uint8)
	cv2.imwrite('/Users/julie/Desktop/' + save_name, overlay_segmentation)
	title = "Seeded Grown Region"
	cv2.namedWindow(title, cv2.WINDOW_NORMAL)
	cv2.imshow(title, overlay_segmentation)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	image_path = '/Users/julie/Desktop/marie.jpg'
	rescale = 25
	image = preprocess_image(image_path, rescale)

	nonadaptive_grown_region = np.zeros(image.shape)
	explored_grown_region = np.zeros(image.shape)
	final_grown_region = np.zeros(image.shape)
	seeds = []
	orientations = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]  # 8-connected

	pick_seeds(image)
	original_image = image  # preserve unnormalized image
	image = image.astype('float32') / 255.0  # normalize [0, 1]

	save_name1 = "nonadapted_rg.png"
	save_name2 = "adapted_rg1.png"
	save_name3 = "adapted_rg2.png"

	nonadaptive_region_growing(seeds, image)
	grown_region_result(original_image, nonadaptive_grown_region, save_name1)

	learned_mean, learned_std, region_size = first_region_growing(seeds, image)
	grown_region_result(original_image, explored_grown_region, save_name2)

	second_region_growing(seeds, image, learned_mean, learned_std, region_size)
	grown_region_result(original_image, final_grown_region, save_name3)


'''
segmented_region = np.uint8(segmented_region)
print(segmented_region[segmented_region == 255].shape[0])
segmented_image = Image.fromarray(segmented_region, 'L')
segmented_image.show()

roi = cv2.selectROI(windowName = "Select Region of Interest", img = im, fromCenter = False, showCrosshair = False)

y1 = int(roi[1])
y2 = int(roi[3])
x1 = int(roi[0])
x2 = int(roi[2])

crop_im = im[y1:y1+y2, x1:x1+x2]
cv2.imshow("Selected Region of Interest", crop_im)
cv2.imwrite('/Users/julie/Desktop/cropped_roi.jpg', crop_im)
cv2.waitKey(0)
'''
