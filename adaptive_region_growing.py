import random
import numpy as np
import cv2

# dimensional reduction and gradient smoothing
def import_and_preprocess(path, scale):
    image = cv2.imread(filename = path, flags = cv2.IMREAD_GRAYSCALE)

	# when shrinking, INTER_AREA looks best
	# when enlarging, INTER_CUBIC looks best, INTER_LINEAR is faster and looks o-k
	image = cv2.resize(src = image, dsize = (int(image.shape[1] * scale), int(image.shape[0] * scale)),
					   interpolation = cv2.INTER_AREA)

	image = cv2.GaussianBlur(src = image, ksize = (5, 5), sigmaX = 0, sigmaY = 0,
					 borderType = cv2.BORDER_REFLECT_101)
	return image

def retain_seeds(event, x, y, flags, userdata):
	global seeds
	if event == cv2.EVENT_LBUTTONDOWN:
		seeds.append((x, y))

def click_seeds(image):
	window = "1. Click Centrally in Desired Region, 2. Press ESC to Exit"
	image = np.array(image, dtype = np.uint8)
	cv2.namedWindow(winname = window, flags = cv2.WINDOW_AUTOSIZE)
	cv2.imshow(winname = window, mat = image)
	cv2.setMouseCallback(window_name = window, on_mouse = retain_seeds)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def within_image(coordinate, image_shape):
	return (0 <= coordinate[0] < image_shape[0]) and (0 <= coordinate[1] < image_shape[1])

def get_neighbor_coordinate(current_coordinate, neighbor, image_shape):
	neighbor_coordinate = (current_coordinate[0] + neighbor[0], current_coordinate[1] + neighbor[1])

	# ascertain that center pixel's connectivity is not out of bounds
	if within_image(neighbor_coordinate, image_shape):
		return neighbor_coordinate
	else:
		return None

def get_neighbor_intensity(current_coordinate, neighbor, image, image_shape):
	neighbor_coordinate = get_neighbor_coordinate(current_coordinate, neighbor, image_shape)

	if neighbor_coordinate is not None:
		intensity = image[neighbor_coordinate[0], neighbor_coordinate[1]]
		return intensity
	else:
		return None

# determine pixel's 8-connected neighbors' intensities mean and standard deviation
def get_neighborhood_mean_and_std(current_coordinate, neighbors, image):
	neighborhood = []

	neighborhood.append(image[current_coordinate[0], current_coordinate[1]])

	for neighbor in neighbors:
		neighbors_intensity = get_neighbor_intensity(current_coordinate, neighbor, image, image.shape)

		if neighbors_intensity is None:
			continue

		neighborhood.append(neighbors_intensity)

	return np.mean(neighborhood), np.std(neighborhood)

def recruit_neighbors_to_active_front(current_coordinate, neighbors, active_front, segmented_region, image_shape):
	for neighbor in neighbors:
		neighbor = get_neighbor_coordinate(current_coordinate, neighbor, image_shape)

		if neighbor is None:
			continue

		if segmented_region[neighbor[0], neighbor[1]] == 0:
			active_front.append(neighbor)
			segmented_region[neighbor[0], neighbor[1]] = 150

	return active_front

# determine the most homogeneous and heterogeneous active front pixels
# in other words, ensure least perturbance to region similarity
def active_front_distances(active_front, mean, image, similarity_metric = "distance"):
	distances = [abs(image[active_front_pixel[0], active_front_pixel[1]] - mean) for active_front_pixel in active_front]

	if len(distances) == 0:
		return -1, -1000, 1000

	smallest_distance = mean - np.amin(distances)
	largest_distance = np.amax(distances) + mean

	if similarity_metric == "distance":
		closest_active_front_pixel = np.argmin(distances)
		return closest_active_front_pixel, smallest_distance, largest_distance
	else:
		random_active_front_pixel = random.choice(list(enumerate(distances)))[0]
		return random_active_front_pixel, smallest_distance, largest_distance

def nonadaptive_region_growing(seeds, image):
	global nonadaptive_region
	print("Performing Non-Adaptive Region Growing: ")

	for seed in seeds:
		print("Growing from Seed ({}, {})...".format(seed[0], seed[1]))

		current_coordinate = [seed[0], seed[1]]

		# if the seed has been acquired by a previous seed's grown region, skip
		if nonadaptive_region[current_coordinate[0], current_coordinate[1]] == 255:
			print("Seed ({}, {}) was previously segmented.".format(seed[0], seed[1]))
			continue

		# similarity criterion is initialized as seed's 8-connectivity neighborhood intensities mean
		mean, _ = get_neighborhood_mean_and_std(current_coordinate, neighbors, image)
		all_seeded_regions_means.append(mean)

		lower_threshold = mean * 0.5
		smallest_distance = mean

		upper_threshold = mean * 1.5
		largest_distance = mean

		active_front = []

		while smallest_distance > lower_threshold and largest_distance < upper_threshold:
			nonadaptive_region[current_coordinate[0], current_coordinate[1]] = 255
			active_front = recruit_neighbors_to_active_front(current_coordinate, neighbors, active_front,
															 nonadaptive_region, image.shape)
			closest_active_front_pixel, smallest_distance, largest_distance = active_front_distances(active_front, mean, image)

			if closest_active_front_pixel == -1:
				break

			current_coordinate = active_front[closest_active_front_pixel]
			del active_front[closest_active_front_pixel]

		print("Non-Adaptively Grown Region mean for seed ({}, {}): {}".format(seed[0], seed[1], np.round(mean, 3)))

# update homogeneity parameters as region grows
def adaptive_updates(prev_mu, new_x, std, region_size, weight):
	new_mu = (prev_mu * (region_size - 1) + new_x) / region_size
	new_std = np.sqrt((1 / region_size) * ((region_size - 1) * (std ** 2) + (new_x - new_mu) * (new_x - prev_mu)))
	c = 20.0 / np.sqrt(region_size)
	lower_threshold = new_mu - (new_std * weight + c)
	upper_threshold = new_mu + (new_std * weight + c)

	return new_mu, new_std, lower_threshold, upper_threshold

# exploratory growing for obtaining estimates of homogeneity parameters
def first_region_growing(seeds, image, weight = 1.5):
	global explored_region

	print("Performing Exploratory Adaptive Region Growing: ")
	for seed in seeds:
		print("Growing from seed ({}, {})...".format(seed[1], seed[0]))

		current_coordinate = [seed[0], seed[1]]

		if explored_region[current_coordinate[0], current_coordinate[1]] == 255:
			print("Seed ({}, {}) was previously segmented.".format(seed[1], seed[0]))
			continue

		region_size = 1

		# homogeneity criterion initialized through seed selection as before
		mean, std = get_neighborhood_mean_and_std(current_coordinate, neighbors, image)

		# thresholds asserted from Pohle and Toennies (2002)
		c = 20.0/np.sqrt(region_size)

		lower_threshold = mean - (std * weight + c)
		smallest_distance = mean

		upper_threshold = mean + (std * weight + c)
		largest_distance = mean

		active_front = []
		while smallest_distance > lower_threshold and largest_distance < upper_threshold:
			explored_region[current_coordinate[0], current_coordinate[1]] = 255
			active_front = recruit_neighbors_to_active_front(current_coordinate, neighbors, active_front,
															 explored_region, image.shape)
			random_active_front_pixel, smallest_distance, largest_distance = active_front_distances(active_front, mean, image,
																									similarity_metric = "random")

			if random_active_front_pixel == -1:
				break

			current_coordinate = active_front[random_active_front_pixel]
			new_x = float(image[current_coordinate[0], current_coordinate[1]])
			region_size += 1

			mean, std, lower_threshold, upper_threshold = adaptive_updates(mean, new_x, std, region_size, weight)
			del active_front[random_active_front_pixel]

		print("Exploratory Adaptively Grown Region mean, standard deviation for seed ({}, {}): {}, {}".format(
			seed[1], seed[0], np.round(mean, 3), np.round(std, 3)))

	return mean, std, region_size

# second region growing is parameterized using prior explored region's homogeneity estimate
# weight is increased to 2.58 to encompass 99% of region assuming normally distributed
def second_region_growing(seeds, image, learned_mean, learned_std, region_size, weight = 2.58):
	global adaptive_region

	print("Performing Final Adaptive Region Growing: ")
	for seed in seeds:
		print("Growing from seed ({}, {})...".format(seed[0], seed[1]))
		current_coordinate = [seed[0], seed[1]]
		if adaptive_region[current_coordinate[0], current_coordinate[1]] == 255:
			print("Seed ({}, {}) was previously segmented.".format(seed[0], seed[1]))
			continue

		# second region growing is parameterized using prior explored region's homogeneity estimate
		mean, std = learned_mean, learned_std

		# previous denominator to not skew our precious learned mean
		region_size = region_size

		# thresholds asserted from Pohle and Toennies (2002)
		lower_threshold = mean - (std * weight)
		smallest_distance = mean

		upper_threshold = mean + (std * weight)
		largest_distance = mean

		active_front = []

		while smallest_distance > lower_threshold and largest_distance < upper_threshold:
			adaptive_region[current_coordinate[0], current_coordinate[1]] = 255
			active_front = recruit_neighbors_to_active_front(current_coordinate, neighbors, active_front, adaptive_region, image.shape)
			closest_active_front_pixel, smallest_distance, largest_distance = active_front_distances(active_front,
																									 learned_mean,
																									 image)
			if closest_active_front_pixel == -1:
				break

			current_coordinate = active_front[closest_active_front_pixel]
			new_x = float(image[current_coordinate[0], current_coordinate[1]])
			region_size += 1

			# finetune homogeneity parameters
			mean, std, lower_threshold, upper_threshold = adaptive_updates(mean, new_x, std, region_size, weight)

			del active_front[closest_active_front_pixel]  # remove segmented pixel from active front

		print("Fine-tuned Adaptive Region Growing mean and standard deviation for seed ({}, {}): {}, {}".format(
			seed[0], seed[1], np.round(mean, 3), np.round(std, 3)))

def save_and_display_grown_region(segmented_region, original_image, title, save_name):
	overlay_segmentation = np.asarray(np.maximum(segmented_region, original_image), dtype = np.uint8)
	cv2.imwrite('/Users/julie/Desktop/' + save_name, overlay_segmentation)
	cv2.namedWindow(winname = title, flags = cv2.WINDOW_AUTOSIZE)
	cv2.imshow(winname = title, mat = overlay_segmentation)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':

	image = import_and_preprocess(path = 'IMG_1033.jpg', scale = 0.25)

	# seed pixel's 8-connectivity neighbors in N, S, E, W, NW, NE, SW, SE directions
	neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

	seeds = []
	click_seeds(image)

	# preserve a copy before normalizing values between 0, 1
	original_image = image
	image = image.astype('float32') / 255.0

	# separate matrix to track segmentation progress
	nonadaptive_region = np.zeros(image.shape)
	nonadaptive_region_growing(seeds, image)
	save_and_display_grown_region(nonadaptive_region, original_image, title = "Non-Adaptively Grown Region",
								  save_name = "nonadapted_rg.png")

	explored_region = np.zeros(image.shape)
	learned_mean, learned_std, region_size = first_region_growing(seeds, image)
	save_and_display_grown_region(explored_region, original_image, title = "Explored Region",
								  save_name = "adapted_rg1.png")

	adaptive_region = np.zeros(image.shape)
	second_region_growing(seeds, image, learned_mean, learned_std, region_size)
	save_and_display_grown_region(adaptive_region, original_image, title = "Adaptively Grown Region",
								  save_name = "adapted_rg2.png")
