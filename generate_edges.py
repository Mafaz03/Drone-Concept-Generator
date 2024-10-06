import cv2 
import numpy as np

class EdgeGeneration():

    def __init__(self, image: np.ndarray):
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.edges = None


    def generate_edges(self, reduce_factor=1, blur_size=5) -> np.ndarray:
        width, height = self.image.shape
        image_resized = cv2.resize(self.image, (height // reduce_factor, width // reduce_factor))
        blurred_image = cv2.GaussianBlur(image_resized, (blur_size, blur_size), 0)
        self.edges = cv2.Canny(blurred_image, threshold1=70, threshold2=200) // 255

        # Update the height and width to the resized image dimensions
        self.edges_height, self.edges_width = self.edges.shape  # Correct this line

        self.edges = cv2.resize(self.edges, (self.edges.shape[0]* reduce_factor, self.edges.shape[1]*reduce_factor))

        return self.edges

    
    def generate_filters_random(self, filter_size, ratio):
        if filter_size <= 2:
            filter_size = 3 

        total_elements = filter_size * filter_size

        num_ones = int(total_elements * ratio)
        num_zeros = total_elements - num_ones

        filter_array = [1] * num_ones + [0] * num_zeros

        # np.random.shuffle(filter_array)
        filter_array_1 = np.array(filter_array).reshape(filter_size, filter_size)

        # np.random.shuffle(filter_array)
        filter_array_2 = np.array(filter_array).reshape(filter_size, filter_size)

        self.filter_size = filter_size

        self.filter_array_1, self.filter_array_2 = filter_array_1, filter_array_2

        return filter_array_1, filter_array_2
    
    def generate_filters_checkerboard_fill(self, ) :
        h, w = self._alter_edges().shape[:2]

        re = np.r_[ w*[0,1] ]             
        ro = np.r_[ w*[1,0] ]   

        self.filter_array_1, self.filter_array_2 = np.row_stack(h*(re, ro)), 1-np.row_stack(h*(re, ro)) 

        return self.filter_array_1, self.filter_array_2
    
    
    def generate_filters_checkered(self, filter_size):
        if filter_size <= 2: filter_size = 3 
        
        one, two = [i%2 for i in range(filter_size)], [(i+1)%2 for i in range(filter_size)]
        if filter_size % 2 == 0:
            filter = np.array([one, two] * (filter_size//2))
        else:
            filter = np.array([one, two] * (filter_size//2) + [one])

        self.filter_size = filter_size
        self.filter_array_1, self.filter_array_2 = filter, 1 - filter
        
        return filter, 1 - filter
    

 
    def _alter_edges(self):
        # Calculate row to add based on the current edge height
        row_to_add = (self.filter_size - (self.edges_height % self.filter_size)) % self.filter_size

        if row_to_add != 0:
            zero_row = np.zeros((row_to_add, self.edges.shape[1]), dtype=np.uint8)  # Use edges.shape[1]
            self.edges = np.vstack((zero_row, self.edges))

        self.edges_height, self.edges_width = self.edges.shape[:2]

        # Calculate column to add based on the current edge width
        column_to_add = (self.filter_size - (self.edges_width % self.filter_size)) % self.filter_size
        if column_to_add != 0:
            zero_column = np.zeros((self.edges_height, column_to_add), dtype=np.uint8) 
            self.edges = np.hstack((zero_column, self.edges))
        
        return self.edges

    def step(self):
        
        new_edges_height, new_edges_width = self._alter_edges().shape[:2]

        for row in range(0, new_edges_height, self.filter_size): 
            for col in range(0, new_edges_width, self.filter_size): 
                if row + self.filter_size <= new_edges_height and col + self.filter_size <= new_edges_width:
                    slice = self.edges[row: row + self.filter_size, col: col + self.filter_size]
                    if row % 2 == 0:
                        self.edges[row: row + self.filter_size, col: col + self.filter_size] = slice * self.filter_array_1
                    else:
                        self.edges[row: row + self.filter_size, col: col + self.filter_size] = slice * self.filter_array_2
    
        return self.edges