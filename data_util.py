import os 
import cv2 as cv

class ReadData:

    def __init__(self, input_type, input_image_dir=None, input_video_path=None):
        """
        Initializes an instance of the class with the 
        specified input type, image directory, and video path.
        """
        if input_type != 'images' and input_type != 'video':
            raise ValueError('Input type is not correct!')

        if input_type == 'images':
            self.path_images = self.get_sorted_image_paths(path_img_dir=input_image_dir) 
            self.data_generator = self.get_next_image_from_image_directory
        elif input_type == 'video':
            self.video_path = input_video_path
            self.data_generator = self.get_next_image_from_video
        
    def get_sorted_image_paths(self, path_img_dir):
        """
        Retrieve and return a list of sorted full paths for image files in the specified directory.

        Parameters:
        - path_img_dir (str): The path of the directory containing image files.

        Returns:
        - list: A sorted list of full paths for image files in the directory.
        """

        # Get a list of all files in the folder
        files = os.listdir(path_img_dir)

        # Filter only the files with image extensions (you can customize this list)
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        image_files = [file for file in files if any(file.lower().endswith(ext) for ext in image_extensions)]

        # Sort the image files by name
        sorted_image_files = sorted(image_files)

        # Create a list of full paths for the sorted image files
        sorted_image_paths = [os.path.join(path_img_dir, file) for file in sorted_image_files]

        return sorted_image_paths

    def get_next_image_from_image_directory(self):
        """
        Generator function that yields images from the specified image directory.
        """
        for path_img_i in self.path_images:
            img_i = cv.imread(filename=path_img_i)
            if len(img_i.shape) == 3 and img_i.shape[2] == 3: 
                img_i = cv.cvtColor(img_i, cv.COLOR_BGR2RGB)
            yield img_i
        return
    
    def get_next_image_from_video(self):
        """
        Generator function to read and yield consecutive frames from a video file.
        """
        # Input video reader
        in_video = cv.VideoCapture(self.video_path)

        while True:
            ret, img_i = in_video.read()
            if ret == False:
                break
            if len(img_i.shape) == 3 and img_i.shape[2] == 3: 
                img_i = cv.cvtColor(img_i, cv.COLOR_BGR2RGB)
            yield img_i
        
        in_video.release()
        return
    