import cv2 as cv
import numpy as np

class DrawingTrackingInfo:

    def __init__(self):
        self.trajectory_len = 50
        self.max_color = 150
        self.tracks_id_colors = np.random.randint(low=0, high=255, size=(self.max_color, 3), dtype='uint8')
        self.tracks = {}

    def draw_tracking_info(self, image, bounding_boxes, tracking_ids):
        """
        Create image of bounding boxes and tracking IDs on the input image.

        Parameters:
        - image: The input image (numpy array).
        - bounding_boxes: List of n * 4 bounding boxes in the format (x, y, width, height).
        - tracking_ids: List of tracking IDs corresponding to each bounding box.

        Returns:
        - None (displays the image with bounding boxes and tracking IDs).
        """
        # Create a copy of image
        image_cp = np.copy(image)
        # Iterate through each bounding box and tracking ID
        for bbox, track_id in zip(bounding_boxes, tracking_ids):
            x1, y1, x2, y2 = bbox
            
            # Draw the bounding box on the image
            color_r, color_g, color_b = self.tracks_id_colors[track_id % self.max_color, :]
            color_r, color_g, color_b = int(color_r), int(color_g), int(color_b)
            color = tuple([color_r, color_g, color_b])
            cv.rectangle(image_cp, (x1, y1), (x2, y2), color, 2)

            # Keep record of previous position of each unique track id
            if track_id not in self.tracks:
                self.tracks[track_id] = [bbox]
            else:
                self.tracks[track_id].append(bbox)

            # Draw trajectory of tracked object in some of the last frames
            for bbox_i in self.tracks[track_id][-self.trajectory_len:]:
                circle_x, circle_y = (bbox_i[0] + bbox_i[2])//2, bbox_i[3]
                cv.circle(image_cp, (circle_x, circle_y), 3, color, 2) 

            # Display the tracking ID near the bounding box
            text = f"ID: {track_id}"
            cv.putText(image_cp, text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image_cp



