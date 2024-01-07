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


class DrawingCountingInfo:
    """
        A class that counts the number of unique cars entering specified regions in frames and displays this information.
    """
    def __init__(self, regions):
        self.max_color = 10
        self.regions_colors = np.random.randint(low=0, high=255, size=(self.max_color, 3), dtype='uint8')
        # Keep record of unique id tracks that entered each region
        self.regions_count = {i:set() for i in range(len(regions))}
        
        # if polygons for counting objects are defined, convert
        # them to appropriate data type for using with opencv
        if regions is not None:
            for polygon_idx in range(len(regions)):
                regions[polygon_idx] = np.array(regions[polygon_idx], dtype=np.int32)
        self.polygons = regions

    def draw_counting_info(self, image, bounding_boxes, tracking_ids):
    
        # Create a copy of image
        image_cp = np.copy(image)
        
        # Draw the polygons on the image
        for polygon_id, polygon_i in enumerate(self.polygons):
            color_r, color_g, color_b = self.regions_colors[polygon_id % self.max_color, :]
            color_r, color_g, color_b = int(color_r), int(color_g), int(color_b)
            color = tuple([color_r, color_g, color_b])
            # Reshape the array to a 2D array with 1 row and as many columns as needed
            polygon_i = polygon_i.reshape((-1, 1, 2))
            # Draw the polygon on the image
            cv.polylines(image_cp, [polygon_i], isClosed=True, color=color, thickness=2)
            polygon_start = (int(round(polygon_i[0, 0, 0])), int(round(polygon_i[0, 0, 1])))
            cv.putText(image_cp, "{}".format(polygon_id+1), polygon_start, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 10), 2)
            
        # Iterate through each bounding box and check it is inside regions or not
        if bounding_boxes is not None:
            for bbox, track_id in zip(bounding_boxes, tracking_ids):
                x1, y1, x2, y2 = bbox
                bottom_center_x = (x1+x2)/2.0
                bottom_center_y = y2
                bottom_center = (int(round(bottom_center_x)), int(round(bottom_center_y)))

                # check the bottom of the bounding box is in which region
                for polygon_idx, polygon_i in enumerate(self.polygons):
                    result = cv.pointPolygonTest(polygon_i, bottom_center, False)
                    if result == True:
                        self.regions_count[polygon_idx].add(track_id)

        # Display number of counted vehicle for each region
        text = ""
        for set_idx in self.regions_count:
            text += ", Region-{}: {}".format(set_idx+1, len(self.regions_count[set_idx]))
        cv.putText(image_cp, text, (15, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image_cp

