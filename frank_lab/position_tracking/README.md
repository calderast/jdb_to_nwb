Some code to find the centroids of each hex given a video file(s). These coordinates should be the same across multiple sessions as long as the maze and camera do not move.

`hex_coordinates.csv` contains hex centroids in video pixel coordinates for Xulu's maze, based on the videos in the `video_files/` folder. In spyglass, we assign a rat's raw position (in pixels) to the corresponding hex using these coordinates.

This replaces `Xulu_hex_coordinates.xlsx`, which contains hex centroids in both pixels and cm (given the scale factor = 0.16 cm per pixel) for Xulu's maze, based on the videos in the `video_files/` folder. (The centroids are the same in these 2 files, but we use a csv with columns 'hex', 'x', 'y' for adding centroids to the nwbfile to match Berke Lab format.)