# Hex Centroids
Much of our analysis relies on assigning a rat's x,y position to a corresponding hex in the maze.
The [Get_Hex_Centroids.ipynb](Get_Hex_Centroids.ipynb) notebook is used to create a csv file of centroids of each hex (in video pixel coordinates) given a video file of the maze. You may wish to create a new centroids file for each session, or use the same file across multiple sessions as long as the maze and camera do not move.

The `frank_lab` folder contains some video files and associated hex coordinate files for different maze positions for the Frank Lab.

# Electrophyisology
The [`electrophysiology`](electrophysiology/README.md) folder contains information about the various probes used by the Berke Lab in the hex maze task, including channel maps and electrode coordinates.

# Photometry
The [`photometry`](photometry/README.md) folder contains information about the photometry setup and viruses used by the Berke Lab in the hex maze task.