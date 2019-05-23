# Target Localization using Feature Matching
B.Tech Project 5th semester

Read the report and PPT for more details.
The code is written in Python 3.6 and uses libraries OpenCV, OpenCV Contrib, Numpy, Tkinter and Pillow. Make sure these are installed properly.

To run the code, find the index of camera for video capture. In our case, it is 2. However, it can easily be changed by modifying the camindex variable in the ``runall.py`` file.

To run the file, execute the following command on linux terminal. We are using Kubuntu 18.10.

``python3 runall.py 2> errorlog.txt``

This will start execution and write any errors into errorlog.txt file.
To stop execution, do a keyboard interrupt on terminal by pressing Ctrl+C.
Results from all runs are stored in stats.txt file. To get the results in proper format from all the data, execute the genstats.py file with the following command:

``python3 genstats.py``
