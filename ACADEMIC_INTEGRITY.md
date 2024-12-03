## Course Number: ENGR 13300

## Semester: Fall 2024

### Name: Manav Gagvani

Description:


    This repository is split into 3 main files:
    1. `model.py` - Contains definition of PyTorch model that is used for end-to-end autonomous driving. Its main function contains a training loop that will train the model on the data in `data/`. The model architecture is the Nvidia PilotNet CNN (Bojarkski et al.)

    2. `record_data.py` - This file contains the code necessary to record data which comprises of a dataset. This data is stored as JPG images while corresponding steering and throttle values are written in a CSV file. This file uses the `MetaDrive` library which is a lightweight graphical simulator for cars. 

    3. `utils.py` contains miscellaneous pieces of code. These include the `Data` class which is the universal data loader and parses the format used in `record_data.py`. Furthermore, it includes functions for evaluating a trained model and visualizing a dataset as a video. It can be run with varying command line arguments to perform different functinos.

Assignment Information:

    Assignment:     Individual Project

    Team ID:        LC1 - 27

    Author:         Manav Gagvani, mgagvani@purdue.edu

    Date:           11/21/2024

Academic Integrity Statement:

    I have not used source code obtained from any unauthorized source, either modified or   unmodified; nor have I provided another student access to my code.  The project I am  submitting is my own original work.