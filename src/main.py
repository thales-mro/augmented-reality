import numpy as np
from ar import AR


def main():
    """
    Entrypoint for the code of project 02 Group 08 MO446/2sem2019
    """

    # Create the Augmented Reality object
    ar = AR()

    # Generate the image
    ar.execute_video("input/i-0.mp4")


main()

