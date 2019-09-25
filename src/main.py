from argparse import ArgumentParser
from ar import AR


def main():
    """
    Entrypoint for the code of project 02 Group 08 MO446/2sem2019
    """
    
    # Create the parser
    parser = ArgumentParser()
    parser.add_argument('--source', dest='comparison', action='store_false')
    parser.add_argument('--frame', dest='comparison', action='store_true')

    ## Parse the args
    args = parser.parse_args() 
    
    # Create the Augmented Reality object
    ar = AR('input/i-2.jpg', 'input/i-1.jpg')
    
    # Set the output filename
    if bool(args.comparison):
        output_name = "o-0.mp4"
    else:
        output_name = "o-1.mp4"    

    # Generate the ar video
    ar.execute("input/i-0.mp4", "output/" + output_name, compare_frame_by_frame=bool(args.comparison), max_frames=10)


main()
