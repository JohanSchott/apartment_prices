import cv2
import imageio
import argparse
import os


def pngs_to_movie(png_filenames, movie_filename='output.mp4', codec='mp4v', fps=20.0):
    frame = cv2.imread(png_filenames[0])
    #cv2.imshow('video', frame)
    height, width, channels = frame.shape
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec) # Be sure to use lower case
    out = cv2.VideoWriter(movie_filename, fourcc, fps, (width, height))
    for png_filename in png_filenames:
        frame = cv2.imread(png_filename)
        out.write(frame) # Write out frame to video
        #cv2.imshow('video', frame)
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
    print("The output video is {}".format(movie_filename))


def pngs_to_gif(png_filenames, movie_filename='output.gif', fps=20.0):
    images = []
    for filename in png_filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(movie_filename, images, duration=1 / fps)


def main():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
    ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
    args = vars(ap.parse_args())

    # Arguments
    dir_path = '.'
    ext = args['extension']
    output = args['output']

    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)

    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    codec = 'mp4v'
    codev = 'gif'
    fourcc = cv2.VideoWriter_fourcc(*codec) # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    for image in images:

        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)

        out.write(frame) # Write out frame to video

        cv2.imshow('video',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))

if __name__ == "__main__":
    main()
