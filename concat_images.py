from PIL import Image

import sys

def main (args) :
    base_source_path = 'position_process/quality_plot/distplot'
    image_files = []

    for index in range(1,101) :
        image_files.append(Image.open(base_source_path + '/' + str(index) + '.png'))

    images_width, images_height = image_files[0].width, image_files[1].height

    # We want 10*10 grid
    final_image_width, final_image_height = images_width*10, images_height*10
    final_image_file = Image.new('RGB', (final_image_width, final_image_height))
    counter = 0

    for i in range(0,10) :
        for j in range(0,10) :
            final_image_file.paste(image_files[counter], (j*images_width, i*images_height))
            counter += 1

    final_image_file.save('position_process/quality_plot/final.png')
if __name__ == "__main__":
    # TO RUN: python3 concat_plot.py <source>
    main(sys.argv)