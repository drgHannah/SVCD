import os
from svcd import SVCDutils, SVCDSegmentation

if __name__ == "__main__":
    loc = os.path.dirname(os.path.realpath(__file__))
    image = SVCDutils.load_image(os.path.join(loc,"../img/kea.jpg"))
    scribble, colors = SVCDutils.load_segmentation_from_image(
        os.path.join(loc,"../img/kea_scribble.png"))
    svcd = SVCDSegmentation(max_iter=1000)
    segmentation, energies = svcd(image, scribble)
    SVCDutils.save_segmentation_as_image("kea_seg.png", segmentation, colors)
