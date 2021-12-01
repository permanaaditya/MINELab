import pixellib
import argparse
from pixellib.instance import custom_segmentation

class SegmentExtractor:
    def __init__(self, args):
        self.model = args.model
        self.source = args.source
        self.output = args.output

    def run(self):
        segment_image = custom_segmentation()
        segment_image.inferConfig(num_classes= 3, class_names= ["Body", "Bow", "Erhu"])
        segment_image.load_model(self.model)
        segment_image.segmentImage(self.source, output_image_name=self.output,
                                   extract_segmented_objects= True, save_extracted_objects=True)

if __name__ == '__main__':
    print("# Segment extraction is starting...\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="model", type=str, help="Training model")
    parser.add_argument("--source", "-s", default="image", type=str, help="Image source")
    parser.add_argument("--output", "-o", default="output", type=str, help="Image output")
    args = parser.parse_args()

    SegmentExtractor(args)

    print("\n# Segment extraction is finished.")