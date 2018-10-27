import os
import Augmentor
dataPath = "/Users/oliver/Developer/Python/Design/FinalApp/scripts/created_dataset"
p = Augmentor.Pipeline(dataPath)
p.rotate90(probability=0.5)
p.rotate270(probability=0.5)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.crop_random(probability=1, percentage_area=0.5)
p.resize(probability=1.0, width=120, height=120)

p.sample(2500)
