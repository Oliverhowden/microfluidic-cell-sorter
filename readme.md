# Microsorter

Microsort is an object detector and classifier for sorting cells in a microfluidic channel developed for my Biomedical Engineering Honours research.
Each time a cell passes through the channel it is classified, and a `serial.write` is used to control the physical/electrical valve actuation on chip via Arduino based on the classification result.


![Cell Sorter](https://i.imgur.com/k1kYN2T.jpg)

Simple architecture:
  - Background Subtraction
  - Contours
  - Transfer Learning (Mobilenet224) for classification

### Usage


Install Conda first!
https://www.anaconda.com/download/

Then:
```sh
$ cd microfluidic-cell-sorter
$ conda env create -f microsorter.yml
$ conda activate microsorter
$ cd app
$ python main.py
```

# Areas for improvement
- Currently single threaded and it haults on classification
- Occlusion prevention, on one hand would be good though also not an issue as a production use channel would only be cell*cell dimensions.

I have more data acquisitions if anyone is interested.
Enjoy!
