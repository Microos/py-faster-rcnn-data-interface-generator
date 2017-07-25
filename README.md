# Python Faster-RCNN Data Interface Generator

The purpose of this tool is to generate dataset interface of [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn)(FRCN) that can help you to deploy training on your own dataset as fast as possible.


#### Usage
This tool will generate dataset interface like `pascal_voc.py`, `voc_eval.py` and `factory.py`. 
Also a set of net prototxt files like `py-faster-rcnn/models/pascal_voc`. 
And a bash script similar to `py-faster-rcnn/experiments/scripts/faster_rcnn_end2end.sh`

Before using this generator, please make sure you have already set up a voc-like tree structure of your dataset(in your `py-faster-rcnn/data/` folder):

	YOUR_DATASET_devkit
		└── data
    		├── Annotations
    		│   └── *.xml
			│
   			├── ImageSets
    		│   └── Main
    		│       ├── train.txt
    		│       └── test.txt
    		└── JPEGImages
        			└── *.jpg

Then you can run the `generate.py` to generate interface, prototxt and script files

  **arguments:**
  
`--froot: absolute path to your faster-rcnn root.`

`--idname: name your dataset, whatever you want. It will be used in the future training.`

`--cls: class names in your dataset, use comma to split them (e.g: cat,dog,tiger) `

`--dvkt: absolute path to your dataset devkit(mentioned above)`

 **notice that**
 
 Please notice that, the tool will do everything automatically for you except for generating of `factory.py`-like file. 
 
 In case of changing this `factory.py` file without any backup, the tool will not overwrite it but create a new `<idname>_factory.py` file for you. 

 You have 2 options to complete the last step manually:
 
 1. Backup the original `factory.py` file and replace it with the generating one.
 2. Copy the code block displayed in the task summary into the `factory.py`.
  
 
 



#### Reference
[deboc's tutorial](https://github.com/deboc/py-faster-rcnn/blob/master/help/Readme.md)
