#Python Faster-RCNN Data Interface Generator

The purpose of this tool is to generate dataset interface of Faster-RCNN(FRCN) that can help you to deploy training on your own dataset as fast as possible.


####Usage
This tool will generate dataset interface like `pascal_voc.py`, `voc_eval.py` and `factory.py`. 
Also a set of net prototxt files like `py-faster-rcnn/models/pascal_voc`. 
And a bash script similar to `py-faster-rcnn/experiments/scripts/faster_rcnn_end2end.sh`

Before using this generator, please make sure you have already set up a voc-like tree structure of your dataset:

	PASCAL_VOC_devkit
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

`--idname: name your dataset, whatever you want. It will be used in the furture training.`

`--cls: class names in your dataset, use comma to split them (e.g: cat,dog,tiger) `

`--dvkt: absolute path to your dataset devkit(mentioned above)`
