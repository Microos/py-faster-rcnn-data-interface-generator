#!/usr/bin/env python
#encoding: utf8
import os,sys
import argparse
from argparse import RawTextHelpFormatter
import templates

DEBUG = False
std_dataset_tree_str = '''
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

'''
default_mark = 'O(..)O'




#from http://stackoverflow.com/questions/12700893/how-to-check-if-a-string-is-a-valid-python-identifier-including-keyword-check
#used to make sure `idname` is a valid identifier
import ast
def isidentifier(ident):
    if not isinstance(ident, str):
        raise TypeError('expected str, but got {!r}'.format(type(ident)))
    try:
        root = ast.parse(ident)
    except SyntaxError:
        return False

    if not isinstance(root, ast.Module):
        return False

    if len(root.body) != 1:
        return False

    if not isinstance(root.body[0], ast.Expr):
        return False

    if not isinstance(root.body[0].value, ast.Name):
        return False

    if root.body[0].value.id != ident:
        return False
    return True


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='A tool to generate Faster R-CNN dataset interfaces.\n'
                                                 'Please make sure your that dataset devkit has a same tree structure of pascal_voc and also the xml annotation format.\n'
                                                 '\nA VOC-like devkit tree structure:{}\n'.format(std_dataset_tree_str),
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('--froot', dest='FASTER_RCNN_ROOT',
                        help='The full path to your Faster RCNN ROOT folder',
                        default=default_mark, type=str)

    parser.add_argument('--idname', dest='NAMEYOURDATASET',
                        help='Assign a name to your dataset. It will be used in the future training as and ID',
                        default=default_mark, type=str)

    parser.add_argument('--cls', dest='CLASSESNAME',
                        help='A List of classes, separate by a comma(e.g. cat,dog,tiger)',
                        default=default_mark, type=str)


    parser.add_argument('--dvkt', dest='DEVKITPATH',
                        help='The full path to your dataset devkit folder',
                        default=default_mark, type=str)

    
    # if not DEBUG and len(sys.argv) < 4:
    #      parser.print_help()
    #
    #      sys.exit(1)

    args = parser.parse_args()
    return args, parser

def parse_classes():
    clses = (arg.CLASSESNAME).split(',')

    #check to prevent: 'a,b,c,'
    for cls in clses:
        if not len(cls) > 0:
            print("[!] Your input classes doesn't match the format. Input:",arg.CLASSESNAME)
            print('[!] Example: a,b,c,d,e')
            sys.exit(1)
    clses = [c.strip() for c in clses]
    return  "'"  +   "','".join(clses) + "'", len(clses) #gives you 'a','b','c','d','e' AND number of classes

def check_exists(FASTER_RCNN_ROOT, DEVKITPATH):
    if not  (os.path.isdir(FASTER_RCNN_ROOT) and os.path.isabs(FASTER_RCNN_ROOT)):
        print("[!] Your Faster_RCNN_ROOT: '{}' not found. No such a directory. \n\tPlease make sure it is a absolute path.".format(FASTER_RCNN_ROOT))
        sys.exit(1)


    if not (os.path.isdir(DEVKITPATH) and os.path.isabs(DEVKITPATH)):
        print("[!] Your dataset Devkit: '{}' not found. No such a directory. \n\tPlease make sure it is a absolute path.".format(DEVKITPATH))
        sys.exit(1)


    for tf in [os.path.join(FASTER_RCNN_ROOT,'experiments','scripts'),os.path.join(FASTER_RCNN_ROOT,'lib','datasets'),os.path.join(FASTER_RCNN_ROOT, 'models')]:
        if os.path.isdir(tf) == False:
            print("[!] Directory :'{}' not found.".format(tf))
            sys.exit(1)



def gen_files(NAMEYOURDATASET, TOKEN_1, CLASSESNAME,TOKEN_2, DEVKITPATH, TOKEN_3):
    x_str = templates.x_template
    eval_str = templates.eval_template
    fac_str = templates.fac_template
    sfac_str = templates.short_fac_template
    sh_str = templates.sh_template
    cfg_str = templates.cfg_template

    x_str = x_str.replace(TOKEN_1, NAMEYOURDATASET).replace(TOKEN_2, CLASSESNAME).replace(TOKEN_3, DEVKITPATH)
    eval_str = eval_str.replace(TOKEN_1, NAMEYOURDATASET).replace(TOKEN_2, CLASSESNAME).replace(TOKEN_3, DEVKITPATH)
    fac_str = fac_str.replace(TOKEN_1, NAMEYOURDATASET).replace(TOKEN_2, CLASSESNAME).replace(TOKEN_3, DEVKITPATH)
    sh_str = sh_str.replace(TOKEN_1, NAMEYOURDATASET).replace(TOKEN_2, CLASSESNAME).replace(TOKEN_3, DEVKITPATH)
    sfac_str = sfac_str.replace(TOKEN_1, NAMEYOURDATASET).replace(TOKEN_2, CLASSESNAME).replace(TOKEN_3, DEVKITPATH)
    cfg_str = cfg_str.replace(TOKEN_1, NAMEYOURDATASET)

    return x_str, eval_str, fac_str, sh_str, sfac_str, cfg_str

def gen_pt_files(NUMCLSES, NAMEYOURDATASET):
    new_ptxt_dir = os.path.join(os.path.dirname(__file__), 'prototxt_templates_new')
    if(os.path.isdir(new_ptxt_dir)):
        os.system("rm -rf {}".format(new_ptxt_dir))
    os.system("cp -r {} {} ".format(os.path.join(os.path.dirname(__file__),'prototxt_templates'),new_ptxt_dir))

    for dir_name, sub_dirs, file_list in os.walk(new_ptxt_dir):
        content = []
        for fl in file_list:
            if(0):
                print("Dealing:",os.path.join(dir_name,fl))
            with open(os.path.join(dir_name,fl),'r') as f:
                content = f.readlines()
            if 'train_net' in content[0]: #this is a solver
                content[0] = content[0].replace('pascal_voc',NAMEYOURDATASET)
            content = [l.replace('MMICC',str(1+NUMCLSES)).replace('MM4CC',str(4*(1+NUMCLSES))).replace('IDNAME',NAMEYOURDATASET) for l in content]
            with open(os.path.join(dir_name,fl),'w') as f:
                f.write(str(''.join(content)))
    return new_ptxt_dir

def write_files(FASTER_RCNN_ROOT,NAMEYOURDATASET, new_ptxt_dir, x_str, eval_str, fac_str, sh_str, sfac_str, cfg_str):

    actions = [1,1,1,1,1]
    # move ptxt_dir
    summary = "\n------------------------------------------\nDone! Summary:\n"
    target_ptxt_dir = os.path.join(FASTER_RCNN_ROOT,'models',NAMEYOURDATASET)
    if os.path.isdir(target_ptxt_dir):
        ans = raw_input("[!] Folder '{}' exists, do you want to overwrite it?(y/N) ".format(target_ptxt_dir)).strip()
        if(ans  == 'y' or ans == 'Y'):
            print("[+] Overwrite folder: '{}'".format(target_ptxt_dir))
            summary += "\t[+] Overwrite folder: '{}'".format(target_ptxt_dir)
            summary += '\n'

            os.system("rm -rf {}".format(target_ptxt_dir))
            os.system("mv {} {}".format(new_ptxt_dir, target_ptxt_dir))
        else:
            print("\t[x] Skip of creating prototxt folder'{}'".format(target_ptxt_dir))
            summary += ("\t[x] Skip of creating prototxt folder'{}'".format(target_ptxt_dir))
            summary += '\n'
            actions[0] = 0

    else:
        print("[+] Create a folder to '{}'".format(target_ptxt_dir))
        os.system("mv {} {}".format(new_ptxt_dir, target_ptxt_dir))
        summary +=  "\t[+] Create a folder to '{}'".format(target_ptxt_dir)
        summary += '\n'
    print()

    x_name = NAMEYOURDATASET + '.py'
    eval_name = NAMEYOURDATASET + '_eval.py'
    fac_name = '{}_factory.py'.format(NAMEYOURDATASET)
    for i, (name, content) in enumerate(zip([x_name,eval_name,fac_name],[x_str, eval_str, fac_str])):
        target_path = os.path.join(FASTER_RCNN_ROOT,'lib','datasets',name)
        if os.path.exists(target_path):
            ans = raw_input(
                "[!] Python file '{}' exists, do you want to overwrite it?(y/N) ".format(target_path)).strip()
            if (ans == 'y' or ans == 'Y'):
                print ("[+] Overwrite file: '{}'".format(target_path))
                os.system("rm {}".format(target_path))
                summary += "\t[+] Overwrite file: '{}'".format(target_path)
                summary += '\n'
                with open(target_path, 'w') as f:
                    f.write(content)
            else:
                print("\t[x] Skip of creating'{}'".format(target_path))
                summary += "\t[x] Skip of creating'{}'".format(target_path)
                summary += '\n'
                actions[1+i] = 0

        else:
            print("[+] Create a file to '{}'".format(target_path))
            summary += "\t[+] Create a file to '{}'".format(target_path)
            summary += '\n'
            with open(target_path, 'w') as f:
                f.write(content)
        print()


    sh_target_path = os.path.join(FASTER_RCNN_ROOT, 'experiments', 'scripts', '{}_faster_rcnn_end2end.sh'.format(NAMEYOURDATASET))
    if os.path.exists(sh_target_path):
        ans = raw_input(
            "[!] Script file '{}' exists, do you want to overwrite it?(y/N) ".format(sh_target_path)).strip()
        if (ans == 'y' or ans == 'Y'):
            print("[+] Overwrite file: '{}'".format(sh_target_path))
            os.system("rm {}".format(sh_target_path))
            summary += "\t[+] Overwrite file: '{}'".format(sh_target_path)
            summary += '\n'
            with open(sh_target_path, 'w') as f:
                f.write(sh_str)
            os.system('chmod +x {}'.format(sh_target_path))
        else:
            print("\t[x] Skip of creating'{}'".format(sh_target_path))
            summary += "\t[x] Skip of creating'{}'".format(sh_target_path)
            summary += '\n'
            actions[-1] = 0

    else:
        print("[+] Create a file to '{}'".format(sh_target_path))
        summary += "\t[+] Create a file to '{}'".format(sh_target_path)
        summary += '\n'
        with open(sh_target_path, 'w') as f:
            f.write(sh_str)
        os.system('chmod +x {}'.format(sh_target_path))

    if(actions[3] or actions[-1]):
        summary += '\nMake sure that the choices you made above can make the generated files consistent. After that: '
    if(actions[3] == 1):
        summary += '\n @) Complete an extra step by yourself:\n'
        summary += "\t[Option1] \n\tBack up your original '{}' file. \n\tAnd rename '{}' to 'factory.py'\n".format(
            os.path.join(FASTER_RCNN_ROOT, 'lib', 'datasets', 'factory.py'),
            os.path.join(FASTER_RCNN_ROOT, 'lib', 'datasets', fac_name))
        summary += "\n\t[Option2] \n\tInsert the following code block to '{}':\n".format(os.path.join(FASTER_RCNN_ROOT, 'lib', 'datasets', 'factory.py'))
        summary += "\n{}\n".format(sfac_str)
    if(actions[-1] == 1):
        cfg_path = os.path.join(FASTER_RCNN_ROOT, 'experiments', 'cfgs', '{}_end2end.yml'.format(NAMEYOURDATASET))
        with open(cfg_path, 'w') as f:
            f.write(cfg_str)
        summary += "\n @) Train the net using the script '{}'. \n\tFor example:\n".format(sh_target_path)
        summary += "\t $ cd {}\n".format(FASTER_RCNN_ROOT)
        summary += "\t $ {} 0 VGG16 {}\n".format(os.path.join('experiments', 'scripts', '{}_faster_rcnn_end2end.sh'.format(NAMEYOURDATASET)), NAMEYOURDATASET)



    summary += '------------------------------------------\n'

    print(summary)
if __name__ == '__main__':
    print()
    arg, parser_ = parse_args()
    if(len(sys.argv)!=9 or arg.FASTER_RCNN_ROOT == default_mark or arg.NAMEYOURDATASET == default_mark or arg.DEVKITPATH == default_mark or arg.CLASSESNAME == default_mark):
        parser_.print_help()
        sys.exit(1)

	
    FASTER_RCNN_ROOT = arg.FASTER_RCNN_ROOT
    NAMEYOURDATASET = arg.NAMEYOURDATASET; TOKEN_1 = 'MMICC'
    CLASSESNAME, NUMCLSES = parse_classes(); TOKEN_2 = 'QYCC'
    DEVKITPATH = arg.DEVKITPATH; TOKEN_3 = 'DEVKITPATH'
    if (not isidentifier(NAMEYOURDATASET)):
        print("[!] --idname '{}' is not a valid identifier in python. ".format(NAMEYOURDATASET))
        sys.exit(1)

    check_exists(FASTER_RCNN_ROOT, DEVKITPATH)
    x_str, eval_str, fac_str, sh_str, sfac_str, cfg_str= gen_files(NAMEYOURDATASET, TOKEN_1, CLASSESNAME,TOKEN_2, DEVKITPATH, TOKEN_3)
    new_ptxt_dir = gen_pt_files(NUMCLSES,NAMEYOURDATASET)

    write_files(FASTER_RCNN_ROOT,NAMEYOURDATASET, new_ptxt_dir, x_str, eval_str, fac_str, sh_str, sfac_str, cfg_str)

