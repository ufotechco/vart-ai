# VeinsDetector
# Esta clase comprende la unión de las distintas clases
# que comprenden el sistema general 
# Desarrollado por UFOTECH S.A.S.

import argparse
from sys import stderr
from subprocess import call
import os
from pathlib import Path
from configparser import ConfigParser

input_config = ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
input_config.read(config_path)
base_directory = input_config['DEFAULT']['base_directory']
base_output_directory = input_config['DEFAULT']['base_output_directory']
nerve_weights = input_config['NERVE_LOC']['nerve_weights']
improve_default_file = input_config['IMAGE_IMPROVE']['default_file']

temp_nerve_directory = input_config['DEFAULT']['temp_nerve_directory']
temp_vessel_directory = input_config['DEFAULT']['temp_vessel_directory']

img_improve_file=base_directory+"VartImageEnhancer.py"
nerve_detection_file = temp_nerve_directory+"VartYoloDetector.py"
veins_detection_file = temp_vessel_directory+"VartVesselPredictor.py"
pixel_count_file=base_directory+"VeinsLabel.py"

# Verificar existencia de carpeta output
try:
    os.makedirs(base_output_directory, exist_ok=True)
except:
    pass

# Verificar existencia de carpeta improved
try:
    os.makedirs(os.path.join(base_output_directory, "improved"), exist_ok=True)
except:
    pass

# Verificar existencia de carpeta labels
try:
    os.makedirs(os.path.join(base_output_directory, "labels"), exist_ok=True)
except:
    pass

# Verificar existencia de carpeta mask
try:
    os.makedirs(os.path.join(base_output_directory, "mask"), exist_ok=True)
except:
    pass

# Verificar existencia de carpeta noir
try:
    os.makedirs(os.path.join(base_output_directory, "noir"), exist_ok=True)
except:
    pass

# Verificar existencia de carpeta PIL
try:
    os.makedirs(os.path.join(base_output_directory, "PIL"), exist_ok=True)
except:
    pass


parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default=None, help='Image file directory to be processed')
args = vars(parser.parse_args())

input_dir = args["source"]

if input_dir is None:
    raise "Debe indicar una fuente de imagen"

for first_file in os.listdir(input_dir):
    ##check if current path is a file
   if os.path.isfile(os.path.join(input_dir, first_file)):

        found_file=os.path.join(input_dir,first_file)

        image_base = Path(found_file).stem
        image_ext = Path(found_file).suffix

        #found_file_m=os.path.join(base_output_directory+"noir/",first_file.split(".")[0]+"-noir."+first_file.split(".")[1])
        #found_papila = os.path.join(base_output_directory,first_file.split(".")[0]+".txt")
        found_file_m=os.path.join(base_output_directory+"noir/",image_base+"-noir"+image_ext)
        found_papila = os.path.join(base_output_directory+"labels/",image_base+".txt")


        try:
            image_retcode = call("python3 "+img_improve_file+" --source "+found_file,shell=True)
            if image_retcode < 0:
                print("Image Child was terminated by signal", -image_retcode, file=stderr)
                exit() 
            
        except OSError as e:
            print("Execution failed:",e,file=stderr)
            exit()

        try:
            vessel_retcode = call("python3 "+veins_detection_file,shell=True)
            if vessel_retcode < 0:
                print("Vessel Child was terminated by signal", -vessel_retcode, file=stderr)
                exit() 
            
        except OSError as e:
            print("Execution failed:",e,file=stderr)
            exit()

        try:
            nerve_retcode = call("python3 "+nerve_detection_file+" --source "+ found_file + " --weights "+nerve_weights+" --save-txt --nosave --save-conf --project "+base_output_directory,shell=True)
            if nerve_retcode < 0:
                print("Nerve Child was terminated by signal", -nerve_retcode, file=stderr) 
                exit()
            
        except OSError as e:
            print("Execution failed:",e,file=stderr)
            exit()

        try:
            if os.path.isfile(found_papila):
                label_retcode = call("python3 "+pixel_count_file+" --source "+found_file_m+" --is-papila --papila-loc "+found_papila,shell=True)
            else:
                label_retcode = call("python3 "+pixel_count_file+" --source "+found_file_m,shell=True)
            if label_retcode < 0:
                print("Label Child was terminated by signal", -label_retcode, file=stderr)
                exit() 
            
        except OSError as e:
            print("Execution failed:",e,file=stderr)
            exit()

        os.remove(found_papila)
        os.remove(found_file_m)
        os.remove(found_file)