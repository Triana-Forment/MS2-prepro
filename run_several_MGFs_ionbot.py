import glob
import os

# Create a list with MGF files
files = [i for i in glob.glob("/public/compomics/triana/master/ionbot/PXD000561/*.mgf")]

for file in files:
        # Extract the file name from the full path
        name = file.split("/")[-1]
        # Extension is the distinctive part from the file name (each sample)
        extension = name.split("_")[5][0:3] 
        # Directory is the working folder
        directory = "/".join(file.split("/")[0:-1])
        # Output path is directory + extension --> so ionbot output is in a unique folder for each sample
        output_path = directory+"/"+extension+"/"
        # Parts of the command to run Ionbot
        part1 = "docker container run -v /public/compomics/triana/master/ionbot/PXD000561/:/public/compomics/triana/master/ionbot/PXD000561/ -w /public/compomics/triana/master/ionbot/PXD000561/ gcr.io/om>
        part2 = " -a 32 -eRI --top-x-tags 30 "
        comand  =  part1+output_path+part2+file
        # Run the comand in the terminal
        os.system(comand)



