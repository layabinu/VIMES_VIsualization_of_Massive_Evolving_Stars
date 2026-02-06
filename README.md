


https://github.com/user-attachments/assets/30f19727-5089-44c6-8598-965a7b3220e3



Log Default


# VIMES: VIsualization of Massive Evolving Stars
Creates a stellar evolution animation for binary systems simulated using COMPAS.
Animation is reflective of the radius, separation, percieved color etc of the stars. 

ALL code related to temperature-color relationship is directly from the TULIPS code
https://bitbucket.org/elaplace/tulips/src/master/
https://ui.adsabs.harvard.edu/abs/2022A%26C....3800516L/abstract


Code is run using the run.sh file. 
Before doing so, update the data file input path in the preprocess.py file, it is currently set to a default example file provided with the code. 
in terminal run 

chmod +x "_______/Final Animation/run.sh"

where ______ should be replaced with your path up until that point



then you can run
./run.sh log tulips

this is just an example, you have the options of log/linear or tulips/default
and the animation will be created using pygame

if you want to save as an mp4 file, after log/linear and tulips/default, write the name of the mp4 file you want it saved to (ex: evolution.mp4)
If you are running this through colab or do not want the pygame display window, you can also add in the word "headless" after the mp4 file name
the full command would look something like "./run.sh linear default evolution.mp4 headless"


things required for this... still needs to be updated  but so far

pip install numpy pygame pillow



https://github.com/user-attachments/assets/7c057ed1-494a-4d30-a08d-a19cd3588de0



Linear Tulips


