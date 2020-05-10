##################### COMPSYS 302 Project - Dog Breed Detector #####################

------------ SET UP STEPS ------------

1) Download the split dataset from this Google drive link "https://drive.google.com/open?id=1FgO9cDOZTMZKK2roJi39TvLfDn4dovpl" and place the images folder into to the root.

Make sure it follows the ImageFolder structure eg.
images->train->class1
      |      |->class2
      |      |->class3
      |
      ->val--->class1
	     |->class2
	     |->class3


2) Downgrade scipy from the latest version or whichever to version 1.3.3. This can be done by running "pip install --upgrade scipy==1.3.3".
Inception v3 which is one of the models we use takes around 3min to load otherwise. This problem has been encountered by many and is documented here: https://github.com/scipy/scipy/issues/11299

Inception still eventually loads however so this step is optional.



------------ TRAINING STEPS ------------ 

1) When the program runs, it first prompts the user whether or not the user wants to train a model. If the user doesn't choose to, it skips to the prediction part of the program.

2) Choose the the model to train. Once the training is complete it will save the model and proceed to the prediction part of the program.



------------ PREDICTION STEPS ------------ 

1) The user is prompted to enter the .jpg file name which is the image of the dog you would like to check.

2) Choose the the model that has been trained previously. Check the models folder if unsure which are available.

3) The program will now show its prediction along with a confidence rating in a percentage. The user can then choose to see the next top 5 results.