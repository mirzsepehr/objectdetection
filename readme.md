# Object Detection
In this project we are going to detect __cellphones__ and __seatbelts__. Program is going to detect whether the driver has seatbelt or not, and additionally does the driver holds a phone in his hand or not.
To implement this project i have used yolov5 model. You can checkout**[github repo](https://github.com/ultralytics/yolov5)** of this model. </br>

I have downloaded the required dataset available in **[Roboflow](https://roboflow.com/)**.</br> Due to insufficient dataset I had to add a few hundreds to my dataset through searching in Internet and labeling them via **[Roboflow](https://roboflow.com/)**.</br>
**[This](https://roboflow.com/how-to-label/yolov5)** is the way you can label your dataset in order to utilize them in yolov5 models. As you know, Yolov5 models come in variety of complexity. The simplest one is Yolov5nwhich is way too simple and the chances of overfitting is high. There are some other models for instance, Yolov5s, Yolov5m, Yolov5l, and Yolov5x. I have used Yolov5m to train my models. I have written the details of my training procedure and parameters at the end of this document. 
Now let's dive deep into the code and see what happens!

## Models & Procedure
There are two models used in my project. I have trained both of them which are in the app folder of the project. "**best.pt**" is the phone detection model and the "**best_seatbelt.pt**" file is the seatbelt detection model. </br>

This project has written in **[fastapi](https://fastapi.tiangolo.com/)** and **[Pytorch](https://pytorch.org/docs/stable/index.html)** frameworks.
You can dive deep into the code and see the code's explanation in the main.ipynb file in the "app" folder.

## Training Process
Training yolov5 models are easy to learn which you can find at __[this link](https://docs.ultralytics.com/yolov5/quickstart_tutorial/)__. You can pass some hyperparameters in order to train your model in a decent way by **--hyps** command. You should write your hyperparameters in a YAML file format. However, there are some default hyperparameter files you can find in the ultralytics/yolov5 project.
> git  clone  https://github.com/ultralytics/yolov5

Go to the following path

>  yolov5\data\hyps

In this directory there are some hyperparameter examples you can work with, however in order to get to the high F1scores and a better confusion matrix you should regularize hyperparameters by yourself. I used starting learning rate (lr0) of 0.001 and lrf of 0.01 for both phone and seatbelt detection.
In yolov5 lrf is not the final learning rate. Final learning rate is calculated by multiplication of lr0 and lrf. 
I used SGD optimizer with the momentum of 0.937 (its default value).
You can get all of my hyper parameters in the hyp_seatbelt.yaml and hyp_phone.yaml in the hyps folder in the project. 
</br>
</br>Have fun!
Best regards.
 



 

