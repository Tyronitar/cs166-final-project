# cs166-final-project

My Final Project for Caltech CS 166 Spring 2022

In this project, I implement the method in [Feature-Based Image Metamorphosis (Beier and Neely 1992)](https://www.cs.princeton.edu/courses/archive/fall00/cs426/papers/beier92.pdf) to morph images of one face into another.

<p align="center">
  <img src="examples/williams.gif" />
  <img src="examples/jennie-rih.gif" />
</p>

## Usage

Run the `morph` module with your two images to morph them. The code will output an animated gif of the morphing sequence. There are a number of settings that can be edited, so use the `--help` to see what they are.

The program will not work if a human face cannot be detected in both images. As a result, using too small images, or using images without people can create issues. If you would like to use faces that are not detectable, you can also provide manual annotations of the landmarks via a .pts file (See [lion_landmarks.pts](lion_landmarks.pts) for an example). The format of the file is 68 lines of the (x, y) coordinates of each landmark in the dlib order. The order and positions of facial landmarks is shown below for your convenicnce.

To manually annotate images, I recommend using [FLAT - Facial Landmarks Annotation Tool](https://github.com/luigivieira/Facial-Landmarks-Annotation-Tool). Below is an example morph using a non-human face.
<p align="center">
  <img src="examples/obama-lion.gif" />
</p>

## Dependencies
All of the dependencies are noted in the [Pipfile](Pipfile). The main ones to note are:
- [dlib](http://dlib.net/)
- [opencv](https://github.com/opencv/opencv-python)
