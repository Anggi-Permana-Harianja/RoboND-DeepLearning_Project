[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##

# Links #

My Jupyter Notebook for training without output -> [HERE](https://github.com/tiedyedguy/RoboND-DeepLearning_Project/blob/master/code/model_training.ipynb)
My Jupyter Notebook saved with output 0.46 Score (Warning, slow load) -> [HERE](http://www.prostarrealms.com/model_training.html)
My Weights file -> [HERE](https://github.com/tiedyedguy/RoboND-DeepLearning_Project/blob/master/code/model_weights)

# Architecture #

The architecutre of the neural network that I created looks like this:

Encoder -> Encoder -> Encoder -> Encoder -> 1x1 -> Decoder -> Decoder -> Decoder with Skip -> Decoder with Skip

Each Encoder step is one stride 2 and 32 deep layer.

Each Decoder step is a bilinear upsample and then two 32 deep layers.

The 1x1 is the same size as the last Encoder, but is just 8 filters deep.

So, let's talk about what it does to each image.  We start with an image that is 256 x 256 pixels.
This is actually an increase from the default 128 x 128.  

The first encoder takes each image and walks over it with a stride 2.  This means while looking at the image, the finished
output from the layer is half the size or 128 x 128 pixels.

The next encoders does the same thing and we are now at 64 x 64. Encoder three takes us down to 32 x 32.  The final
encoder takes us down to a 16 x 16 pixel image.

The next step is the 1x1 layer.  This takes the 16 x 16 image and just changes the depth of the layer to 8.

Next is time to expand the images back up.

The first decoder takes the 16 x 16 layer and doubles it to 32 x 32.  The second decoder takes it up to 64 x 64.

The third decoder takes it up to 128 x 128 and then takes the data out of encoder 1 which is also 128 x 128 and puts
them together.  In the same way, the final decoder outputs our 256 x 256 again and also takes back in our original image
to help it.

[Here is the diagram of my NN](https://github.com/tiedyedguy/RoboND-DeepLearning_Project/blob/master/NN.png)
[Here is the model.summary() output](https://github.com/tiedyedguy/RoboND-DeepLearning_Project/blob/master/model_summary.txt)


# Parameteres #

My Hyper Parameters:

```
img_size = 256
learning_rate = 0.003
batch_size = 40
num_epochs = 100
steps_per_epoch =400
validation_steps = 50
workers = 8
```

I did not use any kind of system for my hyper parameters.  I was able to use my own GPU for training, so testing did not
take too long. So I mostly trial & errored to get to these points.  How I did that will be explained in each
section:

img_size: As mentioed previous, I changed this to the actual size of the images, instead of cutting the image
in half instantly.

learning_rate:  For the learning rate I started at 0.01 and would only go down by 0.001 each time.  Every time
I tried to go below 0.003 I did not see an increase in final score, so that is how I got to 0.003.

batch_size:  For batch size, I would always set this to the maxium number I could that wouldn't give me a memory error
on my GPU.  So there was always a little trial and error whenever I changed the model to find the right size.

num_epochs: I always kept this right around 100.  I never really changed from this.

steps_per_epoch:  This number I always made in to 2000 / batch_size.  The reason is that there are 2000ish training
files.

validation_steps: I never changed this parameter.

workers:  At the start, I tested setting this from 4 to 8 and noticed a slight gain, so kept it from 8 ever since.

# Techniques #

In my model, I use a special techniques of the 1x1 layer and fully connected layers.

The 1x1 layer is a cheap way of making a little more depth to our model.  This layer looks at each pixel by pixel
of it's input and also has a lower depth than the layer before hand.  The 1x1 in my model is when our image is 16x16
so the image is already 1/16 the size, so image that it is look at each 16th block of the image and doing a quick 
search for anything there.  

I also only use 8 filters in this layer to reduce the complexity of this layer before starting the decoding.  The
reason for that is explained next.

Fully connected layers just means that all layers in the output of one layer is sent to the input of the next one.  
Each of the layers in this model are fully connected.  So, the 1x1 layer means that its 8 filters are connected
to the next section.  This lower than my normal of 32 helps increase the speed of the model because if I would just
connect the last encoder to the first encoder it would be connecting 32 outputs to 32 inputs.  Now it is connecting
32 outputs to 8 inputs, then 8 outputs to 32 inputs.

# Image Manipulation #

Why do we encode / decode the image in our system?

The major two reasons are speed and second is that sometimes too much info doesn't help.

First thing first, is speed.  Every time you half the amount of pixels in length and width you are actually decreases the
total amount of pixels of the picture by a fourth.  This means, all thigns else equal, each layer would work 4x as fast.

The second reason is that if you think about it, sometimes you can have too much information in those pixels.  When
you lower the resolution, you are looking at the picture in a differnet light.  Maybe things will pop-out more
the "blurier" you get.  Somteimes if you have all the pixels, the system might look at things too exactly.  In our
follow-me example, maybe if we always use full resolution pictures, maybe it over guess on the Hero over grass because
there has never been a picture of a more blurry Hero.

# Follow-me Scenarios #

This training model that we made was specifically made for this hero.  I doubt if you have another set of training data
for another object like a cat or dog that my model will have the accuracy it does.  The reason is because it is easy
to notice the Hero is the only character filled highly in the color red.  I'm thinking that my model is mostly just
using that fact as how it is working.  So if a dog or a cat stands out as much as the hero, then we stand a chance.

