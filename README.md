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

So, let's talk about what it does to each image.  We start with an image that is 256 x 256 pixels. This is actually
an increase from the default 128 x 128.  I decided to do this because, well, honestly I found it odd we would instantly
decide that half of our data wasn't useful.  While maybe there is something to that, because we do still encoder later,
at least we control how that encoding works.

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

## Here is my Drawing of it ##
![Here is my crude diagram of my NN](https://github.com/tiedyedguy/RoboND-DeepLearning_Project/blob/master/NN.png)

## Here is Kera's Output and Shape of it ##
![Here is Keras's Output And shape](https://github.com/tiedyedguy/RoboND-DeepLearning_Project/blob/master/KerasVersion.png)

## Here is the model.summary() output ##
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 256, 256, 3)   0                                            
____________________________________________________________________________________________________
separable_conv2d_keras_1 (Separa (None, 128, 128, 32)  155         input_1[0][0]                    
____________________________________________________________________________________________________
batch_normalization_1 (BatchNorm (None, 128, 128, 32)  128         separable_conv2d_keras_1[0][0]   
____________________________________________________________________________________________________
separable_conv2d_keras_2 (Separa (None, 64, 64, 32)    1344        batch_normalization_1[0][0]      
____________________________________________________________________________________________________
batch_normalization_2 (BatchNorm (None, 64, 64, 32)    128         separable_conv2d_keras_2[0][0]   
____________________________________________________________________________________________________
separable_conv2d_keras_3 (Separa (None, 32, 32, 32)    1344        batch_normalization_2[0][0]      
____________________________________________________________________________________________________
batch_normalization_3 (BatchNorm (None, 32, 32, 32)    128         separable_conv2d_keras_3[0][0]   
____________________________________________________________________________________________________
separable_conv2d_keras_4 (Separa (None, 16, 16, 32)    1344        batch_normalization_3[0][0]      
____________________________________________________________________________________________________
batch_normalization_4 (BatchNorm (None, 16, 16, 32)    128         separable_conv2d_keras_4[0][0]   
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 16, 16, 8)     264         batch_normalization_4[0][0]      
____________________________________________________________________________________________________
batch_normalization_5 (BatchNorm (None, 16, 16, 8)     32          conv2d_1[0][0]                   
____________________________________________________________________________________________________
bilinear_up_sampling2d_1 (Biline (None, 32, 32, 8)     0           batch_normalization_5[0][0]      
____________________________________________________________________________________________________
separable_conv2d_keras_5 (Separa (None, 32, 32, 32)    360         bilinear_up_sampling2d_1[0][0]   
____________________________________________________________________________________________________
batch_normalization_6 (BatchNorm (None, 32, 32, 32)    128         separable_conv2d_keras_5[0][0]   
____________________________________________________________________________________________________
separable_conv2d_keras_6 (Separa (None, 32, 32, 32)    1344        batch_normalization_6[0][0]      
____________________________________________________________________________________________________
batch_normalization_7 (BatchNorm (None, 32, 32, 32)    128         separable_conv2d_keras_6[0][0]   
____________________________________________________________________________________________________
bilinear_up_sampling2d_2 (Biline (None, 64, 64, 32)    0           batch_normalization_7[0][0]      
____________________________________________________________________________________________________
separable_conv2d_keras_7 (Separa (None, 64, 64, 32)    1344        bilinear_up_sampling2d_2[0][0]   
____________________________________________________________________________________________________
batch_normalization_8 (BatchNorm (None, 64, 64, 32)    128         separable_conv2d_keras_7[0][0]   
____________________________________________________________________________________________________
separable_conv2d_keras_8 (Separa (None, 64, 64, 32)    1344        batch_normalization_8[0][0]      
____________________________________________________________________________________________________
batch_normalization_9 (BatchNorm (None, 64, 64, 32)    128         separable_conv2d_keras_8[0][0]   
____________________________________________________________________________________________________
bilinear_up_sampling2d_3 (Biline (None, 128, 128, 32)  0           batch_normalization_9[0][0]      
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 128, 128, 64)  0           bilinear_up_sampling2d_3[0][0]   
                                                                   batch_normalization_1[0][0]      
____________________________________________________________________________________________________
separable_conv2d_keras_9 (Separa (None, 128, 128, 32)  2656        concatenate_1[0][0]              
____________________________________________________________________________________________________
batch_normalization_10 (BatchNor (None, 128, 128, 32)  128         separable_conv2d_keras_9[0][0]   
____________________________________________________________________________________________________
separable_conv2d_keras_10 (Separ (None, 128, 128, 32)  1344        batch_normalization_10[0][0]     
____________________________________________________________________________________________________
batch_normalization_11 (BatchNor (None, 128, 128, 32)  128         separable_conv2d_keras_10[0][0]  
____________________________________________________________________________________________________
bilinear_up_sampling2d_4 (Biline (None, 256, 256, 32)  0           batch_normalization_11[0][0]     
____________________________________________________________________________________________________
concatenate_2 (Concatenate)      (None, 256, 256, 35)  0           bilinear_up_sampling2d_4[0][0]   
                                                                   input_1[0][0]                    
____________________________________________________________________________________________________
separable_conv2d_keras_11 (Separ (None, 256, 256, 32)  1467        concatenate_2[0][0]              
____________________________________________________________________________________________________
batch_normalization_12 (BatchNor (None, 256, 256, 32)  128         separable_conv2d_keras_11[0][0]  
____________________________________________________________________________________________________
separable_conv2d_keras_12 (Separ (None, 256, 256, 32)  1344        batch_normalization_12[0][0]     
____________________________________________________________________________________________________
batch_normalization_13 (BatchNor (None, 256, 256, 32)  128         separable_conv2d_keras_12[0][0]  
____________________________________________________________________________________________________
conv2d_2 (Conv2D)                (None, 256, 256, 3)   99          batch_normalization_13[0][0]     
====================================================================================================
Total params: 17,321
Trainable params: 16,537
Non-trainable params: 784
```
Why do we encode / decode the image in our system?

The major two reasons are speed and second is that sometimes too much info doesn't help.

First thing first, is speed.  Every time you half the amount of pixels in length and width you are actually decreases the
total amount of pixels of the picture by a fourth.  This means, all thigns else equal, each layer would work 4x as fast.

The second reason is that if you think about it, sometimes you can have too much information in those pixels. 

Let's talk for a second about what we are actually doing when we downsample in the encoding and then upsample in the decoding.
What this means is we are taking an image, let's say this one:

![First Image](https://github.com/tiedyedguy/RoboND-DeepLearning_Project/blob/master/PreEncoding.jpeg)

This is one of our 256x256 training files.  The first thing we do is downsample it with an encoder by 50% (based on our stride).
This takes it to something like this:

![Seocnd Image](https://github.com/tiedyedguy/RoboND-DeepLearning_Project/blob/master/Encode1.jpeg)

Not too bad ... yet.  But, we do this three more times, until we get to the 1x1 block with something like this:

![1x1](https://github.com/tiedyedguy/RoboND-DeepLearning_Project/blob/master/Pre1x1.jpeg)

That's all it has to work with.  Next we start upscaling it, which means adding back pixels, so let's just do that with a simple
method.  Here's one upscaling:

![After1x1](https://github.com/tiedyedguy/RoboND-DeepLearning_Project/blob/master/After1x1.jpeg)

Not that clear, how about we do our final three decodes, what does it look like?

![Final](https://github.com/tiedyedguy/RoboND-DeepLearning_Project/blob/master/Final.jpeg)

No where near our starting.  Now this is not exactly how our model is doing the downscaling/upscaling, but it is just an example.
Keep this example in mind as you read these next paragraphs.

When you lower the resolution, you are looking at the picture in a differnet light.  Maybe things will pop-out more
the "blurier" you get.  Somteimes if you have all the pixels, the system might look at things too exactly.  In our
follow-me example, maybe if we always use full resolution pictures, maybe it over guess on the Hero over grass because
there has never been a picture of a more blurry Hero.  

Even though we are using encoding / decoders, with unlimited computing power, we wouldn't.  There is a disadvantage
here and that is the chance of losing important information.  No matter how you encode / decode you are still having
to make some change to the data for the sake of speed/resources.  In our system, every time we encode we are 
take our image and cutting it by, at most, a 4th in detail.  This will have an impact on our model.  Luckily though,
even with this disadvantage, it is an amazing way of doing DL.

One important thing that I did in my model was keep all but the 1x1 at 32 filters.  Normally when you start to encode /
decode you increase the filters.  Increasing the filters is increasing the complexity of the model, but because you are
encoding the overall size of the model is still shrinking and keeping it manageable.  Plus as you encoder, you have
more filters looking for "bigger picture" type things.  Normally the first few encodings can find things like edges
and basic shapes, as you encode, it can start to find things like faces or more complex charactersists.  

So to shrink the picture size and increase the number of filters is standard, but I'm not doing that.  I have all of my 
layers at 32 filters with the exception of the 1x1.  I wish I had some complex reasoning for this, but the truth is that 
while messing with the batch_size hyper parameter I kept getting upset that my GPU was running out of memory.  I could just 
decrease the batch size, but when I was getting batch sizes less than 40, I felt like it was wrong.  I figured that even 
though I could increase the steps_per_epoch, getting the batch_size to something "higher" was more imporant than anything 
else.

So, originally I had the filters going from 32 -> 64 -> 128 -> 256 through the encoders, I quickly started striking
them down.  It just so happens that at 32 all across is where I ended.  While it doesn't feel "right" it has worked
for our little sample here.

Another thing to point out is that I have two skip connectors at the 128 x 128 and the 256 x 256 stages.  Skip 
connectors are when you take an image from earlier in the model and move it forward though the system.  The advantage
here is of clarity.  The image after getting encoded then decoded will lose some information in the process.  By
brining the pre-encoded image over, you are giving the model the clear image to work with.  This normally helps in
increasing accuracy around borders between values.  Look at my example above, by using the skip connectors the model will
have the same picture as our starting model at the end plus the data of the down/up scaled one!

For the future, I feel as though my model has been a great start at a CNN.  I was able to get 4 levels of 
encoding/decoding to work, which takes us from a 256x256 to a 16x16 image.  There is probably little information we
can get if we squish the image to 8x8.  So, expansion of my model's next steps will have to be in what we do at each
layer.  As previously discussed, the model should be improved by adding more filters at each step, so this would 
be a great place to start.  After that, there is which skip connectors to use or not.  Lastly, there are training
improvements that can happen, like adding dropout or using other optimizers.

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

So when should use each one?  Well, if you have unlimited resources, go ahead and use fully connected to your leasure, you 
lucky person you.  Fully connected are for when you need to full paths open but our costly so you want to use them as you can.
The place to put in a 1x1 is when you are done with an encoding path or a place where you want to just reduce your complexity for
a layer.  Maybe you have a fully connected layer with a high amount of filters but don't neccisarily want them all connected to the
next layer.  This would be a place for the 1x1.

# Follow-me Scenarios #

This model that I made, while it works with the Follow-me Hero, nothing is specifically tailored for this unique problem.
If we were to send to this model images of a dog as the hero, or something else.  There is a good chance the model will
still be able to perform.  Probably not exactly as well as with this Hero, but could be worse or better!  I would like to
think if the hero model was changed to an animalg, the stark difference in posture would be enough for the 
model to figure out.
