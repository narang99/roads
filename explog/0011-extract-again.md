Working on correctly extracting and shrinking taco dataset again, I am revisiting that model one last time.  


test how individual categories look after bbox resizing (we are blind resizing right now).  
Looking at the stats of images, we might need to start with 5px images for finding correct garbage pieces (although how this affects our model is yet to be seen).   

The first thing I need to do i save the image level as part of the file name, so that I can inspect them one by one when needed. Before that, I'm going to look at each supercategory's image sizes, and how they look after shrinking to our desired levels.



How good something looks is a function of how much space a box takes in a given image. Some images are amazing and shrink well. What doesnt?
- The example is too big, in this case shrinking would have to add padding. I'm not sure how good adding a lot of padding is
- The example is too small, in this case, the behavior seems quite undefined

- Cigarette butts at "5px" are horrible
  - But these are the most prominent too. They look like single straight lines and very pixelated. It is useful if it is white in color.  


Changes:
- I've fixed the bbox calculation a bit more
  - Due to the random nature of the leftover splitting, the algorithm would pad the image unnecessarily (it would decide to use a length in left that would spill over, even though there is place in right side). 
  - I'm assuming now we would only get a bbox inside the image, and padding would only be done when absolutely needed (this needs more checking out)
  - Note that i only do scaling based on height and not on width, this can give some unevenness (some objects wont ever see smaller widths)
  - It might be useful to use "bbox area" maybe, but that would be slightly complicated, it is easier to just randomly decide whether to do width or height
  - I would do the height / width random sclaing later. for now, i need to find out what crops are useful



The problem?
- Some shrinked garbage photos are not near reality, they need to be taken out
  - too pixelated
  - too large
  - too small
I myself don't know what the actual problem is though


# Zero in on the problem
Go through many images, and their resizes, see if they look reasonable. if they dont, find the corresponding category in original photos and check them side by side

- The annotations in taco are very detailed. Some annotations would have like stems in front of them, and the masks won't have 1s in the stem part, should i dilate the masks to make it easier for the model? Although dilation might have weird effects too. To check

- I'm quite clueless right now on what is right or wrong, I need to decrease my standards for it
- Aluminium foils are quite small generally, and easily resizable to 5px height
- bigger stuff needs to be caught better
- the model seems to not catch 5px things, maybe cuz of the 7x7 resnet block
- I would prefer catching white stuff which is wonky / algae like for stuff in 5px range

What is the minimum are that creates a sort of "seeing" for me. We need all the bright ones to be detected when in the smallest quantity I guess. We can start with the white ones.  

- There might be value in doing with 50x50 also. TACO is nice, it also has interesting clusters which has helped my model actually. We can add extra brightness differences for plastics though, need to test that (to simulate browning)
- when i collapse all classes and their annotations, it makes it harder for the model to have well defined annotations in clusters. Its just a big blob of annotations in this case. Is that good? Is that bad? For highly pixelated clusters, I would say that having a blob is useful. I'm not sure what to do here

Ohk, Im not satisfied because I am myself not sure if an annotation's good or not. I need categories. When finding what's wrong, its hard to say whats wrong. When finding whats right, its hard to say whats right.  
- The problem here is that i'm myslef not sure how the annotations should look like.  

Time has come, to categorize the garbage and show if we are sure or not i feel.  

- white objects > 5px in any dimension are creating a sort of glow that i can see. I need better examples though.  
  - Lets try this, i guess i should be able to do something after I've seen results mixed with this.  
- Ohk. Now for extremely small objects, I'm prioritising bright objects. Other than this, I need to understand the characteristics of my model to see what it is detecting at 100px ranges.  
  - Once im fine with the white detector, i need to figure out how well it works with the original model
  - We need to understand the behavior of the original model (what is it detecting? What do i want it to detect?)


- Ohk, I've got a good understanding of what dataset i need for white detection. 

### What is my model detecting?
This is the second problem, im not very sure rn

### Bigger objects
We need a bigger model which detects larger objects. As I had previously thought, I need to bucket my object categories in different "size" groups, which different models detect separately.  

### object groups

visible objects which are very small objects are generally bright. I need to find the bounds for these. although i would say bright objects in any viewpoint are visible. For the 50x50 model:
- Bright objects
- Cigarettes

So now, at a top level, i need to decide what goes where (in which bucket). It is useful to just do size ranges for different object  groups, which are legible to me. Then bucket those. Then  we train a model on it.  
The main model is currently trained on all data, which might be the correct approach in the end, I will need to see. I dont see good performance of 200x200 though (larger sizes dont catch the smaller trash, maybe because the dataset is a "closeup"). Dataset being a closeup dataset is the biggest reason why I create separate models.  


Thje architecture can go through the data in one pass, using rules on each annotation and deciding the buckets we can add it to. That is secondary right now though.  
I think by not padding the output and making sure it always fits, the 100 model performed reasonably well (padding would have been a mistake). Since we dont pad, we are able to see the object from its smallest to the biggest with reasonable look.  


Lets first train on white objects. I will have a final list of objects to their annotation ids. i just need to push them to a simple new DS directory. and then create bboxes of levels for 50x50. these would be 5 -> 35 height/width objects i guess.  

xresnet18 training schedule (50 iters)
[13, 52.161930084228516, 59.86985397338867, 0.8413525955034964, '00:33']
[14, 52.20627212524414, 59.89109802246094, 0.8413234746862651, '00:32']
[15, 52.15180587768555, 60.01018524169922, 0.8410886118835035, '00:31']
[16, 52.1817741394043, 59.9805908203125, 0.8411759652681869, '00:30']
[17, 52.04269790649414, 60.02083206176758, 0.840883538847507, '00:31']
[18, 51.79948425292969, 59.991580963134766, 0.8410392552125382, '00:31']
[19, 51.90467834472656, 59.97174835205078, 0.8410763679599706, '00:32']

The model trained on only 5/10 bbox sizes is not better than the model trained on [5,10,...25] sizes. Quite interesting. Maybe i should simply train a model on 5-25 sizes for all images instead of just white ones, we might see more interesting results.  
What they say about data seems to be true lol.  
- this did not work. white seems to be very useful i guess. I would need to train the 100 model on more white data too
- crumpled paper gives most of the models difficulties, ill have to check that out too.  
 - it might be useful to just train a 132 size model instead of 100, but i dont want to train more models before finding the faults with the current one.  

Negmask is removing some of the model's outputs (all the smaller ones). I'll need to do one more pass of training negmask with more examples.  
- I will need to train a bigger cut for whites

Lets start with negmask now. List of things possible
- double the size of the input.  
- Train a model by taking activations of the original negmask model, training each layer on each activation. But we want each layers output shrinked. This is quite an interesting experiment.  
