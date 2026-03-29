- I've got a A small model, A medium model. smallnet
- A high recall negmasker for mediujm, a stable for medium, an sm model
- Segmentations


I want to run these on delhi dataset, the dataset needs to be chunked though. Since colab is ephemeral, I would prefer losing at max 20 minutes of work. 256 smallnet in 5 seconds -> 50 images per second in smallnet. 3000 images per minute

since 3 models, 1000 per minute
negmask assume similar speed along with segmask. So assume 300 images per minute. 
300 * 20 = 6000 images per chunk

Once we are done with a chunk. we tar and push it.  We can use the dvctar script for that. Will need some testing, but its okay. We push to the same location.  

Inference is ridiculously slow. on gpu. big batch sizes too. Ill nede to debug it tomorrow, for now im gonna let it run and eat some compute units

# fixing performance

Getting elev mask before and 0ing out stuff is not working (is it because of normalization?). I will need to write code which only takes tiles of interest then. Doing this is useful, but we need tiles of interest.  
This needs me to black out the parts which are not part of the scenery we want to analyse. now the problem is that the mask can be sparse (due to trees and shit. I dont know what it has assigned to sky portions or the sparse portions. I need to dilate this mask. Im concerned about losing stuff if i dilate like this. After dilating, we see a 2x speedup.  
Although, i feel its fine to remove black tiles by default, we might see some useful speedups.  
Now onto the problem of what is happening with the GPU. My own machine runs the model in 1.2 seconds, why is colab taking so long?


# Plan
- smallnet takes 1.5 seconds on my laptop. thats ~13 hours of runtime for the whole of delhi data, not a problem
- we do segformer on cloud, since that uses gpu very nicely
- once both are done, we combine and then use masks. Easy peasy im hoping
- then negmasks on local. together it will take two days, which i can use to finish more of the course, or analyse problems (i would prefer doing the course)





## fixing walls

We can't remove walls from mapi mask directly. Many times trash is at the lower end of walls and it needs to be removed. The major problems are with walls which are not colored well, or with walls with photos (this is very common).  How do i get good examples in this?   


- Current ive got a shit load of examples. What is the best way though. I need colored walls, painted walls, every kind of wall. Negmask should always go nah on this. Is it possible to pass the label of the region of interest from mapillary to the network to improve performance?  

- I could get walls from other cities and use those as an examples (cities outside india). 
- This would need me to come up with also walls with photos, or fungal stuff, etc  


What is the core problem?
- The model does not have awareness if something is a wall (mapillary does)
- actual trash lies around at the bottom of the walls, mapillary can also mark them as walls (the small things). So the model needs to consider everything


The model needs to know what a wall is and what an edge is. 

The first easiest fix is to find all wall interesections which are not just at the edge of the wall.  
- How do i define edge?
- amount of intersection with wall and non-wall
  - if 100% all good
  - this would also mark trash lying at the wall base as trash though (if it fully intersects)
  - wall base is higher side of the y coordinate (if wall bbox is y1->y2 and the trash bbox is y3->y4, if y3 is below 10% above y2, then it might be trash)


- how do i find these?
  - get crops which were detected by negmask
  - get the region masks of that
  - find the wall it intersects with, find its bbox. now this might be slightly hard
    - i could use region crops to get mask of individual wall, then find intersection with this mask
  - find the mask wiht highest intersection
    - then do the bbox rules



- another problem: the leaking of the mask is a real problem in smallnet. it connects to stuff outside the box after one point (is it because of strides? Should i actually run negmask specifically on each stride?). I will need to generate masks again though. (that is okay i guess, still will need to validate).  Even without strides it is connected lol

- the ds does not look bad, i can now train the model again.  




## train wall model
- I just put all the wall stuff in the new model dataset. now I need a way to see the differences between the older model and the newer model. I can create a new dataset for testing (tis a validation dataset, it uses the whole clean dir though). We run this using valid tfms on old and new models to see the differences
- The new model is not bad at finding walls. 
  - now i do some data trimming, for all wall images, we want to remove the ones which were predicted as trash adn were trash

- looking at all the walls, it does not look like it has enough context lols. tis sucks

Foviate shrink looks quite promising right now. next steps? 
- Regenerate the dataset using foviate shrink
 - I first need to generate the clean crops with full dimensions using crop level
 - then do foviate shrink
- blur pad ds would be fine with these new data points


- We have one more problem, what about taco?
- I can start with the latest model actually and hopefully it will work out (since it already knows how to "look")
- what do i do about TACO dataset? no clue right now. will need to look at the data to see how foviate shrinking works on it (its alreayd i think not very "street level" shit though, although since foviate still simply keeps the main object in focus, it might just work out)
- I also need to do the same thing for walls.


- Lets start with crop_level



# Foveated ds learning
- DS is foveated now. e
 - will add metrics in some time
 - have added taco and walls too

## larger kernels and noise
It might be useful in the end to add some noise (not random, deterministic) and add larger kernels in the first layer. The larger kernels are the context (that is the intuition). They will take care of area around the noise, the smaller ones would be very noisy around the noise, im not sure if that is good though (if they are noisy around the noise, how do we make sure to dampen that noise?). Just using it directly does not sound right (although if you believe that gradient descent will work out, you can say it would kinda figure out how to dampen the noise for areas outside the ROI. not sure. Since ROI has higher pixel values, it might work (note that since we are doing this, stuff in the dark would most likely not work).  

A better way would be to create a gaussian and an inverted gaussian mask, which is passed to the network. We multiply inverted mask to 9x9 outputs, and the gaussian mask to the 3x3 outputs. We might not even need simple step downer if we do this.  This is basically me using two different backbones for different purposes (I would use a much smaller backbone for 9x9 though). We merge them quite soon (not too late).  

Training schedule. Simplest? Do the standard one where we keep changing the noise at every iteration.  
The original schedule started with very random tfms where we had juust noise, juust blur, noise + blur, noise + step down.  We used to add noise everytime to a small training layer. It might be useful to do that firrrst
How do we train the 9x9 backbone though? I believe it might be useful to just train the backbone in the end, to get that extra squeeze out of accuracy I think.  

For now, my recall is quite bad. lets see what we can do about it.  

- I have a problem, currently i only have access to foveated datasets. I should keep track of larger datasets which are not foveated (originals from which we got the foveated ones). This makes sure that later we can use these again to get the correct dataset.  
- i have removed bad examples from this only lol.  
  - now lets get the walls examples from mapillary

- I downloaded 1 lakh+ images lol. Im hoping this is not a waste. But i cant run clip query on all of them
  - shiz i did not expect this lol.



- Walls: now do i separately detect it? This would make it a bit harder for the model
  - for now lets do the simple thing, detect only other
- some useful and interesting walls are counted as buildings ill need to see if i can fix that.  


- Im going to train a xresnet34 now. hopefully it works out lets see.
  - should i do the context thing, where i add deterministic noise everytime?
  - for now ignoring.


- new problem is mostly it recognising stuff near walls as wall due to the dataset (it might have mayn examples of that)
- it has also become bad at finding clusters
- i can do the trick of finding masks which are inside walls
- the other thing i can do is do one final pass without the mapi walls data

I tried removing the walls ds and doing a single fine tune with frozen layers. the model has become unstable.   


Now the next step is to simply retrain the main model with more selective walls data. I should also use a lot less data i think.  

- So the first model is still better (vanilla step edge).  
  - We know there is a context problem with that model though
- It would be useful to create a new model of the foveated type, with the same data as the original model
  - We will also transfer learn this 
  - And we will also do one more normal learning. 
  - Lets see what wins
- For walls, its best to get more data which is 100px above the bottom of the wall instead of all, the model seems to think all trash near walls is a problem

- This time, we make both L1 and final datasets
 - First make using the standard dataset we have (using source dir). Everything would be extremely high resolution (the original resolution of the data)
  - This is L1
 - We then implement shrinking later

- Two places
  - Crop level
  - TACO useful categories
- Now from L1 -> foveated (this would be a single function which takes L1 dir, uses labels to create correct dataset)

- After this we will also add walls (but not right now)


- Now i need to decide on something important. There are two paddings now. The first is the pad while making foveated images. in this case, the pad is the patch which wont be shrinked
- The other is the pad for step down. Step down def needs a pad (like 5px)
- Do we need a pad for foveated? I honestly dont think so

- Ohk, im adding some padding in foveated bbox (3px only). I dont want it to distort half objects ish? The other part is that i need to return the original bbox, not the padded one which can be used by step downer later 
- im keeping one difference, im also adding all taco data lol

# Foveated train

We load the main stable model ("md") as the initial model for now. Its show reports output without any training on foveated dataset
```
              precision    recall  f1-score   support

       other       0.73      0.91      0.81      1142
       trash       0.90      0.71      0.80      1332

    accuracy                           0.80      2474
   macro avg       0.82      0.81      0.80      2474
weighted avg       0.82      0.80      0.80      2474
```

Im using weighted random torch sampler now, for training. it gets more manual examples than the taco examples.  

A problem is that i dont like the 90% accuracy, it will still miss 1 in 10 garbage pieces in a photo. Is it a specific type? Im not sure. I need to release now though. For the walls problem, we can basically use the wall segmentation mask from mapi as a heuristic worst case. The last part in obvious failure modes is trees and sky (where sky meets trees, we need data from elevated vegetation map [the way we got from our walls data]. The easiest way is to add more examples of elevated vegetation blindly (some 1000 examples).   

This should conclude the current issues.  Now onto hawkers, LED markers and lane markers.  
Lane markers should be easy, take lane markers from outside india, run smallnet, find masks, mark as neg. We can do for india also assuming that the majority of the markers wont be problematic (although now the data would have bad examples).  

Hawkers need annotating. This I will add as a "problem" of the model.  
One important part is clusters of garbage, its best to annotate it, add some examples for smallnet. This should be a days effort (in the end though when we have enough data from multiple cities, I can also use google to get some photos of clusters of garbage and annotate them).  
It would also be best to take some first 100 example results from ddg as they might be the first things people test the model on (optimising demo).  


- get n samples for different queries from ddg, run the model, see results, fix annotations, retrain
- lane markers, LED markers (next problems)
- see how we can increase the accuracy of the model.  

Maybe I can find the maximal activations for lane markers and turn them off manually, and see how the model behaves on the final test set xD. Lets see.  This is an interesting experiment I can do I guess after training.  


Foveated model best reports (trained for some 10 iterations) (foveated-224/iter-6-xresnet18.pth)
```
              precision    recall  f1-score   support

       other       0.92      0.94      0.93      1142
       trash       0.95      0.93      0.94      1332

    accuracy                           0.94      2474
   macro avg       0.94      0.94      0.94      2474
weighted avg       0.94      0.94      0.94      2474

              Pred other  Pred trash
Actual other        1070          72
Actual trash          87        1245
```

The older best model
```
              precision    recall  f1-score   support

       other       0.97      0.94      0.95      1391
       trash       0.94      0.96      0.95      1254

    accuracy                           0.95      2645
   macro avg       0.95      0.95      0.95      2645
weighted avg       0.95      0.95      0.95      2645
```

This is much better than the foveated one though, I think I can train a bit more. I'll run `lr_find` and do some more iterations

The last model seems overfitted, im concered. I will redo the first model with the same learning rate for a single epoch, the single epoch seemed nice. New metrics

```
              precision    recall  f1-score   support

       other       0.93      0.94      0.94      1142
       trash       0.95      0.94      0.94      1332

    accuracy                           0.94      2474
   macro avg       0.94      0.94      0.94      2474
weighted avg       0.94      0.94      0.94      2474

              Pred other  Pred trash
Actual other        1076          66
Actual trash          81        1251
```
Still okay though, it actually had a lot of decrease in the training loss. Its F1 score is higher than the useful model which made this model go bad. Sucks.  It would be useful to see where these two models disagree.  
I think a new problem with the foveated models is that half filled objects are not well predicted (it skews them a lot since the bounding box is tight)
I tried finding differences between the new and the old model in valid set, the new seems to be doing better. Lets look at its performance in the main test.ipynb.  Baseline is still `md` model only.  

One thing though, I need to see how im doing the shrinking in infer set.  

In this model, step_down padding is 3. Shrink padding is also 3 (which is not factored in step down padding though)
Infer dataset first pads and then passes to foviate + step down. We set that to 3, foviate shrink = 0, step down = 0 and we good.   
I should train one more model maybe? Not sure. We need to do walls too btw, lets see what happens with that. I should also check with the md model.  

At this point, the model is actually not bad on the training set lol (as i had concluded the last time). I just need better data.  

The new model is quite bad on the test set i think, did we overfit? No wait, the older model is performing weirdly, did i change the settings a bit? Its not step downing lol. I need to get better faithful representations of ds which works with both foveate and older model. Its hurting experimentation.   
I had removed the older code which was creating a new crop centered at the main crop in blurpadinfer ds. This made it hard for the thing to work.  It seems the main model has learnt to look at the middle somehow (idk if there was code in training that did that). Does not matter. I need to just make sure inference works as expected.  I'll change the code for new model.  

Somethign has gone wrong with the inference of the older model....
Noooooooooooooooooooooooooooooooooo.  Matter hogya ; _ ;.   
hmmmmmmmmmmmmmmmmmmmmmm.  

Small bug, problem solved.  Both models seem reasonably similar in performance. They seem to have learnt the same things.  
Foveated one seems to have a better recall for trash (it actually looks like a slightly more stable version of the original model). I also need to check out the lesser iterations of the model which did not overfit.  

Seems like a reasonable win now I guess, now onto walls. We need 
- variety of wall types
- at different places in wall
- different sizes
- should not contain trash (tis the biggest problem)

The trash thing should be reasonably solved by using only 100px above wall category thing.  
The variety can be done by taking from different photos

The different places and sizes are a problem.  

I need to group by data by some metric somehow.  
- mask size
- mask position relative to the wall? -> tis hard to define? (i could use the center of their bbs to calculate direction but this is complicated)

Ohk. best thing to do is:
- Create the full dataset first using 100px constraint
- From that, create a hist with bins for mask size
- Sample randomly from the mask size ones, we would like to maximise the diversity (which means the number of photos used)

One useful thing would be to define baseline model results as actual results, and metrics to see where we are going right/wrong


Next steps:
- baseline model on test results
- Add wall samples


# Baseline test results dataset
- The dataset is generated using a reasonable baseline model.
- I would like to browse a report for each example where the new model differs from the baseline
  - how many regions were "different"
  - the amount of total pixel difference
  - these two should be good to sort the model results

- If we like the new model results, we create a new baseline.  
 - for now using the ensemble of two models as baseline

Im having gemini create a test report script. Will test baseline with latest model for testing.  
Test script is done, quite good actually. im happy.  

Both models are making mistakes in different placecs, Ill use the foveated model alone as baseline for now, it seems more "stable".  This is the "latest" model.  

I'll just train one more from the previous checkpoint (after one it, it started overfitting). We'll compare that with this model.   
- well i accidentally overwrote the previous model lol. nvm. The concept does not change.  


# Wall samples
Now about this, what can we do?
- First i need to extract the wall pair finding code. 


