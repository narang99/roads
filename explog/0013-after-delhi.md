# Looking at big trash finds

The metric is simple, the count of colored pixels. Sampled from 1000 images.  

Problems:
- Painted walls
- Railings/fences
- rock like walls
- sunlights
- blurry images (these need to be handled for sure)
- segmask model overflows a bit outside the actual object
- railway tracks
- smallnet has issues with collections of trash (this is one of the bigger painpoints, i think we can train it with the additional data from delhi)
- puddles of water
- rain water on car glass / lens glass
- hawkers

Now it seems its best to assign new labels to these and train the model on them (we do have a lot of unlabelled data for that case though). A new backbone can be trained which does that, then we trim the older data, then train the older backbone again. 

The easiest way to get more data is to simply get images where the model has mis predicted everything and add them. I now need faster data gathering stuff though. It is useful to train another model which has high other recall and check the differences too (the other way round)


- Next approach. train a model with high other recall, go through examples and add them to training set. The process of going through examples is quite slow. I am hoping to find better alternatives. But not sure.   
- The only way i can think of is using these high recall models, and going through categories.
  - It would be useful to add categories for data that we know is a problem in the model (or train a new one with those new categories)
  - I would need labelled data though. 
  - The easiest way is to get fence/wall data from masks, get preds which have those, then train the model on negatives from that.  
  - This would be quite useful i guess.  
  - Gather data from mapillary
    - walls, fences. We will mark everything as other in bulk
    - Lane markings
    - LED markers if possible (not sure though)
    - I need to see if the zebra colored curbs are kept separately

- for now I have run the high other recall model
- next step, find data where wall and fence is colored


- wall and fence can directly be marked as other fast
- rail track is present as a mapillary segment label. we find all rail tracks.  
