## How to train source model
```sh
python train_source_model.py -n [exp-name] -l [num_layer] -m CNN -s 224 -r [path_to_dataset]
```
### Specify a GPU
In the `get_trainer(args)` function, set the parameter `device`:
        
- devices=[0,1] # Use both GPU
- devices=[1] # Use TITAN RTX 
- devices=[0] # Use RTX 3090 