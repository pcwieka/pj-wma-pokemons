# PJ_WMA_Pokemons

## Conda configuration

```
conda env create -f conda.yml
```

```
conda env update -f conda.yml
```

```
conda activate pja-wma
```

## Application

### Create graphical dataset

To run the application execute:

```
python create_graphical_datasets.py --source_dir ./pokemons/test --output_dir ./output/pokemons/test --image_format png --image_size 256
```

```
python create_graphical_datasets.py --source_dir ./pokemons/train --output_dir ./output/pokemons/train --image_format png --image_size 256
```

Add `--grayscale` if you want to convert images to grayscale.

### Tran model

```
python train_visual.py -d ./output/pokemons/train/ -m ./output/model/trained_model -c ./output/model/history.csv --batch_size 32 --train_split 0.7 --initial_learning_rate 0.001 --epochs 3
```



