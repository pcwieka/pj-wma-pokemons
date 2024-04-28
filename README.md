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

To run the application execute:

```
python create_graphical_datasets.py --source_dir ./pokemons/test --output_dir ./output/pokemons/test --image_format jpg --image_size 128
```

```
python create_graphical_datasets.py --source_dir ./pokemons/train --output_dir ./output/pokemons/train --image_format jpg --image_size 128
```

Add `--grayscale` if you want to convert images to grayscale.


