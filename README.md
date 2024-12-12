# Intra-modal Consistency for Image-text Retrieval through Soft-label Distillation
## Oeverview
 We propose a novel ITR approach ICSD to implement soft-label distillation by transferring the knowledge from both internal intra-modal similarity and external intra-modal similarity to the cross-modal similarity. In addition, our designed (i) PDM greatly distinguishes different features to avoid erroneous matching information, and (ii) ISF organically fuses internal intra-modal similarity and external intra-modal similarity to effectively guide the model training.
![](model/Architecture.png)

## code structure
```bash
./configs
  ./vitb32    <= configs for vitb32
  ./vitl14    <= configs for vitl14

./datasets    <= datasets

./icsd
  ./models    <= models

evaluation.py <= evaluation

retrieval.py  <= retrieval

```


## Tasks
### Training
```bash
# For COCO:
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 15160 retrieval.py --config "./configs/vitb32/coco/icsd.yaml"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 15160 retrieval.py --config "./configs/vitl14_336/coco/icsd.yaml"

# For Flick30k:
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 15160 retrieval.py --config "./configs/vitb32/flickr/icsd.yaml"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 15160 retrieval.py --config "./configs/vitl14_336/flickr/icsd.yaml"

```

##