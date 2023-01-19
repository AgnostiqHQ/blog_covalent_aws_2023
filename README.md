# Covalent Machine Learning Study
## Supplementary Code

This repository contains the complete workflow script (`workflow.py`) corresponding to this post on AWS Blogs (*link to be included upon publication*).

The solution here is adapted from [this script](https://www.kaggle.com/code/mateuszbuda/brain-segmentation-pytorch/script), originally written by Mateusz Buda. The complete input data can be downloaded [here](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

## Instructions

Before running the workflow, ensure that you have valid AWS credentials and that AWS Batch is correctly configured. The input data should be uploaded to an S3 Bucket as a ZIP file, say `data_full.zip`.

After the above, proceed with the following:

* Install the required packages (includes Covalent): `pip install -r requirements.txt`.
* Run the shell command `covalent start` to start Covalent.

We recommend running the experiment through the `argeparse` CLI included in `workflow.py`. 

* Run `python workflow.py --help` to see CLI options that specify the scope of the experiment.

For example, we used the following command to run the experiment in the blog post linked above:

```
    python workflow.py -B 16 -E 20 -Z 64 128 192 256 -L 0.000075 0.0001 0.000125 -d data_full
```

Alternatively, just call the workflow function (`workflow`) normally by passing an arbitrary list of parameters.
