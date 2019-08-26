# MultiKE
Source code and datasets for IJCAI-2019 paper "_[Multi-view Knowledge Graph Embedding for Entity Alignment](https://www.ijcai.org/proceedings/2019/0754.pdf)_".

> **Note that**, the current version of MultiKE is implemented based on OpenEA (a tookit for embedding-based entity alignment, to be released). Thus, you would get errors due to the lack of dependency when directly running this code. We are still working on developing a stand-alone version. 

## Dataset
We use two datasets, namely DBP-WD and DBP-YG, which are based on DWY100K proposed in [BootEA](https://www.ijcai.org/proceedings/2018/0611.pdf). 

### DBP-WD and DBP-YG
In "data/BootEA_datasets.zip", we give the full data of the two datasets we used. Each dataset has the following files:

* ent_links: all the entity links without traning/test/valid splits;
* 631: entity links with traning/test/valid splits, contains three files, namely train_links, test_links and valid_links;
* rel_triples_1: relation triples in the source KG, list of triples like (h \t r \t t);
* rel_triples_2: relation triples in the target KG;
* attr_triples_1: attribute triples in the source KG;
* attr_triples_2: attribute triples in the target KG;
* entity_local_name_1: entity local names in the source KG, list of pairs like (entity \t local_name);
* entity_local_name_2: entity local names in the target KG;
* predicate_local_name_1: predicate local names in the source KG, list of pairs like (predicate \t local_name);
* predicate_local_name_2: predicate local names in the target KG.

### Raw datasets
The raw datasets of DWY100K can also be found [here](https://github.com/nju-websoft/BootEA/tree/master/dataset).

## Dependencies
* Python 3
* Tensorflow 1.x 
* Numpy
* OpenEA (Coming soon)

## Run

To run the experiments, use:

    bash run.sh -m mode -d dataset_folder_path
* mode: training mode, using either ITC or SSL;
* dataset_folder_path: the folder path of dataset to run.

For example, to run the experiments on DBP-WD with ITC mode, use:

    bash run.sh -m ITC -d BootEA_DBP_WD_100K/

> If you have any difficulty or question in running code and reproducing experiment results, please email to qhzhang.nju@gmail.com, zqsun.nju@gmail.com and whu@nju.edu.cn.

## Citation
If you use this model or code, please kindly cite it as follows:      

```
@inproceedings{MultiKE,
  author    = {Qingheng Zhang and Zequn Sun and Wei Hu and Muhao Chen and Lingbing Guo and Yuzhong Qu},
  title     = {Multi-view Knowledge Graph Embedding for Entity Alignment},
  booktitle = {IJCAI},
  pages     = {5429--5435},
  year      = {2019}
}
```
