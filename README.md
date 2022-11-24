# Improving Legal Judgment Prediction through Reinforced Criminal Element Extraction (CEEN)
# Data preprocessing

1. Unzip the CAIL dataset and cail_thulac.npy in '**./data**'. 
   https://drive.google.com/file/d/1o52ND7q4ethnJXBw2KIYsqP08QKoMvz5/view?usp=sharing
2. Run '**data/tongji3.py**' to get '_{}cs.json' 
3. Run '**data/make_Legal_basis_data.py**' to get '{}_processed_thulac_Legal_basis.pkl'
4. Run '**data/small_cail_add4elements.py**' to annotate small CAIL. Run '**data/big_cail_add4elements.py**' to annotate big CAIL.

# Training

Warm-up model without reinforcement learning

1.python main1.py 

Loading model and using reinforcement learning

2.python main2.py

# References

If you find our work useful, please cite our paper as follows:

```
@article{Lyu-etal-2022-ceen,
  author    = {Yougang Lyu and
               Zihan Wang and
               Zhaochun Ren and
               Pengjie Ren and
               Zhumin Chen and
               Xiaozhong Liu and
               Yujun Li and
               Hongsong Li and
               Hongye Song},
  title     = {Improving legal judgment prediction through reinforced criminal element
               extraction},
  journal   = {Inf. Process. Manag.},
  volume    = {59},
  number    = {1},
  pages     = {102780},
  year      = {2022},
}
```
