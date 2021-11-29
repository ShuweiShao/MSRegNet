# A multi-scale unsupervised learning for deformable image registration

Shuwei Shao,
Zhongcai Pei,
Weihai Chen,
Wentao Zhu,
Xingming Wu and
Baochang Zhang – **IJCARS 2021**

[[Link to paper]](https://link.springer.com/article/10.1007/s11548-021-02511-0)

## ✏️ 📄 Citation

If you find our work useful or interesting, please cite our paper:

```latex
@article{shao2021multi,
  title={A multi-scale unsupervised learning for deformable image registration},
  author={Shao, Shuwei and Pei, Zhongcai and Chen, Weihai and Zhu, Wentao and Wu, Xingming and Zhang, Baochang},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  pages={1--10},
  year={2021},
  publisher={Springer}
}
```
## 📈 Results

To train a model, run:
```bash
CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
python train_s2s_2d.py \
     <your_data_path> \
    --model vm2  \
    --batch_size 2  \
    --lambda 0.1 \
    --data_loss mse \
    --epochs 1500 \
    --steps_per_epoch 100 \
    --model_dir <your_dir_to_save_model> 
```
