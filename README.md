# UndergraduateResearch
卒業研究の実験のために書いたソースコードです。<br>
全てのコードを1から書きました。<br>
(自動生成されるプログラムである[cython_wl_kernel.cpp](cython_wl_kernel.cpp)を除く)

論文[Neural Architecture Search using Bayesian Optimisation with Weisfeiler-Lehman Kernel](https://arxiv.org/abs/2006.07556v1)をベースとしています。

卒業論文は[こちら](https://drive.google.com/file/d/1C9roMHSPTDO5KhsnYaoSr7gMs4ZYpqzj/view)

# 実行環境
- CentOS 7.7-1908
- Python 3.6.8
- GCC 4.8.5

# 必要なモジュール
- Cython
- numpy 
- matplotlib
- yaml
- requests
- nats_bench
- tqdm

# 実行例
`python3 nasbowl2.py srcc -T 1500 --trials 10 --k_size_max 400 --eval_freq 10 --name sample`