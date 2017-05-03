# 機械学習プログラム集

機械学習のアルゴリズムを実装してみた集
様々な機械学習のアルゴリズムをここにまとめてます。

### k近傍法(knn.py)

分類器。
学習データ(素性ベクトルとそのラベル)をすべて保存しておいて、新しい入力に対して一番近いkこの訓練データを取ってきてその中で多数決でラベルを予測する。
タスクは[MNIST](http://yann.lecun.com/exdb/mnist/)の手書き数字認識です。

### L2正則化ガウスカーネル回帰(l2_gaussion_kernel.py)

L2正則化したガウスカーネル回帰

### スパースガウスカーネル回帰(sparse_gaussian_kernel.py)

L1正則化することでスパースにしたガウスカーネル回帰

### 多層ニューラルネット(mlp.py)

2層のニューラルネットをtensorflowやchainerを使わずにnumpyだけで実装しました。
タスクは[MNIST](http://yann.lecun.com/exdb/mnist/)の手書き数字認識で、一層目の活性化関数にReLU、最終層の活性化関数にsoftmaxを使っています。

### ナイーブベイズ

ナイーブベイズを使ったシンプルなテキスト分類。
データセットは[20Newsgroup](http://qwone.com/~jason/20Newsgroups/)。
精度はF値が0.7696ぐらい。

参考:
* http://aidiary.hatenablog.com/entry/20100618/1276877116