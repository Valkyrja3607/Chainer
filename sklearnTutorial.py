from sklearn.datasets import load_boston

dataset = load_boston()


x = dataset.data
t = dataset.target

print(x.shape)
print(t.shape)

# データセットを分割する関数の読み込み
from sklearn.model_selection import train_test_split
# 訓練用データセットとテスト用データセットへの分割
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)


from sklearn.linear_model import LinearRegression
# モデルの定義
reg_model = LinearRegression()

# モデルの訓練
reg_model.fit(x_train, t_train)

# 訓練後のパラメータ w
print(reg_model.coef_)
# 訓練後のバイアス b
print(reg_model.intercept_)

# 精度の検証 (1に近い方が正確)
print(reg_model.score(x_train, t_train))

#推論(予測値)
print(reg_model.predict(x_test[:1]))
#入力に対する目標値
print(t_test[0])

#訓練済みモデルの性能を、テスト用データセットを使って決定係数を計算することで評価
print(reg_model.score(x_test, t_test))

#過学習？？
#以下改善

from sklearn.preprocessing import StandardScaler
#標準化のインスタンス作成
scaler = StandardScaler()
#データセットの各入力変数ごとの平均と分散の値を計算
scaler.fit(x_train)
# 平均
print(scaler.mean_)
# 分散
print(scaler.var_)
#標準化
x_train_scaled = scaler.transform(x_train)
x_test_scaled  = scaler.transform(x_test)

reg_model = LinearRegression()
# モデルの訓練
reg_model.fit(x_train_scaled, t_train)

# 精度の検証（訓練データ）
print(reg_model.score(x_train_scaled, t_train))
# 精度の検証（テストデータ）
print(reg_model.score(x_test_scaled, t_test))


#結果変わらず

from sklearn.preprocessing import PowerTransformer
#べき変換
scaler = PowerTransformer()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

reg_model = LinearRegression()
reg_model.fit(x_train_scaled, t_train)

# 訓練データでの決定係数
print(reg_model.score(x_train_scaled, t_train))
# テストデータでの決定係数
print(reg_model.score(x_test_scaled, t_test))

#改善した


from sklearn.pipeline import Pipeline

# パイプラインの作成 (scaler -> svr)
pipeline = Pipeline([
    ('scaler', PowerTransformer()),
    ('reg', LinearRegression())
])

# scaler および reg を順番に使用
pipeline.fit(x_train, t_train)

# 訓練用データセットを用いた決定係数の算出
print(pipeline.score(x_train, t_train))
# テスト用データセットを用いた決定係数の算出
linear_result = pipeline.score(x_test, t_test)
print(linear_result)

